"""Evaluate real PC application instances stored in the project XLSX format.

The workbook is treated strictly as input data.  It must contain the sheets
``Part Feature Matrix``, ``Relation Feature Matrix``, and
``Relative Weight Matrix`` used by the real-application workbooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict

from Evaluation.common import ROOT, save_dataframe, save_json
from Evaluation.evaluate_curriculum_range_baselines import final_checkpoint
from Evaluation.gamma_sweep import (
    evaluate_cpccd,
    evaluate_mcl,
    evaluate_nco_custom,
    evaluate_ortools,
)


METHODS = ("ortools", "cpccd", "mcl", "nco-curriculum")
METHOD_LABELS = {
    "ortools": "CP-SAT",
    "cpccd": "CPCCD",
    "mcl": "MCL",
    "nco-curriculum": "RL-based NCO",
}
COLORS = {
    "ortools": "#4e9bc6",
    "cpccd": "#f28e2b",
    "mcl": "#59a14f",
    "nco-curriculum": "#9467bd",
}

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances",
        nargs="+",
        type=Path,
        required=True,
        help="One or more PC application XLSX files.",
    )
    parser.add_argument("--gammas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--original-run", type=Path, default=None)
    parser.add_argument("--retry-run", type=Path, default=None)
    parser.add_argument("--retry-gammas", default="0.5,1.0")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--nco-batch-size", type=int, default=1)
    parser.add_argument("--ortools-time-limit", type=float, default=300.0)
    parser.add_argument("--ortools-workers", type=int, default=1)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/application_instances_material4"),
    )
    return parser.parse_args()


def _column_number(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha())
    result = 0
    for char in letters.upper():
        result = result * 26 + ord(char) - ord("A") + 1
    return result


def _read_xlsx_rows(path: Path) -> dict[str, dict[int, dict[int, str]]]:
    """Read cached XLSX cell values without adding an openpyxl dependency."""
    ns = {"m": MAIN_NS, "r": OFFICE_REL_NS, "p": PACKAGE_REL_NS}
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                for item in root.findall("m:si", ns)
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        result: dict[str, dict[int, dict[int, str]]] = {}

        for sheet in workbook.find("m:sheets", ns):
            name = sheet.attrib["name"]
            relation_id = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
            target = targets[relation_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            root = ET.fromstring(archive.read(target))
            rows: dict[int, dict[int, str]] = {}
            sheet_data = root.find("m:sheetData", ns)
            if sheet_data is None:
                result[name] = rows
                continue
            for row in sheet_data.findall("m:row", ns):
                values: dict[int, str] = {}
                for cell in row.findall("m:c", ns):
                    column = _column_number(cell.attrib["r"])
                    value_node = cell.find("m:v", ns)
                    inline = cell.find("m:is", ns)
                    cell_type = cell.attrib.get("t")
                    if cell_type == "s" and value_node is not None:
                        value = shared[int(value_node.text)]
                    elif cell_type == "inlineStr" and inline is not None:
                        value = "".join(
                            node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t")
                        )
                    else:
                        value = value_node.text if value_node is not None else ""
                    values[column] = value
                rows[int(row.attrib["r"])] = values
            result[name] = rows
    return result


def _numeric_matrix(
    rows: dict[int, dict[int, str]], start_row: int, size: int
) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            raw = rows.get(start_row + i, {}).get(2 + j, "0")
            matrix[i, j] = float(raw or 0.0)
    return matrix


def _section_start(
    rows: dict[int, dict[int, str]], title: str, *, header_offset: int = 2
) -> int:
    def normalize(value: object) -> str:
        return str(value).strip().lstrip("*•- ").strip()

    for row_number, values in rows.items():
        if normalize(values.get(1, "")) == normalize(title):
            for candidate in sorted(number for number in rows if number > row_number):
                if normalize(rows[candidate].get(1, "")) == "Part":
                    return candidate + 1
            return row_number + header_offset
    raise ValueError(f"Section not found in workbook: {title}")


def load_application_xlsx(path: Path) -> tuple[TensorDict, dict[str, object]]:
    sheets = _read_xlsx_rows(path)
    required = {"Part Feature Matrix", "Relation Feature Matrix", "Relative Weight Matrix"}
    missing = sorted(required - set(sheets))
    if missing:
        raise ValueError(f"{path.name}: missing sheets {missing}")

    part_rows = sheets["Part Feature Matrix"]
    part_records: list[tuple[int, dict[int, str]]] = []
    for row_number, values in part_rows.items():
        try:
            part_id = int(float(values.get(1, "")))
        except (TypeError, ValueError):
            continue
        part_records.append((part_id, values))
    part_records.sort(key=lambda item: item[0])
    num_parts = len(part_records)
    expected_ids = list(range(1, num_parts + 1))
    actual_ids = [part_id for part_id, _ in part_records]
    if actual_ids != expected_ids:
        raise ValueError(f"{path.name}: part IDs must be contiguous 1..N; got {actual_ids}")

    source_material = np.asarray(
        [int(float(values.get(3, 0))) for _, values in part_records], dtype=np.int64
    )
    maintfreq = np.asarray(
        [int(float(values.get(4, 0))) for _, values in part_records], dtype=np.int64
    )
    isstandard = np.asarray(
        [int(float(values.get(5, 0))) for _, values in part_records], dtype=np.int64
    )
    source_material_codes = sorted(int(value) for value in np.unique(source_material))
    if len(source_material_codes) > 4:
        raise ValueError(
            f"{path.name}: found {len(source_material_codes)} material types, "
            "but the trained model supports at most 4"
        )
    material_code_map = {
        source_code: encoded_code
        for encoded_code, source_code in enumerate(source_material_codes)
    }
    material = np.asarray(
        [material_code_map[int(value)] for value in source_material], dtype=np.int64
    )

    relation_rows = sheets["Relation Feature Matrix"]
    relative_rows = sheets["Relative Weight Matrix"]
    W = _numeric_matrix(relative_rows, _section_start(relative_rows, "Relative Weight Matrix", header_offset=2), num_parts)
    mat_var = _numeric_matrix(
        relation_rows, _section_start(relation_rows, "Material Type Difference"), num_parts
    )
    maint_diff = _numeric_matrix(
        relation_rows,
        _section_start(relation_rows, "Maintenance Frequency Difference"),
        num_parts,
    )
    rel_motion = _numeric_matrix(
        relation_rows, _section_start(relation_rows, "Relative Motion"), num_parts
    )

    for name, matrix in {
        "W": W,
        "mat_var": mat_var,
        "maint_diff": maint_diff,
        "rel_motion": rel_motion,
    }.items():
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{path.name}: {name} contains a non-finite value")
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-6):
            raise ValueError(f"{path.name}: {name} must be symmetric")
    if np.any(W < 0) or not np.allclose(np.diag(W), 0.0, atol=1e-8):
        raise ValueError(f"{path.name}: W must be nonnegative with a zero diagonal")

    # The trained model uses four material channels followed by maintenance and
    # standard-part flags.  Index 0 is the separator node used by the PC env.
    padded = num_parts + 1
    node_features = np.zeros((1, padded, 6), dtype=np.float32)
    node_features[0, 1:, :4] = np.eye(4, dtype=np.float32)[material]
    node_features[0, 1:, 4] = maintfreq
    node_features[0, 1:, 5] = isstandard
    edge_features = np.zeros((1, padded, padded, 3), dtype=np.float32)
    edge_features[0, 1:, 1:, 0] = mat_var
    edge_features[0, 1:, 1:, 1] = maint_diff
    edge_features[0, 1:, 1:, 2] = rel_motion

    def with_separator(vector: np.ndarray, fill: int = -1) -> np.ndarray:
        result = np.full((1, padded), fill, dtype=np.int64)
        result[0, 1:] = vector
        return result

    def matrix_with_separator(matrix: np.ndarray) -> np.ndarray:
        result = np.zeros((1, padded, padded), dtype=np.float32)
        result[0, 1:, 1:] = matrix
        return result

    valid_part_mask = np.ones((1, padded), dtype=bool)
    dataset = TensorDict(
        {
            "node_features": torch.from_numpy(node_features),
            "edge_features": torch.from_numpy(edge_features),
            "topology_id": torch.zeros(1, dtype=torch.long),
            "num_parts": torch.tensor([num_parts], dtype=torch.long),
            "material_type_count": torch.tensor(
                [int(np.unique(material).size)], dtype=torch.long
            ),
            "valid_part_mask": torch.from_numpy(valid_part_mask),
            "material": torch.from_numpy(with_separator(material)),
            "maintfreq": torch.from_numpy(with_separator(maintfreq)),
            "isstandard": torch.from_numpy(with_separator(isstandard)),
            "W": torch.from_numpy(matrix_with_separator(W)),
            "mat_var": torch.from_numpy(matrix_with_separator(mat_var)),
            "maint_diff": torch.from_numpy(matrix_with_separator(maint_diff)),
            "rel_motion": torch.from_numpy(matrix_with_separator(rel_motion)),
            "relation_consistent": torch.ones(1, dtype=torch.bool),
        },
        batch_size=[1],
    )
    metadata: dict[str, object] = {
        "source_file": str(path.resolve()),
        "num_parts": num_parts,
        "source_material_codes": source_material_codes,
        "material_code_map": {
            str(source): encoded for source, encoded in material_code_map.items()
        },
        "material_codes": sorted(int(value) for value in np.unique(material)),
        "material_type_count": int(np.unique(material).size),
        "maintenance_high_count": int(maintfreq.sum()),
        "standard_part_count": int(isstandard.sum()),
        "relation_edge_count": int(np.triu(W > 1e-8, 1).sum()),
        "relative_motion_edge_count": int(np.triu(rel_motion > 0.5, 1).sum()),
        "W_symmetric": bool(np.allclose(W, W.T)),
    }
    return dataset, metadata


def append_and_save(raw: pd.DataFrame, rows: list[dict], path: Path) -> pd.DataFrame:
    updated = pd.concat([raw, pd.DataFrame(rows)], ignore_index=True, sort=False)
    updated = updated.drop_duplicates(
        subset=["application", "gamma", "method"], keep="last"
    ).sort_values(["application", "gamma", "method"])
    save_dataframe(updated, path)
    return updated


def save_summary_and_plots(raw: pd.DataFrame, output: Path) -> None:
    summary = raw[
        [
            "application",
            "num_parts",
            "gamma",
            "method",
            "score",
            "num_groups",
            "feasible",
            "wall_elapsed_sec",
            "solver_status",
        ]
    ].sort_values(["application", "gamma", "method"])
    save_dataframe(summary, output / "summary.csv")

    for application, subset in raw.groupby("application", sort=True):
        gammas = sorted(subset["gamma"].astype(float).unique())
        methods = [method for method in METHODS if method in set(subset["method"])]
        x = np.arange(len(gammas), dtype=float)
        total_width = 0.78
        width = total_width / max(len(methods), 1)
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for offset, method in zip(offsets, methods):
            values = (
                subset[subset["method"] == method]
                .set_index("gamma")
                .reindex(gammas)["score"]
            )
            ax.bar(
                x + offset,
                values,
                width,
                label=METHOD_LABELS[method],
                color=COLORS[method],
                alpha=0.92,
            )
        ax.set_xticks(x, [f"{gamma:g}" for gamma in gammas])
        ax.set_xlabel("Gamma", fontsize=19)
        ax.set_ylabel("Score", fontsize=19)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=15)
        fig.tight_layout()
        fig.savefig(output / f"{application}_method_comparison.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    raw = pd.DataFrame() if args.restart or not raw_path.exists() else pd.read_csv(raw_path)
    gammas = [float(value) for value in args.gammas.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; available methods: {METHODS}")

    run_root = args.run_root.resolve() if args.run_root is not None else None
    original = args.original_run.resolve() if args.original_run is not None else None
    retry = args.retry_run.resolve() if args.retry_run is not None else None
    retry_gammas = {float(value) for value in args.retry_gammas.split(",") if value.strip()}
    checkpoints = (
        {
            gamma: final_checkpoint(
                gamma,
                run_root=run_root,
                original=original,
                retry=retry,
                retry_gammas=retry_gammas,
            )
            for gamma in gammas
        }
        if "nco-curriculum" in methods
        else {}
    )

    loaded: list[tuple[str, TensorDict, dict[str, object]]] = []
    metadata: dict[str, object] = {}
    used_names: set[str] = set()
    for path in args.instances:
        resolved = path.resolve()
        dataset, instance_metadata = load_application_xlsx(resolved)
        application = resolved.stem
        if application in used_names:
            raise ValueError(f"Duplicate inferred application name: {application}")
        used_names.add(application)
        loaded.append((application, dataset, instance_metadata))
        metadata[application] = instance_metadata
        print(
            f"Loaded {application}: N={instance_metadata['num_parts']}, "
            f"edges={instance_metadata['relation_edge_count']}, "
            f"materials={instance_metadata['material_codes']}"
        )

    if args.replace_existing and not raw.empty:
        applications = {item[0] for item in loaded}
        replace = (
            raw["application"].isin(applications)
            & raw["method"].isin(methods)
            & raw["gamma"].astype(float).map(
                lambda value: any(np.isclose(value, gamma) for gamma in gammas)
            )
        )
        raw = raw.loc[~replace].copy()
        save_dataframe(raw, raw_path)

    config_path = output / "config.json"
    existing_config: dict[str, object] = {}
    if config_path.exists() and not args.restart:
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_config = {}
    recorded_instances = dict(existing_config.get("instances", {}))
    recorded_instances.update(metadata)
    recorded_checkpoints = dict(existing_config.get("checkpoints", {}))
    if not raw.empty and "ckpt" in raw.columns:
        checkpoint_rows = raw.dropna(subset=["ckpt"])
        for _, row in checkpoint_rows.iterrows():
            recorded_checkpoints[f"{float(row['gamma']):g}"] = str(row["ckpt"])
    recorded_checkpoints.update(
        {f"{gamma:g}": str(path) for gamma, path in checkpoints.items()}
    )
    recorded_gammas = set(float(value) for value in existing_config.get("gammas", []))
    recorded_methods = set(str(value) for value in existing_config.get("methods", []))
    if not raw.empty:
        recorded_gammas.update(raw["gamma"].astype(float).tolist())
        recorded_methods.update(raw["method"].astype(str).tolist())
    recorded_gammas.update(gammas)
    recorded_methods.update(methods)
    save_json(
        config_path,
        {
            "instances": recorded_instances,
            "gammas": sorted(recorded_gammas),
            "methods": [method for method in METHODS if method in recorded_methods],
            "device": args.device,
            "ortools_time_limit": args.ortools_time_limit,
            "ortools_workers": args.ortools_workers,
            "checkpoints": recorded_checkpoints,
        },
    )

    for application, dataset, instance_metadata in loaded:
        for gamma in gammas:
            for method in methods:
                if not raw.empty:
                    done = (
                        (raw["application"] == application)
                        & np.isclose(raw["gamma"].astype(float), gamma)
                        & (raw["method"] == method)
                    )
                    if bool(done.any()):
                        print(f"Skip completed: {application}, gamma={gamma:g}, {method}")
                        continue
                print(f"Evaluate {application}, gamma={gamma:g}, method={method}")
                if method == "ortools":
                    rows = evaluate_ortools(
                        dataset,
                        1,
                        gamma,
                        args.ortools_time_limit,
                        args.ortools_workers,
                        "ortools",
                    )
                elif method == "cpccd":
                    rows = evaluate_cpccd(dataset, 1, gamma, 1234, None)
                elif method == "mcl":
                    rows = evaluate_mcl(dataset, 1, gamma)
                else:
                    rows = evaluate_nco_custom(
                        dataset, 1, gamma, checkpoints[gamma], args.device, args.nco_batch_size
                    )
                    for row in rows:
                        row["method"] = "nco-curriculum"
                        row["algorithm"] = "nco-curriculum"
                        row["curriculum_checkpoint"] = str(checkpoints[gamma])
                for row in rows:
                    row["application"] = application
                    row["source_file"] = instance_metadata["source_file"]
                    row["num_parts"] = instance_metadata["num_parts"]
                raw = append_and_save(raw, rows, raw_path)
                save_summary_and_plots(raw, output)

    save_summary_and_plots(raw, output)
    print(f"Saved application evaluation: {output}")


if __name__ == "__main__":
    main()
