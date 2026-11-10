"""Markov Cluster Algorithm baseline for the PC benchmark.

Kim and Moon (2020) form a weighted part-relation graph, run MCL, and then
check whether identified modules are feasible for consolidation.  PC benchmark
instances do not contain their paper-specific SCCI values, so this adaptation
uses the benchmark relation-strength matrix ``W`` as the MCL similarity matrix.
The resulting clusters are conservatively split until they satisfy the same PC
group constraints used to evaluate CP-SAT, CPCCD, and NCO solutions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rl4co.envs.pc.evaluator import check_r3, group_feasible
from third_party.markov_clustering import (
    __version__ as markov_clustering_version,
    converged,
    get_clusters,
    iterate,
    prune,
    run_mcl,
)


@dataclass(frozen=True)
class MCLDiagnostics:
    iterations: int | None
    converged: bool


class MCLSolver:
    def __init__(
        self,
        *,
        expansion: int = 2,
        inflation: float = 2.0,
        loop_value: float = 1.0,
        iterations: int = 100,
        pruning_threshold: float = 0.001,
    ):
        self.expansion = int(expansion)
        self.inflation = float(inflation)
        self.loop_value = float(loop_value)
        self.iterations = int(iterations)
        self.pruning_threshold = float(pruning_threshold)
        self.implementation = f"vendored markov_clustering {markov_clustering_version}"

        self.last_raw_clusters: list[list[int]] = []
        self.last_groups: list[list[int]] = []
        self.last_diagnostics = MCLDiagnostics(iterations=0, converged=False)
        self.last_repair_splits = 0

    def solve(self, inst) -> tuple[list[list[int]], float]:
        start = time.perf_counter()
        n = int(inst["num_parts"])
        if n <= 0:
            self.last_raw_clusters = []
            self.last_groups = []
            return [], time.perf_counter() - start

        weights = np.asarray(inst["W"], dtype=float)[:n, :n].copy()
        weights[~np.isfinite(weights)] = 0.0
        weights = np.maximum(weights, 0.0)
        weights = 0.5 * (weights + weights.T)
        np.fill_diagonal(weights, 0.0)

        if float(weights.sum()) <= 0.0:
            raw_clusters = [[node] for node in range(n)]
            diagnostics = MCLDiagnostics(iterations=0, converged=True)
        else:
            result = run_mcl(
                weights,
                expansion=self.expansion,
                inflation=self.inflation,
                loop_value=self.loop_value,
                iterations=self.iterations,
                pruning_threshold=self.pruning_threshold,
            )
            check = iterate(result, self.expansion, self.inflation)
            if self.pruning_threshold > 0:
                check = prune(check, self.pruning_threshold)
            # The upstream API does not return its stopping iteration. Keep it
            # unknown instead of reporting a fabricated value.
            diagnostics = MCLDiagnostics(
                iterations=None,
                converged=bool(converged(result, check)),
            )
            raw_clusters = self._make_partition(get_clusters(result), n)

        groups = []
        for cluster in raw_clusters:
            repaired = self._repair_cluster(cluster, inst)
            groups.extend(repaired)
        groups = self._repair_r3(groups, inst)
        groups = sorted((sorted(group) for group in groups), key=lambda group: group[0])
        repair_splits = max(0, len(groups) - len(raw_clusters))

        flattened = [node for group in groups for node in group]
        if sorted(flattened) != list(range(n)):
            raise RuntimeError("MCL repair did not produce a complete node partition")
        if not all(group_feasible(group, inst) for group in groups):
            raise RuntimeError("MCL repair produced an infeasible group")
        if not check_r3(groups, inst):
            raise RuntimeError("MCL repair failed the assembly-access constraint")

        self.last_raw_clusters = [sorted(cluster) for cluster in raw_clusters]
        self.last_groups = groups
        self.last_diagnostics = diagnostics
        self.last_repair_splits = repair_splits
        return groups, time.perf_counter() - start

    @staticmethod
    def _make_partition(clusters: list[tuple[int, ...]], n: int) -> list[list[int]]:
        """Resolve rare overlaps/omissions in attractor clusters deterministically."""
        assigned: set[int] = set()
        partition: list[list[int]] = []
        ordered = sorted(
            ({int(node) for node in cluster if 0 <= int(node) < n} for cluster in clusters),
            key=lambda cluster: (-len(cluster), tuple(sorted(cluster))),
        )
        for cluster in ordered:
            available = sorted(cluster - assigned)
            if available:
                partition.append(available)
                assigned.update(available)
        partition.extend([[node] for node in range(n) if node not in assigned])
        return sorted(partition, key=lambda group: group[0])

    @staticmethod
    def _repair_cluster(cluster: list[int], inst) -> list[list[int]]:
        """Split one MCL cluster only as much as required for PC feasibility."""
        if group_feasible(cluster, inst):
            return [sorted(cluster)]

        weights = np.asarray(inst["W"], dtype=float)
        degree = {node: float(weights[node, cluster].sum()) for node in cluster}
        order = sorted(cluster, key=lambda node: (-degree[node], node))
        groups: list[list[int]] = []

        for node in order:
            candidates: list[tuple[float, int, list[int]]] = []
            for idx, group in enumerate(groups):
                trial = sorted(group + [node])
                if group_feasible(trial, inst):
                    affinity = float(weights[node, group].sum())
                    candidates.append((affinity, -idx, trial))
            if candidates:
                _, neg_idx, trial = max(candidates, key=lambda item: (item[0], item[1]))
                groups[-neg_idx] = trial
            else:
                groups.append([node])
        return sorted(groups, key=lambda group: group[0])

    @staticmethod
    def _repair_r3(groups: list[list[int]], inst) -> list[list[int]]:
        """Conservatively split groups if an optional global R3 checker rejects them."""
        repaired = [sorted(group) for group in groups]
        while not check_r3(repaired, inst):
            candidates = [idx for idx, group in enumerate(repaired) if len(group) > 1]
            if not candidates:
                break
            idx = max(candidates, key=lambda value: (len(repaired[value]), -value))
            group = repaired.pop(idx)
            repaired.extend([[node] for node in group])
        return repaired
