from __future__ import annotations

import numpy as np

from baseline.mcl_solver import MCLSolver
from rl4co.envs.pc.evaluator import evaluate_groups
from third_party.markov_clustering import get_clusters, run_mcl


def test_mcl_core_finds_disconnected_cliques() -> None:
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = 1.0
    matrix[3:, 3:] = 1.0
    np.fill_diagonal(matrix, 0.0)

    result = run_mcl(matrix)
    clusters = get_clusters(result)

    assert set(clusters) == {(0, 1, 2), (3, 4, 5)}


def test_mcl_solver_repairs_constraint_conflicts() -> None:
    weights = np.array(
        [
            [0.0, 1.0, 0.9, 0.0],
            [1.0, 0.0, 0.8, 0.0],
            [0.9, 0.8, 0.0, 0.2],
            [0.0, 0.0, 0.2, 0.0],
        ]
    )
    material = np.array([0, 0, 1, 1])
    adjacency = weights > 0
    mat_var = (material[:, None] != material[None, :]) & adjacency
    zeros = np.zeros_like(adjacency, dtype=bool)
    inst = {
        "num_parts": 4,
        "W": weights,
        "material": material,
        "mat_var": mat_var,
        "maint_diff": zeros,
        "rel_motion": zeros,
        "isstandard": np.zeros(4, dtype=bool),
    }

    solver = MCLSolver()
    groups, _ = solver.solve(inst)
    metrics = evaluate_groups(groups, inst, gamma=0.5)

    assert sorted(node for group in groups for node in group) == [0, 1, 2, 3]
    assert metrics["feasible"] == 1.0


def test_mcl_solver_zero_graph_returns_singletons() -> None:
    zeros = np.zeros((3, 3), dtype=bool)
    inst = {
        "num_parts": 3,
        "W": np.zeros((3, 3), dtype=float),
        "mat_var": zeros,
        "maint_diff": zeros,
        "rel_motion": zeros,
        "isstandard": np.zeros(3, dtype=bool),
    }

    groups, _ = MCLSolver().solve(inst)
    assert groups == [[0], [1], [2]]


def test_mcl_solver_handles_four_material_types() -> None:
    material = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    weights = np.ones((8, 8), dtype=float) - np.eye(8, dtype=float)
    mat_var = material[:, None] != material[None, :]
    zeros = np.zeros_like(mat_var, dtype=bool)
    inst = {
        "num_parts": 8,
        "W": weights,
        "material": material,
        "mat_var": mat_var,
        "maint_diff": zeros,
        "rel_motion": zeros,
        "isstandard": np.zeros(8, dtype=bool),
    }

    groups, _ = MCLSolver().solve(inst)

    assert sorted(node for group in groups for node in group) == list(range(8))
    assert all(len(set(material[group])) == 1 for group in groups)
    assert evaluate_groups(groups, inst, gamma=0.5)["feasible"] == 1.0
