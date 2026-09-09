from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rl4co.envs.pc.evaluator import SCORE_EPS, group_feasible, relation_adjacency


@dataclass
class ORToolsPCResult:
    groups: list[list[int]]
    elapsed_sec: float
    status: str
    objective_value: float | None
    best_objective_bound: float | None


class ORToolsPCSolver:
    """Time-limited CP-SAT solver for PC grouping.

    Pairwise conflicts are modeled directly, and each non-singleton group is
    constrained to be connected over the physical relation graph.
    """

    def __init__(
        self,
        gamma: float = 0.3,
        time_limit_sec: float = 120.0,
        num_workers: int = 8,
        coefficient_scale: int = 1_000_000,
    ):
        self.gamma = float(gamma)
        self.time_limit_sec = float(time_limit_sec)
        self.num_workers = int(num_workers)
        self.coefficient_scale = int(coefficient_scale)
        self.last_status: str | None = None
        self.last_objective_value: float | None = None
        self.last_best_objective_bound: float | None = None

    def solve(self, inst) -> tuple[list[list[int]], float]:
        result = self.solve_with_status(inst)
        return result.groups, result.elapsed_sec

    def solve_with_status(self, inst) -> ORToolsPCResult:
        try:
            from ortools.sat.python import cp_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OR-Tools is not installed. Install it in this environment with: "
                "pip install ortools"
            ) from exc

        start = time.perf_counter()
        n = int(inst["num_parts"])
        w = np.asarray(inst["W"], dtype=float)
        strengths = np.sum(w, axis=1)
        two_m = float(np.sum(strengths))
        if n <= 0:
            return ORToolsPCResult([], 0.0, "EMPTY", None, None)
        if two_m <= SCORE_EPS:
            groups = [[i] for i in range(n)]
            return ORToolsPCResult(groups, time.perf_counter() - start, "ZERO_WEIGHT", 0.0, 0.0)

        model = cp_model.CpModel()
        z = [[model.NewBoolVar(f"z_{i}_{g}") for g in range(n)] for i in range(n)]
        u = [model.NewBoolVar(f"u_{g}") for g in range(n)]
        y = {
            (i, j, g): model.NewBoolVar(f"y_{i}_{j}_{g}")
            for i in range(n)
            for j in range(i + 1, n)
            for g in range(n)
        }

        for i in range(n):
            model.Add(sum(z[i][g] for g in range(n)) == 1)
            for g in range(i + 1, n):
                model.Add(z[i][g] == 0)

        for g in range(n):
            for i in range(n):
                model.Add(z[i][g] <= u[g])
            model.Add(sum(z[i][g] for i in range(n)) >= u[g])
            model.Add(z[g][g] == u[g])

        compatible_pair = self._compatible_pair_matrix(inst)
        adj = relation_adjacency(inst)
        isstandard = np.asarray(inst.get("isstandard", np.zeros(n, dtype=bool))).astype(bool)
        for i in range(n):
            for j in range(i + 1, n):
                pair_allowed = bool(
                    compatible_pair[i, j]
                    and not isstandard[i]
                    and not isstandard[j]
                )
                for g in range(n):
                    var = y[(i, j, g)]
                    model.Add(var <= z[i][g])
                    model.Add(var <= z[j][g])
                    model.Add(var >= z[i][g] + z[j][g] - 1)
                    if not pair_allowed:
                        model.Add(var == 0)

        self._add_connectivity_constraints(model, z, u, y, adj, n)

        terms = []
        for g in range(n):
            for i in range(n):
                coeff = -self.gamma * (float(strengths[i]) ** 2) / (two_m * two_m)
                int_coeff = int(round(coeff * self.coefficient_scale))
                if int_coeff:
                    terms.append(int_coeff * z[i][g])

            for i in range(n):
                for j in range(i + 1, n):
                    observed_coeff = 2.0 * float(w[i, j]) / two_m
                    expected_coeff = self.gamma * 2.0 * float(strengths[i]) * float(strengths[j]) / (
                        two_m * two_m
                    )
                    int_coeff = int(round((observed_coeff - expected_coeff) * self.coefficient_scale))
                    if int_coeff:
                        terms.append(int_coeff * y[(i, j, g)])

        model.Maximize(sum(terms) if terms else 0)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_sec
        solver.parameters.num_search_workers = self.num_workers
        status = solver.Solve(model)
        status_name = solver.StatusName(status)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            groups = [[i] for i in range(n)]
        else:
            groups = []
            for g in range(n):
                group = [i for i in range(n) if solver.Value(z[i][g]) == 1]
                if group:
                    groups.append(group)
            groups = self._repair_if_needed(groups, inst)

        elapsed = time.perf_counter() - start
        objective_value = None
        best_bound = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            objective_value = float(solver.ObjectiveValue()) / float(self.coefficient_scale)
            best_bound = float(solver.BestObjectiveBound()) / float(self.coefficient_scale)

        self.last_status = status_name
        self.last_objective_value = objective_value
        self.last_best_objective_bound = best_bound
        return ORToolsPCResult(groups, elapsed, status_name, objective_value, best_bound)

    @staticmethod
    def _add_connectivity_constraints(model, z, u, y, adj: np.ndarray, n: int) -> None:
        directed_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if bool(adj[i, j]):
                    directed_edges.append((i, j))
                    directed_edges.append((j, i))

        if not directed_edges:
            return

        capacity = max(n - 1, 1)
        for g in range(n):
            flow = {
                (i, j): model.NewIntVar(0, capacity, f"flow_{g}_{i}_{j}")
                for i, j in directed_edges
            }
            for i, j in directed_edges:
                a, b = (i, j) if i < j else (j, i)
                model.Add(flow[(i, j)] <= capacity * y[(a, b, g)])

            group_size = sum(z[i][g] for i in range(n))
            root_out = sum(flow[(g, j)] for j in range(n) if (g, j) in flow)
            root_in = sum(flow[(j, g)] for j in range(n) if (j, g) in flow)
            model.Add(root_out - root_in == group_size - u[g])

            for i in range(n):
                if i == g:
                    continue
                in_flow = sum(flow[(j, i)] for j in range(n) if (j, i) in flow)
                out_flow = sum(flow[(i, j)] for j in range(n) if (i, j) in flow)
                model.Add(in_flow - out_flow == z[i][g])

    @staticmethod
    def _compatible_pair_matrix(inst) -> np.ndarray:
        shape = np.asarray(inst["W"]).shape
        mat_var = np.asarray(inst.get("mat_var", np.zeros(shape, dtype=bool))).astype(bool)
        maint_diff = np.asarray(inst.get("maint_diff", np.zeros(shape, dtype=bool))).astype(bool)
        rel_motion = np.asarray(inst.get("rel_motion", np.zeros(shape, dtype=bool))).astype(bool)
        return ~(mat_var | maint_diff | rel_motion)

    @staticmethod
    def _repair_if_needed(groups: list[list[int]], inst) -> list[list[int]]:
        repaired: list[list[int]] = []
        for group in groups:
            if group_feasible(group, inst):
                repaired.append(sorted(group))
                continue
            for node in group:
                repaired.append([int(node)])
        return repaired
