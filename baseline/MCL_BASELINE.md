# PC-adapted Markov Cluster Algorithm baseline

This baseline adapts the Markov Cluster Algorithm (MCL) used by Kim and Moon
(2020) to this repository's PC benchmark.

## Input and output mapping

- Input graph: the PC instance relation-strength matrix `W`.
- MCL result: a complete partition of part indices.
- Feasibility repair: an MCL cluster is split only when required by the same
  material, maintenance, relative-motion, standard-part, connectivity, and R3
  constraints used by the other PC solvers.
- Score: `rl4co.envs.pc.evaluator.evaluate_groups`, exactly the same evaluator
  used for CP-SAT, CPCCD, and NCO.

The source paper constructs edge weights from its own SCCI disassembly data.
The generated PC instances do not contain SCCI. Therefore `W` is the explicit
benchmark-native replacement for SCCI similarity. Results should be described
as a **PC-adapted MCL baseline**, not an exact reproduction of the paper's
SCCI preprocessing.

## Default MCL settings

| Parameter | Default |
| --- | ---: |
| Expansion | 2 |
| Inflation | 2.0 |
| Self-loop value | 1.0 |
| Maximum iterations | 100 |
| Pruning threshold | 0.001 |

These defaults match the attached `markov_clustering` implementation. Version
`0.0.6.dev` is vendored under `third_party/markov_clustering` and imported
directly by `baseline/mcl_solver.py`. The only compatibility patch replaces
the optional scikit-learn L1 normalization with a mathematically equivalent
NumPy/SciPy fallback when scikit-learn is unavailable. Its MIT license is
preserved with the vendored source and in `baseline/THIRD_PARTY_NOTICES.md`.

## Evaluation example

```bat
cd /d C:\rl4co_pc

"C:\Users\jyr03\anaconda3\envs\pc_env\python.exe" Evaluation\gamma_sweep.py --num-parts 20 --seed 1234 --limit 100 --gammas "0,0.25,0.5,0.75,1.0" --methods "ortools,cpccd,mcl,nco-custom" --device cuda --nco-batch-size 100 --ortools-time-limit 300 --ortools-workers 1 --output-dir "outputs\evaluation\gamma_sweep\n20_material4_mcl"
```

The raw result includes MCL convergence, raw cluster, repair-split, timing,
group, feasibility, and objective-component columns. Material type count does
not require an MCL code change; it is read from each instance's constraint
matrices.
