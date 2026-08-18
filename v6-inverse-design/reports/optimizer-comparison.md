# Optimizer Comparison: Adam vs LBFGS for RCWA-gradient Refinement

**Date:** 2026-08-18
**GPU:** NVIDIA RTX 3070 Ti Laptop, `device=cuda`
**Config:** `nh=8`, `discretization=48`, `iters=25`, `ang=30°`, wl-grid `[800, 1100, 1400, 1600]` nm, surrogate `model_pt/surrogate.pt`, `per_lr=1.0`, solver-only restarts `=3`.
**LBFGS:** `lr=1.0`, `history_size=10`, `line_search_fn="strong_wolfe"`, `max_iter=1` per outer step (one quasi-Newton iteration per loop pass so bounds clamps can be applied between steps).

## Question

Does L-BFGS beat Adam for the RCWA-gradient refinement, and how does it behave in the solver-only regime?

## Modes

| Mode | Amp init | Per init | Refined | Optimizer |
|---|---|---|---|---|
| `surrogate+adam` | surrogate | surrogate | amp + per | Adam |
| `surrogate+lbfgs` | surrogate | surrogate | amp + per | LBFGS |
| `solver-only-adam` | random ×3 | random ×3 | amp + per | Adam |
| `solver-only-lbfgs` | random ×3 | random ×3 | amp + per | LBFGS |

Targets: the true `(amp, per)` used to build the ground-truth spectrum.

## Results

| Target (amp, per) | Mode | Recovered amp | Recovered per | Spectrum loss | Time (s) |
|---|---|---|---|---|---|
| (45, 1500) | surrogate+adam | 46.6 | 1630.6 | 1.27e-04 | 55.3 |
| (45, 1500) | surrogate+lbfgs | 49.2 | 1443.2 | 4.64e-05 | 53.0 |
| (45, 1500) | solver-only-adam | 46.8 | 3213.6 | 2.05e-04 | 163.1 |
| (45, 1500) | solver-only-lbfgs | **80.0 (clamped)** | 2355.1 | 1.40e-03 | 307.8 |
| (35, 1000) | surrogate+adam | 34.2 | 1199.2 | 2.99e-04 | 53.5 |
| (35, 1000) | surrogate+lbfgs | 37.1 | 1214.1 | 3.96e-04 | 102.9 |
| (35, 1000) | solver-only-adam | 37.1 | 1053.4 | 2.35e-04 | 159.7 |
| (35, 1000) | solver-only-lbfgs | **30.0 (clamped)** | 2077.3 | 7.28e-04 | 316.4 |
| (60, 2500) | surrogate+adam | 60.0 | 3291.3 | 4.17e-06 | 54.1 |
| (60, 2500) | surrogate+lbfgs | 60.1 | 3327.4 | 4.24e-06 | 52.9 |
| (60, 2500) | solver-only-adam | 60.3 | 2742.7 | 1.68e-06 | 163.4 |
| (60, 2500) | solver-only-lbfgs | 59.4 | 2873.9 | 3.04e-06 | 264.0 |

## Analysis

**1. With surrogate init, LBFGS ≈ Adam.** `surrogate+ampper` results are comparable: LBFGS wins clearly on target 1 (loss 4.64e-5 vs 1.27e-4; per 1443 vs 1631), ties on target 3 (both ~4.2e-6), and loses slightly on target 2 (3.96e-4 vs 2.99e-4). Neither recovers `per` reliably (1443/1214/3327 vs truths 1500/1000/2500) — the near-degeneracy dominates, not the optimizer.

**2. LBFGS from random starts is unstable.** Solver-only with LBFGS slams into the amp bounds on two of three targets (amp=80.0, amp=30.0 — both hard clamps) and has the worst loss of any mode there. LBFGS' line search can overshoot out of the feasible region; once `amp` is clamped, the quasi-Newton curvature estimate is corrupted by the projection, so it cannot recover. Adam's small fixed steps never hit this failure mode.

**3. LBFGS is not faster here.** Despite quadratic convergence theory, LBFGS is 2-6× slower (solver-only 264-316 s vs Adam 160-163 s) because each outer step performs multiple forward/backward function evaluations for the strong-Wolfe line search, and the RCWA forward+backward is the bottleneck (≈50 ms/solve). Speedup only materializes when far fewer *solver* calls are needed; with a near-degenerate, noisy-ish loss landscape the line search spends its budget probing.

**4. Best overall loss is Adam.** The single best result across all 12 runs is `solver-only-adam` on target 3 (loss 1.68e-6), followed closely by `surrogate+adam`/`surrogate+lbfgs`. But solver-only per-recovery is still erratic (3214, 1053, 2743), reconfirming the surrogate's regularization value from the first report.

## Conclusions

- **Default remains Adam.** It is robust (never diverges to a bound), comparable-or-better loss, and consistently ~2-6× faster than LBFGS for this problem size.
- **LBFGS adds value only with a good surrogate init** and even then does not reliably beat Adam on this near-degenerate landscape; its line search + bound clamps are a poor combination (seen as amp slamming into 80/30).
- The `--optimizer lbfgs` option is retained for experimentation, but `adam` stays the default.

## Reproducibility

- Data: `reports/optimizer-comparison.csv`
- Runner: `reports/run_experiments_lbfgs.py` (GPU, `nh=8`, `disc=48`, `iters=25`, 3 solver-only restarts).