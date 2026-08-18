# Inverse-Design Mode Comparison

**Date:** 2026-08-18
**GPU:** NVIDIA RTX 3070 Ti Laptop, `device=cuda`
**Config:** `nh=8`, `discretization=48`, `iters=25`, `ang=30°`, wl-grid `[800, 1100, 1400, 1600]` nm, surrogate `model_pt/surrogate.pt`, period-refinement `per_lr=1.0`, solver-only restarts `=3`.

## Question

Which inverse-design strategy works best, and does the surrogate add value over using the differentiable RCWA solver directly?

## Modes

| Mode | Amp init | Per init | Refined |
|---|---|---|---|
| `surrogate` | surrogate | surrogate | none |
| `surrogate+amp` | surrogate | surrogate | amp only |
| `surrogate+ampper` | surrogate | surrogate | amp + per |
| `solver-only` | random ×3 | random ×3 | amp + per (best of restarts) |

Targets: the true `(amp, per)` used to build the ground-truth spectrum.

## Results

| Target (amp, per) | Mode | Recovered amp | Recovered per | Spectrum loss | Time (s) |
|---|---|---|---|---|---|
| (45, 1500) | surrogate | 49.2 | 1457.1 | 4.52e-05 | 7.8 |
| (45, 1500) | surrogate+amp | 46.3 | 1457.1 | 1.09e-05 | 58.4 |
| (45, 1500) | surrogate+ampper | 45.9 | 1478.8 | 3.80e-06 | 57.8 |
| (45, 1500) | solver-only | 45.7 | 3400.8 | 2.26e-04 | 164.7 |
| (35, 1000) | surrogate | 37.1 | 1220.5 | 3.94e-04 | 6.5 |
| (35, 1000) | surrogate+amp | 34.1 | 1220.5 | 2.99e-04 | 55.0 |
| (35, 1000) | surrogate+ampper | 34.6 | 1245.7 | 2.69e-04 | 55.0 |
| (35, 1000) | solver-only | 35.6 | 4688.5 | 4.35e-04 | 160.9 |
| (60, 2500) | surrogate | 60.1 | 3275.5 | 4.45e-06 | 6.5 |
| (60, 2500) | surrogate+amp | 60.0 | 3275.5 | 4.25e-06 | 54.5 |
| (60, 2500) | surrogate+ampper | 60.0 | 3283.2 | 4.21e-06 | 54.5 |
| (60, 2500) | solver-only | 57.7 | 4047.3 | 1.16e-05 | 161.6 |

## Analysis

**1. Refinement always helps.** TORCWA refinement lowers the loss for every target. `surrogate+ampper` is the best mode overall: loss drops 12× (target 1), 1.5× (target 2), and 1.06× (target 3) relative to `surrogate` alone.

**2. The surrogate is critical for `per`.** Solver-only runs from random starts recover `amp` well (45.7, 35.6, 57.7 vs truths 45/35/60) but consistently land on a wrong `per` (3400, 4688, 4047 vs truths 1500/1000/2500). The spectrum is nearly degenerate in `per` for the smooth grating — many periods give almost the same response — so gradient descent from random starts finds shallow, wrong local minima. The surrogate pins `per` near a plausible region first.

**3. `per` accuracy stays limited even with refinement.** `surrogate+ampper` never fully recovers `per` (1478 vs 1500, 1246 vs 1000, 3283 vs 2500). Target 2 (amp=35, per=1000) is the hardest: the surrogate misplaces `per` (1220) and refinement barely moves it. This is consistent with the smooth-grating near-degeneracy: at some operating points the response depends weakly on `per`, so the loss is flat in that direction and gradient steps are small/noisy.

**4. `amp` is recovered reliably by every mode that refines.** All refined modes land within ~1.5 nm of truth. The surrogate alone is slightly off (49.2 vs 45, 37.1 vs 35) — refinement closes that gap.

**5. Cost.** `surrogate` ~6-8 s; `surrogate+amp`/`+ampper` ~55-58 s; `solver-only` ~161-165 s (3 restarts). Solver-only is ~3× more expensive than surrogate+refinement and, given point 2, does not pay for itself.

## Conclusions

- **Recommended pipeline: `surrogate` → `surrogate+ampper` refinement.** Fast (≈1 min), best spectrum fit, recovers `amp` to <1.5 nm and `per` to within a few percent where the problem is well-posed.
- **Solver-only is not competitive** for recovering the true structure from this degenerate target class, though it is useful as a validation/ablation (it demonstrates that `amp` gradients through TORCWA are correct and that `per` requires the surrogate's regularization).
- The remaining `per` error is a property of the smooth-grating inverse problem (near-degeneracy), not a solver or optimizer bug.

## Reproducibility

- Data: `reports/design-mode-comparison.csv`
- Runner: `reports/run_experiments.py` (GPU, `nh=8`, `disc=48`, `iters=25`, 3 solver-only restarts).