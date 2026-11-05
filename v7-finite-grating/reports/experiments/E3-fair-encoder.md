# Experiment 3 — fair encoder comparison and redesign

## Objective

Settle two questions left open by Experiment 2:

1. Was the encoder loss an artefact of an **unfair comparison** (E2 trained on
   8 λ/step for ≤25 epochs, one full-batch step per geometry, and optimised
   absolute `R/T/A`) rather than a fair test?
2. Does a **redesigned encoder** — per-ridge tokens instead of global summary
   statistics, amplitude-only config — change the answer?

## Why a config head should not be expected to help

Two structural arguments motivate the result below.

- **Information.** The M2 feature vector already describes the five-ridge
  geometry: period, amplitude, base, ridge-scale mean/std/min, missing count
  and missing index. A global encoder is a nonlinear re-parameterisation of the
  same eight numbers, so it cannot add information the residual head does not
  already see. The ridge encoder adds the full per-ridge pattern; with at most
  one missing ridge on five, the summary statistics are already close to
  sufficient.
- **Redundancy.** The correction head is a function of `(R, T, A, λ, geometry)`.
  An effective-geometry config changes only `(R, T)`. The residual can absorb
  that change, so at a joint optimum the config is **unidentifiable**: the
  differentiable path can lower training error only through data efficiency,
  never through extra accuracy once the residual has converged.

## Protocol (all arms identical except the model)

- **Data:** `dataset_ideal_nh8.npz` — ideal unit cell, `nh=8`, 24 geometries ×
  16 wavelengths, MEEP-de-embedded targets. Split by geometry: 18 train /
  6 held out (the M2 split).
- **Input:** the stored M2 feature matrix `X` (13 static columns + RCWA
  `R, T, A`); encoders additionally consume geometry descriptors or per-ridge
  tokens.
- **Target / loss:** standardised residual `(dR, dT)`; `A = 1 − R − T` is
  derived. MSE, batch 64, AdamW `lr=1e-3`, `wd=1e-3`, ≤800 epochs, early
  stopping on held-out RMSE(A) (`patience 150`, evaluated every 10 epochs).
- **Arms:**
  - `fixed` — no encoder (the M2 control);
  - `global` — the Experiment 2 encoder (8-d descriptors → 16-d latent);
  - `ridge` — redesign: a DeepSets encoder over five tokens
    `[height scale, missing flag, position]`, pooled and combined with five
    global scalars.
- **Seeds:** 3 for the fair ladder and for n=12/18; 6 for n=6.
- **Sanity check:** the `fixed` arm reproduces the M2 baseline
  (0.0016 at seed 0 vs 0.0014–0.0016 for `train_m2.py`).

## Fair ladder (18 training geometries, mean ± std over 3 seeds)

| encoder | RMSE R | RMSE T | RMSE A | params |
|---|---:|---:|---:|---:|
| fixed | 0.0065 ± 0.0020 | 0.0072 ± 0.0021 | **0.0021 ± 0.0003** | 35 458 |
| global (E2) | 0.0055 ± 0.0009 | 0.0065 ± 0.0008 | 0.0024 ± 0.0001 | 57 234 |
| ridge (redesign) | 0.0053 ± 0.0017 | 0.0062 ± 0.0022 | **0.0021 ± 0.0005** | 90 130 |

The ridge encoder **ties** the fixed control on absorption (and is marginally
better on R/T, within seed noise). The original global encoder is still worse
on A despite a similar R/T. Neither encoder improves the metric that matters.

## Sample efficiency (held-out RMSE(A), mean ± std over seeds)

| n_train | fixed | global | ridge |
|---:|---:|---:|---:|
| 6 (6 seeds) | 0.0095 ± 0.0047 | 0.0077 ± 0.0037 | 0.0095 ± 0.0051 |
| 12 (3 seeds) | 0.0026 ± 0.0004 | 0.0026 ± 0.0003 | 0.0026 ± 0.0002 |
| 18 (3 seeds) | 0.0021 ± 0.0003 | 0.0024 ± 0.0001 | 0.0021 ± 0.0005 |

Medians at n=6: fixed 0.0082, global 0.0071, ridge 0.0079. The apparent
global-encoder edge in the low-data regime (0.0077 vs 0.0095) is smaller than
the seed spread, and two of the six n=6 runs never improved past the first
evaluation. At n=12 and n=18 all three arms are indistinguishable. This is a
**null result for data efficiency**, not evidence of a gain.

![scaling](e3_scaling.png)

## Interpretation

- **The E2 negative result survives a fair comparison.** Matched budget, loss,
  features and schedule do not rescue the encoder; the fair ladder is
  0.0021 (fixed / ridge) vs 0.0024 (global).
- **The redesign fixes the harm, not the ceiling.** Per-ridge tokens recover
  the fixed-model accuracy (E2's global encoder was 0.0025 here), but they do
  not beat it. This is consistent with the information argument: the residual
  already has the geometry.
- **No data-efficiency gain.** Even at six training geometries (96 samples),
  where an encoder might be expected to regularise, the ordering is inside the
  seed noise.
- **Nothing left to learn for forward prediction.** The corrected ideal-cell
  model is at ~0.002 absorption RMSE and the residual, not the encoder, is doing
  the work.

## Decision

- **Stop the encoder thread for forward prediction.** Keep the ideal unit cell
  at `nh=8` plus the residual correction.
- **Keep the differentiable solver** — its well-conditioned amplitude gradient
  is the right tool for *inverse design*, where the config is the unknown, not
  a nuisance parameter. That is the natural next use of E2/E3 machinery.
- Re-open the encoder question only if the input changes materially: raw
  per-ridge geometry without summary statistics, several missing ridges,
  out-of-distribution geometry shifts, or a much larger dataset.

## Reproduce

```bash
# rcwa-v7 env
python scripts/run_e3_suite.py --mode fair --mode scaling \
    --seeds 0 1 2 --n-train 6 12 18
python scripts/run_e3_suite.py --mode scaling --n-train 6 --seeds 3 4 5
python scripts/run_e3_suite.py --mode report
```

A single run (used for debugging and the in-loop path):

```bash
python scripts/train_e3.py --encoder ridge --epochs 800 --patience 150 --tag ridge
```

## Artifacts

- `reports/m2/e3/e3_report.md`, `e3_suite.json`, `e3_scaling.png`
- `reports/m2/e3/runs/` (one JSON per run; not tracked)
- code: `src/models_e3.py`, `scripts/train_e3.py`, `scripts/run_e3_suite.py`,
  `tests/test_e3.py`
