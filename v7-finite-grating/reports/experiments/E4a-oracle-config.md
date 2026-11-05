# E4a — does an equivalent periodic config exist?

## Objective

Experiment 3 showed that a learned encoder does not beat a fixed-geometry
residual — but it could not tell whether that is because the encoder failed or
because **no single periodic configuration can represent a finite defective
grating**. E4a answers the existence question directly: for each geometry,
optimise a four-parameter effective unit cell against MEEP and compare the best
achievable error with the fixed ideal cell and the padded supercell.

This is an **oracle upper bound** for any config-predicting encoder: an encoder
cannot do better than the per-geometry optimum.

## Effective-periodic family

A unit cell with four knobs:

| knob | range (× nominal) | role |
|---|---|---|
| period | 0.60–1.60 | shifts diffraction orders |
| amplitude | 0.10–2.50 | corrugation height |
| base | 0.20–2.00 | uniform TiN thickness |
| **fill** | 0.20–1.00 | duty cycle: TiN over `fill·period`, bare substrate elsewhere |

The **fill** knob is the one Experiment 2/3 lacked. A missing ridge is a
coverage defect, which an amplitude or period change cannot express.

Protocol:

- differentiable ideal-cell RCWA, `nh=8`, `nx=200`, 40 layers, the same TiN
  dispersion as the references;
- optimise on 5 wavelengths (Adam, 14 steps, 2 restarts, best point tracked
  along each restart);
- select the winner on the **full 16-λ absorption** error, with the nominal
  ideal cell always included as a candidate — so the oracle can never be worse
  than the baseline;
- stratified 8-geometry subset: 4 jitter-only, 4 missing-ridge.

## Results (raw RMSE(A), mean ± std over geometries)

| subset | n | ideal cell | oracle config | supercell (nh=32) | oracle gain |
|---|---:|---:|---:|---:|---:|
| all | 8 | 0.0207 ± 0.0146 | 0.0171 ± 0.0121 | 0.0166 ± 0.0034 | +17% |
| jitter only | 4 | 0.0063 | **0.0058** | 0.0135 | +8% |
| missing ridge | 4 | 0.0351 | 0.0284 | **0.0198** | +19% |

The aggregate "all" row is misleading: the two defect classes have **opposite
orderings**. Stratify before drawing conclusions.

### Per geometry

| spec | miss | miss idx | p/p0 | a/a0 | b/b0 | fill | ideal | oracle | supercell |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 000 | 1 | 2 | 1.600 | 1.300 | 1.613 | 0.703 | 0.0358 | 0.0300 | 0.0182 |
| 005 | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0325 | 0.0326 | 0.0211 |
| 013 | 1 | 3 | 1.600 | 1.218 | 1.948 | 0.717 | 0.0396 | **0.0184** | 0.0182 |
| 022 | 1 | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0325 | 0.0326 | 0.0216 |
| 003 | 0 | — | 0.752 | 0.808 | 1.295 | 0.924 | 0.0066 | 0.0053 | 0.0142 |
| 009 | 0 | — | 0.861 | 1.162 | 1.162 | 0.916 | 0.0082 | 0.0074 | 0.0147 |
| 015 | 0 | — | 1.000 | 1.000 | 1.000 | 1.000 | 0.0047 | 0.0047 | 0.0119 |
| 019 | 0 | — | 1.000 | 1.000 | 1.000 | 1.000 | 0.0057 | 0.0058 | 0.0131 |

## Interpretation

- **Jitter-only: an equivalent periodic config exists.** The ideal/oracle cell
  is already *better* than the padded supercell (0.0058 vs 0.0135), consistent
  with Experiment 1's finding that the supercell is under-resolved at
  `padding=1.0, nh=32`. There is nothing for a richer config to add.
- **Missing ridge: only partially.** The oracle closes ~19% of the error on
  average, but almost all of that is a **single geometry**: spec 013 drops from
  0.0396 to 0.0184, matching the supercell (0.0182). The other three
  missing-ridge geometries gain nothing and the optimiser falls back to the
  nominal cell. A localised hole is not representable by one period with an
  effective medium; whether an approximation works is geometry-dependent.
- **The effective config is not unique.** Different `(p, a, b, fill)` combine
  to similar spectra (specs 000 and 013 take very different routes), which is
  another reason an encoder would struggle to pin one down.
- **Encoder-only ceiling.** An encoder can at best recover the oracle. For
  missing ridges that ceiling is ~0.028, still 43% above the supercell's
  0.020. So encoder-only training cannot close the hard class — matching E3's
  null result for a structural reason.
- **This points to the Fourier / learned low-order supercell.** It can
  represent the *localised* pattern inside a padded cell, unlike a single
  effective period, while keeping the solve at `nh=8`.

## Caveats

- 8 of 24 geometries (stratified); the full set is a ~50 min run.
- 14 Adam steps × 2 restarts per geometry; the oracle is a lower bound on the
  achievable error and could improve with more optimisation.
- Two geometries hit the period bound `p/p0=1.6`, so the family may be
  constrained there.
- The `fill=1` mask leaves a 1–2 grid-point soft edge (≈1e-4 in R), which is
  why the nominal fallback is occasionally 0.0001 above the ideal baseline.

## Reproduce

```bash
# rcwa-v7 env
python scripts/e4a_oracle_config.py \
    --specs 0 3 5 9 13 15 19 22 --iters 14 --restarts 2 --n-fit-wl 5 \
    --tag oracle_subset
```

## Artifacts

- `reports/m2/e4a/oracle_subset.{md,json,png}`
- code: `scripts/e4a_oracle_config.py`, `src/rcwa_diff.py` (`fill` parameter)
- tests: `tests/test_e4a.py`
