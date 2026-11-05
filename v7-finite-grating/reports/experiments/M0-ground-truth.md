# M0 — validated FDTD ground truth

## Objective

Establish a trustworthy full-wave reference for the reflected/transmitted
spectra of a **finite, defective** 1D grating before using it to judge RCWA.

## Setup

- **Solver:** MEEP 2D FDTD with PML absorbing boundaries. The physical x–z
  problem (invariant along physical y) is mapped to MEEP x–y.
- **Structure:** a finite aperture of N sine-corrugated ridges on a semi-infinite
  substrate (n = 1.5), in air. Defects: per-ridge height jitter and/or a missing
  ridge.
- **Material:** TiN fitted from tabulated `n,k` (Beliaev) to a Drude–Lorentz
  model; the same fit is used by RCWA, so both solvers share one dispersion.
- **Output definition:** total power across horizontal planes, normalised by an
  empty-cell control run (the incident power).
- **Resolution:** the 30–50 nm TiN features require ≈140–150 px/µm (see below).

## Gates and results

### 1. TiN fit and MEEP medium

| quantity | value |
|---|---:|
| fit band | 0.500–1.497 µm |
| `n` relative RMSE | 0.0002 |
| `k` relative RMSE | 0.0001 |
| `meep.Medium.ε(f)` vs analytic model | 5.2e-16 |

### 2. Planar interface vs analytic Fresnel

| pol | angle | MEEP R | Fresnel R | abs. error |
|---|---:|---:|---:|---:|
| s | 0° | 0.0387 | 0.0400 | 0.0013 |
| s | 30° | 0.0551 | 0.0578 | 0.0027 |
| p | 0° | 0.0391 | 0.0400 | 0.0009 |
| p | 30° | 0.0248 | 0.0252 | 0.0004 |

### 3. Energy conservation (lossless grating)

- max |R + T − 1| = **0.0045**.

### 4. Real defective TiN grating

- A bounded spectrum with absorption rising toward shorter wavelengths (TiN
  loss), saved in `reports/m0_defective_grating.csv`.

## Resolution finding

A 50 nm slab is under-resolved at 60–100 px/µm (R biased high); at 150 px/µm it
matches the transfer-matrix result. Use **resolution ≥ ~140** for references.
Absorption is less sensitivity-sensitive than R: res=100 differs from res=140 by
<2% in A.

## Reproduce

```bash
# meep-v7 env
python scripts/validate_m0.py --resolution 50
python -m pytest tests/
```

## Artifacts

- `reports/m0_defective_grating.csv`
- `data/tin_fit_beliaev.json`
- code: `src/grating.py`, `src/tin_dispersion.py`, `src/meep_grating.py`,
  `src/fresnel.py`
