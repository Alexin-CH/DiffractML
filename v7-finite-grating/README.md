# v7-finite-grating

The main aim of DiffractML: approximate the full-wave response of a **finite,
defective** diffractive structure with fast periodic RCWA plus a learned
correction.

```
geometry ──▶ NN encoder ──▶ TORCWA (periodic supercell) ──▶ NN correction ──▶ R(λ), T(λ)
                                                                    ▲
                                              ground truth ─────────┘
                                              MEEP FDTD (finite, PML)
```

This directory currently contains the **M0** deliverable: the FDTD
ground-truth simulator and its validation.

## Why FDTD, and what "finite" means

TORCWA is periodic in `x` (lattice) and finite in `z` (layered stack). A real
grating is *truncated* in `x`: a finite number of ridges with edges, plus
fabrication defects. RCWA can only represent that by wrapping the finite
aperture in a large periodic supercell with air padding. The research question
is whether that approximation converges to the true finite-structure spectrum
as the padding grows.

To answer it we need an independent full-wave reference on the **non-periodic**
structure. MEEP (FDTD) with PML boundaries does exactly that. `R(λ)` and
`T(λ)` are **total power across a plane**, normalised by the incident power
measured in an empty-cell control run.

## Contents

- `src/grating.py` — finite-grating geometry and the defect model:
  - sine-corrugated ridges over a uniform TiN base,
  - **height jitter** (per-ridge amplitude multiplier),
  - **missing ridge** (a ridge removed down to the substrate).
- `src/tin_dispersion.py` — fits tabulated TiN `n,k` (labo / Beliaev) to a
  Drude–Lorentz model and builds the matching `meep.Medium`.
- `src/meep_grating.py` — the FDTD simulator: broadband (normal incidence) or
  fixed-angle (narrowband-per-λ) spectra, plus the empty-cell normalisation.
- `src/fresnel.py` — analytic planar-interface R/T, used to validate the
  source/flux machinery.
- `tests/` — geometry, dispersion-fit, Fresnel, and energy-conservation tests.
- `scripts/validate_m0.py` — end-to-end M0 validation with a pass/fail summary.
- `environment.yml` — dedicated MEEP conda environment.

## Environment

MEEP is distributed via conda (not pip) and needs a dedicated environment; do
**not** install it into the RCWA `mlai` env.

```bash
~/Apps/miniconda3/bin/conda env create -f v7-finite-grating/environment.yml
~/Apps/miniconda3/bin/conda activate meep-v7
```

## Usage

Fit the TiN dispersion once (cached to `data/`):

```bash
python src/tin_dispersion.py --dataset beliaev --n-osc 4 --wl-range 0.5 1.5
```

Simulate a finite grating (normal incidence, broadband pulse, s-polarisation):

```bash
python src/meep_grating.py \
    --n-ridges 10 --period 1.5 --amplitude 0.045 --base-thickness 0.03 \
    --wl-range 0.5 1.5 --n-freq 40 --pol s --resolution 40 \
    --output data/example_spectrum.csv
```

With defects (height jitter + one missing ridge):

```bash
python src/meep_grating.py --n-ridges 10 --jitter-std 0.15 --n-missing 1 --seed 0 ...
```

## Validation

```bash
python scripts/validate_m0.py
python -m pytest tests/ -v
```

`validate_m0.py` checks: (1) the Drude–Lorentz fit reproduces `n,k` and MEEP's
`Medium.epsilon(f)` matches the model; (2) a planar interface reproduces the
analytic Fresnel reflectance; (3) a lossless finite grating conserves energy;
(4) a real defective TiN grating produces a bounded spectrum.

## Next (M1)

Build the padded-supercell TORCWA baseline and benchmark it against this FDTD
reference, sweeping padding and Fourier order.
