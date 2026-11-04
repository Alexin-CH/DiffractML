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

This directory is the **M0** deliverable: the FDTD ground-truth simulator and
its validation. No ML yet — first we make the reference trustworthy.

## Why FDTD, and what "finite" means

TORCWA is periodic in `x` (lattice) and finite in `z` (layered stack). A real
grating is *truncated* in `x`: a finite number of ridges with edges, plus
fabrication defects. RCWA can only represent that by wrapping the finite
aperture in a large periodic supercell with air padding. The research question
is whether that approximation converges to the true finite-structure spectrum
as the padding grows.

To answer it we need an independent full-wave reference on the **non-periodic**
structure. MEEP (FDTD) with PML boundaries does exactly that. (The conversation
that motivated the project called this the "FEM reference"; FDTD is the
pragmatic open-source full-wave choice for finite aperiodic structures — it
solves the same Maxwell problem without the unit-cell assumption.)

`R(λ)` and `T(λ)` are **total power across a plane**: the power flux integrated
over a plane above/below the structure, divided by the incident power measured
in an empty-cell control run. This is the quantity the RCWA baseline must
reproduce (sum over all propagating diffraction orders).

## Contents

- `src/grating.py` — finite-grating geometry and the defect model:
  - sine-corrugated ridges over a uniform TiN base,
  - **height jitter** (per-ridge amplitude multiplier),
  - **missing ridge** (a ridge removed down to the substrate).
- `src/tin_dispersion.py` — fits tabulated TiN `n,k` (labo / Beliaev) to a
  Drude–Lorentz model and builds the matching `meep.Medium`. MEEP cannot ingest
  tabulated `n,k` directly; it needs an analytic oscillator model.
- `src/meep_grating.py` — the FDTD simulator: broadband (normal incidence) or
  fixed-angle (narrowband-per-λ) spectra, plus the empty-cell normalisation.
  Also `run_grating_periodic` for a Bloch-periodic unit cell.
- `src/rcwa_supercell.py` — the TORCWA padded-supercell baseline (defective
  profile sampled into layers, shared TiN Drude–Lorentz model).
- `src/metrics.py` — de-embedding (`A=(1−R−T)·Lx/W`), spectral RMSE, resonance
  position / linewidth matching.
- `src/benchmark.py` — reference I/O, RCWA sweep, comparison and reporting.
- `src/fresnel.py` — analytic planar-interface R/T, used to validate the
  source/flux machinery.
- `tests/` — geometry, dispersion-fit, Fresnel, energy, metrics and RCWA tests.
- `scripts/validate_m0.py` — end-to-end M0 validation with a pass/fail summary.
- `scripts/make_reference.py` — generate a MEEP reference (MEEP env).
- `scripts/run_benchmark.py` — run the RCWA sweep vs a reference (torch env).

## Environment

Two environments are needed because MEEP and the TORCWA/torch stack do not
coexist cleanly:

| half | env | packages |
|---|---|---|
| FDTD reference (`make_reference.py`, M0) | `meep-v7` | pymeep, numpy, scipy |
| RCWA baseline (`run_benchmark.py`) | `mlai` | torch, torcwa (editable) |

MEEP is distributed via conda (not pip) and needs a dedicated environment; do
**not** install it into the RCWA `mlai` env.

```bash
~/Apps/miniconda3/bin/conda env create -f v7-finite-grating/environment.yml
~/Apps/miniconda3/bin/conda activate meep-v7
```

If you prefer to create it manually:

```bash
~/Apps/miniconda3/bin/conda create -n meep-v7 -c conda-forge \
    python=3.12 pymeep scipy numpy pandas matplotlib pytest
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

Oblique incidence (fixed angle; one narrowband run per wavelength):

```bash
python src/meep_grating.py --angle 30 --n-freq 12 ...
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

## M1: padded-supercell RCWA vs MEEP

The finite grating is wrapped in a supercell of width `Lx = W + 2·padding` and
solved with TORCWA (`order=[nh,0]`, `ny=1`). Two subtleties drive the design:

- **Padding dilutes the signal.** The incident plane wave also passes through
  the air-over-substrate padding, so raw R/T are dominated by the bare
  substrate. We de-embed to the aperture: `A = (1−R−T)·Lx/W` (exact, since the
  bare substrate absorbs nothing) and `R = R0 + (R−R0)·Lx/W`.
- **Resolution.** The ~30–50 nm TiN features need MEEP `resolution ≥ ~140`
  px/µm; thinner films are under-resolved at lower values and bias R upward.

### Workflow

```bash
# 1. FDTD reference  (meep-v7 env)
python scripts/make_reference.py \
    --n-ridges 6 --period 1.2 --amplitude 0.05 --base-thickness 0.03 \
    --jitter-std 0.12 --n-missing 1 --seed 0 \
    --wl-range 0.7 1.1 --n-freq 10 --resolution 140 --pol s \
    --pad 1.5 --out reports/reference/grating6_defects

# 2. RCWA sweep + report  (mlai env)
python scripts/run_benchmark.py \
    --reference reports/reference/grating6_defects --pol s \
    --padding 0.5 1.0 2.0 4.0 --nh 8 16 32 64 \
    --grid-nm 10 --n-layers 40 \
    --out reports/m1_grating6_defects_s --plot
```

### First result (`reports/m1_grating6_defects_s.md`)

6 ridges (one missing), period 1.2 µm, amplitude 50 nm, TiN on n=1.5, s-pol,
normal incidence, 0.7–1.1 µm:

| padding (µm) | nh | RMSE A | RMSE R | RMSE T |
|---:|---:|---:|---:|---:|
| 0.5 | 8 | 0.0302 | 0.0065 | 0.0362 |
| 0.5 | 64 | **0.0124** | 0.0081 | 0.0081 |
| 2.0 | 64 | 0.0138 | 0.0094 | 0.0074 |
| 4.0 | 8 | 0.0936 | 0.0264 | 0.1176 |
| 4.0 | 64 | 0.0150 | 0.0087 | 0.0088 |

Two clear trends:

1. At fixed padding, accuracy improves with `nh` (8 → 64).
2. At fixed `nh`, accuracy **degrades** with padding: a larger supercell
   under-resolves each ridge for a given number of Fourier orders. This is the
   central tension of the approach — enough padding to decouple periodic
   images, but enough orders to resolve the aperture.

Best configuration for this structure: padding 0.5 µm, `nh=64`, RMSE(A)=0.0124
(~1.2% of the absorption signal). This is the raw-RCWA error floor that the
learned correction (M2) must beat.

## Next (M2)

Train the hybrid model on paired (RCWA supercell, MEEP) spectra: first a
residual correction on the RCWA spectrum, then the pre-solver encoder. Split by
geometry, and report accuracy vs runtime against both raw RCWA and MEEP.
