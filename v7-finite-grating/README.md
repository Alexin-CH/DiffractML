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

## Why FDTD, and what "finite" means

TORCWA is periodic in `x` (lattice) and finite in `z` (layered stack). A real
grating is *truncated* in `x`: a finite number of ridges with edges, plus
fabrication defects. RCWA can only represent that by wrapping the finite
aperture in a large periodic supercell with air padding. The research question
is whether that approximation converges to the true finite-structure spectrum
as the padding grows.

To answer it we need an independent full-wave reference on the **non-periodic**
structure. [MEEP](https://github.com/NanoComp/meep) (FDTD) with PML boundaries does exactly that.

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
- `src/rcwa_diff.py` — differentiable ideal-unit-cell RCWA (Experiment 2).
- `src/models_e2.py` — shared geometry encoder (`z_geo` + config head) and the
  residual correction head.
- `src/fresnel.py` — analytic planar-interface R/T, used to validate the
  source/flux machinery.
- `tests/` — geometry, dispersion-fit, Fresnel, energy, metrics, RCWA and E2 tests.
- `scripts/validate_m0.py` — end-to-end M0 validation with a pass/fail summary.
- `scripts/make_reference.py` — generate a MEEP reference (MEEP env).
- `scripts/run_benchmark.py` — run the RCWA sweep vs a reference (torch env).
- `scripts/benchmark_ideal_vs_supercell.py` — Experiment 1 comparison and
  ideal-backbone dataset export.
- `scripts/train_e2.py` — Experiment 2 end-to-end encoder → RCWA → residual.

## Diagrams

Self-contained workflow diagrams live in [`diagrams/`](diagrams/):

- [M0/M1 pipeline](diagrams/workflow-m0-m1.html)
- [M2 learned correction](diagrams/workflow-m2.html)
- [Experiment 2 training loop](diagrams/workflow-e2.html)

## Environment

Two environments are needed because MEEP and the TORCWA/torch stack do not
coexist cleanly. Ready-to-use configs are provided:

| half | conda env | config |
|---|---|---|
| FDTD reference (M0) | `meep-v7` | [`environment.yml`](environment.yml) |
| RCWA / ML (M1, M2, E1, E2) | `rcwa-v7` | [`environment-rcwa.yml`](environment-rcwa.yml) / [`requirements.txt`](requirements.txt) |

MEEP is distributed via conda (not pip) and needs its own environment.

**Conda (both halves):**

```bash
# FDTD half
conda env create -f v7-finite-grating/environment.yml
conda activate meep-v7

# RCWA/ML half — run from the repository root
conda env create -f v7-finite-grating/environment-rcwa.yml
conda activate rcwa-v7
pip install -e TORCWA/
```

**Plain pip (RCWA/ML half), if you prefer a venv:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r v7-finite-grating/requirements.txt
pip install -e TORCWA/
```

For a CUDA build of PyTorch, install `torch` following
<https://pytorch.org/get-started/locally/> before the other requirements.

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

# 2. RCWA sweep + report  (rcwa-v7 env)
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

## M2: learned spectral correction (the hybrid model)

The hybrid model is a residual correction on top of the de-embedded fast RCWA
spectrum. For each (geometry, λ):

- **features** — geometry, defect summaries (`mean/std/min` ridge scale,
  number and position of a missing ridge), λ and λ/period, and the fast RCWA
  `(R, T, A)`;
- **target** — residual `[MEEP_R − RCWA_R, MEEP_T − RCWA_T]`;
- `A = 1 − R − T` is derived, so the corrected spectrum automatically sums to 1.

Workflow (same two-env split as M1):

```bash
# 1. MEEP references for 24 seeded geometries  (meep-v7 env)
python scripts/generate_m2_references.py --n-specs 24 --resolution 105 \
    --wl-range 0.7 1.15 --n-freq 16 --pol s

# 2. paired dataset (fast RCWA + de-embedding)  (rcwa-v7 env)
python scripts/build_m2_dataset.py --n-specs 24 --pol s --device cpu

# 3. train + evaluate, split by geometry  (rcwa-v7 env)
python scripts/train_m2.py --epochs 800 --hidden 64 --depth 3 \
    --weight-decay 1e-3 --patience 150 --tag m2
```

### Result (`reports/m2/report_m2.md`)

24 geometries (18 train / 6 held-out by geometry), 16 wavelengths each:

| channel | raw RCWA | corrected | improvement |
|---|---:|---:|---:|
| RMSE R | 0.0141 ± 0.0023 | 0.0027 ± 0.0014 | +81% |
| RMSE T | 0.0094 ± 0.0026 | 0.0040 ± 0.0016 | +58% |
| RMSE A | 0.0131 ± 0.0039 | 0.0019 ± 0.0010 | +85% |

The correction generalises across unseen geometries and cuts the absorption
error by ~85% — the hybrid hypothesis holds for this family.

### Experiment 1 — is the padded supercell needed?

Reusing the 24 MEEP references, the actual defective supercell baseline was
compared with an **ideal unit cell** (one clean period, blind to defects):

| subset | n | ideal RMSE(A) | supercell RMSE(A) |
|---|---:|---:|---:|
| all | 24 | 0.0235 | 0.0148 |
| no missing ridge | 10 | **0.0062** | 0.0135 |
| missing ridge | 14 | 0.0359 | **0.0157** |

- For **jitter-only** gratings the ideal unit cell is *better* than the
  `padding=1.0, nh=32` supercell (which is under-resolved).
- For **missing-ridge** gratings the supercell wins, because the ideal unit
  cell cannot see the defect at all.
- Solve cost at equal `nh` is identical (13.47 vs 13.50 s), because RCWA cost
  is set by `nh`/eigendecomposition, not supercell width.
- **But the ideal cell converges at `nh=8`** (4.65 s) while the supercell needs
  `nh=32` (13.5 s). After the M2 correction both reach the same absorption
  accuracy (ideal 0.0018 vs supercell 0.0019): **~2.9× cheaper for the same
  corrected absorption**, though the supercell keeps a small edge on R.

Details: `reports/m2/experiment1_ideal_vs_supercell.md`.

## Next

- Add the pre-solver **encoder** (compact geometry description → solver grid)
  for out-of-distribution designs, with a larger geometry dataset.
- Extend to p-polarisation, oblique incidence, and resonance-rich structures
  to exercise the resonance-position/linewidth metrics.
- Inverse design through the corrected hybrid model.
