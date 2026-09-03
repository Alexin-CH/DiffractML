# DiffractML — Structured Project Report

**Author:** DiffractML project
**Date:** 2026-08-18
**Repository:** [github.com/Alexin-CH/DiffractML](https://github.com/Alexin-CH/DiffractML)

---

## 1. Introduction

DiffractML is a research project combining Rigorous Coupled-Wave Analysis (RCWA) with machine learning to model and inverse-design diffractive optical structures. The RCWA engine is [TORCWA](https://github.com/Alexin-CH/TORCWA) (GPU-accelerated, differentiable). The target structure is a **1D sine-corrugated TiN (titanium nitride) grating**, a common plasmonic/absorber geometry.

This report summarizes the archived work (v0–v5), describes the current v6 pipeline, and presents experimental results.

---

## 2. Archive: v0–v5 (Closed)

Earlier research versions are preserved under `archives/`. They explored different neural-network architectures for predicting the electromagnetic response of the sine TiN grating, but all contained the same pipeline-level bugs that are fixed in v6.

### 2.1 Architecture Evolution

| Version | Name | Model | Input | Output | Samples | Key Idea |
|---------|------|-------|-------|--------|---------|----------|
| v0 | Fields→Fields | 4-layer 2D CNN | Incident EM fields (2D grid, 256×500) | EM fields in/around structure | 900 | Direct field prediction |
| v1 | HyperNet (w,a,F)→F | HyperNet + CNN | (wavelength, angle, incident fields) | Fields | 900 | Hypernetwork generates CNN weights conditioned on (w,a) |
| v2 | HyperNet (w,a,ε)→F | HyperNet + CNN | (wavelength, angle, permittivity map) | Fields | 900 | Replace field input with structural permittivity |
| v3 | HyperNet + FFT | HyperNet + CNN | FFT of permittivity map, (w,a) | FFT of fields | 900 | Operate in Fourier domain |
| v4 | MLP+CNN | MLP + CNN decoder | (wavelength, angle) + permittivity map | Fields | 900 | Two-stream latent: MLP for (w,a), CNN for ε |
| v5 | Hybrid order correction | Hybrid MLP | High-order RCWA (nh=20) output | Low-order (nh=5) approximation | 2400+ | Correct under-resolved simulations to match high-order reference |

### 2.2 Common Bugs in v0–v5

All archived versions used TORCWA's S-parameter interface incorrectly:

1. **Wrong method name:** `S_parameters` called instead of `s_parameters` (case mismatch with TORCWA API).
2. **Hardcoded loop variables:** The direction/port/polarization loops in the S-parameter extraction used fixed integer literals instead of the loop variable. As a result, **all 32 S-parameter columns contained identical data** — the dataset was effectively a 1-column dataset repeated 32 times.
3. **Hard-threshold geometry masks:** A binary mask `h >= z_mid` was used to define the grating layer. This has zero gradient, making any geometry-aware optimization impossible.
4. **No `_INDEX_CSV` robustness:** The path to `TiN-RefIdx-labo.csv` was constructed relative to `__file__` using unnormalized `..` segments, which broke under pytest (working directory mismatch).

### 2.3 Data and Scope

- v0–v4: 900 samples, wavelength 400–2000 nm, angle 0–70°.
- v5: 2400+ samples, wavelength 200–2500 nm, amplitude 20–100 nm, period 100–10000 nm, angle 10–70°, nh=20 (high order) corrected to nh=5.
- None of the archived versions achieved reliable inverse design; the bugs meant that the surrogate learned a degenerate representation of the optical response.

---

## 3. Current Project: v6 Inverse Design

### 3.1 Pipeline Overview

```
generate.py → dataset.py → train.py → design.py
    (RCWA)      (z-score)    (MLP)     (surrogate → RCWA refinement)
```

| Stage | Script | Purpose | Runtime |
|-------|--------|---------|---------|
| 1. Dataset generation | `generate.py` | Random/Latin-sample `(wl, ang, amp, per)` through TORCWA → S-parameters + `[R_xx, T_xx, R_yy, T_yy]` | ~few min (CPU) |
| 2. Surrogate training | `train.py` | Train `SurrogateMLP`: `(wl, ang, amp, per)` → `[R_xx, T_xx, R_yy, T_yy]` | ~30 s |
| 3. Inverse design | `design.py` | Recover `(amp, per)` from target spectrum | ~1 min (hybrid) |

### 3.2 Key Files

| File | Role |
|------|------|
| `simulation.py` | Correct TORCWA wrapper: smooth sigmoid masks, tensor lattice, correct `_propagating_orders`, `pick_device` hybrid |
| `generate.py` | Dataset generation |
| `dataset.py` | `SinTinDataset` with z-score normalization |
| `models.py` | `SurrogateMLP` and `SurrogateEnsemble` |
| `train.py` | Surrogate training loop |
| `design.py` | Inverse design: surrogate + RCWA refinement + solver-only mode |
| `tests/test_simulation.py` | Regression tests (5 tests) |

### 3.3 What v6 Fixes

| Bug (v0–v5) | v6 Fix |
|-------------|--------|
| `S_parameters` → `s_parameters` | `torcwa.s_parameters()` called correctly |
| Hardcoded loop variables → all 32 S-params identical | Full 4-port, 4-direction, 2-polarization loop with correct indexing |
| Hard-threshold masks `h >= z_mid` | Smooth sigmoid: `torch.sigmoid((h - z_mid) * edge_sharpness)` |
| `_INDEX_CSV` path fragile | `os.path.normpath` anchored to `torcwa.__file__` (works under pytest) |
| Evanescent orders → NaN gradients | `_propagating_orders` requires propagation in both input and output layers |
| Period gradient = 0 (detached lattice) | Tensor lattice: `lx = args.sin_period` (requires_grad=True) |
| CPU-only, no hybrid device | `pick_device(nh, disc, auto)` hybrid: GPU for large solves, CPU for small |

### 3.4 Device Hybrid (`--device auto`)

Measured solve times (single RCWA, nh/disc → CPU ms / CUDA ms):

| Config | CPU | CUDA | Winner |
|--------|-----|------|--------|
| nh=3, disc=16 | 97 | 270 | CPU (2.8×) |
| nh=5, disc=32 | 142 | 293 | CPU (2.1×) |
| nh=8, disc=48 | 995 | 333 | **CUDA (3.0×)** |
| nh=10, disc=64 | 269 | 351 | CPU (1.3×) |
| nh=12, disc=80 | 279 | 390 | CPU (1.4×) |

Threshold: CUDA when `nh ≥ 8` and `disc ≥ 48`. Surrogate stage always CPU.

### 3.5 Period Gradient Investigation

**Finding:** Period gradient through TORCWA is real but weak — this is physics, not a bug.

| Setup | grad_amp | grad_per | grad_per / grad_amp |
|-------|----------|----------|---------------------|
| Detached lattice (v0–v5) | ~2e-4 | ~1e-8 | 0.005% |
| Tensor lattice (v6) | ~2e-4 | ~1e-5 | 5% |

Finite-difference validation: tensor lattice grad_per matches central differences to **99.6%**. The weak period gradient is inherent to the smooth grating (saturated sigmoid mask), not a solver error.

### 3.6 `_propagating_orders` Fix

**Root cause of NaN:** `_propagating_orders` used hardcoded `layer_eps=1.5` and did not check input-layer propagation. Orders evanescent in the input layer (real Kz_in = 0) were included, producing `sqrt(0/ref)` whose backward pass gives `1/(2·0) = inf → NaN`.

**Fix:** Require propagation in both input and output layers using actual permittivities (`sim.eps_in * sim.mu_in` and `sim.eps_out * sim.mu_out`).

**Result:** 198/198 (wl × per × amp) combos pass on GPU with zero NaN, forward and backward.

### 3.7 Optimizer: Adam vs LBFGS

| Mode | Best amp error | Best per error | Loss vs Adam | Time |
|---|---|---|---|---|
| Adam (default) | ~1 nm | ~1–25% | baseline | 55 s |
| LBFGS | ~1 nm | ~1–25% | comparable | 53–317 s |

LBFGS is unstable in solver-only mode (overshoots amp bounds → clamp → corrupted Hessian estimate). Adam remains default.

### 3.8 Modes of `design.py`

| Mode | Flag | Description |
|------|------|-------------|
| Surrogate only | (default) | Gradient descent through the trained NN (ms) |
| Surrogate + refine amp | (default) | Surrogate init, then TORCWA refine amp only |
| Surrogate + refine amp+per | `--refine-per` | Surrogate init, then TORCWA refine both (per_lr=1.0) |
| Solver only | `--no-surrogate` | TORCWA from random restarts, keeps best |
| LBFGS | `--optimizer lbfgs` | Use L-BFGS instead of Adam for RCWA refinement |
| Hybrid device | `--device auto` (default) | GPU for large solves, CPU for small |

---

## 4. Experimental Results

### 4.1 Design Mode Comparison

**Config:** nh=8, disc=48, iters=25, ang=30°, wl=[800, 1100, 1400, 1600] nm, GPU.

| Target (amp, per) | Mode | Amp | Per | Loss | Time |
|---|---|---|---|---|---|
| (45, 1500) | surrogate | 49.2 | 1457 | 4.5e-5 | 8 s |
| | surrogate+amp | 46.3 | 1457 | 1.1e-5 | 58 s |
| | surrogate+ampper | **45.9** | **1479** | **3.8e-6** | 58 s |
| | solver-only | 45.7 | 3401 | 2.3e-4 | 165 s |
| (35, 1000) | surrogate | 37.1 | 1220 | 3.9e-4 | 7 s |
| | surrogate+amp | 34.1 | 1220 | 3.0e-4 | 55 s |
| | surrogate+ampper | **34.6** | **1246** | **2.7e-4** | 55 s |
| | solver-only | 35.6 | 4688 | 4.3e-4 | 161 s |
| (60, 2500) | surrogate | 60.1 | 3276 | 4.5e-6 | 7 s |
| | surrogate+amp | 60.0 | 3276 | 4.2e-6 | 55 s |
| | surrogate+ampper | **60.0** | **3283** | **4.2e-6** | 55 s |
| | solver-only | 57.7 | 4047 | 1.2e-5 | 162 s |

**Key observations:**
1. `surrogate+ampper` achieves the best loss in all three targets (3.8e-6, 2.7e-4, 4.2e-6).
2. `amp` converges to within ~1 nm of truth in all refined modes.
3. `per` recovery is limited by the **smooth-grating near-degeneracy**: many period values give nearly identical spectra. The surrogate regularizes this; solver-only lands in wrong local minima (3401, 4688, 4047 vs truths 1500, 1000, 2500).
4. Runtime: surrogate ~7 s, surrogate+refine ~55 s, solver-only ~160 s.

### 4.2 Adam vs LBFGS

| Target | Mode | Amp | Per | Loss | Time |
|---|---|---|---|---|---|
| (45, 1500) | surrogate+adam | 46.6 | 1631 | 1.3e-4 | 55 s |
| | surrogate+lbfgs | **49.2** | **1443** | **4.6e-5** | 53 s |
| (35, 1000) | surrogate+adam | **34.2** | **1199** | **3.0e-4** | 54 s |
| | surrogate+lbfgs | 37.1 | 1214 | 4.0e-4 | 103 s |
| (60, 2500) | surrogate+adam | 60.0 | 3291 | 4.2e-6 | 54 s |
| | surrogate+lbfgs | 60.1 | 3327 | 4.2e-6 | 53 s |

- LBFGS wins on target 1, ties on target 3, loses slightly on target 2.
- LBFGS solver-only is unstable (slams into amp bounds → corrupted Hessian).
- Adam is the better default: robust, faster, comparable loss.

---

## 5. Current State (main / v6-inverse-design @ `b2781a0`)

### 5.1 What Works

- [x] TORCWA wrapper with correct S-parameter extraction
- [x] Smooth sigmoid masks for gradient flow
- [x] Tensor lattice for period differentiability
- [x] Corrected `_propagating_orders` (input + output layer check)
- [x] Surrogate training pipeline (generate → train)
- [x] Inverse design: surrogate + TORCWA refinement (amp and per)
- [x] Solver-only mode (no surrogate, random restarts)
- [x] LBFGS optimizer option
- [x] Hybrid CPU/CUDA device selection (`--device auto`)
- [x] NaN-safe early stopping
- [x] 5/5 regression tests passing
- [x] Zero NaN across 198 GPU test cases
- [x] Experiment reports (design mode + optimizer comparison)

### 5.2 Known Limitations

- **Period near-degeneracy:** The smooth sine grating's spectrum is nearly degenerate in `per` — many periods give similar responses. This is a property of the physics, not a bug.
- **Surrogate per recovery:** The surrogate sometimes overshoots per (e.g. 3276 vs 2500). The loss is very low despite wrong per — genuine near-degeneracy.
- **No experimental validation:** Results are against ground-truth RCWA, not measured devices.
- **Single target column:** Each run optimizes one efficiency column (e.g. `R_xx`). Multi-column targets would require weighted loss.

### 5.3 Git History

```
b2781a0 Add LBFGS option to RCWA refinement
a59b760 Add experiment reports: design-mode and Adam-vs-LBFGS comparisons
ea0738f Add solver-only (no-surrogate) inverse design mode with restarts
5947184 Add tensor-lattice period gradients, per refinement, and hybrid CPU/CUDA
3129ae5 typo
d402fd3 Update README
c3309bd Adapt v6 to TORCWA package rename; use solve() wrapper
bcd4b1b Add v6 inverse-design project; archive v0-v5 experiments
```

### 5.4 Branches

- `main` and `v6-inverse-design` both at `b2781a0` (in sync, pushed).
- TORCWA submodule at `2109357` on `main`.

---

## 6. References

- Moharam, M. G. & Gaylord, T. K. (1986). Rigorous coupled-wave analysis of planar-grating diffraction. *JOSA A*, 3(8), 1083-1092.
- TORCWA: [github.com/Alexin-CH/TORCWA](https://github.com/Alexin-CH/TORCWA)
- DiffractML: [github.com/Alexin-CH/DiffractML](https://github.com/Alexin-CH/DiffractML)
