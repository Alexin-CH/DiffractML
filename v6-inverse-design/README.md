# v6-inverse-design

Physics-informed machine learning for **inverse design** of a 1D sine-corrugated TiN grating. A fast neural **surrogate** predicts the RCWA optical response, and inverse design recovers the grating geometry `(amp, per)` that reproduces a
target response spectrum.

The RCWA solver is [TORCWA](https://github.com/Alexin-CH/TORCWA) (GPU-accelerated and differentiable). This project fixes several bugs in the archived v0–v5 code (see `src/simulation.py` header) and demonstrates full end-to-end design.

## Structure

- **`simulation.py`** — correct, GPU-capable, differentiable TORCWA wrapper. Fixes the archived bugs: `S_parameters` → `s_parameters`, hardcoded direction/port/polarization loops (all 32 S-param columns were identical), and hard-threshold masks → smooth sigmoid masks so gradients flow w.r.t. geometry. Total power efficiencies `R_xx, T_xx, R_yy, T_yy` are summed over propagating diffraction orders only (evanescent orders produce NaN gradients).
- **`generate.py`** — dataset generation: random/latin-sampled `(wl, ang, amp, per)` run through TORCWA, producing S-parameters + power efficiencies.
- **`dataset.py`** — `SinTinDataset`: `(wl, ang, amp, per)` → `[R_xx, T_xx, R_yy, T_yy]`, z-scored.
- **`models.py`** — `SurrogateMLP` (and `SurrogateEnsemble`).
- **`train.py`** — trains the surrogate forward model.
- **`design.py`** — inverse design:
  1. Gradient descent **through the surrogate** (milliseconds).  
  2. Refinement **through the RCWA solver** (seconds).  
  The period gradient is physically weak in TORCWA (lattice/k-vectors are detached floats), so refinement optimizes `amp` only; `per` is supplied by the surrogate.

## Usage

All scripts run with Python ≥ 3.10 and PyTorch (CUDA optional; `--device cuda` works if available).

### TORCWA dependency

The RCWA engine is consumed as the `torcwa` package (the TORCWA submodule was renamed from `src` to `torcwa`). Install it editable so code edits are picked up immediately without reinstalling:

```bash
pip install -e TORCWA/   # from the repository root; editable install
```

The package then imports as `import torcwa`. The mlai conda environment used during development already has this install.

### Workflow

```bash
# 1. Generate a dataset (nh=8, disc=48, 300 samples ~ a few minutes on CPU)
python src/generate.py --n-samples 300 --nh 8 --discretization 48 \
    --wl-range 700 1700 --amp-range 20 80 --per-range 500 4000

# 2. Train the surrogate
python src/train.py --epochs 200

# 3. Inverse design: match the R_xx spectrum of a ground-truth structure
python src/design.py --wl-grid 800 900 1000 1100 1200 1300 1400 1500 1600 \
    --ang 30 --target-amp 45 --target-per 1500 --target-col R_xx
```
The `design.py` self-check builds the target spectrum from a known structure (`--target-amp`, `--target-per`) and reports how well the recovered `(amp, per)` reproduces it.

## Design notes

- **Spectrum targets, not single-wavelength:** matching one efficiency at one wavelength is ill-posed (many `(amp, per)` give the same response). Using a target *spectrum* makes the problem well-posed; the recovered structure then matches the full spectrum to ~0.6% RMSE.
- **`amp` vs `per`:** `amp` has a strong gradient through the RCWA solver and refines precisely (e.g. 35 → 45.4, truth 45.0). `per` only enters through the geometry mask with saturated sigmoid gradients (~1e-11 at default sharpness), so it is optimized via the surrogate. Residual `per` differences (e.g. 1891 vs 1500) are genuine near-degeneracies of the smooth grating, not solver errors.
- **Numerical safety:** refinement stops early on non-finite loss/gradients (borderline evanescent/propagating orders) rather than poisoning the optimizer.

## Tests

```bash
python -m pytest tests/
```

Verifies: S-parameters are distinct (original bug), per-polarization `R+T < 1` with TiN absorption, zero cross-polarization, `amp` gradient flows, and response changes with `amp`.