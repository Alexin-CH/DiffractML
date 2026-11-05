# M1 — padded-supercell RCWA benchmark

## Objective

Measure how well fast periodic RCWA reproduces the finite-grating FDTD
reference when the finite aperture is wrapped in a padded supercell.

## Setup

- **Structure:** 6 ridges, period 1.2 µm, amplitude 50 nm, base 30 nm, one
  missing ridge; s-polarisation, normal incidence, 0.7–1.1 µm.
- **Reference:** MEEP FDTD at 140 px/µm.
- **RCWA:** TORCWA supercell of width `Lx = W + 2·padding`, `order=[nh,0]`,
  grid 10 nm, 40 layers.
- **Metric:** de-embedded absorption `A = (1−R−T)·Lx/W` (baseline-free; the bare
  substrate absorbs nothing). R/T are compared after subtracting the
  bare-substrate Fresnel baseline.

## Result (16 configurations)

Primary metric is RMSE(A); the best row is bold.

| padding (µm) | nh | RMSE A | RMSE R | RMSE T | max\|ΔA\| |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 8 | 0.0302 | 0.0065 | 0.0362 | 0.0352 |
| 0.5 | 16 | 0.0188 | 0.0050 | 0.0165 | 0.0237 |
| 0.5 | 32 | 0.0135 | 0.0075 | 0.0092 | 0.0194 |
| 0.5 | 64 | **0.0124** | 0.0081 | 0.0081 | 0.0183 |
| 1.0 | 8 | 0.0339 | 0.0106 | 0.0441 | 0.0386 |
| 1.0 | 16 | 0.0210 | 0.0054 | 0.0173 | 0.0272 |
| 1.0 | 32 | 0.0147 | 0.0082 | 0.0093 | 0.0218 |
| 1.0 | 64 | 0.0134 | 0.0091 | 0.0081 | 0.0205 |
| 2.0 | 8 | 0.0679 | 0.0149 | 0.0545 | 0.0750 |
| 2.0 | 16 | 0.0323 | 0.0063 | 0.0382 | 0.0375 |
| 2.0 | 32 | 0.0163 | 0.0077 | 0.0104 | 0.0241 |
| 2.0 | 64 | 0.0138 | 0.0094 | 0.0074 | 0.0218 |
| 4.0 | 8 | 0.0936 | 0.0264 | 0.1176 | 0.1048 |
| 4.0 | 16 | 0.0339 | 0.0115 | 0.0448 | 0.0397 |
| 4.0 | 32 | 0.0202 | 0.0058 | 0.0156 | 0.0287 |
| 4.0 | 64 | 0.0150 | 0.0087 | 0.0088 | 0.0241 |

**Best:** padding 0.5 µm, `nh=64`, RMSE(A) = **0.0124** (~1.2% of the
absorption signal). This is the raw-RCWA error floor the learned correction
must beat.

## Interpretation

1. At fixed padding, accuracy improves with `nh` (8 → 64).
2. At fixed `nh`, accuracy **degrades** with padding: a larger supercell
   under-resolves each ridge for a given number of Fourier orders. This is the
   central tension of the supercell approach — enough padding to decouple
   periodic images, enough orders to resolve the aperture.
3. The absorption spectrum is smooth over this band, so no resonance peaks are
   matched; resonance-position / linewidth metrics are reported as empty rather
   than fabricated.

## Reproduce

```bash
# meep-v7 env
python scripts/make_reference.py --n-ridges 6 --period 1.2 --amplitude 0.05 \
    --base-thickness 0.03 --jitter-std 0.12 --n-missing 1 --seed 0 \
    --wl-range 0.7 1.1 --n-freq 10 --resolution 140 --pol s --pad 1.5 \
    --out reports/reference/grating6_defects

# rcwa-v7 env
python scripts/run_benchmark.py --reference reports/reference/grating6_defects \
    --pol s --padding 0.5 1.0 2.0 4.0 --nh 8 16 32 64 --grid-nm 10 \
    --n-layers 40 --out reports/m1_grating6_defects_s --plot
```

## Artifacts

- `reports/m1_grating6_defects_s.{md,csv,png}`
- `reports/reference/grating6_defects_s.{csv,json}`
- code: `src/rcwa_supercell.py`, `src/metrics.py`, `src/benchmark.py`
