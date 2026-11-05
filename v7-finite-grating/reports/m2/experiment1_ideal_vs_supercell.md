# Experiment 1 — ideal unit cell vs padded supercell

- geometries: 24 (from the M2 dataset)
- RCWA: nh=32, grid=10 nm, 40 layers, pol=s; supercell padding=1 µm
- ideal unit cell = one clean period, no defects (blind to defects)
- supercell = actual defective profile, padded

## Raw RCWA accuracy vs MEEP (RMSE of de-embedded spectra)

| subset | n | ideal RMSE(A) | supercell RMSE(A) | ideal RMSE(R) | supercell RMSE(R) | ideal RMSE(T) | supercell RMSE(T) |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 24 | 0.0235 ± 0.0148 | 0.0148 ± 0.0039 | 0.0880 | 0.0140 | 0.1094 | 0.0104 |
| no_missing | 10 | 0.0062 ± 0.0014 | 0.0135 ± 0.0013 | 0.0125 | 0.0164 | 0.0142 | 0.0108 |
| missing | 14 | 0.0359 ± 0.0028 | 0.0157 ± 0.0047 | 0.1420 | 0.0124 | 0.1774 | 0.0100 |

## Solve cost (seconds per spectrum of 16 wavelengths)

- ideal unit cell: 13.47 s (n=24)
- supercell: 13.50 s (n=5)
- speed-up: 1.00×

Note: RCWA cost is dominated by the Fourier-order eigendecomposition, which depends on `nh`, not on the supercell width. The unit cell is cheaper mainly because it is under-resolved in the supercell for the same `nh`; a fair accuracy-matched comparison may need more orders for the supercell (see the M1 sweep).

## Per-geometry RMSE(A)

| spec | missing | jitter std | ideal | supercell |
|---:|---:|---:|---:|---:|
| 000 | 1 | 0.103 | 0.0357 | 0.0182 |
| 001 | 1 | 0.090 | 0.0401 | 0.0109 |
| 002 | 1 | 0.107 | 0.0366 | 0.0189 |
| 003 | 0 | 0.077 | 0.0066 | 0.0142 |
| 004 | 0 | 0.068 | 0.0089 | 0.0157 |
| 005 | 1 | 0.100 | 0.0323 | 0.0211 |
| 006 | 1 | 0.132 | 0.0343 | 0.0185 |
| 007 | 1 | 0.090 | 0.0321 | 0.0208 |
| 008 | 1 | 0.137 | 0.0317 | 0.0204 |
| 009 | 0 | 0.080 | 0.0081 | 0.0147 |
| 010 | 0 | 0.107 | 0.0072 | 0.0150 |
| 011 | 0 | 0.096 | 0.0051 | 0.0129 |
| 012 | 0 | 0.125 | 0.0048 | 0.0119 |
| 013 | 1 | 0.052 | 0.0396 | 0.0182 |
| 014 | 1 | 0.088 | 0.0374 | 0.0100 |
| 015 | 0 | 0.112 | 0.0047 | 0.0119 |
| 016 | 0 | 0.101 | 0.0055 | 0.0131 |
| 017 | 1 | 0.081 | 0.0373 | 0.0099 |
| 018 | 1 | 0.062 | 0.0380 | 0.0104 |
| 019 | 0 | 0.054 | 0.0057 | 0.0131 |
| 020 | 0 | 0.105 | 0.0050 | 0.0120 |
| 021 | 1 | 0.056 | 0.0392 | 0.0108 |
| 022 | 1 | 0.047 | 0.0323 | 0.0216 |
| 023 | 1 | 0.048 | 0.0361 | 0.0103 |

## Interpretation

- **No missing ridge (n=10):** the ideal unit cell is *better* raw than the
  supercell (`RMSE(A)` 0.0062 vs 0.0135). At `padding=1.0`/`nh=32` the
  supercell is under-resolved, so it adds error rather than removing it.
- **Missing ridge (n=14):** the supercell wins clearly (0.0157 vs 0.0359).
  The ideal unit cell is blind to a missing ridge, so the entire defect
  response is left to the network.
- **Cost:** identical (13.47 vs 13.50 s per 16-λ spectrum). RCWA cost is set by
  `nh` and the layer eigendecomposition, not the supercell width, so the ideal
  cell is **not** faster at equal `nh`.

## With the M2 learned correction

| backbone | corrected RMSE(A) | corrected RMSE(R) | corrected RMSE(T) |
|---|---:|---:|---:|
| supercell (M2) | 0.0019 ± 0.0010 | 0.0027 ± 0.0014 | 0.0040 ± 0.0016 |
| ideal unit cell (M2) | 0.0018 ± 0.0005 | 0.0041 ± 0.0043 | 0.0040 ± 0.0034 |

After correction the two backbones are **equivalent for absorption** (and T),
with the ideal cell slightly better conditioned (lower variance); the supercell
retains an edge on R.

**Conclusion:** for this family the supercell is *not required* to reach the
corrected accuracy — the learned correction absorbs the defect/edge
information. But it is also *not slower* at equal `nh`, so the "cheap ideal
period" premise does not hold at fixed orders.

## Order convergence and the real speed win

The ideal unit cell is already converged at `nh=8` (absorption is insensitive
to further orders):

| nh | ideal RMSE(A) | time (s / 16-λ spectrum) |
|---:|---:|---:|
| 8 | 0.0233 ± 0.0138 | 4.76 |
| 16 | 0.0232 ± 0.0137 | 6.25 |
| 32 | 0.0232 ± 0.0137 | 13.22 |

The padded supercell needs `nh=32` (13.5 s) to resolve its ridges, so the cheap
operating point is the ideal cell at `nh=8`.

## Corrected result at the cheap operating point

| backbone | solve cost | corrected RMSE(A) | corrected RMSE(R) | corrected RMSE(T) |
|---|---:|---:|---:|---:|
| supercell, nh=32 | 13.5 s | 0.0019 ± 0.0010 | 0.0027 ± 0.0014 | 0.0040 ± 0.0016 |
| ideal, nh=32 | 13.2 s | 0.0018 ± 0.0005 | 0.0041 ± 0.0043 | 0.0040 ± 0.0034 |
| **ideal, nh=8** | **4.65 s** | **0.0018 ± 0.0005** | 0.0041 ± 0.0043 | 0.0040 ± 0.0034 |

**Final conclusion:** one ideal period at `nh=8` plus the learned correction
matches the padded supercell at `nh=32` in corrected **absorption** accuracy
(0.0018 vs 0.0019) at **~2.9× lower solve cost**. The supercell retains a small
edge on R (0.0027 vs 0.0041), but the primary metric is equal — so the
expensive supercell is not mandatory, and the cheap backbone + correction is
genuinely faster. This directly supports the "one perfect period + NNs" route;
the encoder/latent work in Experiment 2 should target the residual R gap.
