# E5 — generalisation to oblique incidence and p-polarisation

## Data generation

48 MEEP references at `res=140`, 8 geometries × {0°, 30°, 60°} × {s, p} × 16 λ,
on the remote RTX 3060 box (1056 FDTD runs: empty-cell control + structure per
point; ~10.7 h per oblique reference). All 48 completed with zero failures.

Two setup bugs were found and fixed:

1. **Oblique metadata.** Oblique runs store per-wavelength metadata as
   `fdtd_meta.runs`, so `Lx_um` was written as `NaN` and the de-embedding
   silently produced `NaN` spectra. `save_reference` now reads the cell width
   from either shape, and `scripts/repair_reference_meta.py` repairs existing
   files (32 repaired, 0 unrecoverable).
2. **JSON serialisation.** A nested ndarray in the meta killed each job *after*
   the FDTD run; `_jsonable` + per-spec error isolation fixed it.

## De-embedding policy: de-embed at 0°, compare raw at oblique

The area-weighted de-embedding (scale the deviation from the bare substrate by
`Lx/W`) assumes the aperture and the padding add incoherently. That holds at
normal incidence but **fails at oblique**, where it produces unphysical
`T < 0` (e.g. −0.14 at 30°).

Raw RCWA vs raw MEEP is the correct comparison at oblique: both cells contain
the same padding, and the aperture is a small perturbation of it.

| angle | raw RMSE(A) | de-embedded RMSE(A) | used |
|---:|---:|---:|---|
| 0° | 0.036 | **0.016** | de-embedded |
| 30° | **0.007** | 0.053 | raw |
| 60° | 0.086 | 0.188 | raw, **discarded** |

## The 60° data is biased by PML clipping

A bare-interface control at 60° gives `R = 0.107` against the Fresnel value
0.177; widening the cell (`pad=1.5 → 4.0`) raises it to 0.159. The reflected
beam walks ~5 µm sideways over the cell height and is absorbed by the side
PML. The 30° control gives 0.056 vs 0.058 (3%), so 30° is usable.

Re-running 60° needs a wider cell (≈2× the cost, several hundred core-hours),
so it is **excluded** from training and left for a future run.

## Dataset

- 8 geometries × {0°, 30°} × {s, p} × 16 λ = **512 samples**.
- Features: the 16 M2 features + `sin θ`, `cos θ`, `pol_p` (19 total).
- Geometry split: 6 train / 2 held-out.

## Result (2 held-out geometries, 128 samples)

| subset | R raw → corr | T raw → corr | A raw → corr |
|---|---|---:|---:|
| all | 0.0360 → 0.0113 | 0.0584 → 0.0108 | 0.0655 → **0.0061** |
| 0° s | 0.0120 → 0.0041 | 0.0131 → 0.0063 | 0.0175 → 0.0037 |
| 0° p | 0.0540 → 0.0170 | 0.0614 → 0.0134 | 0.1045 → 0.0093 |
| 30° s | 0.0405 → 0.0111 | 0.0346 → 0.0107 | 0.0071 → 0.0044 |
| 30° p | 0.0221 → 0.0090 | 0.0923 → 0.0115 | 0.0766 → 0.0056 |

The residual **transfers across angle and polarisation**: a single 19-feature
MLP corrects unseen geometries to ~0.006 absorption RMSE, versus 0.065 raw.
The worst raw channel is p-pol at normal incidence (0.10), which the correction
reduces nine-fold.

This is weaker than the normal-incidence s-pol result (0.0018) — expected with
6 training geometries, two illumination conditions, and the harder p-pol
response — but it establishes that the hybrid model is not an s-pol/0°
artefact.

## Reproduce

```bash
# remote (meep-v7), resumable, 2 pinned workers
JOBS=2 taskset -c 0,2 bash reports/m2/run_parallel.sh

# local (rcwa-v7)
python scripts/repair_reference_meta.py --data-dir reports/m2/data140
python scripts/build_m2_dataset.py --angles 0 30 --pols s p \
    --specs 0 3 5 9 13 15 19 22 --data-dir reports/m2/data140 \
    --out reports/m2/dataset_generalisation.npz
python scripts/train_m2.py --data reports/m2/dataset_generalisation.npz \
    --epochs 800 --hidden 128 --depth 4 --weight-decay 1e-3 \
    --patience 150 --device cuda --tag gen --out-dir reports/m2/gen
```

## Artifacts

- `reports/m2/data140/` (48 references, 0° and 30° used)
- `reports/m2/dataset_generalisation.npz`
- `reports/m2/gen/report_gen.md`, `summary_gen.json`
- code: `src/benchmark.py`, `scripts/repair_reference_meta.py`,
  `scripts/build_m2_dataset.py`, `scripts/validate_oblique.py`

## Open items

- 60° requires a wider MEEP cell; estimate `pad ≈ 5 µm` and validate against
  Fresnel before regenerating.
- p-pol at normal incidence has a large raw error (0.10); worth checking
  whether it is RCWA convergence (`nh`) or the reference.
- The oblique de-embedding that is correct for the padded cell is not derived;
  we use raw values, which is adequate because the residual learns the
  padding difference.
