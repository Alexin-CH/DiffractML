# E6 — oblique flux accounting

This experiment started as three open items from E5 and turned out to be one
instrument bug plus two real physics questions.

## The bug: T was normalised at the wrong monitor plane

`_run_once` measures the incident flux in an empty cell at **both** flux planes
and then divided *both* R and T by `inc_refl`. The transmitted flux was never
used, even though the comment said "normalise each plane by the incident flux
measured at that same plane".

At normal incidence the two calibrations agree, so 0° data is unaffected. At
oblique incidence the truncated plane-wave source loses part of its beam to the
side PML on the way from the reflection plane down to the transmission plane,
and the loss is not the same at the two planes:

| angle | pad (µm) | `inc_refl/inc_tran` |
|---:|---:|---:|
| 0° | 1.5 | 1.0025 |
| 30° | 1.5 | 1.043 |
| 60° | 1.5 | 1.915 |
| 60° | 4.0 | 1.147 |

So every archived oblique T is biased low by that factor: 4 % at 30°, and
between 10 % and 90 % at 60°.

**This invalidated the E5 diagnosis.** The 60° set was blamed on side-PML
clipping alone and the remedy was "pad ≈ 5 µm". Clipping is real — a bare
interface at `pad=1.5`, 60° gives R = 0.107 against Fresnel 0.1766 — but once T
is normalised correctly the bare interface recovers at `pad = 3`, not 5:

| angle | pad | R (Fresnel) | T (Fresnel) | R+T |
|---|---:|---|---|---:|
| 30° | 1.5 | 0.0577 (0.0578) | 0.9361 (0.9422) | 0.994 |
| 60° | 1.5 | 0.1067 (0.1766) | 1.116 (0.823) | **1.22** ✗ |
| 60° | 3.0 | 0.1737 (0.1766) | 0.8523 (0.823) | 1.026 |
| 60° | 6.0 | 0.1737 (0.1766) | 0.8505 (0.823) | 1.024 |

Beyond `pad = 3` there is no further gain, and a ~2.5 % systematic remains:
the side loss is not identical for the reflected and transmitted beams.

## Fix

- `empty_cell_fluxes()` is factored out of `_run_once`, so the calibration can
  be reproduced from a stored reference's metadata.
- `T = -tran_flux / inc_tran`; both calibrations and their ratio are stored in
  the sidecar meta.
- `save_reference` refuses an unphysical spectrum
  (`max(R+T-1) > 0.02` or a negative efficiency). This would have caught the
  bad 60° data on the day it was written.
- `scripts/flux_selfcheck.py` measures a bare interface against Fresnel at each
  angle and reports the smallest padding that passes; the remote queue reads
  that number instead of assuming one.

## Repairing the archive instead of regenerating it

The per-λ calibration depends only on stored quantities (`fcen`, `fwidth`,
`sx`, `sy`, `y_src`, `y_refl`, `y_tran`, `resolution`, `dpml`), and an
empty-cell run costs ~170 s at res=140 against ~10.7 h for a full reference.
`scripts/repair_reference_flux.py` therefore recomputes `inc_tran`, rescales T
and writes corrected copies to `data140_fix/`, leaving the original files as the
record of what was measured. It also cross-checks the recomputed `inc_refl`
against the archived one, which validates the reconstruction.

The 60° set is *not* repaired — its cell is too narrow, so it is regenerated.

## p-pol at normal incidence is not an `nh` problem

The E5 residual on 0° p-pol (raw RMSE A = 0.1045) was suspected to be RCWA
convergence. It is not:

| nh | A mean | bias vs MEEP res=140 | A(0.7) bias |
|---:|---:|---:|---:|
| 32 | 0.1941 | +0.067 | +0.059 |
| 64 | 0.1877 | +0.061 | +0.053 |
| 96 | 0.1854 | +0.059 | +0.051 |
| 160 | 0.1835 | **+0.057** | **+0.050** |

RCWA converges to a limit that is still ~0.055 above MEEP, while MEEP itself
moves *down* with resolution (A mean 0.158 at res 80 → 0.127 at res 140). Real
space slicing is converged too (`grid_nm` 10 → 5 changes nothing), and swapping
the p-polarised source from the current `Ex`/`Ey` pair to MEEP's canonical `Hz`
magnetic-current recipe changes nothing (A mean 0.158 vs 0.159). The bare
substrate conserves energy exactly at 0°, so the flux machinery is sound there.

This needs a two-sided study: a MEEP resolution sweep (140/200/280) to see which
side is converged, plus a constant-ε contrast ablation to test whether the
discrepancy is specific to the strongly lossy high-index interface.

## The oblique de-embedding is not algebraically wrong

`R_ap = R0 + (R−R0)·Lx/W`, `A_ap = (1−R−T)·Lx/W` is exactly the disjoint-aperture
power balance under uniform illumination (`f = Lx/W > 1` because the beam covers
the padding too), so no formula fix is needed. What breaks at oblique incidence
is the *assumption*: a finite patch plus a laterally walking beam does not make
the padding behave like an infinite bare substrate. Hence the E5 policy of
comparing raw spectra at oblique, now with correct normalisation, and the
option of matching the RCWA padding to the MEEP pad so that raw is
like-for-like.

## Queue (remote, tmux session `fix`)

`reports/m2/start_fix.sh`, resumable at every stage:

| stage | what | cost |
|---|---|---|
| 0 | Fresnel gate → smallest passing pad at 60° | ~1.5 h |
| 1 | repair 30° references → `data140_fix/` (canary) | ~4–7 h |
| 2 | regenerate 60° at the gate's pad (16 references) | ~5.5 days |
| 3 | p-pol resolution sweep + contrast ablation | ~4.5 h |

Resume with `bash ~/DiffractML/v7-finite-grating/reports/m2/start_fix.sh`;
stage 0 reuses `selfcheck.json`, stage 2 skips existing JSONs.

## Reproduce

```bash
# gate (meep-v7)
python scripts/flux_selfcheck.py --angles 30 60 --pads 1.5 3 4 --pols s p
# repair (meep-v7)
python scripts/repair_reference_flux.py --angles 30 \
    --data-dir reports/m2/data140 --out-dir reports/m2/data140_fix
# rebuild the 30 deg dataset (rcwa-v7)
python scripts/build_m2_dataset.py --angles 0 30 --pols s p \
    --specs 0 3 5 9 13 15 19 22 --data-dir reports/m2/data140_fix \
    --out reports/m2/dataset_generalisation_fix.npz
# p-pol study
python scripts/pol_convergence.py --spec 3 --resolutions 140 200 280
python scripts/pol_convergence_rcwa.py
```

## Open items

- 60° after regeneration: re-check the residual ~2.5 % side-loss bias; if it
  matters, add a same-cell bare-substrate control per (geometry, angle, pol).
- p-pol: decide from the sweep whether to raise `nh` for p-pol in the dataset
  or to declare the p-pol target reference-limited.
- The 0° T values shift by 0.25 % with the new normalisation; the archived 0°
  references are left as-is (well inside the noise floor).
