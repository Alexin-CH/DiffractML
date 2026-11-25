# E7 — 60° references: ratio against a matched bare control

E6 fixed the incident-flux normalisation and predicted that 60° would then be
usable at `pad ≈ 3`. The Fresnel gate disagreed: at `res = 140` the bare
interface never reproduced the analytic values.

## What the gate measured

| angle | pad | err s | err p | R+T |
|---|---:|---:|---:|---|
| 30° | 1.5 | 0.017 | 0.008 | 1.015 |
| 60° | 1.5 | 0.320 | 0.522 | 1.20–1.53 |
| 60° | 3 | 0.038 | 0.070 | 1.02–1.09 |
| 60° | 4 | 0.057 | 0.080 | 1.01–1.09 |
| 60° | 6 | 0.041 | 0.069 | 1.01–1.08 |

30° is fine (1–2 %) and is not regenerated. 60° carries a 2–8 % bias with
`R + T > 1`, i.e. the transmitted flux is overestimated, and **the bias does not
depend on the pad**.

## Why the pad cannot fix it

Two things were measured rather than assumed.

1. **The beam clips no matter how wide the cell is.** In an empty cell at 60°
   the downward flux is already 4 % below its source value at the sample plane
   and 13.6 % below at the transmission plane. Since
   `source_width = sx − 2·dpml`, the beam footprint always fills the non-PML
   width, so widening the cell widens the beam by the same factor and the loss
   fraction stays constant. That is the plateau in the table.
2. **Narrowing the source, to leave slack, makes it worse** (s-pol error
   0.025 → 0.066): a narrower beam diffracts more.

The underlying reason is that the calibration and the measurement happen in
*different geometries*: the incident calibration beam travels 1.1 µm at 60° in
vacuum, whereas the transmitted beam travels 0.4 µm at 35° inside the
substrate. No single per-plane incident-flux factor can reconcile the two.

Also tested and rejected: MEEP's canonical `Hz` magnetic-current source for
p-pol. It is *worse* than the existing `Ex`/`Ey` pair (60°: 0.042 vs 0.021;
30°: 0.012 vs 0.005), because the pair is already normalised by the empty-cell
calibration. The source stays as it is.

## Ratio against a matched bare control

Replace the analytic baseline plus incident calibration by a bare-substrate run
in the *same* cell:

```
R = R0_fresnel · R_struct / R_bare
T = T0_fresnel · T_struct / T_bare
```

The incident flux cancels entirely, and every cell-level bias (source near
field, walk, side-PML clipping, refraction) is common to both runs. What
remains is the unbiased aperture/padding mixture, which is exactly the input
the existing area-weighted de-embedding expects — so oblique now uses the same
treatment as normal incidence instead of the E5 "compare raw" compromise.

Validated by a lossless structure, which must satisfy `R + T = 1` with no
external reference:

| pol | raw R+T | ratio-corrected R+T |
|---|---|---|
| s | 0.998 – 1.003 | 0.990 – 0.995 |
| p | **1.020 – 1.048** | 0.991 – 1.002 |

The correction removes the p-pol bias (2–5 %) and leaves s within 1 %. Note the
s-pol raw result was already good in this cell, so the ratio slightly
over-corrects it (~0.7 %); it is applied uniformly because the TiN references
carry the larger bias and because a single policy is easier to defend.

## Campaign design

Because the bias is a property of the *cell*, one control per (angle, pol)
suffices if every geometry shares one cell. Hence `--common-sx`:

- `sx = 14.78 µm` (widest aperture 7.78 + 2·(pad 3 + dpml 0.5)),
- per-spec pad 3.0–4.4 µm (`pad = (sx − W − 2·dpml)/2`),
- six calibration wavelengths per polarisation, interpolated in the builder,
- references: 16 jobs (8 geometries × s/p) at `res = 140`, 16 λ.

Cost: ~1.6× the 30° references (~17 h each), i.e. **~6 days on two workers**.
The 30° set is left as measured: its bias is 1–2 %, and regenerating it would
cost another ~8 days.

## Queue (`reports/m2/start_fix2.sh`, tmux `fix2`)

| stage | what | cost |
|---|---|---|
| A1 | repair the archived 30° references | ~2–5 h |
| A2 | p-pol resolution sweep + contrast ablation | ~4.5 h |
| B | lossless `R+T` gate (must pass, else no campaign) | ~30 min |
| C0 | bare controls at 60°, s and p | ~3 h |
| C1 | 16 references at 60° | ~6 days |
| D | rebuild datasets (`data140_fix`) and retrain | ~1 h |

Stage D writes `dataset_gen_fix.npz` (0°/30°, corrected T) and
`dataset_gen60.npz` (0°/30°/60° with the ratio correction), then trains
`gen60`.

## Open items

- The ~0.7 % s-pol over-correction from the ratio: the control's cell is
  shorter than the structure's by the ridge height, which shifts the source and
  monitors by `max_height` on the pixel grid. It is inside the noise floor but
  could be removed by sizing the control cell to the structure's height.
- 30° keeps a 1–2 % bias; if it ever needs to match 60°, regenerate with the
  originals' cells plus controls (8 cells, not one).
