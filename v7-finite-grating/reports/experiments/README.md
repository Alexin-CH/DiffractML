# v7 experiment reports

One consolidated, self-contained report per experiment. Each states the
objective, setup, results, interpretation, reproduce commands, and artifacts.

| report | question | headline |
|---|---|---|
| [M0 — validated FDTD ground truth](M0-ground-truth.md) | Is the reference trustworthy? | Fresnel/energy gates pass; TiN medium matches to 5e-16 |
| [M1 — padded-supercell benchmark](M1-supercell-benchmark.md) | How well does padded RCWA match FDTD? | best raw RMSE(A) = 1.2% |
| [M2 — learned correction](M2-learned-correction.md) | Can a network learn the RCWA→MEEP residual? | absorption error −85% on held-out geometries |
| [E1 — ideal cell vs supercell](E1-ideal-vs-supercell.md) | Is the padded supercell needed? | ideal `nh=8` matches it at ~2.9× lower cost |
| [E2 — differentiable encoder](E2-differentiable-encoder.md) | Does RCWA-in-the-loop help? | loop works; the encoder does not help yet |
| [E3 — fair encoder comparison](E3-fair-encoder.md) | Was E2's negative result an unfair-comparison artefact? | no: at matched budget fixed, global and ridge encoders give the same absorption accuracy; stop the forward-prediction encoder |

Supporting artifacts stay next to the code (`reports/`, `reports/m2/`); the
experiment reports link to them rather than duplicating raw files.
