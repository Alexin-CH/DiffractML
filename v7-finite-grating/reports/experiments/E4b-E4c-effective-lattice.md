# E4b/E4c — what an equivalent lattice can and cannot be

Two follow-ups to E4a, both still **oracle** tests (per-geometry optimisation,
no network):

- **E4b:** add an effective TiN permittivity to the single-period family. Is the
  missing-ridge gap partly a *material* effect (`ε` different from bulk TiN)?
- **E4c:** at `nh=8`, can a *learned* low-order supercell profile beat the
  ideal cell and the truncated true supercell?

## E4b — effective permittivity

Family: `(period, amplitude, base, fill, ε_scale)` with
`ε_eff = 1 + ε_scale·(ε_TiN − 1)`, `ε_scale = 1 + 0.6·tanh(u)` (identity at
`u=0`). Same 8-geometry stratified subset as E4a, same fit/selection protocol.

### Raw RMSE(A), mean over geometries

| subset | n | ideal cell | E4a oracle (no ε) | **E4b oracle (+ε)** | supercell |
|---|---:|---:|---:|---:|---:|
| all | 8 | 0.0207 | 0.0171 | **0.0157** | 0.0166 |
| jitter only | 4 | 0.0063 | 0.0058 | 0.0063 | 0.0135 |
| missing ridge | 4 | 0.0351 | 0.0284 | **0.0250** | 0.0198 |

### Missing-ridge geometries

| spec | ideal A | oracle A | fill | ε_scale | oracle R / T | supercell R / T |
|---:|---:|---:|---:|---:|---:|---:|
| 000 | 0.0358 | **0.0207** | 0.684 | 1.373 | 0.010 / 0.017 | 0.011 / 0.011 |
| 013 | 0.0396 | **0.0141** | 0.709 | 1.282 | 0.014 / 0.008 | 0.012 / 0.011 |
| 005 | 0.0325 | 0.0326 | 1.000 | 1.000 | 0.144 / 0.177 | 0.011 / 0.013 |
| 022 | 0.0325 | 0.0326 | 1.000 | 1.000 | 0.146 / 0.178 | 0.010 / 0.014 |

**Findings**

- The effective-permittivity knob is a **real gain**: two of four missing-ridge
  geometries move from `R/T ≈ 0.14/0.18` to `0.01/0.02`, matching or beating
  the `nh=32` supercell. This is not an absorption degeneracy — R and T are
  matched individually.
- The optimum is `ε_scale ≈ 1.3`, i.e. the effective medium is **more**
  metallic/absorbing than bulk TiN. That is the physically sensible
  compensation for lost TiN coverage.
- The other two geometries do not move at all (optimiser falls back to the
  nominal cell). Representability remains geometry-dependent.
- Jitter-only geometries lose the small E4a gain when ε is free (the optimiser
  prefers `ε=1`), but they already beat the supercell.

## E4c — low-order supercell at nh=8

Setup: padded cell (aperture + 1 µm), a free TiN height profile on 24 control
points (0–200 nm), solved at `nh=8`, `nx=256`, 40 layers. Selection on the full
`(R, T)` response — matching A alone can be a degenerate win. Four
missing-ridge geometries.

| metric (mean) | ideal cell | true profile @nh8 | **oracle profile @nh8** | true supercell @nh32 |
|---|---:|---:|---:|---:|
| RMSE(A) | 0.0351 | 0.0145 | 0.0406 | **0.0198** |
| RMSE(R) | 0.136 | 0.123 | **0.041** | **0.011** |
| RMSE(T) | 0.175 | 0.130 | **0.027** | **0.012** |

**Findings**

- At **equal `nh=8`**, an optimised profile improves R/T by **3–5×** over the
  truncated true pattern (`0.12 → 0.04`). So the low-order *representation* is
  learnable.
- It is still **~3× worse than `nh=32`** (`0.041` vs `0.011` in R), and it does
  **not** approach the supercell.
- The true pattern's small A error is a **cancellation artefact**: its R and T
  are each off by ~0.12–0.17, but they cancel in `A = 1−R−T`. Optimising R/T
  (the correct physical objective) makes A *worse* (`0.04`), because the
  independent R/T errors add instead of cancelling. A-only comparisons of
  low-order supercells are therefore unreliable.
- Verdict: `nh=8` is too coarse for a trustworthy full response. A learned
  low-order supercell is not a drop-in replacement for `nh=16–32`.

## What this means for the encoder

- **The `ε` knob is the strongest single-period result so far.** An encoder that
  predicts `(fill, ε_scale)` has a genuine target to learn, and for two
  geometries the oracle matches the supercell at a fraction of the cost.
- **The low-order supercell is not the shortcut it looked like.** Its R/T error
  at `nh=8` is still far from the `nh=32` reference, and the A metric can look
  deceptively good there.
- The residual still has a job: raw R/T errors of `0.01–0.04` are far from the
  corrected `~0.005`, and E1 showed the residual saturates on the ideal cell.
  The value of a better backbone is **data efficiency**, not final accuracy.

## Caveats

- 8 geometries (E4b) and 4 missing-ridge geometries (E4c); oracle fits use
  12–14 Adam steps × 2 restarts and are lower bounds on what is achievable.
- E4b selection is on A (with the nominal always a candidate); R/T are reported
  and confirm the two improving geometries are not degeneracies.
- E4c uses a 24-control-point profile; more control points or a Fourier
  parameterisation could improve the `nh=8` result, but the gap to `nh=32` is
  large enough that this is unlikely to close it.

## Reproduce

```bash
# E4b
python scripts/e4a_oracle_config.py --specs 0 3 5 9 13 15 19 22 \
    --iters 14 --restarts 2 --n-fit-wl 5 --eps-knob --tag oracle_eps
# E4c
python scripts/e4c_oracle_supercell.py --specs 0 5 13 22 \
    --iters 12 --restarts 2 --tag oracle_missing
```

## Artifacts

- `reports/m2/e4a/oracle_eps.{md,json,png}`
- `reports/m2/e4c/oracle_missing.{md,json}`
- code: `src/rcwa_diff.py` (`eps_scale`), `src/rcwa_supercell_diff.py`,
  `scripts/e4a_oracle_config.py`, `scripts/e4c_oracle_supercell.py`
- tests: `tests/test_e4a.py`, `tests/test_e4bc.py`
