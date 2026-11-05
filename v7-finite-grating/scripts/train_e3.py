"""Experiment 3: fair fixed-vs-encoder comparison + encoder redesign.

Experiment 2 compared a learned encoder against a no-encoder ablation, but the
two arms also differed in training budget (8 lambda/step, <=25 epochs, one
full-batch step per geometry) and in the loss (absolute R/T/A vs the M2
residual loss).  Experiment 3 removes those confounds:

1. every arm uses the *same* precomputed ideal unit-cell RCWA data
   (``dataset_ideal_nh8.npz``), the same geometry split, the same optimiser and
   the same mini-batch schedule;
2. the residual target is the M2 formulation (standardised dR, dT; A = 1-R-T),
   so the loss is identical across arms; the residual input is the stored M2
   feature matrix ``X`` (geometry, defect statistics, wavelength ratios and the
   RCWA R/T/A), so the ``none`` arm reproduces the M2 baseline;
3. arms differ only in the model:
   ``none`` (fixed geometry, the M2 control), ``global`` (Experiment 2 encoder)
   and ``ridge`` (the redesign: per-ridge DeepSets encoder);
4. optionally the RCWA is put back in the loop (``--in-loop``) with an
   amplitude-only (``--cfg-mode amp``) or amplitude+period config head, so the
   marginal value of the differentiable path can be measured at matched data
   exposure.

A diagnostic mode (``--diagnose``) retrains the plain residual at a grid of
fixed amplitude / period multipliers and reports the per-geometry *oracle*
gain.  That is an upper bound on what any config head could achieve on this
dataset, without spending RCWA-in-the-loop hours.

Examples
--------
    # fair ladder (precomputed, minutes)
    python scripts/train_e3.py --encoder none  --tag e3_fixed
    python scripts/train_e3.py --encoder global --tag e3_global
    python scripts/train_e3.py --encoder ridge  --tag e3_ridge

    # marginal value of the differentiable config head
    python scripts/train_e3.py --encoder ridge --in-loop --cfg-mode amp \
        --epochs 40 --tag e3_e2e_amp
    python scripts/train_e3.py --encoder ridge --epochs 40 --tag e3_control

    # oracle headroom of an effective-amplitude config
    python scripts/train_e3.py --encoder none --diagnose --tag e3_diag
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from m2_data import geometry_descriptors, make_spec  # noqa: E402
from models_e2 import ResidualHead  # noqa: E402
from models_e3 import ConfigHead, GlobalEncoder, RidgeSetEncoder  # noqa: E402

N_RIDGES = 5


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class E3Data:
    """Precomputed ideal-unit-cell RCWA paired with the MEEP reference."""

    def __init__(self, path: str, device: torch.device):
        d = np.load(path, allow_pickle=True)
        self.path = path
        self.group = d["group"].astype(np.int64)
        self.wl = d["wl"].astype(np.float64)
        rcwa = d["RCWA_abs"].astype(np.float32)
        self.y_abs = d["Y_abs"].astype(np.float32)
        self.rcwa_rt = torch.tensor(rcwa[:, :2], device=device)
        self.y = torch.tensor(self.y_abs, device=device)
        # Canonical M2 feature matrix: the first 13 columns are static
        # (geometry, defect stats, wavelength ratios), the last three are the
        # RCWA (R, T, A).  Using the stored X (rather than rebuilding it) keeps
        # the fixed arm numerically identical to the M2 baseline.
        self.X = torch.tensor(d["X"].astype(np.float32), device=device)
        self.static = self.X[:, :13]

        self.ids = np.unique(self.group)
        gpos = {int(g): i for i, g in enumerate(self.ids)}
        self.gidx_np = np.asarray([gpos[int(g)] for g in self.group])
        self.gidx = torch.tensor(self.gidx_np, device=device)

        specs = [make_spec(int(g)) for g in self.ids]
        self.period = np.asarray([s.period for s in specs], dtype=np.float32)
        self.amplitude = np.asarray([s.amplitude for s in specs], dtype=np.float32)
        self.base = np.asarray([s.base_thickness for s in specs], dtype=np.float32)
        self.gdesc = np.stack([geometry_descriptors(s) for s in specs])

        tokens, glob = [], []
        for s in specs:
            scale = np.asarray(s.ridge_scale, dtype=np.float32)
            miss = np.asarray(s.missing, dtype=bool).astype(np.float32)
            pos = (np.arange(N_RIDGES, dtype=np.float32) - N_RIDGES // 2) \
                / max(1, N_RIDGES // 2)
            tokens.append(np.column_stack([scale, miss, pos]))
            nm = float(miss.sum())
            mi = (np.argmax(miss) - N_RIDGES // 2) / N_RIDGES if nm > 0 else -1.0
            glob.append([s.period / 1.5, s.amplitude / 0.05,
                         s.base_thickness / 0.03, nm, mi])
        self.tokens = torch.tensor(np.stack(tokens), device=device)
        self.glob = torch.tensor(np.asarray(glob, np.float32), device=device)

        self.y_res = self.y[:, :2] - self.rcwa_rt

    def _static(self, amp_scale=1.0, period_scale=1.0):
        """Static feature block ``(n, 13)`` for an assumed effective geometry.

        Columns follow ``m2_data.FEATURE_NAMES``: period, amplitude, base,
        defect stats, wl, 1/wl, wl/period, amp/period, base/period.  Only the
        geometry-dependent columns are rebuilt.
        """
        st = self.X[:, :13].clone()
        if amp_scale == 1.0 and period_scale == 1.0:
            return st
        p = st[:, 0] * period_scale
        amp = st[:, 1] * amp_scale
        st[:, 0] = p
        st[:, 1] = amp
        st[:, 10] = st[:, 8] / p
        st[:, 11] = amp / p
        return st

    def rows_for(self, ids):
        return np.isin(self.group, list(ids))


def split_ids(ids, val_frac, seed):
    ids = np.array(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    return sorted(ids[:-n_val].tolist()), sorted(ids[-n_val:].tolist())


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class HybridModel(torch.nn.Module):
    """Encoder (optional) + config head (optional) + residual correction."""

    def __init__(self, encoder: str = "none", cfg_mode: str = "none",
                 latent: int = 16, hidden: int = 128, depth: int = 4):
        super().__init__()
        self.encoder_kind = encoder
        self.latent = latent if encoder != "none" else 0
        if cfg_mode != "none" and encoder == "none":
            raise ValueError("a config head needs an encoder")
        if encoder == "global":
            self.encoder = GlobalEncoder(8, hidden, latent)
        elif encoder == "ridge":
            self.encoder = RidgeSetEncoder(3, 5, hidden, latent)
        elif encoder == "none":
            self.encoder = None
        else:
            raise ValueError(f"unknown encoder {encoder!r}")

        self.cfg = ConfigHead(latent, cfg_mode) if cfg_mode != "none" else None
        self.cfg_mode = cfg_mode
        self.residual = ResidualHead(
            in_dim=16 + self.latent, hidden=hidden, depth=depth, out_dim=2)

    def encode(self, data: E3Data):
        if self.encoder is None:
            return None, None, None
        if self.encoder_kind == "ridge":
            z = self.encoder(data.tokens, data.glob)
        else:
            z = self.encoder(torch.tensor(data.gdesc, device=data.rcwa_rt.device))
        if self.cfg is None:
            return z, None, None
        log_amp, log_period = self.cfg(z)
        return z, log_amp, log_period


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _solve_fn(data, a, tin_params):
    from rcwa_diff import solve_unit_cell

    def solve(period, amplitude, wl):
        return solve_unit_cell(
            period, amplitude, torch.tensor(a.base_scale, device=data.rcwa_rt.device),
            wl, nh=a.nh, nx=a.nx, n_layers=a.n_layers,
            edge_sharpness=100.0, pol=a.pol, substrate_index=1.5,
            device=a.device, tin_params=tin_params, tin_dataset=a.tin_dataset,
        )

    return solve


def _forward_precomputed(model, data, rows, **_):
    z, _, _ = model.encode(data)
    return z, data.static[rows], data.rcwa_rt[rows, 0], data.rcwa_rt[rows, 1]


def _forward_in_loop(model, data, rows, solve):
    z, log_amp, log_period = model.encode(data)
    rows_np = rows.detach().cpu().numpy()
    gr = data.gidx[rows]
    wls = torch.tensor(data.wl[rows_np], device=gr.device)
    cache = {}
    for g in torch.unique(gr).tolist():
        pos = (gr == g).nonzero(as_tuple=True)[0]
        amp = torch.tensor(data.amplitude[g], device=gr.device)
        per = torch.tensor(data.period[g], device=gr.device)
        if log_amp is not None:
            amp = amp * torch.exp(log_amp[g])
        if log_period is not None:
            per = per * torch.exp(log_period[g])
        for w in torch.unique(wls[pos]).tolist():
            cache[(g, float(w))] = solve(per, amp, float(w))

    r_list, t_list = [], []
    for i in range(len(rows_np)):
        r, t = cache[(int(gr[i]), float(wls[i]))]
        r_list.append(r)
        t_list.append(t)
    r = torch.stack(r_list)
    t = torch.stack(t_list)
    return z, data.static[rows], r, t


def _base_features(data, rows):
    r = data.rcwa_rt[rows, 0]
    t = data.rcwa_rt[rows, 1]
    a = 1.0 - r - t
    return torch.cat([data.static[rows], r[:, None], t[:, None], a[:, None]],
                     dim=1)


def _norm_stats(data, rows):
    base = _base_features(data, rows)
    return base.mean(0), base.std(0).clamp(min=1e-8)


def _loss_and_delta(model, data, rows, norm, fwd, **kw):
    ym, ys, xm, xs = norm
    z, static, r, t = fwd(model, data, rows, **kw)
    a = 1.0 - r - t
    phys = torch.cat([static, r[:, None], t[:, None], a[:, None]], dim=1)
    phys = (phys - xm) / xs
    feat = phys if z is None else torch.cat([z[data.gidx[rows]], phys], dim=1)
    delta = model.residual(feat)
    target = (data.y_res[rows] - ym) / ys
    loss = torch.nn.functional.mse_loss(delta, target)
    return loss, delta, r, t


def _evaluate(model, data, rows, norm, fwd, **kw):
    ym, ys, xm, xs = norm
    model.eval()
    with torch.no_grad():
        _, delta, r, t = _loss_and_delta(model, data, rows, norm, fwd, **kw)
        rc = torch.clamp(r + delta[:, 0] * ys[0] + ym[0], 0.0, 1.0)
        tc = torch.clamp(t + delta[:, 1] * ys[1] + ym[1], 0.0, 1.0)
        ac = torch.clamp(1.0 - rc - tc, 0.0, 1.0)
    rows_np = rows.detach().cpu().numpy()
    groups = data.group[rows_np]
    y = data.y_abs[rows_np]

    def agg(pred_col, true_col):
        vals = []
        for g in np.unique(groups):
            m = groups == g
            vals.append(_rmse(pred_col[m], true_col[m]))
        return float(np.mean(vals)), float(np.std(vals))

    out = {}
    for key, (pred, j) in {
        "R": (rc, 0), "T": (tc, 1), "A": (ac, 2)
    }.items():
        out[f"corr_{key}"] = agg(pred.cpu().numpy(), y[:, j])
        raw = {0: r, 1: t, 2: 1.0 - r - t}[j].cpu().numpy()
        out[f"raw_{key}"] = agg(raw, y[:, j])
    out["corr_A_pooled"] = _rmse(ac.cpu().numpy(), y[:, 2])
    return out


def run_experiment(cfg):
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = E3Data(cfg.data, device)

    from tin_dispersion import get_or_fit

    tin_params = get_or_fit(
        cfg.tin_dataset,
        cache_path=os.path.join(_HERE, "..", "data", f"tin_fit_{cfg.tin_dataset}.json"),
    )
    if not hasattr(cfg, "base_scale"):
        cfg.base_scale = 0.03

    train_ids, val_ids = split_ids(data.ids, cfg.val_frac, cfg.seed)
    if cfg.n_train and cfg.n_train < len(train_ids):
        rng = np.random.default_rng(cfg.seed + 10_000)
        train_ids = sorted(rng.choice(train_ids, cfg.n_train,
                                      replace=False).tolist())

    tr_rows_np = np.where(np.isin(data.group, train_ids))[0]
    va_rows_np = np.where(np.isin(data.group, val_ids))[0]
    tr_rows = torch.tensor(tr_rows_np, device=device)
    va_rows = torch.tensor(va_rows_np, device=device)

    ym = data.y_res[tr_rows].mean(0)
    ys = data.y_res[tr_rows].std(0).clamp(min=1e-8)
    xm, xs = _norm_stats(data, tr_rows)
    norm = (ym, ys, xm, xs)

    model = HybridModel(cfg.encoder, cfg.cfg_mode, cfg.latent,
                        cfg.hidden, cfg.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    solve = _solve_fn(data, cfg, tin_params)
    if cfg.in_loop:
        fwd = lambda model, data, rows, **kw: _forward_in_loop(  # noqa: E731
            model, data, rows, solve)
    else:
        fwd = _forward_precomputed

    if cfg.in_loop and cfg.cfg_mode == "none":
        raise SystemExit("--in-loop with --cfg-mode none is the precomputed "
                         "control; drop --in-loop")

    n = len(tr_rows_np)
    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)
    history, best, best_state, since_best, best_epoch = [], float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, cfg.batch_size):
            rows = tr_rows[perm[i:i + cfg.batch_size]]
            loss, *_ = _loss_and_delta(model, data, rows, norm, fwd)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach()) * len(rows)
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            ev = _evaluate(model, data, va_rows, norm, fwd)
            score = ev["corr_A"][0]
            history.append({"epoch": epoch, "train_mse": tot / n,
                            "val_rmse_A": score})
            if score < best - 1e-6:
                best, best_epoch, since_best = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since_best += cfg.eval_every
            if not cfg.quiet:
                print(f"epoch {epoch:4d}  loss={tot / n:.6f}  "
                      f"val_A={score:.4f}  best={best:.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if since_best >= cfg.patience:
                if not cfg.quiet:
                    print(f"early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ev = _evaluate(model, data, va_rows, norm, fwd)
    wall = time.time() - t0

    trained_epochs = (history[-1]["epoch"] + 1) if history else 0
    result = {
        "config": {k: v for k, v in vars(cfg).items()},
        "n_params": n_params,
        "steps_per_epoch": steps_per_epoch,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "trained_epochs": trained_epochs,
        "samples_seen": trained_epochs * n,
        "best_epoch": best_epoch,
        "wall_s": wall,
        "history": history,
        "metrics": {
            key: {"mean": ev[f"{key}"][0], "std": ev[f"{key}"][1]}
            for key in ("raw_R", "raw_T", "raw_A", "corr_R", "corr_T", "corr_A")
        },
        "pooled_corr_A": ev["corr_A_pooled"],
    }

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, f"e3_{cfg.tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    if cfg.diagnose:
        result["sensitivity"] = config_sensitivity(
            data, model, va_rows, norm, solve)
        with open(os.path.join(cfg.out_dir, f"e3_{cfg.tag}.json"), "w") as fh:
            json.dump(result, fh, indent=2)

    m = result["metrics"]
    print(f"\n[{cfg.tag}] params={n_params}  best_epoch={best_epoch}  "
          f"wall={wall:.0f}s")
    for key in ("R", "T", "A"):
        print(f"  corrected RMSE {key}: {m['corr_' + key]['mean']:.4f} "
              f"± {m['corr_' + key]['std']:.4f}   "
              f"(raw {m['raw_' + key]['mean']:.4f})")
    if "sensitivity" in result:
        s = result["sensitivity"]
        print(f"  frozen-amp best gain: {s['amp']['best_gain']:.4f} "
              f"(at {s['amp']['base']:.4f})")
    return result


# --------------------------------------------------------------------------- #
# Config sensitivity: does the trained residual prefer a different effective
# geometry?  (A frozen-residual sweep, not a retrained oracle.)
# --------------------------------------------------------------------------- #
def _corrected_A(model, data, rows, norm, solve, amp_scale=1.0,
                 period_scale=1.0):
    """Corrected A-RMSE per geometry for a fixed effective-geometry shift."""
    ym, ys, xm, xs = norm
    model.eval()
    rows_np = rows.detach().cpu().numpy()
    static = data._static(amp_scale, period_scale)
    gr = data.gidx[rows]
    wls = torch.tensor(data.wl[rows_np], device=gr.device)
    cache = {}
    for g in torch.unique(gr).tolist():
        pos = (gr == g).nonzero(as_tuple=True)[0]
        per = float(data.period[g]) * period_scale
        amp = float(data.amplitude[g]) * amp_scale
        for w in torch.unique(wls[pos]).tolist():
            cache[(g, float(w))] = _solve_one(data, solve, per, amp, float(w))
    r = torch.stack([cache[(int(gr[i]), float(wls[i]))][0]
                     for i in range(len(rows_np))])
    t = torch.stack([cache[(int(gr[i]), float(wls[i]))][1]
                     for i in range(len(rows_np))])
    z, _, _ = model.encode(data)
    a = 1.0 - r - t
    phys = torch.cat([static[rows], r[:, None], t[:, None], a[:, None]], dim=1)
    phys = (phys - xm) / xs
    feat = phys if z is None else torch.cat([z[data.gidx[rows]], phys], dim=1)
    with torch.no_grad():
        delta = model.residual(feat)
    rc = torch.clamp(r + delta[:, 0] * ys[0] + ym[0], 0.0, 1.0)
    tc = torch.clamp(t + delta[:, 1] * ys[1] + ym[1], 0.0, 1.0)
    ac = (1.0 - rc - tc).cpu().numpy()
    y = data.y_abs[rows_np]
    groups = data.group[rows_np]
    per_spec = {}
    for g in np.unique(groups):
        per_spec[int(g)] = _rmse(ac[groups == g], y[groups == g, 2])
    return per_spec


def _solve_one(data, solve, period, amplitude, wl):
    dev = data.rcwa_rt.device
    return solve(torch.tensor(period, device=dev),
                 torch.tensor(amplitude, device=dev), wl)


def config_sensitivity(data, model, rows, norm, solve):
    """Sweep the *frozen* trained residual over effective-geometry shifts.

    This is a consistency check, not an upper bound: the residual was trained
    at scale 1, so the curve partly measures its extrapolation.  The
    ``best_gain`` is how much a per-geometry effective geometry would buy
    *without retraining*.
    """
    amp_grid = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
    per_grid = [0.97, 0.985, 1.0, 1.015, 1.03]
    out = {}
    for name, grid in (("amp", amp_grid), ("period", per_grid)):
        per_spec = {}
        for m in grid:
            kw = {"amp_scale": m} if name == "amp" else {"period_scale": m}
            ps = _corrected_A(model, data, rows, norm, solve, **kw)
            per_spec[float(m)] = ps
            print(f"  diag {name} x{m:.3f}: mean corrected A "
                  f"{np.mean(list(ps.values())):.4f}", flush=True)
        mean_by_m = {m: float(np.mean(list(ps.values())))
                     for m, ps in per_spec.items()}
        base = mean_by_m[1.0]
        best_per_spec = {g: min(ps[g] for ps in per_spec.values())
                         for g in per_spec[1.0]}
        out[name] = {
            "grid": [float(m) for m in grid],
            "mean_by_m": {str(m): v for m, v in mean_by_m.items()},
            "per_spec": {str(m): {str(g): v for g, v in ps.items()}
                         for m, ps in per_spec.items()},
            "base": base,
            "best_frozen": float(np.mean(list(best_per_spec.values()))),
            "best_gain": float(base - np.mean(list(best_per_spec.values()))),
            "best_multiplier": {str(g): float(
                min(per_spec, key=lambda m: per_spec[m][g]))
                for g in per_spec[1.0]},
        }
    return out


# --------------------------------------------------------------------------- #
def parse(argv=None):
    p = argparse.ArgumentParser(description="Experiment 3")
    p.add_argument("--data", default=os.path.join(
        _HERE, "..", "reports", "m2", "dataset_ideal_nh8.npz"))
    p.add_argument("--encoder", choices=["none", "global", "ridge"], default="none")
    p.add_argument("--cfg-mode", choices=["none", "amp", "amp_period"],
                   default="none")
    p.add_argument("--in-loop", action="store_true",
                   help="run the differentiable RCWA inside training")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--n-train", type=int, default=0,
                   help="cap the number of training geometries (0 = all)")
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--nh", type=int, default=8)
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--n-layers", type=int, default=40)
    p.add_argument("--pol", choices=["s", "p"], default="s")
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2"))
    p.add_argument("--tag", type=str, default="e3")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    cfg = parse(argv)
    cfg.base_scale = 0.03
    run_experiment(cfg)


if __name__ == "__main__":
    main()
