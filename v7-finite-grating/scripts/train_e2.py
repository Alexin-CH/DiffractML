"""Experiment 2: end-to-end encoder -> differentiable RCWA -> residual.

Trains a shared geometry encoder that (a) emits a latent ``z_geo`` consumed by
the correction head and (b) predicts an effective amplitude / period for the
differentiable unit-cell RCWA (``rcwa_diff.solve_unit_cell``).  Training is
per geometry over its wavelengths, so gradients flow through TORCWA back into
the encoder.  Early stopping is on held-out corrected absorption RMSE.

Example
-------
    python scripts/train_e2.py --epochs 30 --n-wl 8 --device cuda --tag e2
    python scripts/train_e2.py --epochs 30 --n-wl 8 --device cuda --tag e2_noenc --no-encoder
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
from models_e2 import GeometryEncoder, ResidualHead  # noqa: E402
from rcwa_diff import solve_unit_cell  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description="Experiment 2 end-to-end training")
    p.add_argument("--data", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "dataset.npz"))
    p.add_argument("--pol", type=str, default="s", choices=["s", "p"])
    p.add_argument("--nh", type=int, default=8)
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-wl", type=int, default=8, help="wavelengths per training step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-encoder", action="store_true",
                   help="ablation: z=0 and input geometry (train residual only)")
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2"))
    p.add_argument("--tag", type=str, default="e2")
    return p.parse_args()


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main(argv=None):
    a = parse()
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    d = np.load(a.data, allow_pickle=True)
    Y_abs = d["Y_abs"].astype(np.float32)
    group = d["group"].astype(np.int64)
    wl_all = d["wl"].astype(np.float64)

    ids = np.unique(group)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * a.val_frac)))
    val_ids = sorted(ids[-n_val:].tolist())
    train_ids = sorted(ids[:-n_val].tolist())
    print(f"specs={len(ids)}  train={len(train_ids)}  val={len(val_ids)}")

    from tin_dispersion import get_or_fit

    tin_params = get_or_fit(
        a.tin_dataset,
        cache_path=os.path.join(_HERE, "..", "data", f"tin_fit_{a.tin_dataset}.json"),
    )

    encoder = GeometryEncoder(in_dim=8, hidden=a.hidden, latent=a.latent).to(device)
    residual = ResidualHead(in_dim=a.latent + 3 + 5 + 8, hidden=a.hidden,
                            depth=3).to(device)
    params = list(encoder.parameters()) + list(residual.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-3)
    base_const = 0.03

    def _res_feat(z, R, T, wl, spec, dvec):
        A = 1.0 - R - T
        extra = torch.tensor([
            wl, 1.0 / wl, wl / spec.period,
            spec.amplitude / spec.period, spec.base_thickness / spec.period,
        ], dtype=torch.float32, device=device)
        return torch.cat([z, torch.stack([R, T, A]), extra, dvec])

    def _forward_geometry(idx, wl_list):
        m = group == idx
        wl_np = wl_all[m]
        y = torch.tensor(Y_abs[m], device=device)
        spec = make_spec(idx)
        dvec = torch.tensor(geometry_descriptors(spec), device=device)

        if a.no_encoder:
            z = torch.zeros(a.latent, device=device)
            amp_eff = torch.tensor(spec.amplitude, device=device)
            period_eff = torch.tensor(spec.period, device=device)
        else:
            z, cfg = encoder(dvec)
            amp_eff = torch.tensor(spec.amplitude, device=device) * torch.exp(0.15 * cfg[0])
            period_eff = torch.tensor(spec.period, device=device) * torch.exp(0.15 * cfg[1])

        sel = list(range(len(wl_np))) if wl_list is None else list(wl_list)
        Rraw, Traw, Rc, Tc = [], [], [], []
        loss = torch.zeros((), device=device)
        for j in sel:
            R, T = solve_unit_cell(
                period_eff, amp_eff, torch.tensor(base_const, device=device),
                float(wl_np[j]), nh=a.nh, nx=a.nx, n_layers=a.n_layers,
                edge_sharpness=100.0, pol=a.pol,
                substrate_index=spec.substrate_index, device=a.device,
                tin_params=tin_params, tin_dataset=a.tin_dataset,
            )
            delta = residual(_res_feat(z, R, T, float(wl_np[j]), spec, dvec))
            R_c = torch.clamp(R + delta[0], 0.0, 1.0)
            T_c = torch.clamp(T + delta[1], 0.0, 1.0)
            A_c = torch.clamp(1.0 - R_c - T_c, 0.0, 1.0)
            target = torch.stack([y[j, 0], y[j, 1], y[j, 2]])
            loss = loss + torch.nn.functional.mse_loss(
                torch.stack([R_c, T_c, A_c]), target)
            Rraw.append(float(R.detach())); Traw.append(float(T.detach()))
            Rc.append(float(R_c.detach())); Tc.append(float(T_c.detach()))
        loss = loss / max(1, len(sel))
        return loss, Rraw, Traw, Rc, Tc, sel, y

    def _evaluate():
        encoder.eval(); residual.eval()
        rows = []
        with torch.no_grad():
            for idx in val_ids:
                _, Rr, Tr, Rc, Tc, sel, y = _forward_geometry(idx, None)
                ysel = y[torch.as_tensor(sel, device=device)].cpu().numpy()
                A_r = 1 - np.asarray(Rr) - np.asarray(Tr)
                A_c = 1 - np.asarray(Rc) - np.asarray(Tc)
                rows.append({
                    "spec": idx,
                    "raw_R": _rmse(Rr, ysel[:, 0]), "corr_R": _rmse(Rc, ysel[:, 0]),
                    "raw_T": _rmse(Tr, ysel[:, 1]), "corr_T": _rmse(Tc, ysel[:, 1]),
                    "raw_A": _rmse(A_r, ysel[:, 2]), "corr_A": _rmse(A_c, ysel[:, 2]),
                })
        return rows

    best = float("inf")
    best_state = None
    since = 0
    for epoch in range(a.epochs):
        t0 = time.time()
        encoder.train(); residual.train()
        tot, n = 0.0, 0
        for idx in train_ids:
            m = group == idx
            n_all = int(m.sum())
            n_use = min(a.n_wl, n_all)
            sel = np.linspace(0, n_all - 1, n_use).round().astype(int).tolist()
            loss, *_ = _forward_geometry(idx, sel)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.detach()); n += 1

        rows = _evaluate()
        val_A = float(np.mean([r["corr_A"] for r in rows]))
        if val_A < best - 1e-5:
            best = val_A
            best_state = {
                "encoder": copy.deepcopy(encoder.state_dict()),
                "residual": copy.deepcopy(residual.state_dict()),
            }
            since = 0
        else:
            since += 1
        print(f"epoch {epoch:2d}  loss={tot / max(1, n):.6f}  "
              f"val_rmse_A={val_A:.4f}  best={best:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        if since >= a.patience:
            print(f"early stop at epoch {epoch}")
            break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        residual.load_state_dict(best_state["residual"])
    rows = _evaluate()

    def _agg(key):
        v = np.array([r[key] for r in rows])
        return float(v.mean()), float(v.std())

    os.makedirs(a.out_dir, exist_ok=True)
    report = os.path.join(a.out_dir, f"report_{a.tag}.md")
    lines = [
        "# Experiment 2 — end-to-end encoder → RCWA → residual",
        "",
        f"- backbone: ideal unit cell, nh={a.nh}, nx={a.nx}, {a.n_layers} layers",
        f"- encoder latent={a.latent}, hidden={a.hidden}; "
        f"{'NO-ENCODER ablation' if a.no_encoder else 'learned cfg (amp, period)'}",
        f"- {len(train_ids)} train / {len(val_ids)} val geometries, "
        f"early stop (patience {a.patience}), {a.n_wl} λ/step",
        "",
        "## Held-out metrics (mean ± std over geometries)",
        "",
        "| channel | raw RCWA | corrected |",
        "|---|---:|---:|",
    ]
    for key in ("R", "T", "A"):
        mr, sr = _agg(f"raw_{key}")
        mc, sc = _agg(f"corr_{key}")
        lines.append(f"| RMSE {key} | {mr:.4f} ± {sr:.4f} | {mc:.4f} ± {sc:.4f} |")
    lines += ["", "## Per-geometry RMSE(A)", "",
              "| spec | raw | corrected |", "|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['spec']:03d} | {r['raw_A']:.4f} | {r['corr_A']:.4f} |")
    lines.append("")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(a.out_dir, f"summary_{a.tag}.json"), "w") as fh:
        json.dump({"per_spec": rows, "train_specs": train_ids,
                   "val_specs": val_ids, "config": vars(a)}, fh, indent=2)

    print("\nheld-out:")
    for key in ("R", "T", "A"):
        mr, sr = _agg(f"raw_{key}")
        mc, sc = _agg(f"corr_{key}")
        print(f"  RMSE {key}: raw {mr:.4f}±{sr:.4f}  corr {mc:.4f}±{sc:.4f}")
    print(f"wrote {os.path.normpath(report)}")


if __name__ == "__main__":
    main()
