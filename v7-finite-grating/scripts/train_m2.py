"""Train and evaluate the M2 learned spectral correction (torch env).

Residual regression on top of the de-embedded fast RCWA spectrum.  Evaluation
is split *by geometry* (spec id), never by wavelength.

Example
-------
    python scripts/train_m2.py --epochs 400
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from models_m2 import CorrectionMLP  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description="Train M2 spectral correction")
    p.add_argument("--data", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "dataset.npz"))
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=150,
                   help="early-stopping patience on val RMSE(A)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2"))
    p.add_argument("--tag", type=str, default="run")
    return p.parse_args()


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main(argv=None):
    a = parse()
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    d = np.load(a.data, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y_res = d["Y_res"].astype(np.float32)
    Y_abs = d["Y_abs"].astype(np.float32)
    RCWA = d["RCWA_abs"].astype(np.float32)
    group = d["group"].astype(np.int64)
    wl = d["wl"].astype(np.float32)
    feature_names = [str(s) for s in d["feature_names"]]

    ids = np.unique(group)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * a.val_frac)))
    val_ids = sorted(ids[-n_val:].tolist())
    train_ids = sorted(ids[:-n_val].tolist())
    tr = np.isin(group, train_ids)
    va = np.isin(group, val_ids)
    print(f"samples={len(X)}  specs={len(ids)}  train specs={len(train_ids)}  "
          f"val specs={len(val_ids)}")

    xm, xs = X[tr].mean(0), X[tr].std(0).clip(min=1e-8)
    ym, ys = Y_res[tr].mean(0), Y_res[tr].std(0).clip(min=1e-8)

    Xtr = torch.tensor((X[tr] - xm) / xs, device=device)
    Ytr = torch.tensor((Y_res[tr] - ym) / ys, device=device)
    Xva = torch.tensor((X[va] - xm) / xs, device=device)

    model = CorrectionMLP(in_dim=X.shape[1], hidden=a.hidden, depth=a.depth,
                          out_dim=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    loss_fn = nn.MSELoss()

    n = len(Xtr)
    va_RCWA_np = RCWA[va]
    va_Y_np = Y_abs[va]

    def val_rmse_A():
        model.eval()
        with torch.no_grad():
            rp = (model(Xva) * torch.tensor(ys, device=device)
                  + torch.tensor(ym, device=device)).cpu().numpy()
        R = np.clip(va_RCWA_np[:, 0] + rp[:, 0], 0.0, 1.0)
        T = np.clip(va_RCWA_np[:, 1] + rp[:, 1], 0.0, 1.0)
        A = np.clip(1.0 - R - T, 0.0, 1.0)
        return float(np.sqrt(np.mean((A - va_Y_np[:, 2]) ** 2)))

    best = float("inf")
    best_state = None
    since_best = 0
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, a.batch_size):
            idx = perm[i:i + a.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)

        if epoch % a.eval_every == 0 or epoch == a.epochs - 1:
            score = val_rmse_A()
            if score < best - 1e-6:
                best = score
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                since_best = 0
            else:
                since_best += a.eval_every
            print(f"epoch {epoch:4d}  train_mse={tot / n:.6f}  "
                  f"val_rmse_A={score:.4f}  best={best:.4f}")
            if since_best >= a.patience:
                print(f"early stop at epoch {epoch} (patience {a.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- evaluation (by geometry) ---- #
    model.eval()
    with torch.no_grad():
        res_pred = (model(Xva) * torch.tensor(ys, device=device)
                    + torch.tensor(ym, device=device)).cpu().numpy()
    va_X = X[va]
    va_RCWA = RCWA[va]
    va_Y = Y_abs[va]
    va_group = group[va]

    R_raw, T_raw = va_RCWA[:, 0], va_RCWA[:, 1]
    R_c = np.clip(R_raw + res_pred[:, 0], 0.0, 1.0)
    T_c = np.clip(T_raw + res_pred[:, 1], 0.0, 1.0)
    A_c = np.clip(1.0 - R_c - T_c, 0.0, 1.0)
    Y_corr = np.column_stack([R_c, T_c, A_c])

    rows = []
    for sid in val_ids:
        m = va_group == sid
        row = {"spec": sid}
        for j, key in enumerate(("R", "T", "A")):
            row[f"raw_{key}"] = _rmse(va_RCWA[m, j], va_Y[m, j])
            row[f"corr_{key}"] = _rmse(Y_corr[m, j], va_Y[m, j])
        rows.append(row)

    def _agg(key):
        return (float(np.mean([r[key] for r in rows])),
                float(np.std([r[key] for r in rows]))) if rows else (float("nan"),)

    pooled = {
        key: {
            "raw": _rmse(va_RCWA[:, j], va_Y[:, j]),
            "corr": _rmse(Y_corr[:, j], va_Y[:, j]),
        }
        for j, key in enumerate(("R", "T", "A"))
    }

    os.makedirs(a.out_dir, exist_ok=True)
    ckpt = os.path.join(a.out_dir, f"model_{a.tag}.pt")
    torch.save({
        "state": model.state_dict(),
        "in_dim": X.shape[1], "hidden": a.hidden, "depth": a.depth,
        "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
        "feature_names": feature_names, "args": vars(a),
        "train_specs": train_ids, "val_specs": val_ids,
    }, ckpt)

    report = os.path.join(a.out_dir, f"report_{a.tag}.md")
    lines = [
        "# M2 — learned spectral correction (fast RCWA → MEEP)",
        "",
        f"- dataset: {len(X)} samples, {len(ids)} geometries "
        f"({len(train_ids)} train / {len(val_ids)} val, split by geometry)",
        f"- features ({len(feature_names)}): {', '.join(feature_names)}",
        f"- model: MLP hidden={a.hidden}, depth={a.depth}, residual on (R, T)",
        f"- trained {a.epochs} epochs, lr={a.lr}, wd={a.weight_decay}",
        "",
        "## Validation metrics (mean ± std over held-out geometries)",
        "",
        "| channel | raw RCWA | corrected | improvement |",
        "|---|---:|---:|---:|",
    ]
    for key in ("R", "T", "A"):
        mr, sr = _agg(f"raw_{key}")
        mc, sc = _agg(f"corr_{key}")
        imp = 100.0 * (mr - mc) / mr if mr > 0 else float("nan")
        lines.append(f"| RMSE {key} | {mr:.4f} ± {sr:.4f} | "
                     f"{mc:.4f} ± {sc:.4f} | {imp:+.1f}% |")
    lines += [
        "",
        "## Pooled validation RMSE",
        "",
        "| channel | raw RCWA | corrected |",
        "|---|---:|---:|",
    ]
    for key in ("R", "T", "A"):
        lines.append(f"| {key} | {pooled[key]['raw']:.4f} | "
                     f"{pooled[key]['corr']:.4f} |")
    lines += [
        "",
        "## Per-geometry RMSE(A)",
        "",
        "| spec | raw | corrected |",
        "|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['spec']:03d} | {r['raw_A']:.4f} | {r['corr_A']:.4f} |")
    lines.append("")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # machine-readable summary
    with open(os.path.join(a.out_dir, f"summary_{a.tag}.json"), "w") as fh:
        json.dump({"pooled": pooled, "per_spec": rows,
                   "train_specs": train_ids, "val_specs": val_ids}, fh, indent=2)

    print("\nvalidation (mean ± std over held-out geometries):")
    for key in ("R", "T", "A"):
        mr, sr = _agg(f"raw_{key}")
        mc, sc = _agg(f"corr_{key}")
        imp = 100.0 * (mr - mc) / mr if mr > 0 else float("nan")
        print(f"  RMSE {key}: raw {mr:.4f}±{sr:.4f}  corr {mc:.4f}±{sc:.4f}  "
              f"({imp:+.1f}%)")
    print(f"\nwrote {os.path.normpath(report)}")
    print(f"wrote {os.path.normpath(ckpt)}")


if __name__ == "__main__":
    main()
