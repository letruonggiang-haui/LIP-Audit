"""E3 (MARLIN variant) - Transformer over the released MARLIN sequences.

Trains on the official EngageNet training split and reports on the official
validation split. The test split is not read: E0 has not returned an answer on
the status of the test labels, and under the pre-registration of Section III-F
the test partition is unsealed once, after E9 freezes the configuration.

Architecture follows Table 2 of the dataset paper: four blocks, eight heads,
head size 256, dropout 0.3, a 128-unit head. Per-clip probabilities are stored
for every run so that E7, E8 and E9 are computed post hoc without retraining.

Usage:
  python p4_train_marlin.py --seed 0 --tag marlin_t0
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

P4 = os.path.expanduser("~/p4")
LEVELS = ["Not-Engaged", "Barely-engaged", "Engaged", "Highly-Engaged"]


# --------------------------------------------------------------------------- #
def load_split(split):
    z = np.load(os.path.join(P4, "data", "marlin_%s.npz" % split), allow_pickle=True)
    X = z["X"].astype(np.float32)
    chunk = np.asarray(z["chunk"], dtype=str)
    lab = pd.read_csv(os.path.join(P4, "data", "labels_%s.csv" % split))
    lab["chunk"] = lab["chunk"].astype(str)
    order = {c: i for i, c in enumerate(chunk)}
    lab = lab[lab["chunk"].isin(order)].copy()
    lab["row"] = lab["chunk"].map(order)
    lab = lab[lab["y"].notna()].reset_index(drop=True)  # SNP dropped
    X = X[lab["row"].values]
    y = lab["y"].values.astype(np.int64)
    return X, y, lab


class SeqTransformer(nn.Module):
    def __init__(self, d_in=1024, d_model=256, n_head=8, n_block=4, d_fc=128,
                 n_class=4, p_drop=0.3, t_max=16):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, t_max, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=p_drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_block)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_fc), nn.GELU(),
                                  nn.Dropout(p_drop), nn.Linear(d_fc, n_class))

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.shape[1]]
        h = self.norm(self.enc(h)).mean(dim=1)
        return self.head(h)


# --------------------------------------------------------------------------- #
def metrics(prob, y):
    from sklearn.metrics import f1_score, cohen_kappa_score, recall_score
    pred = prob.argmax(1)
    return {
        "acc": float((pred == y).mean()),
        "uar": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "mae": float(np.abs(pred - y).mean()),
        "floor": float(np.bincount(y, minlength=4).max() / len(y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()
    tag = args.tag or ("marlin_s%d" % args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, ytr, ltr = load_split("train")
    Xva, yva, lva = load_split("validation")
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    print("train %s  validation %s  subjects %d / %d"
          % (Xtr.shape, Xva.shape, ltr["subj"].nunique(), lva["subj"].nunique()))
    print("class counts train", np.bincount(ytr, minlength=4).tolist(),
          "validation", np.bincount(yva, minlength=4).tolist())

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva).to(dev)

    model = SeqTransformer(d_in=Xtr.shape[-1]).to(dev)
    w = torch.tensor(len(ytr) / (4 * np.bincount(ytr, minlength=4).clip(1)),
                     dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_path = os.path.join(P4, "logs", "%s_log.jsonl" % tag)
    best = {"acc": -1.0}
    best_state = None
    bad = 0
    n = len(ytr)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            k = perm[i:i + args.batch]
            xb, yb = Xtr_t[k].to(dev), ytr_t[k].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(k)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(Xva_t), dim=1).cpu().numpy()
        m = metrics(pv, yva)
        m.update(epoch=ep, train_loss=tot / n, lr=sched.get_last_lr()[0])
        with open(log_path, "a") as fh:
            fh.write(json.dumps(m) + "\n")
        print("ep %2d loss %.4f | val acc %.4f uar %.4f f1 %.4f qwk %.4f mae %.4f"
              % (ep, tot / n, m["acc"], m["uar"], m["macro_f1"], m["qwk"], m["mae"]))
        if m["acc"] > best["acc"]:
            best = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_prob = pv
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop at epoch %d" % ep)
                break

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "mu": mu, "sd": sd, "args": vars(args)},
               os.path.join(P4, "runs", "%s_best.pth" % tag))
    np.savez_compressed(os.path.join(P4, "probs", "%s_val_probs.npz" % tag),
                        P=best_prob, y=yva, chunk=lva["chunk"].values.astype(str),
                        subj=lva["subj"].values, rec=(lva["subj"].astype(str) + "|"
                                                      + lva["sess"].astype(str) + "|"
                                                      + lva["vid"].astype(str)).values,
                        t=lva["t"].values)
    summary = {"tag": tag, "seed": args.seed, "best_epoch": best["epoch"],
               "validation": {k: best[k] for k in
                              ("acc", "uar", "macro_f1", "qwk", "mae", "floor")},
               "args": vars(args), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(P4, "runs", "%s_summary.json" % tag), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("BEST", json.dumps(summary["validation"]))


if __name__ == "__main__":
    main()
