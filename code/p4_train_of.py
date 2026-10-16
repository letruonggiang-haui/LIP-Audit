"""E3 - the ICMI 2023 baseline families over the 98-dimensional OpenFace
descriptors of EngageNet.

Five model families share one training loop, one split, one metric set and one
probability dump, so that later protocol variants (E7, E8, E9) are computed post
hoc from stored predictions without retraining anything:

  mlp          the whole-chunk 98-d vector, no temporal axis
  lstm         two-layer bidirectional LSTM over the sub-window sequence
  cnnlstm      two 1-D convolutions then an LSTM
  tcn          dilated causal convolutions with residual blocks
  transformer  four blocks, eight heads, head size 256, dropout 0.3, FC 128

The published validation figures these are measured against are 67.69, 67.51,
67.79 and 69.10 for LSTM, CNN-LSTM, TCN and Transformer. The sub-window sequence
is our reading of the baseline input and is documented as such in
`p4_reduce_openface.py`; a gap between our numbers and the published ones is
reported as a failure to reproduce under the documented configuration, not as an
adjudication of the original.

The test split is not read. E0 has not returned.

Usage:  python p4_train_of.py --model transformer --seed 0
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


# --------------------------------------------------------------------------- #
def load_split(split):
    z = np.load(os.path.join(P4, "feats", "openface98_%s.npz" % split), allow_pickle=True)
    Xw = z["X"].astype(np.float32)
    Xs = z["X_seq"].astype(np.float32)
    chunk = np.asarray(z["chunk"], dtype=str)
    frames_ok = z["frames_ok"]
    lab = pd.read_csv(os.path.join(P4, "data", "labels_%s.csv" % split))
    lab["chunk"] = lab["chunk"].astype(str)
    order = {c: i for i, c in enumerate(chunk)}
    absent = sorted(set(lab["chunk"]) - set(order))
    if absent:
        raise SystemExit(
            "%s: %d of %d labelled chunks have no descriptor (first: %s). "
            "Refusing to train on a subset: the join would drop them silently "
            "and the accuracy would look ordinary and be wrong."
            % (split, len(absent), len(lab), absent[:3]))
    lab["row"] = lab["chunk"].map(order)
    lab = lab[lab["y"].notna()].reset_index(drop=True)  # SNP dropped
    rows = lab["row"].values
    lab["frames_ok"] = frames_ok[rows]
    return Xw[rows], Xs[rows], lab["y"].values.astype(np.int64), lab


# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, d_in, n_class=4, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(p), nn.Linear(128, n_class))

    def forward(self, x):
        return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, d_in, hidden=128, layers=2, n_class=4, p=0.3):
        super().__init__()
        self.rnn = nn.LSTM(d_in, hidden, layers, batch_first=True,
                           bidirectional=True, dropout=p if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * hidden, 128),
                                  nn.GELU(), nn.Dropout(p), nn.Linear(128, n_class))

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h.mean(dim=1))


class CNNLSTM(nn.Module):
    def __init__(self, d_in, ch=128, hidden=128, n_class=4, p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(p))
        self.rnn = nn.LSTM(ch, hidden, 1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * hidden, 128),
                                  nn.GELU(), nn.Dropout(p), nn.Linear(128, n_class))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.head(h.mean(dim=1))


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, dil, p):
        super().__init__()
        self.pad = (3 - 1) * dil
        self.c1 = nn.Conv1d(c_in, c_out, 3, dilation=dil)
        self.c2 = nn.Conv1d(c_out, c_out, 3, dilation=dil)
        self.n1, self.n2 = nn.BatchNorm1d(c_out), nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(p)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = nn.functional.pad(x, (self.pad, 0))
        h = self.drop(nn.functional.gelu(self.n1(self.c1(h))))
        h = nn.functional.pad(h, (self.pad, 0))
        h = self.drop(nn.functional.gelu(self.n2(self.c2(h))))
        return nn.functional.gelu(h + self.down(x))


class TCN(nn.Module):
    def __init__(self, d_in, ch=128, n_block=3, n_class=4, p=0.3):
        super().__init__()
        blocks, c = [], d_in
        for b in range(n_block):
            blocks.append(TCNBlock(c, ch, 2 ** b, p))
            c = ch
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(ch, 128), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(128, n_class))

    def forward(self, x):
        h = self.body(x.transpose(1, 2))
        return self.head(h[:, :, -1])


class SeqTransformer(nn.Module):
    def __init__(self, d_in, d_model=256, n_head=8, n_block=4, d_fc=128,
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


BUILDERS = {"mlp": MLP, "lstm": LSTMNet, "cnnlstm": CNNLSTM,
            "tcn": TCN, "transformer": SeqTransformer}


# --------------------------------------------------------------------------- #
def metrics(prob, y):
    from sklearn.metrics import f1_score, cohen_kappa_score, recall_score
    pred = prob.argmax(1)
    return {"acc": float((pred == y).mean()),
            "uar": float(recall_score(y, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
            "mae": float(np.abs(pred - y).mean()),
            "floor": float(np.bincount(y, minlength=4).max() / len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()
    tag = args.tag or ("of_%s_s%d" % (args.model, args.seed))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xw_tr, Xs_tr, ytr, ltr = load_split("train")
    Xw_va, Xs_va, yva, lva = load_split("validation")
    static = args.model == "mlp"
    Xtr, Xva = (Xw_tr, Xw_va) if static else (Xs_tr, Xs_va)
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    print("%s: train %s validation %s  subjects %d / %d"
          % (args.model, Xtr.shape, Xva.shape, ltr["subj"].nunique(), lva["subj"].nunique()))
    print("clips with no tracked frame: train %d, validation %d"
          % (int((ltr["frames_ok"] == 0).sum()), int((lva["frames_ok"] == 0).sum())))

    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva).to(dev)
    model = BUILDERS[args.model](Xtr.shape[-1]).to(dev)
    w = torch.tensor(len(ytr) / (4 * np.bincount(ytr, minlength=4).clip(1)),
                     dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_path = os.path.join(P4, "logs", "%s_log.jsonl" % tag)
    best, best_state, best_prob, bad = {"acc": -1.0}, None, None, 0
    last, last_prob = None, None
    n = len(ytr)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            k = perm[i:i + args.batch]
            if len(k) < 2:
                continue
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
        m.update(epoch=ep, train_loss=tot / n)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(m) + "\n")
        print("ep %2d loss %.4f | val acc %.4f uar %.4f f1 %.4f qwk %.4f mae %.4f"
              % (ep, tot / n, m["acc"], m["uar"], m["macro_f1"], m["qwk"], m["mae"]),
              flush=True)
        last, last_prob = m, pv
        if m["acc"] > best["acc"]:
            best, bad = m, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_prob = pv
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop at epoch %d" % ep)
                break

    torch.save({"state": best_state, "mu": mu, "sd": sd, "args": vars(args)},
               os.path.join(P4, "runs", "%s_best.pth" % tag))
    # Both checkpoints are dumped. E9 has to rank systems under selection rules
    # this run did not use, and it is required to do that from stored
    # predictions rather than by retraining fifteen runs.
    np.savez_compressed(
        os.path.join(P4, "probs", "%s_val_probs.npz" % tag),
        P=best_prob, P_last=last_prob, y=yva, chunk=lva["chunk"].values.astype(str),
        subj=lva["subj"].values,
        rec=(lva["subj"].astype(str) + "|" + lva["sess"].astype(str) + "|"
             + lva["vid"].astype(str)).values,
        t=lva["t"].values, frames_ok=lva["frames_ok"].values)
    summary = {"tag": tag, "model": args.model, "seed": args.seed,
               "best_epoch": best["epoch"], "last_epoch": last["epoch"],
               "selection_rule": ("epoch chosen by accuracy on the same "
                                  "validation split that is reported; this is "
                                  "the published baselines' protocol and it is "
                                  "optimistic. The last-epoch probabilities are "
                                  "stored in the same file as P_last so that a "
                                  "selection-free reading is available."),
               "validation": {k: best[k] for k in
                              ("acc", "uar", "macro_f1", "qwk", "mae", "floor")},
               "validation_last_epoch": {k: last[k] for k in
                                         ("acc", "uar", "macro_f1", "qwk", "mae")},
               "args": vars(args), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(P4, "runs", "%s_summary.json" % tag), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("BEST", json.dumps(summary["validation"]))


if __name__ == "__main__":
    main()
