"""AU/gaze/pose channel (300x22, full temporal resolution) on the 2-vs-3 axis.

Why it exists: the visual backbone sees only 16-80 of the 300 frames and ignores
AU/gaze/pose entirely, and AU features had not been measured on the 2-vs-3 axis,
the axis that decides the benchmark. Measured with the same yardstick as
p2_eval_auc23.py so that the two are comparable.

Also writes the per-clip probabilities, to measure how far its errors correlate
with those of the visual model. Uncorrelated errors leave room for fusion;
correlated errors make fusion pointless.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.expanduser("~/daisee/seq")


def metrics(p, y):
    pred = p.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    m = y >= 2
    return dict(acc=float((pred == y).mean()), uar=float(np.mean(rec)),
                f1=float(f1_score(y, pred, average="macro")),
                auc_dis=float(roc_auc_score((y <= 1).astype(int), p[:, 0] + p[:, 1])),
                auc23=float(roc_auc_score((y[m] == 3).astype(int),
                                          p[m, 3] / (p[m, 2] + p[m, 3] + 1e-9))))


class TCN(nn.Module):
    """Dilated TCN: 300 steps -> receptive field covers the whole 10s clip."""

    def __init__(self, cin=22, h=128, nc=4):
        super().__init__()
        L, d = [], 1
        c = cin
        for _ in range(6):                       # dilation 1,2,4,8,16,32 -> RF ~127
            L += [nn.Conv1d(c, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.GELU()]
            c, d = h, d * 2
        self.f = nn.Sequential(*L)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h * 2, nc))

    def forward(self, x):                        # x [B,300,C]
        z = self.f(x.transpose(1, 2))            # [B,h,300]
        return self.head(torch.cat([z.mean(-1), z.max(-1).values], -1))


def load(split, center):
    d = np.load(f"{ROOT}/{split}.npz", allow_pickle=True)
    X, ids, y = d["X"].astype(np.float32), d["ids"], d["y"][:, 1].astype(np.int64)
    if center:                                   # subtract each subject's mean (transductive)
        subj = np.array([i[:6] for i in ids])
        for s in np.unique(subj):
            m = subj == s
            X[m] -= X[m].mean((0, 1), keepdims=True)
    return X, y, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, ytr, _ = load("Train", a.center)
    Xva, yva, iva = load("Validation", a.center)
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6      # standardise with Train statistics
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva).to(dev)

    model = TCN(cin=Xtr.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss()
    best, bestP = None, None
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            opt.zero_grad()
            lossf(model(Xtr_t[j].to(dev)), ytr_t[j].to(dev)).backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            P = torch.cat([model(Xva_t[i:i + 256]).softmax(-1).cpu()
                           for i in range(0, len(Xva_t), 256)]).numpy()
        m = metrics(P, yva)
        if best is None or m["auc23"] > best["auc23"]:
            best, bestP = m, P
        print(f"ep{ep} acc {m['acc']:.3f} auc_dis {m['auc_dis']:.3f} auc23 {m['auc23']:.3f}",
              flush=True)
    print(f"\nDINH  acc {best['acc']:.3f} uar {best['uar']:.3f} f1 {best['f1']:.3f} "
          f"AUC_dis {best['auc_dis']:.3f} | **AUC23 {best['auc23']:.3f}** "
          f"(center={a.center}, seed={a.seed})", flush=True)
    if a.out:
        np.savez(os.path.expanduser(a.out), P=bestP, y=yva, ids=iva)
        print("dumped", a.out, flush=True)


if __name__ == "__main__":
    main()
