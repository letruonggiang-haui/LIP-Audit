"""Full-frame RGB channel (frozen ResNet18, 16x512) on the 2-vs-3 axis.

The last piece of the picture: the face crop discards the body and posture, which
is the channel Abedi/Selim use. Measured on the mixed pool and WITHIN subjects,
with and without per-subject centering.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HOME = os.path.expanduser("~")  # the server account that ran this script
ROOT = f"{HOME}/daisee/rgb"


def load(split, center):
    import pandas as pd
    d = np.load(f"{ROOT}/{split}.npz", allow_pickle=True)
    X0 = d["X"].astype(np.float32)
    ids0 = [str(i) for i in d["ids"]]
    lab = pd.read_csv(f"{HOME}/daisee/DAiSEE/Labels/{split}Labels.csv")
    lab.columns = [c.strip() for c in lab.columns]
    lab["stem"] = lab["ClipID"].astype(str).str.replace(r"\.(avi|mp4)$", "", regex=True)
    eng = dict(zip(lab["stem"], lab["Engagement"]))
    keep = [i for i, s in enumerate(ids0) if s in eng]
    X = X0[keep]
    ids = [ids0[i] for i in keep]
    y = np.array([eng[ids0[i]] for i in keep], dtype=np.int64)
    if center:
        subj = np.array([i[:6] for i in ids])
        for s in np.unique(subj):
            m = subj == s
            X[m] -= X[m].mean((0, 1), keepdims=True)
    return X, y, ids


class TCN(nn.Module):
    def __init__(self, cin=512, h=256, nc=4):
        super().__init__()
        L, d, c = [], 1, cin
        for _ in range(4):                       # 16 steps -> RF 31, enough
            L += [nn.Conv1d(c, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.GELU()]
            c, d = h, d * 2
        self.f = nn.Sequential(*L)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h * 2, nc))

    def forward(self, x):
        z = self.f(x.transpose(1, 2))
        return self.head(torch.cat([z.mean(-1), z.max(-1).values], -1))


def within(P, y, ids):
    subj = np.array([s[:6] for s in ids])
    m = y >= 2
    sc = P[m, 3] / (P[m, 2] + P[m, 3] + 1e-9)
    t = (y[m] == 3).astype(int)
    sj = subj[m]
    aucs, ws = [], []
    for s in np.unique(sj):
        k = sj == s
        if len(np.unique(t[k])) == 2:
            aucs.append(roc_auc_score(t[k], sc[k]))
            ws.append(k.sum())
    return roc_auc_score(t, sc), float(np.average(aucs, weights=ws))


def run(center, seed=0, epochs=40):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, _ = load("Train", center)
    Xva, yva, iva = load("Validation", center)
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xva).to("cuda")
    model = TCN(cin=Xtr.shape[2]).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss()
    best = (0, 0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            j = perm[i:i + 128]
            opt.zero_grad()
            lf(model(Xt[j].to("cuda")), yt[j].to("cuda")).backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            P = torch.cat([model(Xv[i:i + 512]).softmax(-1).cpu()
                           for i in range(0, len(Xv), 512)]).numpy()
        a, w = within(P, yva, iva)
        if a > best[0]:
            best = (a, w)
    print(f"RGB center={center}: AUC23 tron {best[0]:.4f} | TRONG-nguoi {best[1]:.4f}",
          flush=True)


run(False)
run(True)
