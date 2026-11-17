"""UNSEALING THE TEST SET - ONCE ONLY. Run only after the author has approved.

Pre-registered protocol (log 19/08):
- 3 checkpoints w5m_s{0,1,2}/last.pth (Train+Val, fixed 20 epochs, NOT selected on Val).
- Report EACH seed + the ensemble averaging probabilities.
- Metrics: 4-level acc (standard benchmark), UAR, macro-F1, AUC_dis, AUC23, per-class recall,
  beside the Test base rate .494.
- For comparison: Abedi & Khan 63,9% (Train+Val, MATCHED protocol); Selim 67,48% (Train-only).
- Per-clip results dumped to npz for re-checking - Test is touched only ONCE.
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from p2_train import P2Model
from p2_train_v7 import Clips5


def metrics(p, y):
    pred = p.argmax(1)
    rec = {int(c): float((pred[y == c] == c).mean()) for c in np.unique(y)}
    m = y >= 2
    return dict(
        acc=float((pred == y).mean()),
        uar=float(np.mean(list(rec.values()))),
        f1=float(f1_score(y, pred, average="macro")),
        auc_dis=float(roc_auc_score((y <= 1).astype(int), p[:, 0] + p[:, 1])),
        auc23=float(roc_auc_score((y[m] == 3).astype(int),
                                  p[m, 3] / (p[m, 2] + p[m, 3] + 1e-9))),
        recall=rec)


@torch.no_grad()
def probs(ckpt, ds, dev="cuda", batch=2):
    ck = torch.load(os.path.expanduser(ckpt), map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("subj_head")}
    model = P2Model(n_subj=2)
    model.load_state_dict(sd, strict=False)
    model = model.to(dev).eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=6, pin_memory=True)
    ps, ys = [], []
    for x, y in dl:                               # x [B,5,C,T,H,W]
        B, W = x.shape[:2]
        xx = x.reshape(B * W, *x.shape[2:]).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _, _ = model(xx)
        ps.append(lg.float().softmax(-1).reshape(B, W, -1).mean(1).cpu())
        ys.append(y)
    del model
    torch.cuda.empty_cache()
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def main():
    ds = Clips5("Test", False)
    y = None
    Ps = []
    for s in range(3):
        P, y = probs(f"~/p2/runs/w5m_s{s}/last.pth", ds)
        Ps.append(P)
        m = metrics(P, y)
        print(f"w5m_s{s}/last  {json.dumps(m)}", flush=True)
    ens = np.mean(Ps, 0)
    m = metrics(ens, y)
    print(f"ENSEMBLE      {json.dumps(m)}", flush=True)
    base = max(np.bincount(y, minlength=4)) / len(y)
    print(f"\nn={len(y)}  base_rate={base:.4f}  Abedi(Train+Val)=0.639  Selim=0.6748",
          flush=True)
    np.savez(os.path.expanduser("~/p2/runs/TEST_FINAL_probs.npz"),
             P0=Ps[0], P1=Ps[1], P2=Ps[2], ens=ens, y=y)
    print("TEST_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
