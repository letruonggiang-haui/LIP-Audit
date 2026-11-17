"""Evaluate ONE checkpoint (any) on Validation, reporting everything INCLUDING auc23.

Why it exists: the 2-vs-3 numbers of the OLD models (.595/.598) were measured with a LINEAR
PROBE on the embedding; the `auc23` of `p2_train_v7.py` is measured with the model's OWN
CLASSIFIER HEAD. The two measurements differ -> comparing them directly is wrong. This script
computes auc23 **with the classifier head** for both kinds of model, on the same Validation
set, so that the comparison is legitimate.

  --data faces   : old set [N,16,...]      (1 view)
  --data faces5  : new set [N,5,16,...]    (mean over 5 windows)

Test STAYS SEALED.

Run: p2_eval_auc23.py --ckpt ~/p2/runs/crudetta_s0/best_auc.pth --data faces
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from p2_train import P2Model


def metrics(p, y):
    pred = p.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    m23 = y >= 2
    return dict(
        acc=float((pred == y).mean()), uar=float(np.mean(rec)),
        f1=float(f1_score(y, pred, average="macro")),
        auc_dis=float(roc_auc_score((y <= 1).astype(int), p[:, 0] + p[:, 1])),
        auc23=float(roc_auc_score((y[m23] == 3).astype(int),
                                  p[m23, 3] / (p[m23, 2] + p[m23, 3] + 1e-9))),
        n23=int(m23.sum()))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", choices=["faces", "faces5"], required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda"

    if args.data == "faces5":
        from p2_train_v7 import Clips5
        ds = Clips5("Validation", False)
        bs = max(1, args.batch // 4)
    else:
        from p2_train import Clips
        ds = Clips("Validation", False)
        bs = args.batch

    ck = torch.load(os.path.expanduser(args.ckpt), map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("subj_head")}
    model = P2Model(n_subj=2)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to(dev).eval()
    print(f"ckpt ep{ck.get('ep')} | missing={len(missing)} unexpected={len(unexpected)} "
          f"| data={args.data}", flush=True)

    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False,
                                     num_workers=6, pin_memory=True)
    ps, ys = [], []
    for batch in dl:
        x, y = batch[0], batch[1]
        if args.data == "faces5":                       # [B,5,C,T,H,W] -> mean over 5 views
            B, W = x.shape[0], x.shape[1]
            xx = x.reshape(B * W, *x.shape[2:]).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _, _ = model(xx)
            p = lg.float().softmax(-1).reshape(B, W, -1).mean(1)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _, _ = model(x.to(dev, non_blocking=True))
            p = lg.float().softmax(-1)
        ps.append(p.cpu())
        ys.append(y)
    m = metrics(torch.cat(ps).numpy(), torch.cat(ys).numpy())
    name = "/".join(args.ckpt.rstrip("/").split("/")[-2:])
    print(f"{name:34s} acc {m['acc']:.3f} uar {m['uar']:.3f} f1 {m['f1']:.3f} "
          f"AUC_dis {m['auc_dis']:.3f} | **AUC23 {m['auc23']:.3f}** (n={m['n23']})", flush=True)


if __name__ == "__main__":
    main()
