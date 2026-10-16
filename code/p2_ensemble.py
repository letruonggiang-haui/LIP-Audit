"""Danh gia ENSEMBLE nhieu checkpoint (+ TTA) tren Validation — do dong gop tung thanh phan.

Dung cho buoc day bang chuan: ensemble cac best_auc.pth cua nhieu seed, trung binh XAC SUAT.
In ca ket qua tung checkpoint rieng le lan ensemble luy tien (1, 1+2, 1+2+3...) de thay
ensemble co thuc su them gia tri hay chi lap lai model tot nhat.

Test VAN NIEM PHONG — script nay chi cham Validation.

Chay: p2_ensemble.py --ckpt ~/p2/runs/a/best_auc.pth ~/p2/runs/b/best_auc.pth [--no_tta]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from p2_train import P2Model
from p2_train_v6 import ClipsAug


def metrics(p, y):
    pred = p.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    dis = (y <= 1).astype(int)
    return dict(acc=float((pred == y).mean()), uar=float(np.mean(rec)),
                f1=float(f1_score(y, pred, average="macro")),
                auc=float(roc_auc_score(dis, p[:, 0] + p[:, 1])))


@torch.no_grad()
def probs_for(ckpt, ds, dev, batch, tta):
    ck = torch.load(os.path.expanduser(ckpt), map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("subj_head")}
    model = P2Model(n_subj=2)
    model.load_state_dict(sd, strict=False)
    model = model.to(dev).eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=6, pin_memory=True)
    ps, ys = [], []
    for x, y, _ in dl:
        x = x.to(dev, non_blocking=True)
        views = [x]
        if tta:
            B, C, T, H, W = x.shape
            c = int(H * 0.875)
            o = (H - c) // 2
            xc = x[:, :, :, o:o + c, o:o + c].reshape(B * C, T, c, c)
            xc = F.interpolate(xc, size=(H, W), mode="bilinear",
                               align_corners=False).reshape(B, C, T, H, W)
            views.append(xc)
            views = views + [torch.flip(v, dims=[4]) for v in views]
        acc = 0
        for v in views:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _ = model(v)
            acc = acc + logits.float().softmax(-1)
        ps.append((acc / len(views)).cpu())
        ys.append(y)
    del model
    torch.cuda.empty_cache()
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no_tta", action="store_true")
    args = ap.parse_args()
    ds = ClipsAug("Validation", False)
    tta = not args.no_tta
    print(f"TTA: {'4 view' if tta else 'tat'} | {len(args.ckpt)} checkpoint", flush=True)

    allp, y = [], None
    for c in args.ckpt:
        p, y = probs_for(c, ds, "cuda", args.batch, tta)
        allp.append(p)
        m = metrics(p, y)
        name = "/".join(c.rstrip("/").split("/")[-2:])
        print(f"  [rieng] {name:38s} acc {m['acc']:.3f} uar {m['uar']:.3f} "
              f"f1 {m['f1']:.3f} AUC {m['auc']:.3f}", flush=True)

    print("  --- ensemble luy tien (trung binh xac suat) ---", flush=True)
    for k in range(2, len(allp) + 1):
        m = metrics(np.mean(allp[:k], axis=0), y)
        print(f"  [ens {k}] acc {m['acc']:.3f} uar {m['uar']:.3f} f1 {m['f1']:.3f} "
              f"AUC {m['auc']:.3f}", flush=True)
    print(f"  san lop da so Validation = {max(np.bincount(y))/len(y):.4f}", flush=True)


if __name__ == "__main__":
    main()
