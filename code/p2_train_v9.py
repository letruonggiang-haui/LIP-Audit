"""GIAI DOAN 4b: backbone VideoMAE (Kinetics-400 fine-tuned, HF) tren faces5.

Vi sao: chan doan giai doan 3 — frozen .659 ~ full-FT .680 -> tran nam o BIEU DIEN
cua MAE-DFER (pretrain DFEW 16k clip). VideoMAE Kinetics pretrain tren video quy mo
lon hon nhieu; day la don "doi bieu dien", don duy nhat chua thu.

Giu NGUYEN goi chong-sup cua v7: temporal aug (1/5 cua so ngau nhien moi epoch),
gop 5 view luc eval, bao cao auc23, luu best theo acc/auc_dis/auc23.
Khac v7: backbone HF VideoMAE [B,T,C,H,W] 224px (resize tu 160), chuan hoa ImageNet.
--merge_val: gop Validation vao Train (giao thuc Abedi & Khan) — khi do eval tren
Validation CHI de ghi duong cong tham khao, KHONG dung chon checkpoint.

Chay: p2_train_v9.py --seed 0 --out ~/p2/runs/vmae_s0
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

os.environ.setdefault("HF_HOME", os.path.expanduser("~/p2/hf"))
from transformers import VideoMAEModel  # noqa: E402

from p2_train_v7 import Clips5  # dung lai y nguyen dataset (160px, chuan hoa MEAN/STD cua v7)

# Clips5 chuan hoa theo MEAN/STD cua MAE-DFER (0.485.../0.229... — trung ImageNet) -> giu.


class VMae(nn.Module):
    def __init__(self, nc=4):
        super().__init__()
        self.bb = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.head = nn.Linear(self.bb.config.hidden_size, nc)

    def forward(self, x):                    # x [B,C,T,H,W] 160px tu Clips5
        x = x.permute(0, 2, 1, 3, 4)         # -> [B,T,C,H,W]
        B, T = x.shape[:2]
        x = F.interpolate(x.reshape(B * T, *x.shape[2:]), size=(224, 224),
                          mode="bilinear", align_corners=False).reshape(B, T, 3, 224, 224)
        z = self.bb(pixel_values=x).last_hidden_state.mean(1)
        return self.head(z)


def metrics(p, y):
    pred = p.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    m = y >= 2
    return dict(acc=float((pred == y).mean()), uar=float(np.mean(rec)),
                f1=float(f1_score(y, pred, average="macro")),
                auc_dis=float(roc_auc_score((y <= 1).astype(int), p[:, 0] + p[:, 1])),
                auc23=float(roc_auc_score((y[m] == 3).astype(int),
                                          p[m, 3] / (p[m, 2] + p[m, 3] + 1e-9))))


@torch.no_grad()
def evaluate5(model, ds, dev, batch):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=6, pin_memory=True)
    ps, ys = [], []
    for x, y in dl:                          # x [B,5,C,T,H,W]
        B, W = x.shape[:2]
        xx = x.reshape(B * W, *x.shape[2:]).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(xx)
        ps.append(lg.float().softmax(-1).reshape(B, W, -1).mean(1).cpu())
        ys.append(y)
    return metrics(torch.cat(ps).numpy(), torch.cat(ys).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)      # effective 32 nhu v7
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--merge_val", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = "cuda"
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"))

    tr = Clips5("Train", True)
    if args.merge_val:
        va_as_train = Clips5("Validation", True)
        tr = torch.utils.data.ConcatDataset([tr, va_as_train])
    va = Clips5("Validation", False)

    model = VMae().to(dev)
    head_ids = {id(p) for p in model.head.parameters()}
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if id(p) not in head_ids],
          "lr": args.lr},
         {"params": list(model.head.parameters()), "lr": args.lr_head}],
        weight_decay=args.wd)
    steps_ep = (len(tr) + args.batch * args.accum - 1) // (args.batch * args.accum)
    total = steps_ep * args.epochs
    warm = steps_ep * args.warmup

    def lr_at(g):
        if g < warm:
            return g / max(1, warm)
        import math
        return 0.5 * (1 + math.cos(math.pi * (g - warm) / max(1, total - warm)))

    lossf = nn.CrossEntropyLoss()
    log = open(out / "log.jsonl", "a")
    best = {"acc": -1, "auc_dis": -1, "auc23": -1}
    gstep = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        dl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                         num_workers=6, pin_memory=True, drop_last=True)
        losses = []
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(x.to(dev, non_blocking=True)),
                             y.to(dev, non_blocking=True)) / args.accum
            loss.backward()
            losses.append(loss.item() * args.accum)
            if (i + 1) % args.accum == 0:
                for gi, g in enumerate(opt.param_groups):
                    g["lr"] = (args.lr if gi == 0 else args.lr_head) * lr_at(gstep)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
        m = evaluate5(model, va, dev, max(1, args.batch))
        rec = dict(ep=ep, loss=float(np.mean(losses)),
                   mins=round((time.time() - t0) / 60, 1), **m)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args)},
                   out / "last.pth")
        for key in ("acc", "auc_dis", "auc23"):
            if m[key] > best[key]:
                best[key] = m[key]
                torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args),
                            "val": m}, out / f"best_{key}.pth")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
