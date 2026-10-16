"""P2 giai doan 3 — huan luyen tren 5 cua so x 16 khung sr4 (khop domain pretraining).

Khac ban truoc o DUNG BA cho, tat ca deu la thieu sot da ghi nhan:
  1. Du lieu `faces5`: moi clip co 5 cua so 2,1s (dung nhu DFEW pretrain) thay vi 16 khung
     trai deu 10s. Dung 80/300 khung thay vi 16/300.
  2. TANG CUONG THOI GIAN luc train: moi lan lay mau chon NGAU NHIEN 1 trong 5 cua so.
     Truoc gio khung co dinh -> khong co temporal aug (han che ghi tu Phase 0).
  3. GOP DA CUA SO luc eval: chay ca 5 cua so, trung binh XAC SUAT (giong test_num_segment).

Bao cao them **AUC muc 2 vs muc 3** — day moi la truc quyet dinh accuracy cua bang chuan
(95% du lieu nam o muc 2,3), chi so ma toan bo du an truoc day khong theo doi.

Chay: p2_train_v7.py --seed 0 --out ~/p2/runs/w5_s0
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from p2_train import MEAN, STD, P2Model, load_pretrained
from p2_train_v6 import param_groups_layer_decay

DAISEE = Path(os.path.expanduser("~/daisee"))
FACES5 = Path(os.path.expanduser("~/p2/faces5"))


class Clips5(torch.utils.data.Dataset):
    """X la [N,5,16,160,160,3]. train: chon ngau nhien 1 cua so. eval: tra ca 5."""

    def __init__(self, split, train_aug):
        self.X = np.load(FACES5 / f"{split}_X.npy", mmap_mode="r")
        ids = [s for s in np.load(FACES5 / f"{split}_ids.npy")]
        lab = pd.read_csv(DAISEE / "DAiSEE" / "Labels" / f"{split}Labels.csv")
        lab.columns = [c.strip() for c in lab.columns]
        lab["stem"] = lab["ClipID"].astype(str).str.replace(r"\.(avi|mp4)$", "", regex=True)
        eng = dict(zip(lab["stem"], lab["Engagement"]))
        self.rows = [(i, s) for i, s in enumerate(ids) if s in eng]
        self.y = np.array([eng[s] for _, s in self.rows], dtype=np.int64)
        self.train_aug = train_aug
        print(f"{split}: {len(self.rows)} clip co nhan / {len(ids)} trich duoc, "
              f"{self.X.shape[1]} cua so", flush=True)

    def __len__(self):
        return len(self.rows)

    def _prep(self, arr, aug):
        x = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
        x = x.permute(3, 0, 1, 2)  # [C,T,H,W]
        if aug:
            s = random.uniform(0.7, 1.0)
            ar = random.uniform(3 / 4, 4 / 3)
            h = min(160, int(round(160 * (s * ar) ** 0.5)))
            w = min(160, int(round(160 * (s / ar) ** 0.5)))
            y0, x0 = random.randint(0, 160 - h), random.randint(0, 160 - w)
            x = F.interpolate(x[:, :, y0:y0 + h, x0:x0 + w], size=(160, 160),
                              mode="bilinear", align_corners=False)
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])
        return (x - MEAN) / STD

    def __getitem__(self, k):
        i, _ = self.rows[k]
        if self.train_aug:
            w = random.randrange(self.X.shape[1])          # tang cuong thoi gian
            return self._prep(self.X[i, w], True), self.y[k]
        xs = torch.stack([self._prep(self.X[i, w], False) for w in range(self.X.shape[1])])
        return xs, self.y[k]                                # [5,C,T,H,W]


@torch.no_grad()
def evaluate5(model, ds, dev, batch):
    """Chay ca 5 cua so, trung binh xac suat. Bao cao them AUC muc2-vs-muc3."""
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=6, pin_memory=True)
    ps, ys = [], []
    for xs, y in dl:
        B, W = xs.shape[0], xs.shape[1]
        x = xs.reshape(B * W, *xs.shape[2:]).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = model(x)
        p = logits.float().softmax(-1).reshape(B, W, -1).mean(1)
        ps.append(p.cpu())
        ys.append(y)
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    pred = p.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    dis = (y <= 1).astype(int)
    m23 = y >= 2
    # AUC muc 3 vs muc 2 tren rieng phan du lieu muc>=2 — truc quyet dinh bang chuan
    auc23 = float(roc_auc_score((y[m23] == 3).astype(int),
                                (p[m23, 3] / (p[m23, 2] + p[m23, 3] + 1e-9))))
    model.train()
    return dict(acc=float((pred == y).mean()), uar=float(np.mean(rec)),
                f1=float(f1_score(y, pred, average="macro")),
                auc_dis=float(roc_auc_score(dis, p[:, 0] + p[:, 1])), auc23=auc23)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--layer_decay", type=float, default=1.0)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    dev = "cuda"

    tr = Clips5("Train", True)
    va = Clips5("Validation", False)
    lt = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=8,
                                     pin_memory=True, drop_last=True, persistent_workers=True)

    model = P2Model(n_subj=2)
    load_pretrained(model)
    model = model.to(dev).train()
    opt = torch.optim.AdamW(
        param_groups_layer_decay(model, args.lr, args.lr_head, args.wd, args.layer_decay),
        betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_ep = len(lt) // args.accum
    total, warm = args.epochs * steps_per_ep, args.warmup * steps_per_ep
    crit = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    def set_lr(step):
        k = step / max(1, warm) if step < warm else \
            0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, total - warm)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * k

    best = {"acc": -1.0, "auc_dis": -1.0, "auc23": -1.0}
    gstep = 0
    log = open(out / "log.jsonl", "a")
    for ep in range(args.epochs):
        t0, losses = time.time(), []
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(lt):
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                le, _, _ = model(x)
                loss = crit(le, y)
            (loss / args.accum).backward()
            losses.append(loss.item())
            if (it + 1) % args.accum == 0:
                set_lr(gstep)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
        # eval nap 5 cua so/clip -> batch hieu dung gap 5. Chia 4 de giu VRAM ~5GB
        # (GPU dung chung, job nguoi khac ~13GB; da OOM mot lan vi khong tinh cho nay).
        m = evaluate5(model, va, dev, max(1, args.batch // 4))
        rec = dict(ep=ep, loss=float(np.mean(losses)), mins=round((time.time() - t0) / 60, 1), **m)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args)}, out / "last.pth")
        for key in ("acc", "auc_dis", "auc23"):
            if m[key] > best[key]:
                best[key] = m[key]
                torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args),
                            "val": m}, out / f"best_{key}.pth")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
