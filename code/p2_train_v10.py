"""GIAI DOAN 4a: v7 + --merge_val (giao thuc Abedi & Khan: gop Validation vao Train).

Khac v7 DUY NHAT o du lieu train. Khi --merge_val:
- Train = Train + Validation, ca hai deu temporal aug.
- Eval moi epoch tren Validation CHI DE GHI THAM KHAO (Val nam trong train, so nay
  khong duoc dung chon checkpoint hay bao cao).
- Checkpoint chinh thuc = last.pth, ngan sach 20 epoch co dinh (dang ky truoc) —
  hop le vi goi w5 da chung minh cuoi ~ dinh. Van luu best_* de doi chieu sau.

Chay: p2_train_v10.py --seed 0 --merge_val --out ~/p2/runs/w5m_s0
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

from p2_train import P2Model, load_pretrained
from p2_train_v6 import param_groups_layer_decay
from p2_train_v7 import Clips5, evaluate5


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
    ap.add_argument("--merge_val", action="store_true")
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
    if args.merge_val:
        tr = torch.utils.data.ConcatDataset([tr, Clips5("Validation", True)])
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
        m = evaluate5(model, va, dev, max(1, args.batch // 4))   # THAM KHAO (Val in-train)
        rec = dict(ep=ep, loss=float(np.mean(losses)), mins=round((time.time() - t0) / 60, 1),
                   ref_only=bool(args.merge_val), **m)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args)}, out / "last.pth")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
