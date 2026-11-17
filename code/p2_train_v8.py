"""P2 stage 3b - ORDINAL (CORAL) instead of 4-class CE. A final attempt aimed at the 2-vs-3 axis.

Why: engagement is a scale with an ORDER 0<1<2<3, but 4-class CE treats level 2 and level 3 as
two UNRELATED categories - while all of the standard benchmark's accuracy sits at the 2|3
boundary. CORAL turns the problem into 3 ordered cumulative questions: P(y>0), P(y>1), P(y>2),
sharing one weight vector w with 3 separate biases (guaranteeing rank consistency). The 2|3
boundary becomes a DIRECT LEARNING TASK (output P(y>2)) instead of a side effect of a 4-way
softmax.

Everything else is kept as in v7 (5-window data, temporal aug, 5-view pooling at eval) so
that the difference comes only from the OBJECTIVE FUNCTION.

Metrics reported exactly as in v7 for direct comparison:
  auc_dis = 1 - sigmoid(logit_1)  -> P(y<=1)      (cleaner than adding two softmax probabilities)
  auc23   = sigmoid(logit_2)      -> P(y>2) on the y>=2 subset only

DECISION THRESHOLD set before running: if AUC23 does not exceed .72 then the backbone is the
real constraint, which no loss function can fix - stop, no further tuning.

Run: p2_train_v8.py --seed 0 --out ~/p2/runs/coral_s0
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

from p2_train import build_backbone
from p2_train_v6 import param_groups_layer_decay
from p2_train_v7 import Clips5

CKPT = os.path.expanduser("~/MAE-DFER/saved/model/finetuning/dfew/fold1.pth")


class CoralModel(nn.Module):
    """Backbone + CORAL head: one shared w, 3 biases -> logit_k = w.z + b_k."""

    def __init__(self, n_cuts=3):
        super().__init__()
        self.backbone = build_backbone()
        self.w = nn.Linear(512, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(n_cuts))
        nn.init.trunc_normal_(self.w.weight, std=0.02)

    def forward(self, x):
        z = self.backbone(x)
        return self.w(z) + self.b, z          # [B,3]


def load_pretrained(model):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    for k in ("model", "module", "state_dict"):
        if isinstance(ck, dict) and k in ck:
            ck = ck[k]
            break
    sd = {k: v for k, v in ck.items() if not k.startswith("head.")}
    miss, unexp = model.backbone.load_state_dict(sd, strict=False)
    print(f"init fold1: missing={len(miss)} unexpected={len(unexp)}", flush=True)


def coral_targets(y, n_cuts=3):
    """y [B] -> [B,3] binary: (y>0, y>1, y>2)."""
    return torch.stack([(y > k).float() for k in range(n_cuts)], dim=1)


def probs_from_cum(logits):
    """Class probabilities from the 3 cumulative logits; clamp + renormalise if not monotone."""
    c = torch.sigmoid(logits)                      # [B,3] = P(y>0),P(y>1),P(y>2)
    ones = torch.ones_like(c[:, :1])
    zeros = torch.zeros_like(c[:, :1])
    upper = torch.cat([ones, c], dim=1)            # P(y>=0..3)
    lower = torch.cat([c, zeros], dim=1)
    p = (upper - lower).clamp_min(1e-6)
    return p / p.sum(1, keepdim=True)


@torch.no_grad()
def evaluate(model, ds, dev, batch):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=6, pin_memory=True)
    L, ys = [], []
    for xs, y in dl:
        B, W = xs.shape[0], xs.shape[1]
        x = xs.reshape(B * W, *xs.shape[2:]).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(x)
        L.append(torch.sigmoid(lg.float()).reshape(B, W, -1).mean(1).cpu())  # pool 5 views
        ys.append(y)
    c, y = torch.cat(L).numpy(), torch.cat(ys).numpy()
    pred = (c > 0.5).sum(1)                                   # rank -> class 0..3
    rec = [float((pred[y == k] == k).mean()) for k in np.unique(y)]
    m23 = y >= 2
    model.train()
    return dict(acc=float((pred == y).mean()), uar=float(np.mean(rec)),
                f1=float(f1_score(y, pred, average="macro")),
                auc_dis=float(roc_auc_score((y <= 1).astype(int), 1.0 - c[:, 1])),
                auc23=float(roc_auc_score((y[m23] == 3).astype(int), c[m23, 2])))


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

    model = CoralModel()
    load_pretrained(model)
    model = model.to(dev).train()

    # reuse the layer-decay grouping of v6; the CORAL head (w,b) acts as "head"
    groups = param_groups_layer_decay(
        type("M", (), {"backbone": model.backbone,
                       "named_parameters": model.named_parameters,
                       "parameters": model.parameters})(),
        args.lr, args.lr_head, args.wd, 1.0)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_ep = len(lt) // args.accum
    total, warm = args.epochs * steps_per_ep, args.warmup * steps_per_ep

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
                lg, _ = model(x)
                loss = F.binary_cross_entropy_with_logits(lg.float(), coral_targets(y))
            (loss / args.accum).backward()
            losses.append(loss.item())
            if (it + 1) % args.accum == 0:
                set_lr(gstep)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
        m = evaluate(model, va, dev, max(1, args.batch // 4))
        rec = dict(ep=ep, loss=float(np.mean(losses)),
                   mins=round((time.time() - t0) / 60, 1), **m)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args)}, out / "last.pth")
        for k in ("acc", "auc_dis", "auc23"):
            if m[k] > best[k]:
                best[k] = m[k]
                torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args),
                            "val": m}, out / f"best_{k}.pth")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
