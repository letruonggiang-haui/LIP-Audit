"""WSR - within-subject ranking (registered 19/08, log section 'PROPOSAL: WSR').

A NEW training signal, not tried by any run so far: directly optimise the question that decides
the standard benchmark - "of two clips of the SAME person, which one is level 3?". Absolute CE
has its gradient swallowed by the between-person prior (within-person AUC23 only .57-.61 across
5 channels); within-person pairwise ranking removes the prior from the loss by construction.

- MAE-DFER backbone + the whole w5 package (Clips5: 5 windows, temporal aug, 5-view eval pooling).
- Batch P=2 persons x K=4 clips. Ranking score = logit3 - logit2. Loss =
  margin-ranking over every same-person pair (y=3, y=2) in the batch + w_ce * CE (absolute
  anchor, keeps the auc_dis head).
- Eval each epoch: acc/auc_dis/auc23 + **within23** (within-person AUC23, mean weighted
  by person) - the pilot's deciding metric. Saves best by within23 + last.

Decision threshold (pre-registered): within23 Val < .65 -> STOP the standard-benchmark goal
permanently.

Run: p2_train_v11.py --seed 0 --out ~/p2/runs/wsr_s0
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

from p2_train import P2Model, load_pretrained
from p2_train_v6 import param_groups_layer_decay
from p2_train_v7 import Clips5


class PKSampler(torch.utils.data.Sampler):
    """Each batch: P persons x K clips. Prefers persons with BOTH level 2 and level 3 (prob 0.8)."""

    def __init__(self, ds, P=2, K=4, seed=0):
        self.P, self.K = P, K
        self.rng = random.Random(seed)
        subj = np.array([s[:6] for _, s in ds.rows])
        self.groups = {}
        for i, s in enumerate(subj):
            self.groups.setdefault(s, []).append(i)
        y = ds.y
        self.mixed = [s for s, idx in self.groups.items()
                      if (y[idx] == 3).any() and (y[idx] == 2).any()]
        self.all = list(self.groups)
        self.n_batches = len(ds) // (P * K)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for _ in range(self.P):
                pool = self.mixed if (self.rng.random() < 0.8 and self.mixed) else self.all
                s = self.rng.choice(pool)
                idx = self.groups[s]
                take = self.rng.sample(idx, self.K) if len(idx) >= self.K \
                    else self.rng.choices(idx, k=self.K)
                batch += take
            yield batch


def rank_loss(score, y, subj, margin):
    """Margin ranking over every same-person pair (y=3 ranked above y=2)."""
    losses = []
    for s in subj.unique():
        k = subj == s
        hi, lo = score[k & (y == 3)], score[k & (y == 2)]
        if len(hi) and len(lo):
            d = hi.unsqueeze(1) - lo.unsqueeze(0)          # [n3,n2]
            losses.append(F.relu(margin - d).mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, ds, dev, batch):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=6, pin_memory=True)
    ps, ys = [], []
    for x, y in dl:                                        # x [B,5,C,T,H,W]
        B, W = x.shape[:2]
        xx = x.reshape(B * W, *x.shape[2:]).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _, _ = model(xx)
        ps.append(lg.float().reshape(B, W, -1).mean(1).cpu())   # mean logit over 5 views
        ys.append(y)
    Z, Y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    P = torch.softmax(torch.tensor(Z), -1).numpy()
    pred = P.argmax(1)
    rec = [float((pred[Y == c] == c).mean()) for c in np.unique(Y)]
    m = Y >= 2
    sc = Z[m, 3] - Z[m, 2]                                 # ranking score
    t = (Y[m] == 3).astype(int)
    subj = np.array([s[:6] for _, s in ds.rows])[m]
    aucs, ws = [], []
    for s in np.unique(subj):
        k = subj == s
        if len(np.unique(t[k])) == 2:
            aucs.append(roc_auc_score(t[k], sc[k]))
            ws.append(int(k.sum()))
    return dict(acc=float((pred == Y).mean()), uar=float(np.mean(rec)),
                f1=float(f1_score(Y, pred, average="macro")),
                auc_dis=float(roc_auc_score((Y <= 1).astype(int), P[:, 0] + P[:, 1])),
                auc23=float(roc_auc_score(t, sc)),
                within23=float(np.average(aucs, weights=ws)))


class WithIdx(torch.utils.data.Dataset):
    """Wraps the dataset to also return the index - needed to look up the subject in the batch."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--P", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--layer_decay", type=float, default=1.0)
    ap.add_argument("--w_ce", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--out", required=True)
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
    subj_tr = np.array([s[:6] for _, s in tr.rows])
    sampler = PKSampler(tr, args.P, args.K, args.seed)
    print(f"nguoi mixed 2&3: {len(sampler.mixed)}/{len(sampler.all)}", flush=True)
    lt = torch.utils.data.DataLoader(WithIdx(tr), batch_sampler=sampler, num_workers=8,
                                     pin_memory=True, persistent_workers=True)
    subj_codes = {s: i for i, s in enumerate(np.unique(subj_tr))}
    subj_id = torch.tensor([subj_codes[s] for s in subj_tr])

    model = P2Model(n_subj=2)
    load_pretrained(model)
    model = model.to(dev).train()
    opt = torch.optim.AdamW(
        param_groups_layer_decay(model, args.lr, args.lr_head, args.wd, args.layer_decay),
        betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_ep = len(sampler) // args.accum
    total, warm = args.epochs * steps_per_ep, args.warmup * steps_per_ep
    ce = nn.CrossEntropyLoss()

    def set_lr(step):
        k = step / max(1, warm) if step < warm else \
            0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, total - warm)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * k

    best = {"within23": -1.0}
    gstep = 0
    log = open(out / "log.jsonl", "a")
    for ep in range(args.epochs):
        model.train()
        t0, l_r, l_c, n_pair_b = time.time(), [], [], 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y, idx) in enumerate(lt):
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            sj = subj_id[idx].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _, _ = model(x)
                lc_ = ce(lg, y)
            score = (lg[:, 3] - lg[:, 2]).float()
            lr_ = rank_loss(score, y, sj, args.margin)
            loss = args.w_ce * lc_.float() + (lr_ if lr_ is not None else 0.0)
            (loss / args.accum).backward()
            l_c.append(lc_.item())
            if lr_ is not None:
                l_r.append(lr_.item())
                n_pair_b += 1
            if (it + 1) % args.accum == 0:
                set_lr(gstep)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
        m = evaluate(model, va, dev, 2)
        rec = dict(ep=ep, loss_rank=float(np.mean(l_r)) if l_r else None,
                   loss_ce=float(np.mean(l_c)),
                   pct_batch_pair=round(n_pair_b / max(1, len(sampler)), 2),
                   mins=round((time.time() - t0) / 60, 1), **m)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args)}, out / "last.pth")
        if m["within23"] > best["within23"]:
            best["within23"] = m["within23"]
            torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args),
                        "val": m}, out / "best_within23.pth")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
