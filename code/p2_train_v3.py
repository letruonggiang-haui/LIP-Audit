"""P2 vong 3 — bat bien danh tinh bang PHAT TACH BIET DANG DONG (khong adversary).

Ly do bo adversary (bang chung 14/08, xem NHAT_KY_P2.md): 6/6 cau hinh GRL that bai.
Adversary bi day ve chance (top-1 .0213 tren chinh clip train, 1.47x chance; dau ra sup
ve 11/69 lop) NHUNG probe tuyen tinh huan luyen lai doc danh tinh .959 tren clip held-out
cung 69 lop. Backbone chi danh bai BO THAM SO HIEN THOI cua adversary. => Bo han ho
adversary huan luyen chung; dung phat tinh trong dong dang dong, khong co tham so de lua.

Co che: LDA-style separability. Voi batch P doi tuong x K clip, embedding z:
  S_b = mean_s ||mu_s - mu||^2      (tan xa GIUA doi tuong — chinh la thu probe khai thac)
  S_w = mean_s mean_{i in s} ||z_i - mu_s||^2   (tan xa TRONG doi tuong)
  L_id = S_b / (S_w + eps)          -> toi thieu hoa
Chia cho S_w chan nghiem tam thuong "co cum toan bo" (thu nho moi thu thi ca hai cung
giam, ti so khong loi); CE engagement giu tin hieu nhiem vu.

BAT BUOC sampler P x K: batch 8 tren 69 doi tuong => hau het clip la mot doi tuong rieng,
S_w uoc luong tu ~1 mau => phat chi la nhieu. Sampler duoi day dam bao K clip/doi tuong.

Tieu chi thanh cong (KHONG phai "identity giam" don thuan): official-AUC do DUOI centering
giu vung trong khi identity-probe giam. Moi cau hinh tu truoc toi nay deu day identity giam
mot chut va tin hieu nhiem vu giam nhieu — sai huong.

Chay: p2_train_v3.py --seed 0 --lambda_id 1.0 --pk_p 4 --pk_k 4 --out ~/p2/runs/idsep10_s0
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from p2_train import Clips, P2Model, evaluate, load_pretrained


class PKSampler(torch.utils.data.Sampler):
    """Moi batch: P doi tuong x K clip. Bo qua doi tuong co < K clip trong vong do."""

    def __init__(self, subjects, p, k, seed=0):
        self.by_subj = {}
        for i, s in enumerate(subjects):
            self.by_subj.setdefault(s, []).append(i)
        self.p, self.k, self.seed, self.epoch = p, k, seed, 0
        self.n_batches = len(subjects) // (p * k)

    def set_epoch(self, ep):
        self.epoch = ep

    def __len__(self):
        return self.n_batches * self.p * self.k

    def __iter__(self):
        rng = random.Random(self.seed * 1000 + self.epoch)
        pool = {s: rng.sample(v, len(v)) for s, v in self.by_subj.items()}
        subs = [s for s, v in pool.items() if len(v) >= self.k]
        order = []
        for _ in range(self.n_batches):
            avail = [s for s in subs if len(pool[s]) >= self.k]
            if len(avail) < self.p:
                break
            for s in rng.sample(avail, self.p):
                for _ in range(self.k):
                    order.append(pool[s].pop())
        return iter(order)


def identity_separability(z, ys):
    """S_b / S_w tren batch. z [B,D] float32, ys [B] subject index."""
    z = z.float()
    uniq = torch.unique(ys)
    mus, within = [], []
    for s in uniq:
        m = ys == s
        if m.sum() < 2:
            continue
        zs = z[m]
        mu = zs.mean(0)
        mus.append(mu)
        within.append(((zs - mu) ** 2).sum(-1).mean())
    if len(mus) < 2:
        return z.new_zeros(())
    mus = torch.stack(mus)
    s_b = ((mus - mus.mean(0)) ** 2).sum(-1).mean()
    s_w = torch.stack(within).mean()
    return s_b / (s_w + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pk_p", type=int, default=4, help="so doi tuong moi batch")
    ap.add_argument("--pk_k", type=int, default=4, help="so clip moi doi tuong")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lambda_id", type=float, default=1.0)
    ap.add_argument("--id_warmup", type=int, default=3, help="epoch dau lambda_id=0")
    ap.add_argument("--late_from", type=int, default=5)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    dev = "cuda"
    batch = args.pk_p * args.pk_k

    tr = Clips("Train", train_aug=True)
    va = Clips("Validation", train_aug=False)
    samp = PKSampler(tr.subjects, args.pk_p, args.pk_k, seed=args.seed)
    lt = torch.utils.data.DataLoader(tr, batch_size=batch, sampler=samp, num_workers=8,
                                     pin_memory=True, drop_last=True, persistent_workers=True)
    lv = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False,
                                     num_workers=4, pin_memory=True)

    model = P2Model(n_subj=2)  # subj_head khong dung, giu de tuong thich p2_probe
    load_pretrained(model)
    model = model.to(dev).train()

    heads = list(model.eng_head.parameters())
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr},
                             {"params": heads, "lr": args.lr_head}], weight_decay=args.wd)
    steps_per_ep = len(lt) // args.accum
    total, warm = args.epochs * steps_per_ep, args.warmup * steps_per_ep

    def lr_scale(step):
        if step < warm:
            return step / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    best_acc, best_acc_late = -1.0, -1.0
    log = open(out / "log.jsonl", "a")
    for ep in range(args.epochs):
        samp.set_epoch(ep)
        lam = args.lambda_id if ep >= args.id_warmup else 0.0
        t0, losses, sep = time.time(), [], []
        opt.zero_grad(set_to_none=True)
        for it, (x, y, ys) in enumerate(lt):
            x, y, ys = x.to(dev, non_blocking=True), y.to(dev), ys.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                le, _, z = model(x)
                loss_e = F.cross_entropy(le, y)
            s = identity_separability(z, ys)  # float32 ngoai autocast cho on dinh
            loss = loss_e + lam * s
            (loss / args.accum).backward()
            losses.append(loss_e.item())
            sep.append(float(s))
            if (it + 1) % args.accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        m = evaluate(model, lv, dev)
        rec = dict(ep=ep, loss=float(np.mean(losses)), sep=float(np.mean(sep)), lam=lam,
                   lr=sched.get_last_lr()[0], mins=round((time.time() - t0) / 60, 1), **m)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args)}, out / "last.pth")
        if m["acc"] > best_acc:
            best_acc = m["acc"]
            torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args),
                        "val": m}, out / "best_acc.pth")
        if ep >= args.late_from and m["acc"] > best_acc_late:
            best_acc_late = m["acc"]
            torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args),
                        "val": m}, out / "best_acc_late.pth")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
