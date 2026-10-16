"""P2 vong 5 — prototype-confusion TOAN CUC (69 doi tuong) qua memory bank EMA.

Bai hoc vong 4 (NHAT_KY_P2.md 14/08): phat prototype 4 lop (chi cac doi tuong CO MAT trong
batch) dat chance o 4 lop (.266 vs .25) nhung dai luong TUONG UNG o 69 lop lai .7898 —
CAO HON ca run chung khong phat (.7436). Mo hinh lam lan 4 doi tuong ngau nhien trong batch
ma khong dong toi cau truc toan cuc. => Loi la PHAM VI, khong phai co che.

Vong 5: giu nguyen prototype-confusion (khong tham so, bat bien ti le) nhung tinh tren
TOAN BO 69 prototype:
  - Memory bank M [69, D] cap nhat EMA tu embedding da chuan hoa cua moi batch (detach).
  - Anchor i so voi ca 69 prototype; prototype cua chinh doi tuong anchor dung ban EMA
    (khong chua z_i vi bank da detach va cap nhat SAU khi tinh loss) -> khong can LOO.
  - logit = -(||z_n - M_t||^2 / D) / tau ; phat = CE ve phan bo DEU tren 69 lop.
Bank chi cap nhat bang thong ke da detach nen khong tao duong gradient vong lai.

Warmup: 3 epoch dau lam=0 nhung VAN cap nhat bank (de bank da hoi tu truoc khi phat bat).

Chay: p2_train_v5.py --seed 0 --lambda_id 2 --out ~/p2/runs/gproto2_s0
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
from p2_train_v3 import PKSampler


class GlobalProtoBank:
    """Memory bank EMA cho 69 prototype, tren khong gian da chuan hoa tung chieu."""

    def __init__(self, n_subj, dim, momentum=0.9, dev="cuda"):
        self.M = torch.zeros(n_subj, dim, device=dev)
        self.seen = torch.zeros(n_subj, dtype=torch.bool, device=dev)
        self.m = momentum

    @torch.no_grad()
    def update(self, z_n, ys):
        for s in torch.unique(ys):
            v = z_n[ys == s].mean(0)
            if self.seen[s]:
                self.M[s] = self.m * self.M[s] + (1 - self.m) * v
            else:
                self.M[s] = v
                self.seen[s] = True


def global_confusion(z, ys, bank, tau=0.5, eps=1e-5):
    """Phat CE-ve-deu tren TOAN BO prototype da thay. Tra ve (loss, top1_69lop)."""
    z = z.float()
    z_n = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + eps)
    if bank.seen.sum() < 2:
        bank.update(z_n.detach(), ys)
        return z.new_zeros(()), 0.0
    idx = bank.seen.nonzero().squeeze(1)              # cac lop da co prototype
    M = bank.M[idx]                                    # [S,D]
    d = z_n.shape[1]
    dist2 = ((z_n.unsqueeze(1) - M.unsqueeze(0)) ** 2).sum(-1) / d
    logits = -dist2 / tau
    loss = -F.log_softmax(logits, dim=-1).mean()       # toi thieu khi deu tren S lop
    pos = torch.full((len(ys),), -1, device=z.device, dtype=torch.long)
    for k, s in enumerate(idx):
        pos[ys == s] = k
    valid = pos >= 0
    top1 = float((logits.argmax(-1)[valid] == pos[valid]).float().mean()) if valid.any() else 0.0
    bank.update(z_n.detach(), ys)                      # cap nhat SAU khi tinh loss
    return loss, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pk_p", type=int, default=4)
    ap.add_argument("--pk_k", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--bank_momentum", type=float, default=0.9)
    ap.add_argument("--lambda_id", type=float, default=2.0)
    ap.add_argument("--id_warmup", type=int, default=3)
    ap.add_argument("--late_from", type=int, default=5)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    dev, batch = "cuda", args.pk_p * args.pk_k

    tr = Clips("Train", train_aug=True)
    va = Clips("Validation", train_aug=False)
    samp = PKSampler(tr.subjects, args.pk_p, args.pk_k, seed=args.seed)
    lt = torch.utils.data.DataLoader(tr, batch_size=batch, sampler=samp, num_workers=8,
                                     pin_memory=True, drop_last=True, persistent_workers=True)
    lv = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False,
                                     num_workers=4, pin_memory=True)

    model = P2Model(n_subj=2)
    load_pretrained(model)
    model = model.to(dev).train()
    bank = GlobalProtoBank(len(tr.subj_to_idx), 512, args.bank_momentum, dev)

    opt = torch.optim.AdamW(
        [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
         {"params": list(model.eng_head.parameters()), "lr": args.lr_head}], weight_decay=args.wd)
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
        t0, losses, cfs, p1s, stds = time.time(), [], [], [], []
        opt.zero_grad(set_to_none=True)
        for it, (x, y, ys) in enumerate(lt):
            x, y, ys = x.to(dev, non_blocking=True), y.to(dev), ys.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                le, _, z = model(x)
                loss_e = F.cross_entropy(le, y)
            cf, p1 = global_confusion(z, ys, bank, tau=args.tau)
            loss = loss_e + lam * cf
            (loss / args.accum).backward()
            losses.append(loss_e.item())
            cfs.append(float(cf.detach()))
            p1s.append(p1)
            stds.append(float(z.detach().float().std(0).mean()))
            if (it + 1) % args.accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        m = evaluate(model, lv, dev)
        rec = dict(ep=ep, loss=float(np.mean(losses)), conf=float(np.mean(cfs)),
                   proto69_top1=float(np.mean(p1s)), z_std=float(np.mean(stds)),
                   n_proto=int(bank.seen.sum()), lam=lam, lr=sched.get_last_lr()[0],
                   mins=round((time.time() - t0) / 60, 1), **m)
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
