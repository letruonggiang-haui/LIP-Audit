"""P2 round 4 - SCALE-INVARIANT identity penalty via prototype-confusion (parameter-free).

Lesson from round 3 (see NHAT_KY_P2.md 14/08): the S_b/S_w penalty was evaded by VARIANCE
COLLAPSE (per-dimension std 0.258 -> 0.0043, i.e. ~11.700x in squared distance) while the
identity-probe STILL read .9955. Reason: the probe uses StandardScaler -> SCALE-INVARIANT. The
model squeezed the variance to lower the trace ratio but kept the discriminative DIRECTION
intact - exactly what the probe reads.

=> Round-4 principle: the penalty must be computed on a representation STANDARDISED EXACTLY AS
   THE PROBE STANDARDISES. Standardising each dimension with batch statistics RIGHT INSIDE the
   penalty -> a global collapse is normalised away and is no longer a solution.

Mechanism: prototype-confusion, no parameters at all to fool (unlike the adversary).
  1. z_n = (z - mean_batch) / (std_batch + eps)          <- scale-invariant
  2. one prototype per subject in the batch, LEAVE-ONE-OUT for the anchor (mandatory: if z_i
     is inside its own prototype then with K=4 it makes up 25%, the classifier is right
     automatically and the penalty optimises an artifact)
  3. logit_{i,t} = -(||z_n_i - mu_t||^2 / D) / tau
  4. L_id = CE(uniform, softmax(logit)) -> minimal when the identity prediction is UNIFORM

Run: p2_train_v4.py --seed 0 --lambda_id 1.0 --pk_p 4 --pk_k 4 --out ~/p2/runs/proto10_s0
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


def prototype_confusion(z, ys, tau=0.5, eps=1e-5):
    """Returns (loss_confusion, top1_prototype). Scale-invariant thanks to batch standardisation."""
    z = z.float()
    z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + eps)
    d = z.shape[1]
    uniq = torch.unique(ys)
    if len(uniq) < 2:
        return z.new_zeros(()), 0.0
    sums, cnts = [], []
    for s in uniq:
        m = ys == s
        sums.append(z[m].sum(0))
        cnts.append(m.sum())
    sums = torch.stack(sums)                       # [P,D]
    cnts = torch.stack(cnts).float().unsqueeze(1)  # [P,1]
    tgt = torch.stack([(uniq == y).nonzero()[0, 0] for y in ys])  # index of the correct prototype

    # leave-one-out: for anchor i of subject s, subtract z_i itself from the sum of s
    mu = sums.unsqueeze(0).expand(len(z), -1, -1).clone()          # [B,P,D]
    cn = cnts.unsqueeze(0).expand(len(z), -1, -1).clone()          # [B,P,1]
    ar = torch.arange(len(z), device=z.device)
    mu[ar, tgt] = mu[ar, tgt] - z
    cn[ar, tgt] = cn[ar, tgt] - 1.0
    mu = mu / cn.clamp_min(1.0)

    dist2 = ((z.unsqueeze(1) - mu) ** 2).sum(-1) / d               # [B,P]
    logits = -dist2 / tau
    logp = F.log_softmax(logits, dim=-1)
    loss = -logp.mean()          # CE to uniform: minimal when the softmax is uniform
    top1 = float((logits.argmax(-1) == tgt).float().mean())
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
    ap.add_argument("--lambda_id", type=float, default=1.0)
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
            cf, p1 = prototype_confusion(z, ys, tau=args.tau)
            loss = loss_e + lam * cf
            (loss / args.accum).backward()
            losses.append(loss_e.item())
            cfs.append(float(cf.detach()))
            p1s.append(p1)
            stds.append(float(z.detach().float().std(0).mean()))  # watch for collapse
            if (it + 1) % args.accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        m = evaluate(model, lv, dev)
        rec = dict(ep=ep, loss=float(np.mean(losses)), conf=float(np.mean(cfs)),
                   proto_top1=float(np.mean(p1s)), z_std=float(np.mean(stds)), lam=lam,
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
