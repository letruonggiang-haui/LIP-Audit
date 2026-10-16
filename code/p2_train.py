"""P2 — fine-tune MAE-DFER tren DAiSEE engagement 4 muc, official split.

Phase 1 (baseline):  p2_train.py --seed 0 --out ~/p2/runs/base_s0
Phase 2 (GRL):       p2_train.py --seed 0 --lambda_grl 0.5 --out ~/p2/runs/grl05_s0

- Backbone: vit_base_dim512_no_depth_patch16_160 (LGI-Former, depth 16), init fold1.pth.
- Dau engagement 4 muc (CE); dau subject (chi train split) qua gradient reversal khi
  lambda_grl > 0. Bao cao: acc 4 muc, UAR, macro-F1, AUC disengagement (p0+p1).
- Du lieu: ~/p2/faces/<Split>_X.npy (uint8 [N,16,160,160,3], memmap) tu p2_extract_faces.py.
- Test KHONG duoc cham trong script nay (chi Train/Validation).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/MAE-DFER"))
import modeling_finetune  # noqa: E402

DAISEE = Path(os.path.expanduser("~/daisee"))
FACES = Path(os.path.expanduser("~/p2/faces"))
CKPT = os.path.expanduser("~/MAE-DFER/saved/model/finetuning/dfew/fold1.pth")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


# ---------------------------------------------------------------- du lieu
class Clips(torch.utils.data.Dataset):
    def __init__(self, split, train_aug, subj_to_idx=None):
        self.X = np.load(FACES / f"{split}_X.npy", mmap_mode="r")
        ids = [s for s in np.load(FACES / f"{split}_ids.npy")]
        lab = pd.read_csv(DAISEE / "DAiSEE" / "Labels" / f"{split}Labels.csv")
        lab.columns = [c.strip() for c in lab.columns]
        lab["stem"] = lab["ClipID"].astype(str).str.replace(r"\.(avi|mp4)$", "", regex=True)
        eng = dict(zip(lab["stem"], lab["Engagement"]))
        self.rows = [(i, s) for i, s in enumerate(ids) if s in eng]
        self.y = np.array([eng[s] for _, s in self.rows], dtype=np.int64)
        self.subjects = [s[:6] for _, s in self.rows]
        if subj_to_idx is None:
            subj_to_idx = {s: k for k, s in enumerate(sorted(set(self.subjects)))}
        self.subj_to_idx = subj_to_idx
        self.y_subj = np.array([subj_to_idx.get(s, -1) for s in self.subjects], dtype=np.int64)
        self.train_aug = train_aug
        print(f"{split}: {len(self.rows)} clip co nhan / {len(ids)} trich duoc, "
              f"{len(set(self.subjects))} doi tuong", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        i, _ = self.rows[k]
        x = torch.from_numpy(np.ascontiguousarray(self.X[i])).float() / 255.0
        x = x.permute(3, 0, 1, 2)  # [C,T,H,W]
        if self.train_aug:
            # random resized crop dong nhat theo thoi gian + hflip
            s = random.uniform(0.7, 1.0)
            ar = random.uniform(3 / 4, 4 / 3)
            h = min(160, int(round(160 * (s * ar) ** 0.5)))
            w = min(160, int(round(160 * (s / ar) ** 0.5)))
            y0, x0 = random.randint(0, 160 - h), random.randint(0, 160 - w)
            x = x[:, :, y0:y0 + h, x0:x0 + w]
            x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])
        x = (x - MEAN) / STD
        return x, self.y[k], self.y_subj[k]


# ---------------------------------------------------------------- mo hinh
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def build_backbone():
    # num_classes=0 se crash: __init__ goi self.head.weight.data.mul_ khong guard
    # (modeling_finetune.py:487-491) -> dung head that roi thay Identity sau.
    m = modeling_finetune.vit_base_dim512_no_depth_patch16_160(
        num_classes=4, all_frames=16, tubelet_size=2, depth=16,
        drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
        use_mean_pooling=True, init_scale=0.001,
        attn_type="local_global", lg_region_size=(2, 5, 10),
        lg_first_attn_type="self", lg_third_attn_type="cross",
        lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
        lg_classify_token_type="region", lg_no_second=False, lg_no_third=False,
    )
    m.head = nn.Identity()  # forward tra ve embedding 512d
    return m


class P2Model(nn.Module):
    def __init__(self, n_subj, adv_hidden=0):
        super().__init__()
        self.backbone = build_backbone()
        self.eng_head = nn.Linear(512, 4)
        if adv_hidden > 0:
            # adversary MLP: dau tuyen tinh bi backbone danh lua qua de (pilot 13/08:
            # identity-probe van .97-.98 sau 25 epoch GRL) -> tang suc adversary
            self.subj_head = nn.Sequential(
                nn.Linear(512, adv_hidden), nn.GELU(), nn.Linear(adv_hidden, n_subj))
            for m in self.subj_head:
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
        else:
            self.subj_head = nn.Linear(512, n_subj)
            nn.init.trunc_normal_(self.subj_head.weight, std=0.02)
        nn.init.trunc_normal_(self.eng_head.weight, std=0.02)

    def forward(self, x, lam=0.0):
        z = self.backbone(x)
        return self.eng_head(z), self.subj_head(GRL.apply(z, lam)), z


def load_pretrained(model):
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    for key in ("model", "module", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    sd = {k: v for k, v in ckpt.items() if not k.startswith("head.")}
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    print(f"init fold1: missing={len(missing)} unexpected={len(unexpected)}", flush=True)


# ---------------------------------------------------------------- danh gia
@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    ps, ys = [], []
    for x, y, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = model(x.to(dev, non_blocking=True))
        ps.append(logits.float().softmax(-1).cpu())
        ys.append(y)
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    rec = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    uar = float(np.mean(rec))
    from sklearn.metrics import f1_score, roc_auc_score
    f1 = float(f1_score(y, pred, average="macro"))
    dis = (y <= 1).astype(int)
    auc_dis = float(roc_auc_score(dis, p[:, 0] + p[:, 1])) if dis.any() else float("nan")
    model.train()
    return dict(acc=acc, uar=uar, f1=f1, auc_dis=auc_dis)


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--freeze_blocks", type=int, default=0)
    ap.add_argument("--lambda_grl", type=float, default=0.0)
    ap.add_argument("--adv_hidden", type=int, default=0,
                    help="0 = adversary tuyen tinh; >0 = MLP 2 tang voi hidden nay")
    ap.add_argument("--grl_warmup", type=int, default=0,
                    help="so epoch dau lam=0 (hoc engagement truoc, bat GRL sau)")
    ap.add_argument("--late_from", type=int, default=5,
                    help="best_acc_late.pth = acc tot nhat trong cac epoch >= nguong nay "
                         "(ne bay dinh acc ep0 khi model chua-adversarial)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    dev = "cuda"

    tr = Clips("Train", train_aug=True)
    va = Clips("Validation", train_aug=False)
    lt = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                     num_workers=8, pin_memory=True, drop_last=True,
                                     persistent_workers=True)
    lv = torch.utils.data.DataLoader(va, batch_size=args.batch * 2, shuffle=False,
                                     num_workers=4, pin_memory=True)

    model = P2Model(n_subj=len(tr.subj_to_idx), adv_hidden=args.adv_hidden)
    load_pretrained(model)
    if args.freeze_blocks > 0:
        for p in model.backbone.patch_embed.parameters():
            p.requires_grad = False
        for blk in model.backbone.blocks[:args.freeze_blocks]:
            for p in blk.parameters():
                p.requires_grad = False
    model = model.to(dev).train()

    heads = list(model.eng_head.parameters()) + list(model.subj_head.parameters())
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr},
                             {"params": heads, "lr": args.lr_head}], weight_decay=args.wd)
    steps_per_ep = len(lt) // args.accum
    total = args.epochs * steps_per_ep
    warm = args.warmup * steps_per_ep

    def lr_scale(step):
        if step < warm:
            return step / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    best_uar, best_acc, best_acc_late, gstep = -1.0, -1.0, -1.0, 0
    log = open(out / "log.jsonl", "a")
    for ep in range(args.epochs):
        lam_ep = args.lambda_grl if ep >= args.grl_warmup else 0.0
        t0, losses, losses_s = time.time(), [], []
        opt.zero_grad(set_to_none=True)
        for it, (x, y, ys) in enumerate(lt):
            x, y, ys = x.to(dev, non_blocking=True), y.to(dev), ys.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                le, ls_, _ = model(x, lam=lam_ep)
                loss_e = F.cross_entropy(le, y)
                loss_s = F.cross_entropy(ls_, ys) if lam_ep > 0 else le.new_zeros(())
                loss = loss_e + loss_s
            (loss / args.accum).backward()
            losses.append(loss_e.item())
            losses_s.append(loss_s.item())
            if (it + 1) % args.accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
        m = evaluate(model, lv, dev)
        rec = dict(ep=ep, loss=float(np.mean(losses)), loss_subj=float(np.mean(losses_s)),
                   lr=sched.get_last_lr()[0], mins=round((time.time() - t0) / 60, 1), **m)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args)}, out / "last.pth")
        if m["uar"] > best_uar:
            best_uar = m["uar"]
            torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args),
                        "val": m}, out / "best.pth")
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
