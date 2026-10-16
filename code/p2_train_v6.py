"""P2 vong 6 — DUNG DUNG CONG THUC FINE-TUNE CUA MAE-DFER (muc tieu: bang chuan).

Phat hien 15/08: cong thuc goc trong `run_class_finetuning.py` cua MAE-DFER dung
  layer_decay 0.75 | label smoothing 0.1 | mixup 0.8 + cutmix 1.0 | RandAugment
  rand-m7-n4 | random erasing 0.25 | test_num_segment 2 x test_num_crop 2 (TTA 4 view)
Toan bo p2_train.py cua ta dung LR DONG DEU, khong smoothing, khong mixup/erasing,
khong TTA -> duoi-chinh-quy-hoa nang so voi cong thuc ma checkpoint duoc thiet ke cho.
Do la ly do duong cong luon dinh o ep0-6 roi sup (overfit).

Script nay bo sung, theo thu tu don bay/chi phi:
  1. LAYER-WISE LR DECAY 0.75  <- don bay lon nhat cho fine-tune ViT
  2. label smoothing 0.1
  3. random erasing + color jitter (aug khong gian manh hon, lam trung tren toan clip)
  4. TTA luc eval: 2 crop x flip (4 view), trung binh xac suat
Mixup/CutMix CHUA bat: voi 4 lop mat can bang nang (lop 0+1 ~5.8%) mixup de lam nhoe
lop hiem; se ablate rieng neu buoc 1-4 chua du.

Test VAN NIEM PHONG. Moi so o day la official Train -> Validation.

Chay: p2_train_v6.py --seed 0 --layer_decay 0.75 --out ~/p2/runs/tuned_s0
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

from p2_train import MEAN, STD, Clips, P2Model, load_pretrained


# ---------------------------------------------------------------- du lieu + aug
class ClipsAug(Clips):
    """Them random erasing + color jitter, ap DONG NHAT tren ca 16 khung."""

    def __init__(self, split, train_aug, reprob=0.25, jitter=0.2):
        super().__init__(split, train_aug)
        self.reprob, self.jitter = reprob, jitter

    def __getitem__(self, k):
        i, _ = self.rows[k]
        x = torch.from_numpy(np.ascontiguousarray(self.X[i])).float() / 255.0
        x = x.permute(3, 0, 1, 2)  # [C,T,H,W]
        if self.train_aug:
            s = random.uniform(0.7, 1.0)
            ar = random.uniform(3 / 4, 4 / 3)
            h = min(160, int(round(160 * (s * ar) ** 0.5)))
            w = min(160, int(round(160 * (s / ar) ** 0.5)))
            y0, x0 = random.randint(0, 160 - h), random.randint(0, 160 - w)
            x = F.interpolate(x[:, :, y0:y0 + h, x0:x0 + w], size=(160, 160),
                              mode="bilinear", align_corners=False)
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])
            if self.jitter > 0:  # brightness/contrast dong nhat theo thoi gian
                b = 1.0 + random.uniform(-self.jitter, self.jitter)
                c = 1.0 + random.uniform(-self.jitter, self.jitter)
                x = ((x * b - x.mean()) * c + x.mean()).clamp(0, 1)
            x = (x - MEAN) / STD
            if random.random() < self.reprob:  # random erasing, cung vi tri moi khung
                eh, ew = random.randint(20, 56), random.randint(20, 56)
                ey, ex = random.randint(0, 160 - eh), random.randint(0, 160 - ew)
                x[:, :, ey:ey + eh, ex:ex + ew] = torch.randn(3, 1, eh, ew)
        else:
            x = (x - MEAN) / STD
        return x, self.y[k], self.y_subj[k]


# ---------------------------------------------------------------- layer decay
def param_groups_layer_decay(model, lr, lr_head, wd, decay):
    """Gan moi tham so mot layer_id -> lr = lr * decay^(n_layers - layer_id).

    patch_embed & lg_region_tokens = 0; blocks[i] = i+1; fc_norm/head = n_layers.
    Khong weight-decay cho bias va cac tham so 1 chieu (chuan cho ViT fine-tune).
    """
    n_blocks = len(model.backbone.blocks)
    n_layers = n_blocks + 1

    def layer_id(name):
        if name.startswith("backbone.patch_embed") or name == "backbone.lg_region_tokens":
            return 0
        if name.startswith("backbone.blocks."):
            return int(name.split(".")[2]) + 1
        return n_layers

    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = n.startswith("eng_head") or n.startswith("subj_head")
        lid = n_layers if is_head else layer_id(n)
        scale = 1.0 if is_head else decay ** (n_layers - lid)
        base = lr_head if is_head else lr
        no_wd = p.ndim <= 1
        key = (lid, is_head, no_wd)
        if key not in groups:
            groups[key] = dict(params=[], lr=base * scale,
                               weight_decay=0.0 if no_wd else wd,
                               layer=lid, head=is_head)
        groups[key]["params"].append(p)
    out = list(groups.values())
    lrs = sorted({round(g["lr"], 8) for g in out})
    print(f"layer-decay {decay}: {len(out)} nhom, {n_layers} tang, "
          f"lr tu {lrs[0]:.2e} den {lrs[-1]:.2e}", flush=True)
    return out


# ---------------------------------------------------------------- eval + TTA
@torch.no_grad()
def evaluate_tta(model, ds, dev, batch, tta=True):
    """TTA: 2 crop (full + center 0.875) x 2 flip = 4 view, trung binh xac suat."""
    from sklearn.metrics import f1_score, roc_auc_score
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=6, pin_memory=True)
    ps, ys = [], []
    for x, y, _ in dl:
        x = x.to(dev, non_blocking=True)
        views = [x]
        if tta:
            # x la [B,C,T,H,W]: cat theo H,W (chieu 3,4) — KHONG phai 2,3 (do la T,H).
            # interpolate nhan 4D nen gop B,C lai va coi T la kenh, giong ham aug luc train.
            B, C, T, H, W = x.shape
            c = int(H * 0.875)
            o = (H - c) // 2
            xc = x[:, :, :, o:o + c, o:o + c].reshape(B * C, T, c, c)
            xc = F.interpolate(xc, size=(H, W), mode="bilinear",
                               align_corners=False).reshape(B, C, T, H, W)
            views.append(xc)
            views = views + [torch.flip(v, dims=[4]) for v in views]  # lat NGANG = chieu 4
        acc = 0
        for v in views:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _ = model(v)
            acc = acc + logits.float().softmax(-1)
        ps.append((acc / len(views)).cpu())
        ys.append(y)
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    pred = p.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    dis = (y <= 1).astype(int)
    model.train()
    return dict(acc=float((pred == y).mean()), uar=float(np.mean(rec)),
                f1=float(f1_score(y, pred, average="macro")),
                auc_dis=float(roc_auc_score(dis, p[:, 0] + p[:, 1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--layer_decay", type=float, default=0.75)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--reprob", type=float, default=0.25)
    ap.add_argument("--jitter", type=float, default=0.2)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    dev = "cuda"

    tr = ClipsAug("Train", True, reprob=args.reprob, jitter=args.jitter)
    va = ClipsAug("Validation", False)
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
        if step < warm:
            k = step / max(1, warm)
        else:
            t = (step - warm) / max(1, total - warm)
            k = 0.5 * (1 + np.cos(np.pi * t))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * k

    best_acc, best_auc, gstep = -1.0, -1.0, 0
    log = open(out / "log.jsonl", "a")
    for ep in range(args.epochs):
        t0, losses = time.time(), []
        opt.zero_grad(set_to_none=True)
        for it, (x, y, _) in enumerate(lt):
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
        m = evaluate_tta(model, va, dev, args.batch * 2, tta=not args.no_tta)
        rec = dict(ep=ep, loss=float(np.mean(losses)), lr=opt.param_groups[-1]["lr"],
                   mins=round((time.time() - t0) / 60, 1), **m)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args)}, out / "last.pth")
        if m["acc"] > best_acc:
            best_acc = m["acc"]
            torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args),
                        "val": m}, out / "best_acc.pth")
        # LUU THEO AUC: dinh AUC luon roi vao ep2-6 va truoc gio KHONG duoc luu -> khong
        # ensemble duoc cac model tot nhat. Day la lo hong da lam mat 3 checkpoint baseline.
        if m["auc_dis"] > best_auc:
            best_auc = m["auc_dis"]
            torch.save({"model": model.state_dict(), "ep": ep, "args": vars(args),
                        "val": m}, out / "best_auc.pth")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
