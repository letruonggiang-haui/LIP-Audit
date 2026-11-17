"""Measure the TOP-1 accuracy of the adversary head (subj_head) on its own Train data.

Why: loss_subj ~4.1 (ln69=4.234) does NOT mean the adversary is at chance - a diffuse CE
can still come with a top-1 of 20-40%. The statement "the adversary reaches chance but
identity remains" holds only if top-1 is also near chance (1/69 = .0145). This script
gets exactly that number.

Run: ~/p2/venv/bin/python p2_adv_acc.py --ckpt ~/p2/runs/mlp025_s0/last.pth
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_train import Clips, P2Model  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    ck = torch.load(os.path.expanduser(args.ckpt), map_location="cpu", weights_only=False)
    sd, cargs = ck["model"], ck.get("args", {})
    adv_hidden = int(cargs.get("adv_hidden", 0))
    # infer n_subj from the shape of the last layer of subj_head (linear: subj_head.weight;
    # MLP: subj_head.2.weight)
    key = "subj_head.2.weight" if adv_hidden > 0 else "subj_head.weight"
    n_subj = sd[key].shape[0]

    ds = Clips("Train", train_aug=False)
    model = P2Model(n_subj=n_subj, adv_hidden=adv_hidden)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"ckpt ep{ck.get('ep')} lam={cargs.get('lambda_grl')} adv_hidden={adv_hidden} "
          f"n_subj={n_subj} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to("cuda").eval()

    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=8, pin_memory=True)
    hits, tot, ce, ent, preds = 0, 0, [], [], []
    for x, _, ys in dl:
        ys = ys.to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, ls_, _ = model(x.to("cuda", non_blocking=True))
        ls_ = ls_.float()
        ce.append(F.cross_entropy(ls_, ys, reduction="sum").item())
        p = ls_.softmax(-1)
        ent.append(float(-(p * p.clamp_min(1e-12).log()).sum(-1).sum()))
        preds.append(ls_.argmax(-1).cpu().numpy())
        hits += int((ls_.argmax(-1) == ys).sum())
        tot += len(ys)
    preds = np.concatenate(preds)
    _, cnt = np.unique(preds, return_counts=True)
    print(f"ADVERSARY tren Train (du lieu train cua chinh no): top-1 {hits/tot:.4f} "
          f"(chance {1/n_subj:.4f}, {n_subj} lop) = {hits/tot*n_subj:.2f}x chance | "
          f"CE {sum(ce)/tot:.3f} (ln{n_subj}={np.log(n_subj):.3f}) | n={tot}", flush=True)
    # distinguish two scenarios: "truly blind" (high entropy, predictions spread evenly) vs
    # "collapsed / confidently wrong" (low entropy, only a few classes predicted)
    print(f"  che do hong: entropy TB {sum(ent)/tot:.3f} (uniform={np.log(n_subj):.3f}) | "
          f"so lop argmax phan biet {len(cnt)}/{n_subj} | lop pho bien nhat chiem "
          f"{cnt.max()/tot:.1%} so clip", flush=True)


if __name__ == "__main__":
    main()
