"""Dump the 4-class probabilities on Validation for ONE checkpoint -> npz.

Why it exists: our Test accuracy estimate ('.60') is a CHAIN OF INFERENCE
(AUC -> d' -> balanced acc -> times .95), never measured. With per-clip probabilities
it can be computed directly: reweight Validation to the exact class proportions of Test,
then read the 4-class accuracy. Test STAYS SEALED - only the published DISTRIBUTION
STATISTICS of Test (.002/.047/.494/.456) are used, not Test labels or data.
"""
import argparse
import os

import numpy as np
import torch

from p2_train import P2Model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", choices=["faces", "faces5"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    if a.data == "faces5":
        from p2_train_v7 import Clips5
        ds, bs = Clips5("Validation", False), max(1, a.batch // 4)
    else:
        from p2_train import Clips
        ds, bs = Clips("Validation", False), a.batch

    ck = torch.load(os.path.expanduser(a.ckpt), map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("subj_head")}
    model = P2Model(n_subj=2)
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda").eval()

    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=6)
    ps, ys = [], []
    for b in dl:
        x, y = b[0], b[1]
        if a.data == "faces5":
            B, W = x.shape[0], x.shape[1]
            xx = x.reshape(B * W, *x.shape[2:]).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _, _ = model(xx)
            p = lg.float().softmax(-1).reshape(B, W, -1).mean(1)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _, _ = model(x.to("cuda"))
            p = lg.float().softmax(-1)
        ps.append(p.cpu())
        ys.append(y)
    P, Y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    ids = np.array(ds.ids) if hasattr(ds, "ids") else np.arange(len(Y))
    np.savez(os.path.expanduser(a.out), P=P, y=Y, ids=ids)
    print(f"OK {a.out}  P{P.shape} y{Y.shape}", flush=True)


if __name__ == "__main__":
    main()
