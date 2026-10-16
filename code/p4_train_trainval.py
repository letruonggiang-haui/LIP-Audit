"""The Train+Validation arm, registered in
DANG_KY_TRUOC_TEST_ENGAGENET_2026-09-05.md section 2(d).

The dataset paper reports one row trained on the training and validation
partitions combined (fusion, 69.29 on test). To have anything comparable we have
to do the same, and the moment validation is folded into training there is no
held-out partition left to choose an epoch on. That is the exact hole through
which a test number gets tuned, so the rule is fixed in advance and read from
disk rather than typed: **the epoch is the median of the epochs the same model
family already selected on validation**, over its three seeds, as recorded in
`runs/of_<model>_s<seed>_summary.json`. Training runs to that epoch and stops.
No early stopping, no monitoring, no checkpoint selection.

Everything else matches p4_train_of.py exactly: same builders, same optimiser,
same schedule, same batch size, same standardisation procedure. The only changes
are the training set and the stopping rule.

Usage:  python p4_train_trainval.py --model tcn --seed 0
        python p4_train_trainval.py --all
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p4_train_of import BUILDERS, load_split  # noqa: E402

P4 = os.path.expanduser("~/p4")


def registered_epoch(model):
    """The median best epoch of this family's three validation runs."""
    eps = []
    for seed in (0, 1, 2):
        p = os.path.join(P4, "runs", "of_%s_s%d_summary.json" % (model, seed))
        if not os.path.exists(p):
            raise SystemExit("missing %s: the epoch rule reads the validation runs "
                             "and will not guess" % p)
        eps.append(int(json.load(open(p))["best_epoch"]))
    return int(np.median(eps)), eps


def train_one(model_name, seed, dev):
    n_epoch, seen = registered_epoch(model_name)
    tag = "tv_%s_s%d" % (model_name, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xw_tr, Xs_tr, ytr, ltr = load_split("train")
    Xw_va, Xs_va, yva, lva = load_split("validation")
    static = model_name == "mlp"
    Xa, Xb = (Xw_tr, Xw_va) if static else (Xs_tr, Xs_va)
    X = np.concatenate([Xa, Xb])
    y = np.concatenate([ytr, yva])
    n_subj = ltr["subj"].nunique() + lva["subj"].nunique()

    flat = X.reshape(-1, X.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    X = (X - mu) / sd
    print("%s: %s over %d clips, %d participants, fixed %d epochs "
          "(median of validation epochs %s)"
          % (tag, model_name, len(y), n_subj, n_epoch + 1, seen), flush=True)

    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y)
    model = BUILDERS[model_name](X.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epoch + 1, 1))
    crit = nn.CrossEntropyLoss()

    for ep in range(n_epoch + 1):
        model.train()
        perm = torch.randperm(len(y))
        tot = 0.0
        for i in range(0, len(y), 64):
            k = perm[i:i + 64]
            if len(k) < 2:
                continue
            xb, yb = Xt[k].to(dev), yt[k].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(k)
        sched.step()
        print("  ep %2d train loss %.4f" % (ep, tot / len(y)), flush=True)

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"state": state, "mu": mu, "sd": sd,
                "args": {"model": model_name, "seed": seed, "epochs": n_epoch + 1}},
               os.path.join(P4, "runs", "%s_best.pth" % tag))
    json.dump({"tag": tag, "model": model_name, "seed": seed,
               "trained_on": "train+validation", "n_clips": int(len(y)),
               "fixed_epochs": n_epoch + 1, "validation_epochs_seen": seen,
               "selection_rule": ("no epoch selection: validation is inside the "
                                  "training set, so the epoch count is the median "
                                  "of the epochs this family selected on validation "
                                  "when validation was held out. Registered before "
                                  "any test descriptor existed."),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(P4, "runs", "%s_summary.json" % tag), "w"), indent=2)
    print("saved %s_best.pth" % tag, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(BUILDERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.all:
        for m in sorted(BUILDERS):
            for s in (0, 1, 2):
                subprocess.run([sys.executable, os.path.abspath(__file__),
                                "--model", m, "--seed", str(s)], check=True)
        return
    if not args.model:
        raise SystemExit("give --model or --all")
    train_one(args.model, args.seed, dev)


if __name__ == "__main__":
    main()
