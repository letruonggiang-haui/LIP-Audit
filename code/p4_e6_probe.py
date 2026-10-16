"""E6 - identity probe on the EngageNet representations.

Follows Section III-D of the manuscript exactly, so the number is comparable
with the 66x chance reported for DAiSEE. After training, the representation
entering the final classifier layer is frozen and standardised per dimension.
The clips of each training participant are split 75/25 at random; an
l2-regularised multinomial logistic regression over the participants is fitted
on the 75 % portion and read on the held-out 25 %. The result is reported as
top-1 accuracy and as a multiple of chance, M = accuracy x |S|.

A nearest-class-mean probe runs alongside, because the two families disagreed on
identical DAiSEE embeddings and a single family invites false conclusions. Both
are repeated after per-participant centring, which is the operation that
collapsed the RGB channel in P1.

Usage:  python p4_e6_probe.py            # every of_* checkpoint
        python p4_e6_probe.py --models transformer tcn
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p4_train_of import BUILDERS, load_split  # noqa: E402

P4 = os.path.expanduser("~/p4")
OUT = os.path.join(P4, "analysis")


def embed(model, X, batch=256):
    """The vector entering the final classifier layer."""
    feats = []
    hook_out = {}

    def hook(_m, inp, _out):
        hook_out["v"] = inp[0].detach().cpu().numpy()

    last = model.head[-1] if hasattr(model, "head") else model.net[-1]
    h = last.register_forward_hook(hook)
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(X), batch):
            model(torch.from_numpy(X[i:i + batch]).to(dev))
            feats.append(hook_out["v"])
    h.remove()
    return np.concatenate(feats)


def logistic_probe(Z, subj, seed=0, C=1.0):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s)
        rng.shuffle(idx)
        k = max(1, int(round(0.75 * len(idx))))
        tr.append(idx[:k])
        te.append(idx[k:])
    tr = np.concatenate(tr)
    te = np.concatenate([t for t in te if len(t)])
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(Z[tr], subj[tr])
    return float((clf.predict(Z[te]) == subj[te]).mean()), len(tr), len(te)


def ncm_probe(Z, subj, seed=0):
    """Nearest class mean, same 75/25 split."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s)
        rng.shuffle(idx)
        k = max(1, int(round(0.75 * len(idx))))
        tr.append(idx[:k])
        te.append(idx[k:])
    tr = np.concatenate(tr)
    te = np.concatenate([t for t in te if len(t)])
    classes = np.unique(subj)
    means = np.stack([Z[tr][subj[tr] == s].mean(0) for s in classes])
    d = ((Z[te][:, None, :] - means[None]) ** 2).sum(-1)
    return float((classes[d.argmin(1)] == subj[te]).mean())


def centre_per_subject(Z, subj):
    Zc = Z.copy()
    for s in np.unique(subj):
        k = subj == s
        Zc[k] = Z[k] - Z[k].mean(0, keepdims=True)
    return Zc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    Xw, Xs, ytr, ltr = load_split("train")
    subj = ltr["subj"].values
    n_subj = len(np.unique(subj))
    print("training split: %d clips, %d participants, chance = 1/%d = %.4f"
          % (len(subj), n_subj, n_subj, 1.0 / n_subj))

    results = []
    for ck in sorted(glob.glob(os.path.join(P4, "runs", "of_*_best.pth"))):
        tag = os.path.basename(ck)[:-len("_best.pth")]
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        model_name = blob["args"]["model"]
        if args.models and model_name not in args.models:
            continue
        static = model_name == "mlp"
        X = (Xw if static else Xs)
        X = (X - blob["mu"]) / blob["sd"]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = BUILDERS[model_name](X.shape[-1]).to(dev)
        model.load_state_dict(blob["state"])
        Z = embed(model, X.astype(np.float32))
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)

        acc, n_tr, n_te = logistic_probe(Z, subj)
        ncm = ncm_probe(Z, subj)
        Zc = centre_per_subject(Z, subj)
        acc_c, _, _ = logistic_probe(Zc, subj)
        ncm_c = ncm_probe(Zc, subj)
        row = {"tag": tag, "model": model_name, "dim": int(Z.shape[1]),
               "n_subjects": n_subj, "chance": 1.0 / n_subj,
               "logistic_acc": acc, "logistic_M": acc * n_subj,
               "ncm_acc": ncm, "ncm_M": ncm * n_subj,
               "logistic_acc_centred": acc_c, "logistic_M_centred": acc_c * n_subj,
               "ncm_acc_centred": ncm_c, "ncm_M_centred": ncm_c * n_subj,
               "n_probe_train": n_tr, "n_probe_test": n_te}
        results.append(row)
        print("%-20s dim %3d | logistic %.4f (%5.1fx)  ncm %.4f (%5.1fx) | "
              "after centring %.4f (%5.1fx) / %.4f (%5.1fx)"
              % (tag, row["dim"], acc, row["logistic_M"], ncm, row["ncm_M"],
                 acc_c, row["logistic_M_centred"], ncm_c, row["ncm_M_centred"]),
              flush=True)

    with open(os.path.join(OUT, "E6_probe.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote", os.path.join(OUT, "E6_probe.json"))

    by = {}
    for r in results:
        by.setdefault(r["model"], []).append(r)
    print("\n%-14s %-18s %-18s %s" % ("model", "logistic M", "ncm M", "logistic M after centring"))
    for m, rs in sorted(by.items()):
        f = lambda k: "%.1f +- %.1f" % (np.mean([r[k] for r in rs]),
                                        np.std([r[k] for r in rs]))
        print("%-14s %-18s %-18s %s" % (m, f("logistic_M"), f("ncm_M"),
                                        f("logistic_M_centred")))


if __name__ == "__main__":
    main()
