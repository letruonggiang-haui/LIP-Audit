"""P5/R8 - identity probe under a recording-disjoint split, and honest centring.

The published probe (p4_e6_probe.py) splits each participant's clips 75/25 at
random, so probe-train and probe-test clips come from the same recording.  What
it reads is therefore identity together with recording context (lighting, camera
placement, session-level appearance).  It also centres each participant with a
mean computed over all of that participant's clips, including the held-out ones,
so the reported collapse after centring is partly an oracle operation.

This script re-reads the same frozen representations three ways:

  split   random      : the published 75/25 random split (reproduction check)
          recording   : one whole recording per participant held out, the rest
                        used for fitting; only the 78 training participants that
                        carry at least two recordings take part

  centring none       : as published, no centring
          oracle      : participant mean over all clips (as published)
          fitted      : participant mean over probe-train clips only, applied
                        unchanged to the held-out clips

Nothing is retrained: the checkpoints in ~/p4/runs are loaded and the vector
entering the final classifier layer is taken exactly as in p4_e6_probe.py.  Only
the EngageNet training split is touched; the sealed test split is not read.

Output: ~/p4/analysis/p5_probe_recdisjoint.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/p4/code"))
from p4_train_of import BUILDERS, load_split  # noqa: E402

P4 = os.path.expanduser("~/p4")
OUT = os.path.join(P4, "analysis")
SEEDS = [0]


def embed(model, X, batch=256):
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


def random_split(subj, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s)
        rng.shuffle(idx)
        k = max(1, int(round(0.75 * len(idx))))
        tr.append(idx[:k])
        te.append(idx[k:])
    return (np.concatenate(tr),
            np.concatenate([t for t in te if len(t)]))


def recording_split(subj, rec, seed):
    """Hold out one whole recording per participant; drop single-recording ones."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s)
        recs = np.unique(rec[idx])
        if len(recs) < 2:
            continue
        held = recs[rng.integers(len(recs))]
        te.append(idx[rec[idx] == held])
        tr.append(idx[rec[idx] != held])
    return np.concatenate(tr), np.concatenate(te)


def centre(Z, subj, tr, mode):
    if mode == "none":
        return Z
    Zc = Z.copy()
    in_tr = np.zeros(len(Z), dtype=bool)
    in_tr[tr] = True
    for s in np.unique(subj):
        k = np.flatnonzero(subj == s)
        if mode == "oracle":
            mu = Z[k].mean(0, keepdims=True)
        else:
            src = k[in_tr[k]]
            if len(src) == 0:
                continue
            mu = Z[src].mean(0, keepdims=True)
        Zc[k] = Z[k] - mu
    return Zc


def fit_logistic(Z, subj, tr, te):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=1000, tol=1e-3)
    clf.fit(Z[tr], subj[tr])
    return float((clf.predict(Z[te]) == subj[te]).mean())


def fit_ncm(Z, subj, tr, te):
    classes = np.unique(subj[tr])
    means = np.stack([Z[tr][subj[tr] == s].mean(0) for s in classes])
    d = ((Z[te][:, None, :] - means[None]) ** 2).sum(-1)
    return float((classes[d.argmin(1)] == subj[te]).mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these model families")
    ap.add_argument("--out", default="p5_probe_recdisjoint.json",
                    help="output file name inside the analysis directory")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    Xw, Xs, ytr, lab = load_split("train")
    subj = lab["subj"].values
    rec = (lab["subj"].astype(str) + "|" + lab["sess"].astype(str) + "|"
           + lab["vid"].astype(str)).values

    multi = np.array([len(np.unique(rec[subj == s])) >= 2 for s in subj])
    n_all = len(np.unique(subj))
    n_multi = len(np.unique(subj[multi]))
    print("train split: %d clips, %d participants, %d recordings; "
          "%d participants carry >= 2 recordings (%d clips)"
          % (len(subj), n_all, len(np.unique(rec)), n_multi, multi.sum()),
          flush=True)

    results = []
    for ck in sorted(glob.glob(os.path.join(P4, "runs", "of_*_best.pth"))):
        tag = os.path.basename(ck)[:-len("_best.pth")]
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        model_name = blob["args"]["model"]
        if args.only and model_name not in args.only:
            continue
        X = (Xw if model_name == "mlp" else Xs)
        X = (X - blob["mu"]) / blob["sd"]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = BUILDERS[model_name](X.shape[-1]).to(dev)
        model.load_state_dict(blob["state"])
        Z = embed(model, X.astype(np.float32))
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)

        for split_name in ("random", "recording"):
            for seed in SEEDS:
                if split_name == "random":
                    tr, te = random_split(subj, seed)
                    classes = n_all
                else:
                    tr, te = recording_split(subj, rec, seed)
                    classes = n_multi
                row = {"tag": tag, "model": model_name, "split": split_name,
                       "seed": seed, "n_classes": classes,
                       "chance": 1.0 / classes,
                       "n_probe_train": int(len(tr)), "n_probe_test": int(len(te))}
                for mode in ("none", "oracle", "fitted"):
                    Zc = centre(Z, subj, tr, mode)
                    a_log = fit_logistic(Zc, subj, tr, te)
                    a_ncm = fit_ncm(Zc, subj, tr, te)
                    row["logistic_acc_" + mode] = a_log
                    row["logistic_M_" + mode] = a_log * classes
                    row["ncm_acc_" + mode] = a_ncm
                    row["ncm_M_" + mode] = a_ncm * classes
                results.append(row)
                print("%-20s %-9s s%d | logistic M none %6.1f  oracle %5.2f  "
                      "fitted %6.2f | ncm M none %6.1f  fitted %6.2f"
                      % (tag, split_name, seed, row["logistic_M_none"],
                         row["logistic_M_oracle"], row["logistic_M_fitted"],
                         row["ncm_M_none"], row["ncm_M_fitted"]), flush=True)

    with open(os.path.join(OUT, args.out), "w") as fh:
        json.dump({"n_participants_all": n_all,
                   "n_participants_multi_recording": n_multi,
                   "seeds": SEEDS, "rows": results}, fh, indent=2)

    print("\nsummary: mean multiple of chance over the 15 checkpoints")
    print("%-10s %-9s %-12s %-12s %-12s" % ("probe", "split", "no centring",
                                            "oracle centre", "fitted centre"))
    for fam in ("logistic", "ncm"):
        for sp in ("random", "recording"):
            rs = [r for r in results if r["split"] == sp]
            f = lambda m: np.mean([r["%s_M_%s" % (fam, m)] for r in rs])
            print("%-10s %-9s %-12.1f %-12.2f %-12.2f"
                  % (fam, sp, f("none"), f("oracle"), f("fitted")))
    print("\nwrote", os.path.join(OUT, args.out))


if __name__ == "__main__":
    main()
