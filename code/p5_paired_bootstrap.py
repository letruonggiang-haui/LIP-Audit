# -*- coding: utf-8 -*-
"""Paired, participant-clustered bootstrap of accuracy and UAR DIFFERENCES between
the reproduced EngageNet systems (review item M2). Runs on the server over the
stored per-chunk probability files in ~/p4/probs and retrains nothing; the test
partition is not re-read, only its stored predictions are used.

For every replicate the same draw of participants (with replacement) is applied
to every system; a system's value is the mean over its seeds of the metric on the
resampled chunks; differences are taken inside the replicate. 2,000 replicates.

  python p5_paired_bootstrap.py  -> ~/p4/analysis/p5_paired_bootstrap.json
"""
import glob
import json
import os

import numpy as np

from label_join import load_probs

# the server copy if present, else the repository's engagenet/ directory
P4 = os.environ.get("P4_DIR") or next(
    (p for p in (os.path.expanduser("~/p4"),
                 os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engagenet"))
     if os.path.isdir(os.path.join(p, "probs"))), os.path.expanduser("~/p4"))
OUT = os.path.join(P4, "analysis", "p5_paired_bootstrap.json")
N_BOOT = 2000
SEED = 20260909


def load(pattern, key="P"):
    runs = {}
    for f in sorted(glob.glob(os.path.join(P4, "probs", pattern))):
        tag = os.path.basename(f).split("_probs")[0]
        if tag.startswith("smoke") or tag.startswith("tcct"):
            continue
        z = load_probs(f)  # labels joined from the released label file (label_join.py)
        if key not in z or "y" not in z:
            continue
        sysname = tag.rsplit("_s", 1)[0].replace("_val", "").replace("_test", "")
        runs.setdefault(sysname, []).append(dict(pred=z[key].argmax(1), y=z["y"].astype(int),
                                                 subj=np.asarray(z["subj"]).astype(str)))
    return runs


def uar(pred, y):
    vals = []
    for k in range(4):
        m = y == k
        if m.sum():
            vals.append((pred[m] == k).mean())
    return float(np.mean(vals))


def acc(pred, y):
    return float((pred == y).mean())


def boot(runs, label):
    systems = sorted(runs)
    # every seed of every system shares the chunk order of the split
    ref = runs[systems[0]][0]
    subj = ref["subj"]
    for s in systems:
        for r in runs[s]:
            assert len(r["y"]) == len(ref["y"]) and (r["y"] == ref["y"]).all(), (label, s)
    uniq = np.unique(subj)
    idx_of = {u: np.flatnonzero(subj == u) for u in uniq}
    rng = np.random.default_rng(SEED)
    A = {s: np.empty(N_BOOT) for s in systems}
    U = {s: np.empty(N_BOOT) for s in systems}
    for b in range(N_BOOT):
        draw = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_of[u] for u in draw])
        for s in systems:
            A[s][b] = np.mean([acc(r["pred"][idx], r["y"][idx]) for r in runs[s]])
            U[s][b] = np.mean([uar(r["pred"][idx], r["y"][idx]) for r in runs[s]])
    point = {s: dict(acc=float(np.mean([acc(r["pred"], r["y"]) for r in runs[s]])),
                     uar=float(np.mean([uar(r["pred"], r["y"]) for r in runs[s]])),
                     seeds=len(runs[s])) for s in systems}
    single = {s: dict(acc_ci=[float(np.percentile(A[s], 2.5)), float(np.percentile(A[s], 97.5))],
                      uar_ci=[float(np.percentile(U[s], 2.5)), float(np.percentile(U[s], 97.5))])
              for s in systems}
    pairs = {}
    for i, s1 in enumerate(systems):
        for s2 in systems[i + 1:]:
            da, du = A[s1] - A[s2], U[s1] - U[s2]
            pairs["%s - %s" % (s1, s2)] = dict(
                d_acc=point[s1]["acc"] - point[s2]["acc"],
                d_acc_ci=[float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
                d_uar=point[s1]["uar"] - point[s2]["uar"],
                d_uar_ci=[float(np.percentile(du, 2.5)), float(np.percentile(du, 97.5))],
                p_sign_flip_acc=float(min((da <= 0).mean(), (da >= 0).mean()) * 2))
    # best sequence model minus the MLP control, chosen inside each replicate
    seqs = [s for s in systems if "mlp" not in s and "marlin" not in s]
    mlp = [s for s in systems if "mlp" in s]
    extra = {}
    if seqs and mlp:
        best_seq = np.max(np.stack([A[s] for s in seqs]), axis=0)
        d = best_seq - A[mlp[0]]
        extra["best_sequence_minus_mlp_acc"] = dict(
            point=max(point[s]["acc"] for s in seqs) - point[mlp[0]]["acc"],
            ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    return dict(n_participants=int(len(uniq)), n_chunks=int(len(subj)), systems=systems,
                point=point, single_ci=single, paired=pairs, extra=extra)


def main():
    res = {}
    # a split whose label file is not supplied is skipped
    for name, pat, key, lab in (("validation_best_epoch", "of_*_val_probs.npz", "P", "val"),
                                ("validation_last_epoch", "of_*_val_probs.npz", "P_last", "val-last"),
                                ("test_frozen", "of_*_test_probs.npz", "P", "test"),
                                ("test_trainval", "tv_*_test_probs.npz", "P", "test-tv")):
        runs = load(pat, key)
        if runs:
            res[name] = boot(runs, lab)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        print("==", k, v["n_participants"], "participants")
        for s in v["systems"]:
            print("  %-16s acc %.3f [%.3f, %.3f]" % (s, v["point"][s]["acc"], *v["single_ci"][s]["acc_ci"]))
        for p, d in v["paired"].items():
            print("  %-36s d_acc %+.3f [%+.3f, %+.3f]" % (p, d["d_acc"], *d["d_acc_ci"]))
        print("  extra", v["extra"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
