"""E8 and E9 - clustered intervals, and how far the ranking survives them.

Both run entirely on the stored per-clip probabilities.

E8 puts a 95 % interval on every metric of every system by resampling
participants with replacement and keeping all their clips, B = 2,000. The naive
clip-level bootstrap is computed beside it, to say how far an interval that
ignores the clustering under-covers. With eleven validation participants the
intervals will be wide; that is the finding, not a defect, because the field
ranks these systems by differences of one to two points.

E9 ranks the systems under every protocol variant the stored predictions permit:
five clip-level metrics, their participant-averaged and recording-averaged
counterparts, and the two epoch-selection rules. Kendall's tau between every
pair of rankings, and a list of the pairwise orderings that flip.

The SNP policies of the E9 plan are not computable here: the validation split
contains no SNP chunk, so exclusion, a fifth class and collapsing into
Not-Engaged coincide on it. That is recorded rather than silently skipped.

Usage:  python p4_e89.py
"""
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

P4 = os.path.expanduser("~/p4")
OUT = os.path.join(P4, "analysis")


def load_all():
    runs = {}
    for f in sorted(glob.glob(os.path.join(P4, "probs", "*_val_probs.npz"))):
        if "smoke" in f:
            continue
        tag = os.path.basename(f)[:-len("_val_probs.npz")]
        z = np.load(f, allow_pickle=True)
        runs[tag] = {"P": z["P"], "P_last": z["P_last"] if "P_last" in z.files else None,
                     "y": z["y"], "subj": z["subj"], "rec": np.asarray(z["rec"], dtype=str),
                     "system": tag.rsplit("_s", 1)[0]}
    return runs


def metrics(pred, y):
    return {"acc": float((pred == y).mean()),
            "uar": float(recall_score(y, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
            "neg_mae": float(-np.abs(pred - y).mean())}


def grouped_accuracy(pred, y, group):
    """Accuracy computed per group and then averaged over groups."""
    return float(np.mean([(pred[group == g] == y[group == g]).mean()
                          for g in np.unique(group)]))


# --------------------------------------------------------------------------- #
# E8
# --------------------------------------------------------------------------- #
def bootstrap(pred, y, clusters, n_boot=2000, seed=20260905):
    rng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uc}
    keys = ["acc", "uar", "macro_f1", "qwk", "neg_mae"]
    draws = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.choice(uc, len(uc), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        m = metrics(pred[idx], y[idx])
        for k in keys:
            draws[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in draws.items()}


def run_e8(runs, n_boot=2000):
    rows = []
    for tag, r in runs.items():
        pred = r["P"].argmax(1)
        y = r["y"]
        point = metrics(pred, y)
        clip_clusters = np.arange(len(y))
        rows.append({"tag": tag, "system": r["system"], "point": point,
                     "ci_subject": bootstrap(pred, y, r["subj"], n_boot),
                     "ci_recording": bootstrap(pred, y, r["rec"], n_boot),
                     "ci_clip": bootstrap(pred, y, clip_clusters, n_boot),
                     "n_subjects": int(len(np.unique(r["subj"]))),
                     "n_recordings": int(len(np.unique(r["rec"])))})
        c = rows[-1]
        print("%-20s acc %.4f  subject [%.4f, %.4f] w=%.4f | clip [%.4f, %.4f] w=%.4f"
              % (tag, point["acc"], c["ci_subject"]["acc"][0], c["ci_subject"]["acc"][1],
                 c["ci_subject"]["acc"][1] - c["ci_subject"]["acc"][0],
                 c["ci_clip"]["acc"][0], c["ci_clip"]["acc"][1],
                 c["ci_clip"]["acc"][1] - c["ci_clip"]["acc"][0]), flush=True)
    return rows


# --------------------------------------------------------------------------- #
# E9
# --------------------------------------------------------------------------- #
def run_e9(runs):
    systems = sorted(set(r["system"] for r in runs.values()))
    variants = {}
    for sel in ("best", "last"):
        for name in ("acc", "uar", "macro_f1", "qwk", "neg_mae",
                     "acc_by_participant", "acc_by_recording"):
            variants["%s/%s" % (sel, name)] = {}

    for sysname in systems:
        tags = [t for t, r in runs.items() if r["system"] == sysname]
        for sel in ("best", "last"):
            vals = defaultdict(list)
            for t in tags:
                r = runs[t]
                Pm = r["P"] if sel == "best" else r["P_last"]
                if Pm is None:
                    continue
                pred, y = Pm.argmax(1), r["y"]
                m = metrics(pred, y)
                for k, v in m.items():
                    vals[k].append(v)
                vals["acc_by_participant"].append(grouped_accuracy(pred, y, r["subj"]))
                vals["acc_by_recording"].append(grouped_accuracy(pred, y, r["rec"]))
            for k, v in vals.items():
                if v:
                    variants["%s/%s" % (sel, k)][sysname] = float(np.mean(v))

    variants = {k: v for k, v in variants.items() if len(v) == len(systems)}
    ranks = {}
    for k, v in variants.items():
        order = sorted(systems, key=lambda s: -v[s])
        ranks[k] = {s: i for i, s in enumerate(order)}
        print("%-28s %s" % (k, " > ".join(order)))

    names = sorted(ranks)
    taus = {}
    for a, b in itertools.combinations(names, 2):
        ra = [ranks[a][s] for s in systems]
        rb = [ranks[b][s] for s in systems]
        taus["%s vs %s" % (a, b)] = float(kendalltau(ra, rb).statistic)

    flips = []
    for s1, s2 in itertools.combinations(systems, 2):
        signs = set(np.sign(variants[k][s1] - variants[k][s2]) for k in names)
        if len(signs) > 1:
            flips.append({"pair": [s1, s2],
                          "orderings": {k: round(variants[k][s1] - variants[k][s2], 4)
                                        for k in names}})
    return {"systems": systems, "variants": variants, "ranks": ranks,
            "kendall_tau": taus, "flipping_pairs": flips,
            "snp_note": ("The validation split contains no SNP chunk, so the three "
                         "SNP policies of the plan coincide on it and are not a "
                         "variant here.")}


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_all()
    print("loaded %d runs\n=== E8 ===" % len(runs))
    e8 = run_e8(runs)
    with open(os.path.join(OUT, "E8_intervals.json"), "w") as fh:
        json.dump(e8, fh, indent=2)

    print("\n=== E9 ===")
    e9 = run_e9(runs)
    with open(os.path.join(OUT, "E9_variants.json"), "w") as fh:
        json.dump(e9, fh, indent=2)

    taus = list(e9["kendall_tau"].values())
    print("\nKendall tau over %d ranking pairs: min %.2f, median %.2f, %d pairs at 1.00"
          % (len(taus), min(taus), float(np.median(taus)), sum(1 for t in taus if t == 1.0)))
    print("pairs of systems whose ordering flips across variants: %d of %d"
          % (len(e9["flipping_pairs"]),
             len(e9["systems"]) * (len(e9["systems"]) - 1) // 2))
    for f in e9["flipping_pairs"]:
        print("  %s vs %s" % tuple(f["pair"]))
    print("\nwrote E8_intervals.json and E9_variants.json")


if __name__ == "__main__":
    main()
