"""Post hoc analyses on the stored EngageNet test probabilities.

The registration (DANG_KY_TRUOC_TEST_ENGAGENET_2026-09-05.md) spends the test
partition once, in p4_eval_test.py, and says that everything afterwards is post
hoc on the per-clip probabilities that run stored. This is that afterwards. It
reads ~/p4/probs/*_test_probs.npz and retrains nothing.

Two analyses, both already defined for validation and reused unchanged:
  E7  the within-person decomposition of p4_e7_decompose.analyse, now over 26
      participants instead of 11; the placeholder block is skipped because the
      test probability files carry no tracked-frame record;
  E9  the ranking of the five descriptor systems under the seven metrics of
      p4_e89, at the stored epoch only, since a test run has no "last epoch"
      variant to offer - the epoch was frozen on validation before the test
      descriptors existed.

Usage:  python p4_e79_test.py [frozen|trainval]
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p4_e7_decompose import analyse  # noqa: E402
from p4_e89 import grouped_accuracy, metrics  # noqa: E402

P4 = os.path.expanduser("~/p4")
OUT = os.path.join(P4, "analysis")


def run_e7(files, label):
    rows = [analyse(f) for f in files]
    for r in rows:
        r["tag"] = os.path.basename(files[rows.index(r)])[:-len("_test_probs.npz")]
        d = r["decisive"]
        print("%-20s decisive blended %.4f within %.4f (p %.4f, n=%d) centred %.4f rho_b %+.2f "
              "| eta2 score %.3f label %.3f"
              % (r["tag"], d["auc_blended"], d["auc_within_person"],
                 d["within_person_null"]["p"], d["n_participants_usable"],
                 d["auc_after_centring"], d["rho_between"],
                 r["ordinal"]["eta2_between_score"], r["ordinal"]["eta2_between_label"]),
              flush=True)
    json.dump(rows, open(os.path.join(OUT, "E7_test_%s.json" % label), "w"), indent=2)
    return rows


def run_e9(files, label):
    runs = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        tag = os.path.basename(f)[:-len("_test_probs.npz")]
        if tag.startswith("marlin"):
            continue
        runs[tag] = {"P": z["P"], "y": z["y"], "subj": z["subj"],
                     "rec": np.asarray(z["rec"], dtype=str),
                     "system": tag.rsplit("_s", 1)[0]}
    systems = sorted(set(r["system"] for r in runs.values()))
    names = ["acc", "uar", "macro_f1", "qwk", "neg_mae", "acc_by_participant", "acc_by_recording"]
    variants = {n: {} for n in names}
    for s in systems:
        vals = {n: [] for n in names}
        for t, r in runs.items():
            if r["system"] != s:
                continue
            pred, y = r["P"].argmax(1), r["y"]
            for k, v in metrics(pred, y).items():
                vals[k].append(v)
            vals["acc_by_participant"].append(grouped_accuracy(pred, y, r["subj"]))
            vals["acc_by_recording"].append(grouped_accuracy(pred, y, r["rec"]))
        for n in names:
            variants[n][s] = float(np.mean(vals[n]))
    ranks = {}
    for n in names:
        order = sorted(systems, key=lambda s: -variants[n][s])
        ranks[n] = {s: i for i, s in enumerate(order)}
        print("%-20s %s" % (n, " > ".join(order)))
    taus = {"%s vs %s" % (a, b): float(kendalltau([ranks[a][s] for s in systems],
                                                  [ranks[b][s] for s in systems]).statistic)
            for a, b in itertools.combinations(names, 2)}
    flips = [[s1, s2] for s1, s2 in itertools.combinations(systems, 2)
             if len(set(np.sign(variants[n][s1] - variants[n][s2]) for n in names)) > 1]
    distinct = len(set(tuple(sorted(r, key=r.get)) for r in ranks.values()))
    out = {"systems": systems, "variants": variants, "ranks": ranks, "kendall_tau": taus,
           "flipping_pairs": flips, "n_distinct_rankings": distinct,
           "note": "stored epoch only: no last-epoch variant exists for a test run"}
    json.dump(out, open(os.path.join(OUT, "E9_test_%s.json" % label), "w"), indent=2)
    tv = list(taus.values())
    print("%d variants, %d distinct rankings, %d of %d pairs flip, tau min %.2f median %.2f"
          % (len(names), distinct, len(flips), len(systems) * (len(systems) - 1) // 2,
             min(tv), float(np.median(tv))))
    return out


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "frozen"
    prefix = {"frozen": ("of_", "marlin"), "trainval": ("tv_",)}[label]
    files = sorted(f for f in glob.glob(os.path.join(P4, "probs", "*_test_probs.npz"))
                   if os.path.basename(f).startswith(prefix))
    if not files:
        raise SystemExit("no test probability files for %s" % label)
    print("=== E7 on test, %s: %d runs ===" % (label, len(files)))
    run_e7(files, label)
    print("\n=== E9 on test, %s ===" % label)
    run_e9(files, label)


if __name__ == "__main__":
    main()
