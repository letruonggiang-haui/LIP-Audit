"""Per-system means of the E7 and E9 test analyses, from results_p4/json/."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(HERE, "results_p4", "json")

for label in ("frozen", "trainval"):
    rows = json.load(open(os.path.join(J, "E7_test_%s.json" % label)))
    by = {}
    for r in rows:
        by.setdefault(r["tag"].rsplit("_s", 1)[0], []).append(r)
    print("=== E7 test, %s ===" % label)
    print("%-16s %-8s %-8s %-8s %-6s %-6s %-12s %s" % (
        "system", "blended", "within", "centred", "rho_b", "eta2", "eta2 range", "n_usable"))
    lab = rows[0]["decisive"]["eta2_between_label"]
    allw = []
    for s, rs in sorted(by.items()):
        d = [r["decisive"] for r in rs]
        w = [x["auc_within_person"] for x in d]
        allw += w
        e = [r["ordinal"]["eta2_between_score"] for r in rs]
        print("%-16s %.3f    %.3f    %.3f    %+.2f  %.3f  %.3f-%.3f  %d" % (
            s, np.mean([x["auc_blended"] for x in d]), np.mean(w),
            np.mean([x["auc_after_centring"] for x in d]),
            np.mean([x["rho_between"] for x in d]), np.mean(e), min(e), max(e),
            d[0]["n_participants_usable"]))
    print("label eta2 (ordinal): %.3f ; decisive-axis label eta2: %.3f" % (
        rows[0]["ordinal"]["eta2_between_label"], lab))
    print("within-person over all runs: %.3f - %.3f ; max p = %.4f" % (
        min(allw), max(allw), max(r["decisive"]["within_person_null"]["p"] for r in rows)))
    e9 = json.load(open(os.path.join(J, "E9_test_%s.json" % label)))
    tv = list(e9["kendall_tau"].values())
    print("E9: %d distinct rankings of %d variants, %d of 10 pairs flip, tau min %.2f median %.2f\n" % (
        e9["n_distinct_rankings"], len(e9["variants"]), len(e9["flipping_pairs"]),
        min(tv), float(np.median(tv))))
