"""Read the two E10 result files and print what the manuscript needs:
per-system test accuracy with seed SD, the gap to the published test column,
participant-clustered interval widths, and the validation figures beside them
for the reproduction-gap comparison. Runs locally on results_p4/json/.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(HERE, "results_p4", "json")

VAL = {"of_mlp": (.6474, None), "of_lstm": (.6533, 67.69), "of_cnnlstm": (.6452, 67.51),
       "of_tcn": (.6586, 67.79), "of_transformer": (.6548, 69.10), "marlin": (.6299, None)}


def widths(runs):
    return {r["tag"]: r["ci_subject"]["acc"][1] - r["ci_subject"]["acc"][0] for r in runs}


for stage, fn in (("frozen", "E10_test.json"), ("trainval", "E10_test_trainval.json")):
    d = json.load(open(os.path.join(J, fn)))
    print("=== %s  (floor %.4f, %d runs, %d participants) ===" % (
        stage, d["floor"], len(d["runs"]), d["runs"][0]["n_subjects"]))
    w = widths(d["runs"])
    by = {}
    for r in d["runs"]:
        by.setdefault(r["system"], []).append(r)
    print("%-16s %-16s %-9s %-7s %-8s %-9s %-7s %s" % (
        "system", "test acc", "pub", "gap", "val acc", "val gap", "CI w", "UAR/QWK/MAE"))
    for s, rs in sorted(by.items()):
        accs = [r["point"]["acc"] for r in rs]
        m, sd = np.mean(accs), np.std(accs)
        pub = d["summary"][s]["published"]
        key = s.replace("tv_", "of_")
        va, vpub = VAL.get(key, (None, None))
        vgap = (va * 100 - vpub) if (va and vpub) else None
        ws = [w[r["tag"]] for r in rs]
        print("%-16s %.4f +- %.4f  %-9s %-7s %-8s %-9s %.3f-%.3f  %.3f/%.3f/%.3f" % (
            s, m, sd, "%.2f" % pub if pub else "-",
            "%+.1f" % (m * 100 - pub) if pub else "-",
            "%.4f" % va if va else "-", "%+.1f" % vgap if vgap is not None else "-",
            min(ws), max(ws),
            np.mean([r["point"]["uar"] for r in rs]),
            np.mean([r["point"]["qwk"] for r in rs]),
            np.mean([r["point"]["mae"] for r in rs])))
    allw = list(w.values())
    print("interval widths over all runs: %.3f - %.3f" % (min(allw), max(allw)))
    accs = {s: np.mean([r["point"]["acc"] for r in rs]) for s, rs in by.items() if s != "marlin"}
    print("spread best-worst among descriptor systems: %.1f points\n" % (
        (max(accs.values()) - min(accs.values())) * 100))
