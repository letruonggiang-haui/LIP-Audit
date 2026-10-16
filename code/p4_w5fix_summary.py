"""Peak-to-final summary for the stability-package ablation (fixed middle window,
no random draw) against the full package, both read from per-epoch logs with the
same rule: peak = best epoch of that metric, final = last epoch.
Usage: python p4_w5fix_summary.py   (reads results_p4/json/w5fix/*.jsonl)"""
import glob, json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "results_p4", "json", "w5fix")
out = {}
for f in sorted(glob.glob(os.path.join(D, "*.jsonl"))):
    name = os.path.basename(f)[:-6]
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rec = {"n_ep": len(rows)}
    for k in ("auc_dis", "auc23", "acc"):
        v = [r[k] for r in rows]; pk = max(v)
        rec[k] = {"peak": round(pk, 4), "peak_ep": v.index(pk), "final": round(v[-1], 4),
                  "delta": round(v[-1] - pk, 4)}
    out[name] = rec
    print(f"{name:10s} n={rec['n_ep']:2d}  AUC_dis {rec['auc_dis']['peak']:.3f}(ep{rec['auc_dis']['peak_ep']})->{rec['auc_dis']['final']:.3f} {rec['auc_dis']['delta']:+.3f}"
          f"  AUC23 {rec['auc23']['peak']:.3f}(ep{rec['auc23']['peak_ep']})->{rec['auc23']['final']:.3f} {rec['auc23']['delta']:+.3f}"
          f"  acc {rec['acc']['peak']:.3f}->{rec['acc']['final']:.3f}")
json.dump(out, open(os.path.join(HERE, "results_p4", "json", "E11_w5fix_ablation.json"), "w"), indent=1)
