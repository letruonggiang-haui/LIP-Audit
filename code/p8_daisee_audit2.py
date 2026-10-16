"""Audit 2 on the deposited DAiSEE prediction files (within-person AUC_23 and rho_b).

The server-side scripts that first produced these readings (audits/p2_decompose.py,
code/repro_audit2.py) read files that carried labels. This script recomputes the
same two quantities from the deposited, label-free files, joining the labels from
the DAiSEE label files on the clip identifiers (code/label_join.py):

  logs/val_probs/w5_s{0,1,2}_val_probs.npz   five-window model, trained on the training
                                             split, read on validation
  logs/val_probs/au_s0_val_probs.npz         action-unit stream, validation
  logs/TEST_FINAL_probs.npz                  sealed-test ensemble (ens)
  logs/repro_test_probs_{best,last}.npz      reference re-run, peak and final epochs

Definitions as in Section III-C of the article: on clips with label >= 2, the
decisive score is s23 = P3 / (P2 + P3); within-person AUC_23 is the clip-weighted
mean over subjects of the AUC separating level 3 from level 2 (subjects with both
levels only); rho_b is the Spearman correlation between a subject's mean s23 and
its share of level-3 clips.

  python p8_daisee_audit2.py --labels <DAiSEE>/Labels --out p8_daisee_audit2.json
"""
import argparse
import io
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import label_join as J

LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def audit2(P, y, ids):
    subj = np.array([str(s)[:6] for s in ids])
    m = y >= 2
    s23 = P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9)
    t, sc, sj = (y[m] == 3).astype(int), s23[m], subj[m]
    us = np.unique(sj)
    aucs, ws = [], []
    for s in us:
        k = sj == s
        if len(np.unique(t[k])) == 2:
            aucs.append(roc_auc_score(t[k], sc[k]))
            ws.append(k.sum())
    fr = np.array([t[sj == s].mean() for s in us])
    ms = np.array([sc[sj == s].mean() for s in us])
    return dict(clips=int(len(y)), subjects=int(len(us)),
                within_person_auc23=round(float(np.average(aucs, weights=ws)), 4),
                pooled_auc23=round(float(roc_auc_score(t, sc)), 4),
                rho_b=round(float(spearmanr(ms, fr).statistic), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="DAiSEE Labels directory")
    ap.add_argument("--out", default="p8_daisee_audit2.json")
    args = ap.parse_args()
    res = {}
    for f in ("w5_s0", "w5_s1", "w5_s2", "au_s0"):
        z = J.daisee_probs(os.path.join(LOGS, "val_probs", f + "_val_probs.npz"), args.labels, "Validation")
        res["validation/" + f] = audit2(z["P"], z["y"], z["ids"])
    z = J.daisee_sealed_test(LOGS, args.labels)
    res["test/sealed_ensemble"] = audit2(z["ens"], z["y"], z["stems"])
    for tag in ("best", "last"):
        z = J.daisee_probs(os.path.join(LOGS, "repro_test_probs_%s.npz" % tag), args.labels, "Test")
        res["test/reference_rerun_" + tag] = audit2(z["P"], z["y"], z["ids"])
    for k, v in res.items():
        print("%-32s within-person AUC23 %.3f  pooled %.3f  rho_b %.2f  (%d clips, %d subjects)" % (
            k, v["within_person_auc23"], v["pooled_auc23"], v["rho_b"], v["clips"], v["subjects"]))
    io.open(args.out, "w", encoding="utf-8").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
