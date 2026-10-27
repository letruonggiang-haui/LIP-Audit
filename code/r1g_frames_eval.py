# -*- coding: utf-8 -*-
"""R1 registration, Section 4G: read the frame-ablation arms from their saved validation
probabilities (every epoch). Primary: final-epoch accuracy, UAR, within-person AUC_23 (Audit 2 of
p8_daisee_audit2.py, the definition behind the .54-.61 band of the main text). Secondary: the peak
validation accuracy, flagged as a maximum over the epochs read.

Arms: 16c, 16u and 50u (seed 0, registered), plus the A50u seeds added on 27/10/2026
(DEVIATIONS_R1.md, entry 6), read from run_50u_s<seed>/ when present.

Degenerate outputs: an epoch whose probabilities have an across-clip standard deviation below 1e-4
for every class gives (to that precision) the same score to every clip. Its accuracy and UAR are
reported (they equal a constant prediction), but AUC_23, rho_b and the band test are undefined and
set to None. The peak is taken over epochs that are neither degenerate nor constant in their
predicted class. Arm status: "trained" when the final epoch is neither; "did_not_learn" otherwise;
"collapsed_stopped" for an added seed stopped by the stop rule. The minimum training loss in
log.jsonl is reported beside the entropy of the training-label prior.

Labels come from the DAiSEE validation label file, joined on the clip identifiers saved with the
probabilities (the server copies also carry y, used only to check the join).

Usage: python r1g_frames_eval.py <dir holding run_16c/, run_16u/, run_50u/[, run_50u_s1/ ...]>
Writes paper/review_R1_20261024/results/r1_4G_frames.json and, for the repository, label-free
copies results/r1_frames_val_probs_<arm>.npz (P of every epoch and ids).
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import recall_score

ROOT = r"E:\OneDrive\R&D\Education\P2-Identity-Invariant-Engagement"
sys.path.insert(0, os.path.join(ROOT, "release_lip_audit", "code"))
import p4_labels as L  # noqa: E402
from p8_daisee_audit2 import audit2  # noqa: E402

LAB = os.path.join(ROOT, "data_local", "daisee_labels")
OUT = os.path.join(ROOT, "paper", "review_R1_20261024", "results")
REGISTERED_BAND = (0.54, 0.61)
DEGENERATE_STD = 1e-4


def read_arm(d, eng):
    z = np.load(os.path.join(d, "val_probs_all_epochs.npz"), allow_pickle=True)
    P, ids = z["P"], np.array([str(s) for s in z["ids"]])
    y = np.array([eng[s] for s in ids])
    assert np.array_equal(y, z["y"]), "label join disagrees with the server copy"
    per_epoch = []
    for e in range(P.shape[0]):
        pred = P[e].argmax(1)
        max_std = float(P[e].std(0).max())
        rec = {"epoch": e, "acc": float((pred == y).mean()),
               "uar": float(recall_score(y, pred, average="macro", zero_division=0)),
               "max_across_clip_std": max_std, "degenerate": max_std < DEGENERATE_STD,
               "constant_prediction": bool(len(np.unique(pred)) == 1)}
        if rec["degenerate"]:
            rec.update(within_person_auc23=None, pooled_auc23=None, rho_b=None)
        else:
            a2 = audit2(P[e], y, ids)
            rec.update(within_person_auc23=a2["within_person_auc23"], pooled_auc23=a2["pooled_auc23"],
                       rho_b=a2["rho_b"])
        per_epoch.append(rec)
    fin = per_epoch[-1]
    good = [r for r in per_epoch if not (r["degenerate"] or r["constant_prediction"])]
    peak = max(good, key=lambda r: r["acc"]) if good else None
    if os.path.exists(os.path.join(d, "COLLAPSED")):
        status = "collapsed_stopped"
    elif fin["degenerate"] or fin["constant_prediction"]:
        status = "did_not_learn"
    else:
        status = "trained"
    band = (None if fin["within_person_auc23"] is None else
            REGISTERED_BAND[0] <= fin["within_person_auc23"] <= REGISTERED_BAND[1])
    losses = [json.loads(l)["loss"] for l in open(os.path.join(d, "log.jsonl"), encoding="utf-8") if l.strip()]
    res = {"status": status, "epochs_read": len(per_epoch), "done": os.path.exists(os.path.join(d, "DONE")),
           "n_degenerate_epochs": sum(r["degenerate"] for r in per_epoch),
           "n_constant_prediction_epochs": sum(r["constant_prediction"] for r in per_epoch),
           "min_train_loss": min(losses), "final_train_loss": losses[-1],
           "final": fin, "peak_accuracy_epoch": peak,
           "in_registered_band": band, "per_epoch": per_epoch}
    return res, P, ids


def main(src):
    df = L.load_daisee("Validation", LAB)
    eng = dict(zip(df["stem"], df["y"].astype(int)))
    arms = ["16c", "16u", "50u"] + sorted(os.path.basename(p)[4:] for p in glob.glob(os.path.join(src, "run_50u_s*")))
    res = {"registered_band_auc23": REGISTERED_BAND, "degenerate_std_threshold": DEGENERATE_STD, "arms": {}}
    for arm in arms:
        d = os.path.join(src, "run_" + arm)
        if not os.path.exists(os.path.join(d, "val_probs_all_epochs.npz")):
            res["arms"][arm] = None
            continue
        r, P, ids = read_arm(d, eng)
        res["arms"][arm] = r
        np.savez_compressed(os.path.join(OUT, "r1_frames_val_probs_%s.npz" % arm), P=P.astype(np.float32), ids=ids)
        f, pk = r["final"], r["peak_accuracy_epoch"]
        print("%-7s %-17s epochs %3d (degenerate %3d) min loss %.4f final acc %.3f uar %.3f AUC23_w %s rho_b %s | peak %s"
              % (arm, r["status"], r["epochs_read"], r["n_degenerate_epochs"], r["min_train_loss"], f["acc"], f["uar"],
                 "-" if f["within_person_auc23"] is None else "%.3f" % f["within_person_auc23"],
                 "-" if f["rho_b"] is None else "%.2f" % f["rho_b"],
                 "-" if pk is None else "%.3f at epoch %d" % (pk["acc"], pk["epoch"])))
    floor = float(np.bincount(np.array(list(eng.values())), minlength=4).max() / len(eng))
    res["majority_floor_validation"] = floor
    ytr = L.load_daisee("Train", LAB)["y"].astype(int).values
    prior = np.bincount(ytr, minlength=4) / len(ytr)
    res["train_prior"] = prior.tolist()
    res["train_prior_entropy"] = float(-(prior[prior > 0] * np.log(prior[prior > 0])).sum())
    with open(os.path.join(OUT, "r1_4G_frames.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("majority floor %.3f; train prior entropy %.4f; wrote r1_4G_frames.json" % (floor, res["train_prior_entropy"]))


if __name__ == "__main__":
    main(sys.argv[1])
