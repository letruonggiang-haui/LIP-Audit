"""P6 - the four post-hoc controls the round-2 panel asked for that need no
retraining and no fresh read of any sealed partition.

  A3  Heterogeneity across the three DAiSEE adjacency kappas.  Table 4 reports
      .144 / .085 / .053 with overlapping intervals and the article then draws a
      binary verdict from two of the three splits.  This bootstraps each split's
      kappa under the same recording-clustered scheme and forms the pairwise
      differences, so the question "are these three distinguishable" gets an
      answer instead of an assumption.  Self-check first: the script must
      reproduce Table 4's pair counts and kappas before its new numbers are
      used.

  A2  Acc_maj with a leave-one-clip-out mode.  The constant-per-person oracle in
      the article estimates each participant's modal label from the very clips
      it is then scored on, so part of its margin is in-sample.  Recomputed by
      taking, for each clip, the mode of that participant's other clips.

  A8  Session-clock control for within-person AUC.  A feature that perceives no
      state at all -- the chunk's position in its recording -- is scored with the
      same within-person AUC the article elevates, to show how much of that
      metric a clock earns.

  A9  Optimism of the reference re-run's 56.9%.  That figure is a maximum over
      100 per-epoch test reads.  The per-epoch trace is stored, so the optimism
      of the maximum is estimated from the trace's own dispersion.

Inputs  : ~/daisee/DAiSEE/Labels/*.csv           (A3)
          ~/p4/probs/*_val_probs.npz, *_test_probs.npz   (A2, A8)
          ~/p2/repro/repro_log.jsonl             (A9)
Output  : ~/p4/analysis/p6_controls.json

Reads no video, trains nothing, and re-reads no sealed partition: A2 and A8 run
on the per-clip probabilities stored at the single registered EngageNet test
read, and A9 on a log written by an already-completed registered run.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

from label_join import load_probs

LAB = os.path.expanduser("~/daisee/DAiSEE/Labels")
PROBS = os.path.expanduser("~/p4/probs")
REPRO = os.path.expanduser("~/p2/repro/repro_log.jsonl")
OUT = os.path.expanduser("~/p4/analysis")
B = 2000
RNG = np.random.default_rng(0)
LEVELS = np.array([0.0, 1.0, 2.0, 3.0])


# ---------------------------------------------------------------- A3

def load_daisee(split):
    """ClipID -> engagement level, for one official split."""
    import csv
    path = os.path.join(LAB, "%sLabels.csv" % split)
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cid = None
            for k in row:
                if k.strip().lower() in ("clipid", "clip id"):
                    cid = row[k].strip()
            if cid is None:
                continue
            stem = os.path.splitext(cid)[0]
            for k in row:
                if k.strip().lower() == "engagement":
                    try:
                        out[stem] = int(float(row[k]))
                    except ValueError:
                        pass
    return out


def adjacent_pairs(labels):
    """Strict index-adjacent pairs inside a seven-character session prefix.

    Returns (session_id, y1, y2) for every pair of ten-character identifiers
    whose trailing indices differ by exactly one.
    """
    sess = defaultdict(dict)
    for stem, y in labels.items():
        if len(stem) != 10 or not stem.isdigit():
            continue
        sess[stem[:7]][int(stem[7:])] = y
    pairs = []
    for s, d in sess.items():
        for t in sorted(d):
            if t + 1 in d:
                pairs.append((s, d[t], d[t + 1]))
    return pairs


def kappa_binary(pairs, idx):
    """Agreement, chance agreement and kappa on the 2-versus-3 decisive axis."""
    a1 = np.array([pairs[i][1] for i in idx])
    a2 = np.array([pairs[i][2] for i in idx])
    keep = (a1 >= 2) & (a2 >= 2)
    a1, a2 = a1[keep], a2[keep]
    if len(a1) < 10:
        return np.nan, np.nan, np.nan, 0
    b1, b2 = (a1 == 3).astype(int), (a2 == 3).astype(int)
    agree = float((b1 == b2).mean())
    p = np.concatenate([b1, b2]).mean()
    chance = p * p + (1 - p) * (1 - p)
    kap = (agree - chance) / (1 - chance) if chance < 1 else np.nan
    return agree, chance, kap, len(a1)


def boot_kappa(pairs, n=B):
    """Recording-clustered bootstrap: resample sessions, rebuild the pair set."""
    by_sess = defaultdict(list)
    for i, (s, _, _) in enumerate(pairs):
        by_sess[s].append(i)
    keys = list(by_sess)
    out = []
    for _ in range(n):
        draw = RNG.choice(len(keys), size=len(keys), replace=True)
        idx = []
        for j in draw:
            idx += by_sess[keys[j]]
        out.append(kappa_binary(pairs, idx)[2])
    return np.array(out, dtype=float)


def run_a3():
    res = {}
    boots = {}
    for split in ("Train", "Validation", "Test"):
        labels = load_daisee(split)
        pairs = adjacent_pairs(labels)
        agree, chance, kap, npair = kappa_binary(pairs, range(len(pairs)))
        bs = boot_kappa(pairs)
        bs = bs[~np.isnan(bs)]
        boots[split] = bs
        res[split] = dict(clips=len(labels), pairs_all=len(pairs),
                          pairs_decisive=npair, agreement=round(agree, 4),
                          chance=round(chance, 4), kappa=round(kap, 4),
                          ci=[round(float(np.percentile(bs, 2.5)), 4),
                              round(float(np.percentile(bs, 97.5)), 4)])
    diffs = {}
    for a, b in (("Train", "Validation"), ("Train", "Test"),
                 ("Validation", "Test")):
        m = min(len(boots[a]), len(boots[b]))
        d = boots[a][:m] - boots[b][:m]
        diffs["%s-%s" % (a, b)] = dict(
            point=round(res[a]["kappa"] - res[b]["kappa"], 4),
            ci=[round(float(np.percentile(d, 2.5)), 4),
                round(float(np.percentile(d, 97.5)), 4)],
            p_two_sided=round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4))
    # Range statistic: how often do three independent draws span at least the
    # observed range, if the three splits shared one kappa?
    m = min(len(v) for v in boots.values())
    centred = {k: v[:m] - v[:m].mean() for k, v in boots.items()}
    stack = np.vstack([centred[k] for k in ("Train", "Validation", "Test")])
    null_range = stack.max(axis=0) - stack.min(axis=0)
    obs_range = max(res[k]["kappa"] for k in res) - min(res[k]["kappa"] for k in res)
    res["heterogeneity"] = dict(
        observed_range=round(float(obs_range), 4),
        null_range_p=round(float((null_range >= obs_range).mean()), 4),
        pairwise=diffs)
    return res


# ---------------------------------------------------------------- A2, A8

def load_probs(tag):
    path = os.path.join(PROBS, tag)
    return dict(load_probs(path))  # labels joined from the released label file


def within_person_auc(score, y, subj):
    """Mean over participants of the AUC separating level 3 from level 2."""
    aucs = []
    for s in np.unique(subj):
        m = subj == s
        yy, ss = y[m], score[m]
        pos, neg = ss[yy == 3], ss[yy == 2]
        if len(pos) < 1 or len(neg) < 1:
            continue
        # Mann-Whitney with ties at one half.
        gt = (pos[:, None] > neg[None, :]).sum()
        eq = (pos[:, None] == neg[None, :]).sum()
        aucs.append((gt + 0.5 * eq) / (len(pos) * len(neg)))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def acc_maj(y, subj, leave_one_out):
    """Constant-per-person oracle, in-sample or leave-one-clip-out."""
    hit = 0
    for s in np.unique(subj):
        m = subj == s
        yy = y[m]
        if not leave_one_out:
            vals, cnt = np.unique(yy, return_counts=True)
            hit += int((yy == vals[cnt.argmax()]).sum())
        else:
            for i in range(len(yy)):
                other = np.delete(yy, i)
                if len(other) == 0:
                    continue
                vals, cnt = np.unique(other, return_counts=True)
                hit += int(yy[i] == vals[cnt.argmax()])
    return hit / len(y)


def run_a2_a8():
    out = {}
    for split, suffix in (("validation", "_val_probs.npz"),
                          ("test_frozen", "_test_probs.npz")):
        files = sorted(f for f in os.listdir(PROBS) if f.endswith(suffix))
        if not files:
            continue
        d = load_probs(files[0])
        y, subj = d["y"], d["subj"]
        t = d["t"] if "t" in d else None
        rec = d["rec"] if "rec" in d else None
        entry = dict(
            clips=int(len(y)), participants=int(len(np.unique(subj))),
            acc_maj_in_sample=round(acc_maj(y, subj, False), 4),
            acc_maj_loo=round(acc_maj(y, subj, True), 4),
            floor=round(float(np.bincount(y).max() / len(y)), 4))
        if t is not None:
            # A8: the clock alone, scored by the article's own within-person AUC.
            clock = t.astype(float)
            auc, npart = within_person_auc(clock, y, subj)
            # Position within its own recording, if the recording field exists.
            if rec is not None:
                rel = np.zeros(len(t), dtype=float)
                for r in np.unique(rec):
                    m = rec == r
                    o = np.argsort(t[m])
                    ranks = np.empty(m.sum(), dtype=float)
                    ranks[o] = np.arange(m.sum(), dtype=float)
                    rel[m] = ranks / max(1, m.sum() - 1)
                auc_rel, _ = within_person_auc(rel, y, subj)
            else:
                auc_rel = float("nan")
            entry["clock_within_person_auc"] = round(auc, 4)
            entry["clock_relative_within_person_auc"] = round(auc_rel, 4)
            entry["clock_participants_scored"] = int(npart)
        # Every model's own within-person AUC on the same clips, for scale.
        model_aucs = {}
        for f in files:
            dd = load_probs(f)
            P = dd["P"]
            score = P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9)
            a, _ = within_person_auc(score, dd["y"], dd["subj"])
            model_aucs[f.replace(suffix, "")] = round(a, 4)
        entry["model_within_person_auc"] = model_aucs
        out[split] = entry
    return out


# ---------------------------------------------------------------- A9

def run_a9():
    if not os.path.exists(REPRO):
        return {"available": False}
    accs = []
    with open(REPRO) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            for k in ("test_acc", "acc", "test_accuracy"):
                if k in r:
                    accs.append(float(r[k]))
                    break
    if len(accs) < 10:
        return {"available": False, "rows": len(accs)}
    a = np.array(accs)
    tail = a[len(a) // 2:]
    # Expected maximum of n independent standard normals, by simulation.
    sim = RNG.standard_normal((4000, len(a)))
    e_max = float(sim.max(axis=1).mean())
    return dict(available=True, epochs=len(a),
                max=round(float(a.max()), 4),
                final=round(float(a[-1]), 4),
                plateau_mean=round(float(tail.mean()), 4),
                plateau_sd=round(float(tail.std(ddof=1)), 4),
                expected_max_minus_mean=round(float(tail.std(ddof=1) * e_max), 4),
                note=("optimism estimated as the plateau SD times the expected "
                      "maximum of as many standard normal draws as there are "
                      "epochs; this assumes the per-epoch fluctuation is "
                      "exchangeable noise around a flat plateau, which is an "
                      "approximation"))


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"A3_kappa_heterogeneity": run_a3(),
           "A2_A8_person_controls": run_a2_a8(),
           "A9_max_optimism": run_a9()}
    path = os.path.join(OUT, "p6_controls.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
