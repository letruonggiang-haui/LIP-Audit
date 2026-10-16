"""Between-split heterogeneity of the DAiSEE Audit 1 kappa under the rule of record.

Supplementary S3.5 tested whether the train, validation and test kappas of
Table 2 differ. The earlier test (p6_controls.py, A3) ran on that script's own
pairing (ten-character identifiers, pair marginals: .142 / .104 / .042). This
script repeats it on the pairing and chance term of p7_audit1_daisee.py, the
analysis of record for Table 2 (.144 / .085 / .053):

  - recording-clustered bootstrap per split (sessions resampled with
    replacement; pairs and the split-marginal chance term rebuilt inside each
    replicate), 2,000 replicates, seed 20260928;
  - pairwise differences of kappa: percentile 95% interval and a two-sided
    bootstrap p, twice the smaller tail of the replicate differences;
  - common-value test: the share of centred replicate triples whose range
    reaches the observed range of the three kappas.

Reads the DAiSEE label files only.
  python p8_kappa_heterogeneity.py --labels <DAiSEE>/Labels --out results/p8_kappa_heterogeneity.json
"""
import argparse
import io
import json

import numpy as np

import p7_audit1_daisee as A

B = 2000


def boot_kappa(pairs, df_dec, rng):
    sess = np.array([p[0] for p in pairs])
    keys = np.unique(np.concatenate([sess, df_dec["sess"].unique()]))
    kidx = {k: i for i, k in enumerate(keys)}
    by_pairs = [[] for _ in keys]
    for i, s in enumerate(sess):
        by_pairs[kidx[s]].append(i)
    by_clips = [[] for _ in keys]
    ydec = df_dec["y"].to_numpy()
    for i, s in enumerate(df_dec["sess"].to_numpy()):
        by_clips[kidx[s]].append(i)
    out = []
    for _ in range(B):
        draw = rng.integers(0, len(keys), len(keys))
        pidx = np.concatenate([np.array(by_pairs[d], int) for d in draw])
        cidx = np.concatenate([np.array(by_clips[d], int) for d in draw])
        if len(pidx) < 10:
            continue
        out.append(A.stats([pairs[i] for i in pidx], ydec[cidx])[2])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=A.LAB, help="DAiSEE Labels directory")
    ap.add_argument("--out", default="p8_kappa_heterogeneity.json")
    args = ap.parse_args()
    rng = np.random.default_rng(20260928)
    kap, boots = {}, {}
    for split in ("Train", "Validation", "Test"):
        df = A.load(split, args.labels)
        dfd = df[df["y"] >= 2].reset_index(drop=True)
        dec = A.decisive(A.pairs_for(df, "s31_allwidths"))
        kap[split] = A.stats(dec, dfd["y"].to_numpy())[2]
        boots[split] = boot_kappa(dec, dfd, rng)
    res = {"kappa": {k: round(v, 4) for k, v in kap.items()}, "n_boot": B, "seed": 20260928, "pairwise": {}}
    for a, b in (("Train", "Validation"), ("Train", "Test"), ("Validation", "Test")):
        m = min(len(boots[a]), len(boots[b]))
        d = boots[a][:m] - boots[b][:m]
        res["pairwise"]["%s-%s" % (a, b)] = dict(
            point=round(kap[a] - kap[b], 4),
            ci=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            p_two_sided=round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4))
    m = min(len(v) for v in boots.values())
    stack = np.vstack([boots[k][:m] - boots[k][:m].mean() for k in ("Train", "Validation", "Test")])
    null_range = stack.max(axis=0) - stack.min(axis=0)
    obs = max(kap.values()) - min(kap.values())
    res["heterogeneity"] = dict(observed_range=round(obs, 4), null_range_p=round(float((null_range >= obs).mean()), 4))
    print(json.dumps(res, indent=1))
    io.open(args.out, "w", encoding="utf-8").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
