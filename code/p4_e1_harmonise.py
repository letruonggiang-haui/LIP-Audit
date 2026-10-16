"""E1 - label-space harmonisation and matching permutation nulls.

Runs the same instrument on both benchmarks in both label spaces, so that the
raw-agreement rows of the contrast table can be compared directly instead of
only through kappa, and runs both permutation nulls on both benchmarks so that
the two columns of the contrast table rest on the same test.

Outputs results_p4/E1_results.json and results_p4/E1_report.md.

Usage:  python p4_e1_harmonise.py [--perm 1000] [--no-test]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4_labels as L  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_p4")

DAISEE_SPLITS = ["Train", "Validation", "Test"]
EN_SPLITS = ["train", "validation", "test"]


def prepare_daisee(split):
    d = L.load_daisee(split)
    d = d[d["y"].notna()].reset_index(drop=True)
    return d


def prepare_engagenet(split):
    e = L.load_engagenet(split)
    n_snp = int(e["y"].isna().sum())
    e = e[e["y"].notna()].reset_index(drop=True)
    e.attrs["n_snp"] = n_snp
    return e


def one_benchmark(name, splits, prep, strict, n_perm, verbose=True):
    """Every E1 quantity for one benchmark, in both label spaces."""
    res = {}
    for split in splits:
        t0 = time.time()
        df = prep(split)
        d1, a1, b1 = L.build_pairs(df, "rec", strict=strict, distance=1)
        _, a2, b2 = L.build_pairs(df, "rec", strict=strict, distance=2)
        dna, ana, bna = L.build_nonadjacent_pairs(df, "subj", "rec")
        y = d1["y"].values.astype(float)
        yna = dna["y"].values.astype(float)
        rec = d1["rec"].values
        subj = d1["subj"].values

        row = {"n_clips_labelled": int(len(d1)),
               "n_recordings": int(d1["rec"].nunique()),
               "n_subjects": int(d1["subj"].nunique())}
        if "n_snp" in df.attrs:
            row["n_snp_removed"] = df.attrs["n_snp"]

        for space, sel in (("four_class", L.space_full), ("decisive", L.space_decisive)):
            keep = sel(d1)
            keep_na = sel(dna)
            block = {"adjacent": L.agreement(y, a1, b1, keep),
                     "distance_two": L.agreement(y, a2, b2, keep),
                     "non_adjacent_same_subject": L.agreement(yna, ana, bna, keep_na),
                     "null_within_recording":
                         L.permutation_null(y, a1, b1, keep, rec, n_perm=n_perm),
                     "null_within_subject":
                         L.permutation_null(y, a1, b1, keep, subj, n_perm=n_perm)}
            row[space] = block
            if verbose:
                ag = block["adjacent"]
                print("  %-10s %-10s pairs=%5d a=%.4f a_iid=%.4f kappa=%.4f "
                      "| null(rec) p=%.4f mean=%.4f | null(subj) p=%.4f mean=%.4f"
                      % (split, space, ag["pairs"], ag["a"], ag["a_iid"], ag["kappa"],
                         block["null_within_recording"]["p"],
                         block["null_within_recording"]["null_mean"],
                         block["null_within_subject"]["p"],
                         block["null_within_subject"]["null_mean"]))
        row["seconds"] = round(time.time() - t0, 1)
        res[split] = row
    return res


def daisee_id_hygiene():
    """DAiSEE ClipIDs are not fixed width, which the session/order convention
    assumes. Report how many clips the convention is applied to as written."""
    out = {}
    for split in DAISEE_SPLITS:
        d = L.load_daisee(split)
        lens = Counter(int(v) for v in d["idlen"].values)
        disk = L.daisee_disk_stems(split)
        stems = set(d["stem"])
        out[split] = {"n_labelled": int(len(d)),
                      "id_length_counts": dict(sorted(lens.items())),
                      "n_wellformed_10char": int(d["wellformed"].sum()),
                      "n_clip_files_on_disk": int(len(disk)),
                      "n_labelled_missing_from_disk": int(len(stems - disk)),
                      "n_files_without_a_label": int(len(disk - stems))}
    return out


def wellformed_only(n_perm):
    """The adjacency statistic restricted to the 10-character ClipIDs, where
    the session and order fields are unambiguous."""
    out = {}
    for split in DAISEE_SPLITS:
        d = L.load_daisee(split)
        d = d[d["y"].notna() & d["wellformed"]].reset_index(drop=True)
        d1, a1, b1 = L.build_pairs(d, "rec", strict=True, distance=1)
        y = d1["y"].values.astype(float)
        out[split] = {
            "four_class": L.agreement(y, a1, b1, L.space_full(d1)),
            "decisive": L.agreement(y, a1, b1, L.space_decisive(d1)),
        }
    return out


def archived_daisee_null(n_perm=2000, seed=0):
    """Reproduce the null behind the p = .76 / .53 quoted in the manuscript.

    The archived `p2_revision_stats.py` builds its pair index with a gap of up
    to two, permutes the four-class label inside each participant and only then
    restricts to pairs at Engaged or above, so the pair set moves between
    replicates. The agreement that null is compared against is therefore the
    gap-two-inclusive one, not the strict-adjacency value the table reports.
    This function reruns it unchanged so the two can be quoted side by side.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for split in ["Validation", "Test"]:
        d = L.load_daisee(split)
        d = d[d["y"].notna()].reset_index(drop=True)
        y = d["y"].values.astype(int)
        subj = d["subj"].values
        rec = d["rec"].values
        t = d["t"].values
        pair = []
        for s in np.unique(rec):
            idx = np.flatnonzero(rec == s)
            idx = idx[np.argsort(t[idx])]
            oo = t[idx]
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    if oo[j] - oo[i] > 2:
                        break
                    pair.append((idx[i], idx[j]))
        pair = np.asarray(pair)

        def agree(yv):
            yi, yj = yv[pair[:, 0]], yv[pair[:, 1]]
            m = (yi >= 2) & (yj >= 2)
            return float((yi[m] == yj[m]).mean())

        obs = agree(y)
        blocks = [np.flatnonzero(subj == s) for s in np.unique(subj)]
        draws = np.empty(n_perm)
        for r in range(n_perm):
            yp = y.copy()
            for b in blocks:
                yp[b] = rng.permutation(y[b])
            draws[r] = agree(yp)
        out[split] = dict(pairs_gap_up_to_two=int(len(pair)), a_gap_up_to_two=obs,
                          null_mean=float(draws.mean()),
                          p_uncorrected=float((draws >= obs).mean()),
                          p_corrected=float((int((draws >= obs).sum()) + 1) / (n_perm + 1)),
                          n_perm=n_perm)
    return out


def fmt_p(p):
    if p is None:
        return "n/a"
    floor = 1.0 / 1001
    return "<= .001" if p <= floor + 1e-12 else ("%.3f" % p).lstrip("0")


def f3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = "%.3f" % x
    return s.lstrip("0") if s.startswith("0.") else s


def write_report(res, path):
    lines = []
    W = lines.append
    W("# E1 - label-space harmonisation and matching permutation nulls")
    W("")
    W("Generated by `code/p4_e1_harmonise.py` on "
      + time.strftime("%Y-%m-%d %H:%M") + ". Permutations: "
      + str(res["settings"]["n_perm"]) + ", seed "
      + str(res["settings"]["seed"]) + ". A p-value of `<= .001` is the floor "
      "1/(B+1) and means no permutation reached the observed value.")
    W("")
    W("## 1. Both benchmarks in both label spaces")
    W("")
    W("Agreement between adjacent clips of one recording. `four_class` is exact "
      "match over the four ordinal levels; `decisive` is the top level against "
      "the one below it, among clips at the second level or above, which is the "
      "axis Table `tab:flicker` reports for DAiSEE.")
    W("")
    W("| Benchmark | Split | Space | Clips | Pairs | a | a_iid | kappa | Floor |")
    W("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for bench in ("DAiSEE", "EngageNet"):
        for split, row in res[bench].items():
            for space in ("four_class", "decisive"):
                g = row[space]["adjacent"]
                W("| %s | %s | %s | %d | %d | %s | %s | %s | %s |"
                  % (bench, split, space.replace("_", "-"), g["n_clips"], g["pairs"],
                     f3(g["a"]), f3(g["a_iid"]), f3(g["kappa"]), f3(g["floor"])))
    W("")
    W("## 2. Both permutation nulls on both benchmarks")
    W("")
    W("`within-recording` shuffles labels among the clips of one recording; "
      "`within-subject` shuffles among all clips of one participant. The second "
      "is the looser null, because it lets a recording's composition change.")
    W("")
    W("| Benchmark | Split | Space | a | Null within-recording | p | Null within-subject | p |")
    W("|---|---|---|---:|---:|---:|---:|---:|")
    for bench in ("DAiSEE", "EngageNet"):
        for split, row in res[bench].items():
            for space in ("four_class", "decisive"):
                b = row[space]
                nr, ns = b["null_within_recording"], b["null_within_subject"]
                W("| %s | %s | %s | %s | %s | %s | %s | %s |"
                  % (bench, split, space.replace("_", "-"), f3(b["adjacent"]["a"]),
                     f3(nr.get("null_mean")), fmt_p(nr.get("p")),
                     f3(ns.get("null_mean")), fmt_p(ns.get("p"))))
    W("")
    W("## 3. Distance-two and non-adjacent pairs")
    W("")
    W("| Benchmark | Split | Space | Adjacent | Distance two | Non-adjacent, same subject |")
    W("|---|---|---|---:|---:|---:|")
    for bench in ("DAiSEE", "EngageNet"):
        for split, row in res[bench].items():
            for space in ("four_class", "decisive"):
                b = row[space]
                W("| %s | %s | %s | %s (%d) | %s (%d) | %s (%d) |"
                  % (bench, split, space.replace("_", "-"),
                     f3(b["adjacent"]["a"]), b["adjacent"]["pairs"],
                     f3(b["distance_two"]["a"]), b["distance_two"]["pairs"],
                     f3(b["non_adjacent_same_subject"]["a"]),
                     b["non_adjacent_same_subject"]["pairs"]))
    W("")
    W("## 4. DAiSEE identifier hygiene")
    W("")
    W("The session and order fields are read as `ClipID[:7]` and "
      "`int(ClipID[7:])`, which assumes a ten-character identifier. The released "
      "label files do not keep that width.")
    W("")
    W("| Split | Labelled clips | Ten-character IDs | Other lengths | Clip files on disk | Labelled but absent from disk |")
    W("|---|---:|---:|---|---:|---:|")
    for split, h in res["daisee_id_hygiene"].items():
        others = ", ".join("%s:%d" % (k, v) for k, v in h["id_length_counts"].items()
                           if int(k) != 10) or "none"
        W("| %s | %d | %d | %s | %d | %d |"
          % (split, h["n_labelled"], h["n_wellformed_10char"], others,
             h["n_clip_files_on_disk"], h["n_labelled_missing_from_disk"]))
    W("")
    W("Restricting the statistic to the ten-character identifiers:")
    W("")
    W("| Split | Space | Clips | Pairs | a | a_iid | kappa |")
    W("|---|---|---:|---:|---:|---:|---:|")
    for split, row in res["daisee_wellformed_only"].items():
        for space in ("four_class", "decisive"):
            g = row[space]
            W("| %s | %s | %d | %d | %s | %s | %s |"
              % (split, space.replace("_", "-"), g["n_clips"], g["pairs"],
                 f3(g["a"]), f3(g["a_iid"]), f3(g["kappa"])))
    W("")
    W("## 5. Provenance of the p = .76 / .53 in the current manuscript")
    W("")
    W("Rerunning the archived `p2_revision_stats.py` null unchanged reproduces "
      "those two values. It builds its pair index with a gap of up to two, and "
      "the agreement it compares against is therefore not the strict-adjacency "
      "value printed in `tab:flicker`. The verdict is the same either way, but "
      "the prose and the table currently describe different pair sets.")
    W("")
    W("| Split | Pairs, gap up to two | a on that pair set | a, strict adjacency (table) | Null mean | p as archived | p with the +1 correction |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for split, h in res["daisee_archived_null"].items():
        strict = res["DAiSEE"][split]["decisive"]["adjacent"]["a"]
        W("| %s | %d | %s | %s | %s | %s | %s |"
          % (split, h["pairs_gap_up_to_two"], f3(h["a_gap_up_to_two"]), f3(strict),
             f3(h["null_mean"]), f3(h["p_uncorrected"]), f3(h["p_corrected"])))
    W("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--no-test", action="store_true",
                    help="drop the EngageNet test split if E0 returns that its "
                         "labels are not public")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    en_splits = [s for s in EN_SPLITS if not (args.no_test and s == "test")]

    print("DAiSEE:")
    daisee = one_benchmark("DAiSEE", DAISEE_SPLITS, prepare_daisee, True, args.perm)
    print("EngageNet:")
    engagenet = one_benchmark("EngageNet", en_splits, prepare_engagenet, False, args.perm)
    print("DAiSEE identifier hygiene:")
    hyg = daisee_id_hygiene()
    wf = wellformed_only(args.perm)
    print("archived DAiSEE null (provenance of p = .76 / .53):")
    arch = archived_daisee_null(n_perm=min(2000, max(200, args.perm)))

    res = {"settings": {"n_perm": args.perm, "seed": args.seed,
                        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "engagenet_test_included": "test" in en_splits},
           "DAiSEE": daisee, "EngageNet": engagenet,
           "daisee_id_hygiene": hyg, "daisee_wellformed_only": wf,
           "daisee_archived_null": arch}
    with open(os.path.join(OUT, "E1_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    write_report(res, os.path.join(OUT, "E1_report.md"))
    print("wrote", os.path.join(OUT, "E1_results.json"))
    print("wrote", os.path.join(OUT, "E1_report.md"))


if __name__ == "__main__":
    main()
