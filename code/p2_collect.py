"""Collect every P2 run + probe into ONE single table (CSV + markdown for the paper).

Reads:
  ~/p2/runs/<tag>/log.jsonl   -> training curve (peak acc, peak acc-late, peak AUC, last ep)
  ~/p2/probe/<tag>_<ck>_probe.log -> identity-probe (88 classes), identity-probe-69 (if any),
                                     official-none/trans AUC for the 4 tasks
Writes: ~/p2/RESULTS.csv + prints the markdown table to stdout.

Run: ~/p2/venv/bin/python p2_collect.py
"""
import csv
import json
import os
import re
from pathlib import Path

RUNS = Path(os.path.expanduser("~/p2/runs"))
PROBE = Path(os.path.expanduser("~/p2/probe"))
TASKS = ["disengagement", "boredom", "confusion", "frustration"]


def curve(tag):
    f = RUNS / tag / "log.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.open() if l.strip()]
    if not rows:
        return None
    late = [r for r in rows if r["ep"] >= 5]
    best = max(rows, key=lambda r: r["acc"])
    bl = max(late, key=lambda r: r["acc"]) if late else None
    bu = max(rows, key=lambda r: r["auc_dis"])
    # auc23 exists only for stage-3 runs (v7/v8) - the deciding axis of the standard benchmark
    has23 = [r for r in rows if "auc23" in r]
    b23 = max(has23, key=lambda r: r["auc23"]) if has23 else None
    return dict(
        n_ep=len(rows),
        peak_acc=best["acc"], peak_acc_ep=best["ep"],
        late_acc=bl["acc"] if bl else None, late_acc_ep=bl["ep"] if bl else None,
        peak_auc=bu["auc_dis"], peak_auc_ep=bu["ep"],
        end_acc=rows[-1]["acc"], end_auc=rows[-1]["auc_dis"],
        peak_auc23=b23["auc23"] if b23 else None,
        peak_auc23_ep=b23["ep"] if b23 else None,
        end_auc23=rows[-1].get("auc23"),
        z_std_last=rows[-1].get("z_std"),
        sep_first=rows[0].get("sep"), sep_last=rows[-1].get("sep"),
        subj_last=rows[-1].get("loss_subj"),
    )


def probe(tag, ck):
    f = PROBE / f"{tag}_{ck}_probe.log"
    if not f.exists():
        return None
    txt = f.read_text(errors="replace")
    d = {}
    m = re.search(r"identity-probe\[none\s*\]: top-1 acc ([\d.]+)", txt)
    if m:
        d["id88"] = float(m.group(1))
    m = re.search(r"identity-probe\[trans\]: top-1 acc ([\d.]+)", txt)
    if m:
        d["id88_trans"] = float(m.group(1))
    m = re.search(r"identity-probe-69\[none\s*\]: top-1 ([\d.]+)", txt)
    if m:
        d["id69"] = float(m.group(1))
    m = re.search(r"prototype-69lop .*top-1 ([\d.]+)", txt)
    if m:
        d["proto69"] = float(m.group(1))
    m = re.search(r"ADVERSARY .*top-1 ([\d.]+)", txt)
    if m:
        d["adv_top1"] = float(m.group(1))
    m = re.search(r"so lop argmax phan biet (\d+)/", txt)
    if m:
        d["adv_nclass"] = int(m.group(1))
    for mode in ("none", "trans"):
        for t in TASKS:
            m = re.search(rf"official-{mode}\s*\|{t}\s*\] AUC ([\d.]+)", txt)
            if m:
                d[f"off_{mode}_{t[:3]}"] = float(m.group(1))
    return d or None


def main():
    tags = sorted(p.name for p in RUNS.iterdir() if p.is_dir())
    rows = []
    for tag in tags:
        c = curve(tag)
        if not c:
            continue
        for ck in ("best_acc", "best_acc_late", "last", "best"):
            p = probe(tag, ck)
            if p:
                rows.append(dict(run=tag, ckpt=ck, **c, **p))
        if not any(r["run"] == tag for r in rows):
            rows.append(dict(run=tag, ckpt="(chua probe)", **c))

    cols = ["run", "ckpt", "n_ep", "peak_acc", "peak_acc_ep", "late_acc", "late_acc_ep",
            "peak_auc", "peak_auc_ep", "end_acc", "end_auc", "sep_first", "sep_last",
            "subj_last", "peak_auc23", "peak_auc23_ep", "end_auc23", "z_std_last",
            "id88", "id88_trans", "id69", "proto69", "adv_top1", "adv_nclass"]
    cols += [f"off_{m}_{t[:3]}" for m in ("none", "trans") for t in TASKS]
    out = Path(os.path.expanduser("~/p2/RESULTS.csv"))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def s(v, f="%.3f"):
        return "—" if v is None else (f % v if isinstance(v, float) else str(v))

    # table numbers are all expressed as MULTIPLES OF CHANCE to compare across label spaces
    print("| run | ckpt | acc-late | đỉnh AUC_dis | **đỉnh AUC23** | id69 (×chance) "
          "| proto69 (×chance) | adv top-1 (×chance) | off-none dis/bor/conf/fru |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        off = "/".join(s(r.get(f"off_none_{t[:3]}")) for t in TASKS)

        def mult(key):
            v = r.get(key)
            return "—" if v is None else f"{v:.3f} ({v*69:.1f}×)"
        print(f"| {r['run']} | {r['ckpt']} | {s(r.get('late_acc'))} | {s(r.get('peak_auc'))} "
              f"| {s(r.get('peak_auc23'))} | {mult('id69')} | {mult('proto69')} "
              f"| {mult('adv_top1')} | {off} |")
    print(f"\n-> {out} ({len(rows)} dong)")


if __name__ == "__main__":
    main()
