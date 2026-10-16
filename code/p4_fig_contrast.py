"""Figure: the two benchmarks under the same instrument, three panels.

  (a) Audit 1: Cohen's kappa of adjacent-clip label agreement, per split.
  (b) Audit 2 on models: within-person AUC on the decisive axis, per system.
  (c) Identity probe: multiple of chance recovered by a retrained probe.

Every value is one already printed in the manuscript (Tables tab:flicker,
tab:en-audit, tab:contrast, tab:identity, Section V-C and V-F); nothing is
recomputed here. Output: figs/figG_contrast.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    # TrueType (Type 42) instead of Type 3 so IEEE PDF eXpress accepts the PDF
    "pdf.fonttype": 42, "ps.fonttype": 42,
    # same face as make_figs.py and the draw.io Fig. 1
    "font.family": "Arial",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})
FS, FS_T = 8, 9   # tick / label / annotation size, panel-title size

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "figs", "figG_contrast.pdf")

C_D, C_E = "#555555", "#0072b2"   # DAiSEE dark grey, EngageNet blue (print-safe)

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))

# (a) kappa per split ------------------------------------------------------
# DAiSEE train / val / test (Table 2: .144 / .085 / .053) beside EngageNet
# train / val / test (Tables 3-4: .53 / .45 / .53), so both corpora show the
# same three splits.
a = ax[0]
# whiskers: recording-clustered 95% intervals (Table 2; Section V-B for EngageNet)
K_D, CI_D = [0.144, 0.085, 0.053], [(0.104, 0.183), (0.000, 0.160), (-0.023, 0.119)]
K_E, CI_E = [0.53, 0.45, 0.53], [(0.50, 0.57), (0.33, 0.54), (0.47, 0.59)]
EB = dict(ecolor="k", elinewidth=0.6, capsize=2, capthick=0.6)
a.bar([0, 1, 2], K_D, color=C_D, width=0.6,
      yerr=[[k - lo for k, (lo, hi) in zip(K_D, CI_D)], [hi - k for k, (lo, hi) in zip(K_D, CI_D)]], error_kw=EB)
a.bar([3.4, 4.4, 5.4], K_E, color=C_E, width=0.6,
      yerr=[[k - lo for k, (lo, hi) in zip(K_E, CI_E)], [hi - k for k, (lo, hi) in zip(K_E, CI_E)]], error_kw=EB)
a.axhline(0, color="k", lw=0.5)
a.set_xticks([0, 1, 2, 3.4, 4.4, 5.4])
a.set_xticklabels(["train", "val", "test", "train", "val", "test"], fontsize=FS)
a.text(1.0, 0.60, "DAiSEE\n(2 vs 3)", ha="center", fontsize=FS, color=C_D)
a.text(4.4, 0.60, "EngageNet\n(4 levels)", ha="center", fontsize=FS, color=C_E)
a.set_ylim(-0.04, 0.72)
a.set_ylabel("Cohen's $\\kappa$, adjacent clips", fontsize=FS)
a.set_title("(a) Audit 1: label consistency", fontsize=FS_T)

# (b) within-person AUC per system -----------------------------------------
b = ax[1]
# (name, value, test?): the sealed-test ensemble (.572) and the reference re-run at its
# test-selected peak (.543) are test-split readings and take the test marker
daisee = [("face, val", 0.607, False), ("face, test", 0.572, True), ("AU stream", 0.589, False),
          ("context", 0.553, False), ("ranking obj.", 0.558, False), ("ResNet-TCN", 0.543, True)]
# EngageNet values are the seed-mean within-person AUC_23 of the participant-
# clustered analysis (Supplementary Table tab:person, v5 onward) plus TCCT-Net;
# updated 25/09/2026 so that the panel matches Table 4 and Section V-C.
engage_val = [("MLP", 0.837), ("CNN-LSTM", 0.824), ("LSTM", 0.813), ("TCN", 0.810),
              ("Transf.", 0.780), ("MARLIN", 0.787), ("TCCT-Net", 0.800)]
engage_test = [("MLP", 0.822), ("CNN-LSTM", 0.812), ("LSTM", 0.783), ("TCN", 0.782),
               ("Transf.", 0.774), ("MARLIN", 0.779)]
xs = 0
for name, v, is_test in daisee:
    if is_test:
        b.plot(xs, v, "s", color=C_D, ms=4, mfc="white")
    else:
        b.plot(xs, v, "o", color=C_D, ms=4)
    xs += 1
xs += 1
for name, v in engage_val:
    b.plot(xs, v, "o", color=C_E, ms=4)
    xs += 1
xs += 1
for name, v in engage_test:
    b.plot(xs, v, "s", color=C_E, ms=4, mfc="white")
    xs += 1
b.axhline(0.5, color="k", lw=0.5, ls=":")
b.set_xticks([2.5, 9.5, 17.0])
b.set_xticklabels(["DAiSEE", "EngageNet\nval", "EngageNet\ntest"], fontsize=FS)
b.set_ylim(0.45, 0.9)
b.set_ylabel("within-person AUC (2 vs 3)", fontsize=FS)
b.set_title("(b) Audit 2 on models", fontsize=FS_T)
b.text(2.5, 0.63, ".54–.61", ha="center", fontsize=FS, color=C_D)
b.text(9.5, 0.86, ".78–.84", ha="center", fontsize=FS, color=C_E)
b.text(17.0, 0.86, ".77–.82", ha="center", fontsize=FS, color=C_E)
b.text(20.2, 0.505, "chance", ha="right", va="bottom", fontsize=FS - 1, color="k")
b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#333333", ms=4, label="validation"),
                  plt.Line2D([], [], marker="s", ls="", color="#333333", mfc="white", ms=4, label="test")],
         frameon=False, fontsize=FS - 1, loc="center right", handletextpad=0.2, borderaxespad=0.2)

# (c) identity probe -------------------------------------------------------
# One dot per system (mean over three seeds), the same encoding as panel (b):
# DAiSEE's single 512-d read-out, then EngageNet's five descriptor systems at
# the classifier input (128-d, results_p4/json/E6_probe.json) and at the pooled
# representation (results_p4/json/E67_checks.json). The annotated ranges are
# the ones printed in Section V-C.
import json
JS = os.path.join(HERE, "results")


def per_system_means(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r[key])
    return {k: sum(v) / len(v) for k, v in by.items()}


m128 = per_system_means(json.load(open(os.path.join(JS, "E6_probe.json"), encoding="utf-8")), "logistic_M")
m256 = per_system_means(json.load(open(os.path.join(JS, "E67_checks.json"), encoding="utf-8"))["probe_pooled_tap"], "logistic_M")
c = ax[2]
# DAiSEE: the penalty-free backbone read over the 88-subject pool (83.2x, Table 4);
# the 66x of an earlier version was the 69-subject reading of Table 6.
c.plot(0, 83.2, "o", color=C_D, ms=4)
for x, vals, mfc in ((1.7, m128, C_E), (3.4, m256, "white")):
    for v in vals.values():
        c.plot(x, v, "o" if mfc == C_E else "s", color=C_E, ms=4, mfc=mfc)
c.set_xticks([0, 1.7, 3.4])
c.set_xticklabels(["DAiSEE\n512-d, 88", "EngageNet\n128-d, 91", "EngageNet\n256-d, 91"], fontsize=FS)
c.set_ylabel("identity probe, $\\times$ chance (pool size)", fontsize=FS)
c.set_ylim(0, 95)
c.set_xlim(-0.7, 4.1)
c.set_title("(c) Identity recoverable", fontsize=FS_T)
c.text(0, 88, "83", ha="center", fontsize=FS, color=C_D)
c.text(1.7, 81, "52–76", ha="center", fontsize=FS, color=C_E)
c.text(3.4, 89, "63–83", ha="center", fontsize=FS, color=C_E)
c.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C_E, ms=4, label="classifier input"),
                  plt.Line2D([], [], marker="s", ls="", color=C_E, mfc="white", ms=4, label="pooled")],
         frameon=False, fontsize=FS - 1, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
print("panel (c) per-system means: 128-d", {k: round(v, 1) for k, v in m128.items()},
      "pooled", {k: round(v, 1) for k, v in m256.items()})

for x in ax:
    x.tick_params(axis="y", labelsize=FS)
    x.spines["top"].set_visible(False)
    x.spines["right"].set_visible(False)
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT)
fig.savefig(OUT[:-4] + ".png", dpi=300)
print("wrote", OUT)
