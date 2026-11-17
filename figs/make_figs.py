"""Figures 2, S1, S2 and S3 of the article (Okabe-Ito colorblind-safe palette).

Every value is one printed in the article's tables (sources noted inline); PDF (vector)
and PNG are written next to this script.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
OUT = os.path.dirname(os.path.abspath(__file__))

BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#8a8a8a"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    # TrueType (Type 42) instead of Type 3 so IEEE PDF eXpress accepts the PDFs
    "pdf.fonttype": 42, "ps.fonttype": 42,
    # Arial (installed) unifies the figure font with the draw.io Fig. 1;
    # mathtext drawn from the same face so kappa / lambda / subscripts match
    "font.family": "Arial",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})
# white box behind a label so gridlines / reference lines do not cross the glyphs
LBOX = dict(facecolor="white", edgecolor="none", pad=0.5)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("saved", name)


# ---------- Fig 2 in the paper: Audit 1 dot plot (Table 2) ----------
fig, ax = plt.subplots(figsize=(3.45, 1.95))
# intervals from the deposited script code/p7_audit1_daisee.py (25/09/2026)
rows = {"Validation": (0.580, 0.541, 0.644, (0.536, 0.628)), "Test": (0.527, 0.501, 0.520, (0.493, 0.561))}
ys = [1, 0]
for (name, (obs, iid, floor, ci)), y in zip(rows.items(), ys):
    ax.plot([iid, obs], [y, y], color="#c8c8c8", linewidth=2.2, zorder=1,
            solid_capstyle="round")
    ax.plot(iid, y, "o", mfc="white", mec=GRAY, mew=1.6, ms=9, zorder=3)
    ax.plot([ci[0], ci[1]], [y, y], color=BLUE, linewidth=1.1, zorder=2, alpha=0.55)
    ax.plot(obs, y, "o", mfc=BLUE, mec=BLUE, ms=9, zorder=3)
    ax.text(iid, y - 0.34, f"{iid:.3f}".lstrip("0"), ha="center", fontsize=8, color=GRAY,
            bbox=LBOX, zorder=4)
    ax.text(obs, y + 0.19, f"{obs:.3f}".lstrip("0"), ha="center", fontsize=8,
            color=BLUE, fontweight="bold", bbox=LBOX, zorder=4)
    if floor:
        ax.plot([floor, floor], [y - 0.2, y + 0.2], color=VERM, linewidth=2.4, zorder=2)
        if y > 0:
            ax.annotate("majority-class\nfloor " + f"{floor:.3f}".lstrip("0"),
                        xy=(floor, y), xytext=(floor + 0.006, y - 0.66), fontsize=8,
                        color=VERM, ha="left", va="center", bbox=LBOX, zorder=4)
        else:
            # test row: label to the right of the interval, joined to its marker
            ax.annotate("majority-class floor " + f"{floor:.3f}".lstrip("0"),
                        xy=(floor, y - 0.2), xytext=(0.568, y - 0.45), fontsize=8,
                        color=VERM, ha="left", va="center", bbox=LBOX, zorder=4,
                        arrowprops=dict(arrowstyle="-", color=VERM, lw=0.6, shrinkA=0, shrinkB=0))
ax.set_yticks(ys)
ax.set_yticklabels(list(rows))
ax.set_xlim(0.48, 0.70)
ax.set_ylim(-0.8, 1.6)
ax.set_xlabel("Adjacent-clip label agreement (2-vs-3 axis)")
ax.xaxis.grid(True, linewidth=0.4, color="#e4e4e4")
ax.set_axisbelow(True)
h_obs = plt.Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=8, label="observed $a$")
h_iid = plt.Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, mew=1.6, ms=8,
                   label="independent labeling $a_{\\mathrm{iid}}$")
h_ci = plt.Line2D([], [], color=BLUE, linewidth=1.1, alpha=0.55, label="95% CI (recording-clustered)")
h_ex = plt.Line2D([], [], color="#c8c8c8", linewidth=2.2, label="excess $a - a_{\\mathrm{iid}}$")
ax.legend(handles=[h_obs, h_iid, h_ci, h_ex], frameon=False, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2, fontsize=7.5)
save(fig, "figB_audit1")

# ---------- Fig 3: identity log bars (Table 3) ----------
mechs = ["Prototype conf. (local, $\\lambda$=2)",
         "Prototype conf. (global)",
         "Separation penalty $S_b/S_w$",
         "Adversary (GRL, 9 runs)"]
probe = [58.3, 67.9, 68.6, 67.4]
NOPEN = 66.2   # penalty-free baseline probe, Table 6 (69-subject pool)
fig, ax = plt.subplots(figsize=(3.6, 2.2))
yp = np.arange(4)
ax.barh(yp, [v - 1.0 for v in probe], 0.58, left=1.0, color=BLUE, edgecolor="none", label="retrained probe (from chance)", zorder=2)
ax.plot(2.4, 3, "D", mfc=VERM, mec="white", mew=0.8, ms=9, zorder=4,
        label="adversary head, held-out ($\\leq$2.4$\\times$)")
ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
ax.text(1.06, 3.5, "chance (1$\\times$)", fontsize=9, ha="left", va="center", color="#333333")
# penalty-free reference (Table 6): only the local prototype falls below it
ax.axvline(NOPEN, color=VERM, linewidth=1.0, linestyle="--", zorder=1)
ax.text(NOPEN / 1.05, 3.5, f"no penalty {NOPEN}$\\times$", fontsize=8, ha="right",
        va="center", color=VERM)
for y, v in zip(yp, probe):
    ax.text(max(v, NOPEN) * 1.09, y, f"{v}$\\times$", va="center", fontsize=9, color=BLUE,
            fontweight="bold", bbox=LBOX, zorder=5)
ax.set_xscale("log")
ax.set_xlim(0.9, 200)
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(["1", "10", "100"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(-0.5, 3.85)
ax.set_yticks(yp)
ax.set_yticklabels(mechs, fontsize=9)
ax.tick_params(axis="x", labelsize=9)
ax.set_xlabel("Multiple of chance (log scale)", fontsize=9)
ax.legend(frameon=False, ncol=1, loc="lower right", bbox_to_anchor=(1.0, 1.0),
          handletextpad=0.4, fontsize=8.5)
save(fig, "figC_identity")

# ---------- Fig 4: transfer slope (IV-E) - redesigned ----------
fig, ax = plt.subplots(figsize=(3.6, 2.5))
# AUC_dis: rises (blue)
ax.plot([0, 1], [0.71, 0.76], "-", color=BLUE, linewidth=2.6, zorder=2,
        solid_capstyle="round")
ax.plot([0, 1], [0.71, 0.76], "s", mfc=BLUE, mec="white", mew=1.0, ms=9, zorder=3)
ax.errorbar(1, 0.76, yerr=[[0.76-0.644], [0.839-0.76]], color=BLUE, capsize=4, elinewidth=1.4, capthick=1.4, zorder=2, alpha=0.8)
ax.errorbar(0, 0.71, yerr=[[0.01], [0.01]], color=BLUE, capsize=4,
            elinewidth=1.6, capthick=1.6, zorder=2)
# AUC_23: collapses (red-orange)
ax.plot([0, 1], [0.68, 0.60], "-", color=VERM, linewidth=2.6, zorder=2,
        solid_capstyle="round")
ax.plot([0, 1], [0.68, 0.60], "o", mfc=VERM, mec="white", mew=1.0, ms=9, zorder=3)
ax.errorbar(1, 0.60, yerr=[[0.60-0.548], [0.656-0.60]], color=VERM, capsize=4, elinewidth=1.4, capthick=1.4, zorder=2, alpha=0.8)
# validation whisker = seed range of the five-window runs (.679-.682, S4.2 / Table S9),
# the same variability definition as the AUC_dis validation whisker (.70-.72);
# test whiskers = subject-clustered 95% intervals of Table 7
ax.errorbar(0, 0.68, yerr=[[0.68-0.679], [0.682-0.68]], color=VERM, capsize=4, elinewidth=1.4, capthick=1.4, zorder=2, alpha=0.8)
# direct labels at the two endpoints
ax.text(-0.09, 0.71, "0.70–0.72", ha="right", va="center", fontsize=8.5, color=BLUE,
        bbox=LBOX, zorder=4)
ax.text(-0.09, 0.68, "0.68", ha="right", va="center", fontsize=8.5, color=VERM,
        bbox=LBOX, zorder=4)
ax.text(1.08, 0.76, "0.76  AUC$_{\\mathrm{dis}}$", ha="left", va="center",
        fontsize=9, color=BLUE, fontweight="bold", bbox=LBOX, zorder=4)
ax.text(1.08, 0.60, "0.60  AUC$_{23}$", ha="left", va="center",
        fontsize=9, color=VERM, fontweight="bold", bbox=LBOX, zorder=4)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Held-out\nvalidation", "Sealed test\n(one shot)"])
ax.set_xlim(-0.8, 1.9)
ax.set_ylim(0.52, 0.86)
ax.set_yticks([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
ax.set_ylabel("AUC")
ax.yaxis.grid(True, linewidth=0.4, color="#e4e4e4")
ax.set_axisbelow(True)
save(fig, "figE_transfer")

# ---------- Fig 5: stability bars (IV-F) ----------
fig, ax = plt.subplots(figsize=(3.45, 2.1))
ctrl_dis, pkg_dis = [-0.126, -0.117], [-0.006, -0.026, -0.026]
ctrl_23, pkg_23 = [-0.078, -0.077], [-0.005, -0.007, -0.015]


def put(x0, vals, color, w=0.16, hatch=None):
    xs = [x0 + i * (w + 0.03) for i in range(len(vals))]
    ax.bar(xs, vals, w, color=color, edgecolor="white", linewidth=0.0, hatch=hatch, zorder=2)
    return xs


x1 = put(0.00, ctrl_dis, VERM, hatch="///")
x2 = put(0.45, pkg_dis, BLUE)
x3 = put(1.25, ctrl_23, VERM, hatch="///")
x4 = put(1.70, pkg_23, BLUE)
ax.axhline(0, color="#333333", linewidth=0.9)
ax.set_xticks([np.mean(x1 + x2), np.mean(x3 + x4)])
ax.set_xticklabels(["AUC$_{\\mathrm{dis}}$", "AUC$_{23}$"])
ax.set_ylabel("Peak-to-final change in AUC")
ax.set_ylim(-0.148, 0.014)
ax.yaxis.grid(True, linewidth=0.4, color="#e4e4e4")
ax.set_axisbelow(True)
h_c = plt.Rectangle((0, 0), 1, 1, facecolor=VERM, edgecolor="white", hatch="///", label="matched control (per seed)")
h_p = plt.Rectangle((0, 0), 1, 1, color=BLUE, label="five-window package (per seed)")
# legend outside the axes (above), so no gridline runs through its text
ax.legend(handles=[h_c, h_p], frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, handletextpad=0.4,
          columnspacing=1.0)
save(fig, "figF_stability")

print("ALL DONE")
