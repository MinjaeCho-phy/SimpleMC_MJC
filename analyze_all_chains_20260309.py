import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from getdist import plots, MCSamples

GR_DIR  = "/home/mcho/SimpleMC_MJC/simplemc/chains/chains_20260309/GR_chains_20260309"
DFT_DIR = "/home/mcho/SimpleMC_MJC/simplemc/chains/chains_20260309/DFT_chains_20260309"
OUTPUT_DIR = "/home/mcho/SimpleMC_MJC/simplemc/chains/chains_20260309"
N_DATA = 1797

MODELS = [
    # ── GR ──────────────────────────────────────────────────────────────────
    {
        "group":   "GR",
        "name":    "LCDM",
        "dir":     GR_DIR,
        "prefix":  "LCDM_phy_HD+Union3+FSC_nested_multi",
        "k":       3,
        "params":  ["Om", "Obh2", "h"],
        "labels":  [r"\Omega_m", r"\Omega_b h^2", r"h"],
    },
    {
        "group":   "GR",
        "name":    "wCDM",
        "dir":     GR_DIR,
        "prefix":  "wCDM_phy_HD+Union3+FSC_nested_multi",
        "k":       4,
        "params":  ["Om", "Obh2", "h", "w"],
        "labels":  [r"\Omega_m", r"\Omega_b h^2", r"h", r"w"],
    },
    {
        "group":   "GR",
        "name":    "owa0CDM",
        "dir":     GR_DIR,
        "prefix":  "owa0CDM_phy_HD+Union3+FSC_nested_multi",
        "k":       5,
        "params":  ["Om", "Obh2", "h", "w", "wa"],
        "labels":  [r"\Omega_m", r"\Omega_b h^2", r"h", r"w", r"w_a"],
    },
    # ── DFT ─────────────────────────────────────────────────────────────────
    {
        "group":   "DFT",
        "name":    "DFTvac",
        "dir":     DFT_DIR,
        "prefix":  "DFTvac_phy_HD+Union3+FSC_nested_multi",
        "k":       3,
        "params":  ["h", "Ok", "Oh"],
        "labels":  [r"h", r"\Omega_k", r"\Omega_{\mathfrak{h}}"],
    },
    {
        "group":   "DFT",
        "name":    "DFTvac_noh",
        "dir":     DFT_DIR,
        "prefix":  "DFTvac_noh_phy_HD+Union3+FSC_nested_multi",
        "k":       2,
        "params":  ["h", "Ok"],
        "labels":  [r"h", r"\Omega_k"],
    },
    {
        "group":   "DFT",
        "name":    "LDFT_w1l2",
        "dir":     "/home/mcho/SimpleMC_MJC/simplemc/chains/DFT",
        "prefix":  "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi",
        "k":       5,
        "params":  ["h", "Ok", "Oh", "OL", "Oe"],
        "labels":  [r"h", r"\Omega_k", r"\Omega_{\mathfrak{h}}",
                    r"\Omega_\Lambda", r"\Omega_\varepsilon"],
    },
    {
        "group":   "DFT",
        "name":    "DFT_w1l2",
        "dir":     "/home/mcho/SimpleMC_MJC/simplemc/chains/DFT",
        "prefix":  "DFT_w1l2_phy_HD+Union3+FSC_nested_multi",
        "k":       4,
        "params":  ["h", "Ok", "Oh", "Oe"],
        "labels":  [r"h", r"\Omega_k", r"\Omega_{\mathfrak{h}}",
                    r"\Omega_\varepsilon"],
    },
    {
        "group":   "DFT",
        "name":    "DFT_l0",
        "dir":     "/home/mcho/SimpleMC_MJC/simplemc/chains/DFT",
        "prefix":  "DFT_l0_phy_HD+Union3+FSC_nested_multi",
        "k":       5,
        "params":  ["h", "Ok", "Oh", "Oe", "w_dft"],
        "labels":  [r"h", r"\Omega_k", r"\Omega_{\mathfrak{h}}",
                    r"\Omega_\varepsilon", r"w_{\rm DFT}"],
    },
]

# Omega params → 8 decimal places, others → 4
def fmt(p, val):
    f = ".8f" if p.startswith("O") else ".4f"
    return f"{val:{f}}"


# ── load chain ────────────────────────────────────────────────────────────────
def load_chain(model):
    path = os.path.join(model["dir"], model["prefix"] + "_1.txt")
    n_params = len(model["params"])
    weights, neg_loglikes, params = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2 + n_params:
                continue
            try:
                w  = float(parts[0])
                nl = float(parts[1])
                ps = [float(parts[2 + i]) for i in range(n_params)]
            except ValueError:
                continue
            if not (np.isfinite(w) and np.isfinite(nl)):
                continue
            weights.append(w)
            neg_loglikes.append(nl)
            params.append(ps)
    return np.array(weights), np.array(neg_loglikes), np.array(params)


def analyze(model):
    weights, neg_loglikes, params = load_chain(model)

    valid    = np.isfinite(neg_loglikes) & (neg_loglikes < 1e10)
    min_chi2 = float(2.0 * np.min(neg_loglikes[valid]))
    dof      = N_DATA - model["k"]
    red_chi2 = min_chi2 / dof
    aic      = min_chi2 + 2 * model["k"]
    bic      = min_chi2 + model["k"] * np.log(N_DATA)

    w = weights.copy()
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    if w.sum() == 0:
        w = np.ones(len(w))
    w /= w.sum()

    means = np.average(params, axis=0, weights=w)
    stds  = np.sqrt(np.average((params - means) ** 2, axis=0, weights=w))

    return {
        "weights":      w,
        "params":       params,
        "neg_loglikes": neg_loglikes,
        "min_chi2":     min_chi2,
        "red_chi2":     red_chi2,
        "dof":          dof,
        "aic":          aic,
        "bic":          bic,
        "means":        means,
        "stds":         stds,
    }


# ── run analysis ──────────────────────────────────────────────────────────────
results = []
for m in MODELS:
    results.append(analyze(m))


# ── print combined table ──────────────────────────────────────────────────────
print("\n===== Model Comparison (HD + Union3 + FSC) =====")
print(f"{'Group':<6} {'Model':<12} {'k':>3} {'chi^2':>12} {'red-chi^2':>11} {'AIC':>10} {'BIC':>10}")
print("-" * 70)
for m, r in zip(MODELS, results):
    print(f"{m['group']:<6} {m['name']:<12} {m['k']:>3} "
          f"{r['min_chi2']:>12.4f} {r['red_chi2']:>11.4f} "
          f"{r['aic']:>10.4f} {r['bic']:>10.4f}")

print("\n===== Parameter Means =====")
ALL_PARAMS = ["Om", "h", "Ok", "Oh", "OL", "Oe", "Obh2", "w", "wa"]
header = f"{'Group':<6} {'Model':<12}"
for p in ALL_PARAMS:
    header += f"  {p:>22}"
print(header)
print("-" * (18 + 24 * len(ALL_PARAMS)))
for m, r in zip(MODELS, results):
    row = f"{m['group']:<6} {m['name']:<12}"
    for p in ALL_PARAMS:
        if p in m["params"]:
            i = m["params"].index(p)
            row += f"  {fmt(p, r['means'][i]):>22}"
        else:
            row += f"  {'-':>22}"
    print(row)


# ── Corner plots: one per model ──────────────────────────────────────────────
gd_samples = []
for m, r in zip(MODELS, results):
    samp = MCSamples(
        samples=r["params"],
        weights=r["weights"],
        loglikes=r["neg_loglikes"],
        names=m["params"],
        labels=m["labels"],
        label=m["name"],
    )
    samp.updateSettings({"fine_bins_2D": 40, "smooth_scale_2D": 0.4})
    gd_samples.append(samp)

for m, samp in zip(MODELS, gd_samples):
    g = plots.get_subplot_plotter(width_inch=6)
    g.settings.axes_fontsize   = 11
    g.settings.axes_labelsize  = 13
    g.settings.legend_fontsize = 12

    g.triangle_plot(
        [samp],
        params=m["params"],
        filled=True,
        contour_lws=[1.5],
    )

    g.fig.suptitle(
        f"{m['name']}  —  HD + Union3 + FSC",
        fontsize=13, y=1.01,
    )

    out_corner = os.path.join(m["dir"], f"corner_{m['name']}.png")
    g.fig.savefig(out_corner, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Corner plot saved → {out_corner}")


# ── Summary figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 13))
for ax in axes:
    ax.axis("off")

# Table 1 – Model Comparison
t1_data = [
    [m["group"], m["name"], str(m["k"]),
     f"{r['min_chi2']:.4f}", f"{r['red_chi2']:.4f}",
     f"{r['aic']:.4f}", f"{r['bic']:.4f}"]
    for m, r in zip(MODELS, results)
]
t1 = axes[0].table(
    cellText=t1_data,
    colLabels=["Group", "Model", "k", "Best Fit χ²", "Reduced χ²", "AIC", "BIC"],
    loc="center", cellLoc="center",
)
t1.auto_set_font_size(False)
t1.set_fontsize(11)
t1.scale(1, 2.0)

GR_COLOR  = "#dce6f1"
DFT_COLOR = "#fdebd0"
for (row, col), cell in t1.get_celld().items():
    if row == 0:
        cell.set_facecolor("#b0c4de")
        cell.set_text_props(fontweight="bold")
    else:
        grp = MODELS[row - 1]["group"]
        cell.set_facecolor(GR_COLOR if grp == "GR" else DFT_COLOR)
    cell.set_edgecolor("#aaaaaa")

# highlight best AIC & BIC
aic_vals = [r["aic"] for r in results]
bic_vals = [r["bic"] for r in results]
best_aic = aic_vals.index(min(aic_vals)) + 1
best_bic = bic_vals.index(min(bic_vals)) + 1
for row in range(1, len(MODELS) + 1):
    if row == best_aic:
        t1[(row, 5)].set_text_props(fontweight="bold", color="#1a5276")
    if row == best_bic:
        t1[(row, 6)].set_text_props(fontweight="bold", color="#7b241c")

axes[0].set_title("Model Comparison Table  (HD + Union3 + FSC)",
                  fontsize=13, pad=10, fontweight="bold")

# Table 2 – Parameter Means (h, Om, Ok, Oh, w, wa)
SHOW_PARAMS = ["h", "Om", "Ok", "Oh", "OL", "Oe", "w", "wa"]
SHOW_LABELS = [r"$h$", r"$\Omega_m$", r"$\Omega_k$",
               r"$\Omega_\mathfrak{h}$", r"$\Omega_\Lambda$",
               r"$\Omega_\varepsilon$", r"$w$", r"$w_a$"]

t2_data = []
for m, r in zip(MODELS, results):
    row_vals = [m["group"], m["name"]]
    for p in SHOW_PARAMS:
        if p in m["params"]:
            i = m["params"].index(p)
            row_vals.append(f"{fmt(p, r['means'][i])}\n±{fmt(p, r['stds'][i])}")
        else:
            row_vals.append("-")
    t2_data.append(row_vals)

t2 = axes[1].table(
    cellText=t2_data,
    colLabels=["Group", "Model"] + SHOW_LABELS,
    loc="center", cellLoc="center",
)
t2.auto_set_font_size(False)
t2.set_fontsize(9)
t2.scale(1, 3.2)

for (row, col), cell in t2.get_celld().items():
    if row == 0:
        cell.set_facecolor("#b0c4de")
        cell.set_text_props(fontweight="bold")
    else:
        grp = MODELS[row - 1]["group"]
        cell.set_facecolor(GR_COLOR if grp == "GR" else DFT_COLOR)
    cell.set_edgecolor("#aaaaaa")

axes[1].set_title("Parameter Means ± σ",
                  fontsize=13, pad=30, fontweight="bold")

# legend patch
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=GR_COLOR,  edgecolor="#aaaaaa", label="GR models"),
    Patch(facecolor=DFT_COLOR, edgecolor="#aaaaaa", label="DFT models"),
]
fig.legend(handles=legend_handles, loc="lower right",
           fontsize=11, framealpha=0.9, bbox_to_anchor=(0.98, 0.01))

plt.tight_layout(pad=2.0)
out = os.path.join(OUTPUT_DIR, "all_summary_20260309.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSummary figure saved → {out}")
