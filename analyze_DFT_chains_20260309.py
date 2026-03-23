import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from getdist import plots, MCSamples

CHAINS_DIR = "/home/mcho/SimpleMC_MJC/simplemc/chains/chains_20260309/DFT_chains_20260309"
OUTPUT_DIR = CHAINS_DIR
N_DATA = 1797

MODELS = [
    {
        "name":    "DFTvac",
        "prefix":  "DFTvac_phy_HD+Union3+FSC_nested_multi",
        "k":       3,
        "params":  ["h", "Ok", "Oh"],
        "labels":  [r"h", r"\Omega_k", r"\Omega_{\mathfrak{h}}"],
    },
    {
        "name":    "DFTvac_noh",
        "prefix":  "DFTvac_noh_phy_HD+Union3+FSC_nested_multi",
        "k":       2,
        "params":  ["h", "Ok"],
        "labels":  [r"h", r"\Omega_k"],
    },
]

COLORS = ["#e84c4c", "#4c7be8"]


# ── load chain ────────────────────────────────────────────────────────────────
def load_chain(prefix, n_params):
    path = os.path.join(CHAINS_DIR, prefix + "_1.txt")
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
    weights, neg_loglikes, params = load_chain(model["prefix"], len(model["params"]))

    # col1 = -logL  →  chi2 = -2 * logL_best = 2 * min(col1)
    # DFT chains can have very large col1 for bad samples → filter
    valid    = np.isfinite(neg_loglikes) & (neg_loglikes < 1e10)
    min_chi2 = float(2.0 * np.min(neg_loglikes[valid]))
    dof      = N_DATA - model["k"]
    red_chi2 = min_chi2 / dof

    # Normalise weights (zero-weight bad samples are excluded naturally)
    w = weights.copy()
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    if w.sum() == 0:
        w = np.ones(len(w))
    w /= w.sum()

    means = np.average(params, axis=0, weights=w)
    stds  = np.sqrt(np.average((params - means) ** 2, axis=0, weights=w))

    aic = min_chi2 + 2 * model["k"]
    bic = min_chi2 + model["k"] * np.log(N_DATA)

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
    r = analyze(m)
    results.append(r)

# ── print Model Comparison Table ──────────────────────────────────────────────
print("\n1. Model Comparison Table")
print(f"{'Model':<12} {'N':>6} {'k':>5} {'Best Fit chi^2':>18} {'Reduced chi^2':>15} {'AIC':>10} {'BIC':>10}")
print("-" * 80)
for m, r in zip(MODELS, results):
    print(f"{m['name']:<12} {N_DATA:>6} {m['k']:>5} "
          f"{r['min_chi2']:>18.4f} {r['red_chi2']:>15.4f} "
          f"{r['aic']:>10.4f} {r['bic']:>10.4f}")

# ── print Parameter Means ─────────────────────────────────────────────────────
print("\n2. Parameter Means")
for m, r in zip(MODELS, results):
    print(f"\n--- {m['name']} ---")
    for i, p in enumerate(m["params"]):
        fmt = ".8f" if p.startswith("O") else ".4f"
        print(f"  {p:<6}: {r['means'][i]:+{fmt}} ± {r['stds'][i]:{fmt}}")


# ── build getdist MCSamples ───────────────────────────────────────────────────
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


# ── Corner plots: one per model ───────────────────────────────────────────────
for m, r, samp in zip(MODELS, results, gd_samples):
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

    out_corner = os.path.join(OUTPUT_DIR, f"DFT_corner_{m['name']}.png")
    g.fig.savefig(out_corner, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"\nCorner plot saved → {out_corner}")


# ── Summary table figure ──────────────────────────────────────────────────────
fig2, axes = plt.subplots(2, 1, figsize=(12, 7))

for ax in axes:
    ax.axis("off")

# Table 1 – Model Comparison
t1_data = [
    [m["name"], str(N_DATA), str(m["k"]),
     f"{r['min_chi2']:.4f}", f"{r['red_chi2']:.4f}",
     f"{r['aic']:.4f}", f"{r['bic']:.4f}"]
    for m, r in zip(MODELS, results)
]
t1 = axes[0].table(
    cellText=t1_data,
    colLabels=["Model", "N", "k",
               "Best Fit χ²", "Reduced χ²", "AIC", "BIC"],
    loc="center", cellLoc="center",
)
t1.auto_set_font_size(False)
t1.set_fontsize(11)
t1.scale(1, 2.5)
for (row, col), cell in t1.get_celld().items():
    cell.set_facecolor("white" if row > 0 else "#dce6f1")
    cell.set_edgecolor("#aaaaaa")
    if row == 0:
        cell.set_text_props(fontweight="bold")
    if row > 0 and col == 4:
        cell.set_text_props(fontweight="bold", color="#1a5276")
axes[0].set_title("1. Model Comparison Table", fontsize=13,
                  pad=10, fontweight="bold")

# Table 2 – Parameter Means
t2_data = []
for m, r in zip(MODELS, results):
    row_vals = [m["name"]]
    for p in ["h", "Ok", "Oh"]:
        if p in m["params"]:
            i = m["params"].index(p)
            fmt = ".8f" if p.startswith("O") else ".4f"
            row_vals.append(f"{r['means'][i]:{fmt}}\n±{r['stds'][i]:{fmt}}")
        else:
            row_vals.append("-")
    t2_data.append(row_vals)

t2 = axes[1].table(
    cellText=t2_data,
    colLabels=["Model", r"$h \pm \sigma$",
               r"$\Omega_k \pm \sigma$", r"$\Omega_\mathfrak{h} \pm \sigma$"],
    loc="center", cellLoc="center",
)
t2.auto_set_font_size(False)
t2.set_fontsize(11)
t2.scale(1, 3)
for (row, col), cell in t2.get_celld().items():
    cell.set_facecolor("white" if row > 0 else "#dce6f1")
    cell.set_edgecolor("#aaaaaa")
    if row == 0:
        cell.set_text_props(fontweight="bold")
axes[1].set_title("2. Parameter Means", fontsize=13,
                  pad=10, fontweight="bold")

plt.tight_layout(pad=2.0)
out_table = os.path.join(OUTPUT_DIR, "DFT_summary_table.png")
fig2.savefig(out_table, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Summary table saved → {out_table}")
