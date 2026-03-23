import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

GR_DIR  = "/home/mcho/SimpleMC_MJC/simplemc/chains/chains_20260309/GR_chains_20260309"
DFT_DIR = "/home/mcho/SimpleMC_MJC/simplemc/chains/chains_20260309/DFT_chains_20260309"
DFT_NEW = "/home/mcho/SimpleMC_MJC/simplemc/chains/DFT"
OUTPUT_DIR = "/home/mcho/SimpleMC_MJC/results"
N_DATA = 1797

MODELS = [
    {"group": "GR",  "name": "LCDM",      "dir": GR_DIR,  "prefix": "LCDM_phy_HD+Union3+FSC_nested_multi",        "k": 3, "params": ["Om","Obh2","h"]},
    {"group": "GR",  "name": "wCDM",      "dir": GR_DIR,  "prefix": "wCDM_phy_HD+Union3+FSC_nested_multi",        "k": 4, "params": ["Om","Obh2","h","w"]},
    {"group": "GR",  "name": "owa0CDM",   "dir": GR_DIR,  "prefix": "owa0CDM_phy_HD+Union3+FSC_nested_multi",     "k": 5, "params": ["Om","Obh2","h","w","wa"]},
    {"group": "DFT", "name": "DFTvac",   "dir": DFT_DIR, "prefix": "DFTvac_phy_HD+Union3+FSC_nested_multi",    "k": 3, "params": ["h","Ok","Oh"]},
    {"group": "DFT", "name": "DFT_w1l2", "dir": DFT_NEW, "prefix": "DFT_w1l2_phy_HD+Union3+FSC_nested_multi", "k": 4, "params": ["h","Ok","Oh","Oe"]},
]


def load_chain(model):
    path = os.path.join(model["dir"], model["prefix"] + "_1.txt")
    n_params = len(model["params"])
    weights, neg_loglikes = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2 + n_params:
                continue
            try:
                w  = float(parts[0])
                nl = float(parts[1])
            except ValueError:
                continue
            if not (np.isfinite(w) and np.isfinite(nl)):
                continue
            weights.append(w)
            neg_loglikes.append(nl)
    return np.array(weights), np.array(neg_loglikes)


def analyze(model):
    weights, neg_loglikes = load_chain(model)
    valid    = np.isfinite(neg_loglikes) & (neg_loglikes < 1e10)
    min_chi2 = float(2.0 * np.min(neg_loglikes[valid]))
    dof      = N_DATA - model["k"]
    return {
        "min_chi2": min_chi2,
        "red_chi2": min_chi2 / dof,
        "aic":      min_chi2 + 2 * model["k"],
        "bic":      min_chi2 + model["k"] * np.log(N_DATA),
    }


results = [analyze(m) for m in MODELS]

# ── print table ───────────────────────────────────────────────────────────────
print("\n===== Model Comparison (HD + Union3 + FSC) =====")
print(f"{'Group':<6} {'Model':<12} {'k':>3} {'chi^2':>12} {'red-chi^2':>11} {'AIC':>10} {'BIC':>10}")
print("-" * 70)
for m, r in zip(MODELS, results):
    print(f"{m['group']:<6} {m['name']:<12} {m['k']:>3} "
          f"{r['min_chi2']:>12.4f} {r['red_chi2']:>11.4f} "
          f"{r['aic']:>10.4f} {r['bic']:>10.4f}")

# ── summary figure (single table) ────────────────────────────────────────────
GR_COLOR  = "#dce6f1"
DFT_COLOR = "#fdebd0"

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis("off")

t_data = [
    [m["group"], m["name"], str(m["k"]),
     f"{r['min_chi2']:.4f}", f"{r['red_chi2']:.4f}",
     f"{r['aic']:.4f}", f"{r['bic']:.4f}"]
    for m, r in zip(MODELS, results)
]
t = ax.table(
    cellText=t_data,
    colLabels=["Group", "Model", "k", "Best Fit χ²", "Reduced χ²", "AIC", "BIC"],
    loc="center", cellLoc="center",
)
t.auto_set_font_size(False)
t.set_fontsize(11)
t.scale(1, 2.2)

for (row, col), cell in t.get_celld().items():
    if row == 0:
        cell.set_facecolor("#b0c4de")
        cell.set_text_props(fontweight="bold")
    else:
        grp = MODELS[row - 1]["group"]
        cell.set_facecolor(GR_COLOR if grp == "GR" else DFT_COLOR)
    cell.set_edgecolor("#aaaaaa")

aic_vals = [r["aic"] for r in results]
bic_vals = [r["bic"] for r in results]
best_aic = aic_vals.index(min(aic_vals)) + 1
best_bic = bic_vals.index(min(bic_vals)) + 1
for row in range(1, len(MODELS) + 1):
    if row == best_aic:
        t[(row, 5)].set_text_props(fontweight="bold", color="#1a5276")
    if row == best_bic:
        t[(row, 6)].set_text_props(fontweight="bold", color="#7b241c")

ax.set_title("Model Comparison  (HD + Union3 + FSC)",
             fontsize=13, pad=12, fontweight="bold")

legend_handles = [
    Patch(facecolor=GR_COLOR,  edgecolor="#aaaaaa", label="GR models"),
    Patch(facecolor=DFT_COLOR, edgecolor="#aaaaaa", label="DFT models"),
]
fig.legend(handles=legend_handles, loc="lower right",
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.98, 0.01))

plt.tight_layout(pad=1.5)
out = os.path.join(OUTPUT_DIR, "summary_short.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSummary figure saved → {out}")
