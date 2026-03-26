#!/usr/bin/env python3
"""
GetDist corner / posterior plots for all models.
HD + Union3 + FSC dataset.

GR models  → multi-parameter triangle plot
DFT models → 1D h posterior (all 4 DFT models overlaid in one figure)

Output files (all in results/):
  corner_LCDM.pdf
  corner_wCDM.pdf
  corner_owa0CDM.pdf
  corner_DFT_h_posteriors.pdf
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import MCSamples, plots as gdplots

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.labelsize': 12,
    'legend.fontsize':10.5,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'axes.linewidth': 1.0,
    'xtick.direction':'in',
    'ytick.direction':'in',
})

COLORS_DFT = {
    "DFTvac":     "#D6604D",
    "DFTvac_noh": "#1A9850",
    "DFT1_w1l2":  "#E08214",
    "DFT_w1l2":   "#762A83",
}
LABELS_DFT = {
    "DFTvac":     r"DFT$_\mathrm{vac}$",
    "DFTvac_noh": r"DFT$_\mathrm{vac}$ (no $\mathfrak{h}$)",
    "DFT1_w1l2":  r"$\Lambda$DFT",
    "DFT_w1l2":   r"DFT ($w=1$, $\lambda=2$)",
}
LS_DFT = {
    "DFTvac":     "--",
    "DFTvac_noh": ":",
    "DFT1_w1l2":  "-.",
    "DFT_w1l2":   (0, (3, 1, 1, 1, 1, 1)),
}

base_gr   = os.path.join(_ROOT, "simplemc/chains/chains_20260309/GR_chains_20260309")
base_dft  = os.path.join(_ROOT, "simplemc/chains/chains_20260309/DFT_chains_20260309")
base_dft1 = os.path.join(_ROOT, "simplemc/chains/DFT")

# ── Helper: load & normalize weights ─────────────────────────────
def load_chain(path):
    d = np.loadtxt(path)
    log_w = np.log(np.where(d[:, 0] > 0, d[:, 0], 1e-300))
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    return d, w

# ── Helper: build MCSamples ───────────────────────────────────────
def make_samples(d, w, col_names, col_labels, label):
    cols = [d[:, i] for i in range(2, 2 + len(col_names))]
    samp = MCSamples(
        samples  = np.column_stack(cols),
        weights  = w,
        names    = col_names,
        labels   = col_labels,
        label    = label,
        settings = {"smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5,
                    "fine_bins_2D": 256, "fine_bins": 512},
    )
    return samp

# ── Helper: triangle plot ─────────────────────────────────────────
def make_triangle(samp, color, width=6.5):
    g = gdplots.get_subplot_plotter(width_inch=width)
    g.settings.axes_fontsize          = 11
    g.settings.lab_fontsize           = 12
    g.settings.legend_fontsize        = 11
    g.settings.alpha_filled_add       = 0.75
    g.settings.solid_contour_palefactor = 0.55
    g.settings.figure_legend_loc      = "upper right"
    g.triangle_plot(
        [samp], filled=True,
        contour_colors=[color],
        line_args=[{"lw": 1.5, "color": color}],
    )
    return g

# ══════════════════════════════════════════════════════════════════
# 1. LCDM  (Om, Obh2, h)
# ══════════════════════════════════════════════════════════════════
print("─── LCDM ───")
d, w = load_chain(os.path.join(base_gr, "LCDM_phy_HD+Union3+FSC_nested_multi_1.txt"))
samp = make_samples(d, w,
    col_names  = ["Om", "Obh2", "h"],
    col_labels = [r"\Omega_m", r"\Omega_b h^2", r"h"],
    label      = r"$\Lambda$CDM",
)
g = make_triangle(samp, "#2166AC")
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"corner_LCDM.{ext}")
    g.fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
plt.close('all')

# ══════════════════════════════════════════════════════════════════
# 2. wCDM  (Om, Obh2, h, w)
# ══════════════════════════════════════════════════════════════════
print("─── wCDM ───")
d, w = load_chain(os.path.join(base_gr, "wCDM_phy_HD+Union3+FSC_nested_multi_1.txt"))
samp = make_samples(d, w,
    col_names  = ["Om", "Obh2", "h", "w"],
    col_labels = [r"\Omega_m", r"\Omega_b h^2", r"h", r"w"],
    label      = r"$w$CDM",
)
g = make_triangle(samp, "#4DAC26", width=7)
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"corner_wCDM.{ext}")
    g.fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
plt.close('all')

# ══════════════════════════════════════════════════════════════════
# 3. owa0CDM  (Om, Obh2, h, w, wa, Ok)
# ══════════════════════════════════════════════════════════════════
print("─── owa0CDM ───")
d, w = load_chain(os.path.join(base_gr, "owa0CDM_phy_HD+Union3+FSC_nested_multi_1.txt"))
samp = make_samples(d, w,
    col_names  = ["Om", "Obh2", "h", "w", "wa", "Ok"],
    col_labels = [r"\Omega_m", r"\Omega_b h^2", r"h", r"w_0", r"w_a", r"\Omega_k"],
    label      = r"$o w_0 w_a$CDM",
)
g = make_triangle(samp, "#762A83", width=8)
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"corner_owa0CDM.{ext}")
    g.fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
plt.close('all')

# ══════════════════════════════════════════════════════════════════
# 4. DFT models: overlaid h posteriors (all 4)
# ══════════════════════════════════════════════════════════════════
print("─── DFT h posteriors ───")

DFT_CHAINS = {
    "DFTvac":     os.path.join(base_dft,  "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFTvac_noh": os.path.join(base_dft,  "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFT1_w1l2":  os.path.join(base_dft1, "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFT_w1l2":   os.path.join(base_dft1, "DFT_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
}

fig, ax = plt.subplots(figsize=(6.5, 4.2))

for key, chain_path in DFT_CHAINS.items():
    d, w = load_chain(chain_path)
    h_vals = d[:, 2]  # h is always col 2 for DFT chains

    samp = MCSamples(
        samples  = h_vals[:, None],
        weights  = w,
        names    = ["h"],
        labels   = [r"h"],
        settings = {"smooth_scale_1D": 0.45, "fine_bins": 512},
    )
    dens = samp.get1DDensity("h")
    x = np.linspace(h_vals.min(), h_vals.max(), 600)
    y = dens.Prob(x)
    y /= y.max()

    mean_h = np.average(h_vals, weights=w)
    ax.plot(x, y, color=COLORS_DFT[key], lw=2.2, ls=LS_DFT[key],
            label=LABELS_DFT[key])
    ax.axvline(mean_h, color=COLORS_DFT[key], lw=0.9, ls=LS_DFT[key], alpha=0.5)

# Add LCDM h for reference
d_lcdm, w_lcdm = load_chain(os.path.join(base_gr, "LCDM_phy_HD+Union3+FSC_nested_multi_1.txt"))
h_lcdm = d_lcdm[:, 4]
samp_l = MCSamples(
    samples=h_lcdm[:, None], weights=w_lcdm, names=["h"], labels=[r"h"],
    settings={"smooth_scale_1D": 0.45, "fine_bins": 512},
)
dens_l = samp_l.get1DDensity("h")
x_l    = np.linspace(h_lcdm.min(), h_lcdm.max(), 600)
y_l    = dens_l.Prob(x_l); y_l /= y_l.max()
ax.plot(x_l, y_l, color="#2166AC", lw=2.2, ls="-", label=r"$\Lambda$CDM (ref.)")
ax.axvline(np.average(h_lcdm, weights=w_lcdm), color="#2166AC", lw=0.9, alpha=0.5)

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"Normalized posterior $P(h)$")
ax.set_xlim(0.53, 0.77)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', framealpha=0.92, edgecolor='#cccccc', fontsize=10)
ax.grid(True, ls='--', alpha=0.3, lw=0.7)
ax.tick_params(direction='in', top=True, right=True)
fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"corner_DFT_h_posteriors.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
plt.close('all')

print("\nAll done.")
