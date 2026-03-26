#!/usr/bin/env python3
"""
Corner / posterior plots for HD + Union3 + FSC chains.

Generates:
  corner_LCDM.pdf       — LCDM 3-parameter corner (Om, Obh2, h)
  corner_h_compare.pdf  — h posterior comparison across all 4 models
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

# ── Publication style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.labelsize': 12,
    'legend.fontsize':10,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'axes.linewidth': 1.0,
    'xtick.direction':'in',
    'ytick.direction':'in',
})

COLORS = {
    "LCDM":       "#2166AC",
    "DFTvac":     "#D6604D",
    "DFTvac_noh": "#1A9850",
    "DFT1_w1l2":  "#E08214",
}
LABELS = {
    "LCDM":       r"$\Lambda$CDM",
    "DFTvac":     r"DFT$_\mathrm{vac}$",
    "DFTvac_noh": r"DFT$_\mathrm{vac}$ (no $\mathfrak{h}$)",
    "DFT1_w1l2":  r"$\Lambda$DFT",
}

base_gr   = os.path.join(_ROOT, "simplemc/chains/chains_20260309/GR_chains_20260309")
base_dft  = os.path.join(_ROOT, "simplemc/chains/chains_20260309/DFT_chains_20260309")
base_dft1 = os.path.join(_ROOT, "simplemc/chains/DFT")

CHAIN_FILES = {
    "LCDM":       os.path.join(base_gr,   "LCDM_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFTvac":     os.path.join(base_dft,  "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFTvac_noh": os.path.join(base_dft,  "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFT1_w1l2":  os.path.join(base_dft1, "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
}

# Chain column layout (after weight, -2logL):
# LCDM:       Om, Obh2, h
# DFTvac:     h, Ok, Oh
# DFTvac_noh: h, Ok
# DFT1_w1l2:  h, Ok, Oh, OL, Oe
PARAM_COLS = {
    "LCDM":       {"Om": 2, "Obh2": 3, "h": 4},
    "DFTvac":     {"h": 2},
    "DFTvac_noh": {"h": 2},
    "DFT1_w1l2":  {"h": 2},
}
PARAM_LABELS = {
    "Om":   r"\Omega_m",
    "Obh2": r"\Omega_b h^2",
    "h":    r"h",
}

# ─── Figure 1: LCDM corner plot ────────────────────────────────────
print("Building LCDM corner plot ...")
d    = np.loadtxt(CHAIN_FILES["LCDM"])
# Normalize weights to avoid overflow (nested sampling weights can be huge)
log_wts = np.log(d[:, 0])
log_wts -= log_wts.max()
wts = np.exp(log_wts)
wts /= wts.sum()
samp = MCSamples(
    samples  = d[:, [2, 3, 4]],
    weights  = wts,
    names    = ["Om", "Obh2", "h"],
    labels   = [r"\Omega_m", r"\Omega_b h^2", r"h"],
    label    = r"$\Lambda$CDM",
    settings = {"smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5,
                "fine_bins_2D": 256, "fine_bins": 512},
)

g = gdplots.get_subplot_plotter(width_inch=6)
g.settings.axes_fontsize      = 11
g.settings.lab_fontsize       = 12
g.settings.legend_fontsize    = 11
g.settings.figure_legend_loc  = "upper right"
g.settings.alpha_filled_add   = 0.75
g.settings.solid_contour_palefactor = 0.5

g.triangle_plot(
    [samp],
    filled    = True,
    contour_colors = [COLORS["LCDM"]],
    line_args  = [{"lw": 1.5, "color": COLORS["LCDM"]}],
)

for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"corner_LCDM.{ext}")
    g.fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
plt.close('all')

# ─── Figure 2: h posterior comparison (all 4 models) ──────────────
print("Building h posterior comparison ...")

fig, ax = plt.subplots(figsize=(6.5, 4))

for key in ["LCDM", "DFTvac", "DFTvac_noh", "DFT1_w1l2"]:
    d       = np.loadtxt(CHAIN_FILES[key])
    log_wts = np.log(d[:, 0]); log_wts -= log_wts.max()
    wts     = np.exp(log_wts); wts /= wts.sum()
    col  = PARAM_COLS[key]["h"]
    h_vals = d[:, col]

    # KDE via getdist
    samp = MCSamples(
        samples  = h_vals[:, None],
        weights  = wts,
        names    = ["h"],
        labels   = [r"h"],
        settings = {"smooth_scale_1D": 0.4},
    )
    dens = samp.get1DDensity("h")
    x    = np.linspace(h_vals.min(), h_vals.max(), 500)
    y    = dens.Prob(x)
    y   /= y.max()  # normalize to peak=1 for visual comparison

    ls = "-" if key == "LCDM" else ("--" if key == "DFTvac" else
                                     (":" if key == "DFTvac_noh" else "-."))
    ax.plot(x, y, color=COLORS[key], lw=2.2, ls=ls, label=LABELS[key])
    ax.axvline(np.average(h_vals, weights=wts), color=COLORS[key],
               lw=0.8, ls=ls, alpha=0.5)

ax.set_xlabel(r'$h$')
ax.set_ylabel(r'Normalized posterior $P(h)$')
ax.set_xlim(0.55, 0.75)
ax.set_ylim(0, 1.12)
ax.legend(loc='upper right', framealpha=0.92, edgecolor='#cccccc')
ax.grid(True, ls='--', alpha=0.3, lw=0.7)
ax.tick_params(direction='in', top=True, right=True)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"corner_h_compare.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
