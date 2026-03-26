#!/usr/bin/env python3
"""
H(z) comparison: LCDM best-fit vs DFT models best-fit, with 1σ confidence bands.
Dataset: HD + Union3 + FSC  (chains_20260309)
"""

import sys, os
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)   # SimpleMC_MJC/
sys.path.insert(0, _ROOT)

import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.models.DFTVacuum import DFTVacuum
from simplemc.models.DFT1Cosmology import DFT1w1l2Cosmology

# ── Publication style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         12,
    'axes.labelsize':    13,
    'legend.fontsize':   11,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'axes.linewidth':    1.0,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.top':         True,
    'ytick.right':       True,
})

COLORS = {
    "LCDM":       "#2166AC",
    "DFTvac":     "#D6604D",
    "DFTvac_noh": "#1A9850",
    "DFT1_w1l2":  "#E08214",
}
LS     = {"LCDM": "-", "DFTvac": "--", "DFTvac_noh": ":", "DFT1_w1l2": "-."}
LABELS = {
    "LCDM":       r"$\Lambda$CDM",
    "DFTvac":     r"DFT$_\mathrm{vac}$",
    "DFTvac_noh": r"DFT$_\mathrm{vac}$ (no $\mathfrak{h}$)",
    "DFT1_w1l2":  r"$\Lambda$DFT",
}

# ── Best-fit parameters ────────────────────────────────────────────
BESTFITS = {
    "LCDM":       dict(Om=0.3515, Obh2=0.0220, h=0.6640),
    "DFTvac":     dict(h=0.6242, Ok=1.0, Oh=0.0),
    "DFTvac_noh": dict(h=0.6238, Ok=1.0),
    "DFT1_w1l2":  dict(h=0.6210, Ok=1.0, Oh=0.0, OL=0.0, Oe=0.0),
}

lcdm = LCDMCosmology(
    Om=BESTFITS["LCDM"]["Om"], Obh2=BESTFITS["LCDM"]["Obh2"], h=BESTFITS["LCDM"]["h"])
dftvac = DFTVacuum(
    h=BESTFITS["DFTvac"]["h"], Ok=BESTFITS["DFTvac"]["Ok"],
    Oh=BESTFITS["DFTvac"]["Oh"], ishzero=False)
dftvac_noh = DFTVacuum(
    h=BESTFITS["DFTvac_noh"]["h"], Ok=BESTFITS["DFTvac_noh"]["Ok"], ishzero=True)
dft1_w1l2 = DFT1w1l2Cosmology(
    h=BESTFITS["DFT1_w1l2"]["h"], Ok=BESTFITS["DFT1_w1l2"]["Ok"],
    Oh=BESTFITS["DFT1_w1l2"]["Oh"], OL=BESTFITS["DFT1_w1l2"]["OL"],
    Oe=BESTFITS["DFT1_w1l2"]["Oe"])

MODELS = {"LCDM": lcdm, "DFTvac": dftvac, "DFTvac_noh": dftvac_noh, "DFT1_w1l2": dft1_w1l2}

def Hz(z, model):
    if hasattr(model, 'hub'):
        return float(model.hub(z))
    H0 = model.h * 100.0
    return H0 * np.sqrt(max(model.RHSquared_a(1.0 / (1.0 + z)), 0.0))

z_arr = np.linspace(0.0, 2.5, 500)
H = {k: np.array([Hz(z, m) for z in z_arr]) for k, m in MODELS.items()}

# ── Confidence bands from posterior chains ─────────────────────────
base_gr   = os.path.join(_ROOT, "simplemc/chains/chains_20260309/GR_chains_20260309")
base_dft  = os.path.join(_ROOT, "simplemc/chains/chains_20260309/DFT_chains_20260309")
base_dft1 = os.path.join(_ROOT, "simplemc/chains/DFT")

CHAIN_PATHS = {
    "LCDM":       os.path.join(base_gr,   "LCDM_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFTvac":     os.path.join(base_dft,  "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFTvac_noh": os.path.join(base_dft,  "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFT1_w1l2":  os.path.join(base_dft1, "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
}

MODEL_FACTORIES = {
    "LCDM":       lambda row: LCDMCosmology(Om=row[2], Obh2=row[3], h=row[4]),
    "DFTvac":     lambda row: DFTVacuum(h=row[2], Ok=row[3], Oh=row[4], ishzero=False),
    "DFTvac_noh": lambda row: DFTVacuum(h=row[2], Ok=row[3], ishzero=True),
    "DFT1_w1l2":  lambda row: DFT1w1l2Cosmology(h=row[2], Ok=row[3], Oh=row[4],
                                                  OL=row[5], Oe=row[6]),
}

N_DRAW = 500

def _load_resample(path, n=N_DRAW, seed=42):
    d = np.loadtxt(path)
    log_w = np.log(np.where(d[:, 0] > 0, d[:, 0], 1e-300))
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    rng = np.random.default_rng(seed)
    return d[rng.choice(len(w), size=n, p=w, replace=True)]

def _sigma1_bands(curves):
    return np.quantile(curves, 0.1587, axis=0), np.quantile(curves, 0.8413, axis=0)

H_bands = {}
for key in MODELS:
    print(f"  H(z) bands [{key}] ...")
    samp = _load_resample(CHAIN_PATHS[key])
    curves = []
    for row in samp:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = MODEL_FACTORIES[key](row)
            curves.append(np.array([Hz(z, m) for z in z_arr]))
        except Exception:
            pass
    H_bands[key] = _sigma1_bands(np.array(curves))

# ── Load OHD data ──────────────────────────────────────────────────
data_dir = os.path.join(_ROOT, "simplemc/data")
hd = np.loadtxt(os.path.join(data_dir, "HDiagramCompilacion-data_31.txt"))
z_hd, H_hd, e_hd = hd[:, 0], hd[:, 1], hd[:, 2]

# ── Plot ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 7))
gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# 1σ bands
for k in MODELS:
    lo, hi = H_bands[k]
    ax1.fill_between(z_arr, lo, hi, color=COLORS[k], alpha=0.18, lw=0, zorder=3)

# Data
ax1.errorbar(z_hd, H_hd, yerr=e_hd,
             fmt='o', ms=5, color='#333333', elinewidth=1.2,
             capsize=3, capthick=1.2, zorder=10, alpha=0.85,
             label=r'OHD ($N=31$)')

# Best-fit curves
for k in MODELS:
    ax1.plot(z_arr, H[k], color=COLORS[k], ls=LS[k], lw=2.2,
             label=LABELS[k], zorder=5)

ax1.set_ylabel(r'$H(z)$  [km s$^{-1}$ Mpc$^{-1}$]')
ax1.set_ylim(45, 390)
ax1.set_xlim(0, 2.5)
ax1.grid(True, ls='--', alpha=0.3, lw=0.7)
plt.setp(ax1.get_xticklabels(), visible=False)

# Residuals: ΔH / H_LCDM × 100 (%)
H_ref = H["LCDM"]
ax2.axhline(0, color=COLORS["LCDM"], lw=1.5, ls='-', zorder=5)

# LCDM uncertainty band in residual panel
lo_lcdm, hi_lcdm = H_bands["LCDM"]
ax2.fill_between(z_arr,
                 (lo_lcdm - H_ref) / H_ref * 100,
                 (hi_lcdm - H_ref) / H_ref * 100,
                 color=COLORS["LCDM"], alpha=0.18, lw=0, zorder=3)

# Data residuals
H_lcdm_at_data = np.array([Hz(z, lcdm) for z in z_hd])
ax2.errorbar(z_hd, (H_hd - H_lcdm_at_data) / H_lcdm_at_data * 100,
             yerr=e_hd / H_lcdm_at_data * 100,
             fmt='o', ms=5, color='#333333', elinewidth=1.2,
             capsize=3, capthick=1.2, zorder=10, alpha=0.85)

for k in ["DFTvac", "DFTvac_noh", "DFT1_w1l2"]:
    lo, hi = H_bands[k]
    ax2.fill_between(z_arr,
                     (lo - H_ref) / H_ref * 100,
                     (hi - H_ref) / H_ref * 100,
                     color=COLORS[k], alpha=0.18, lw=0, zorder=3)
    ax2.plot(z_arr, (H[k] - H_ref) / H_ref * 100,
             color=COLORS[k], ls=LS[k], lw=2.0, zorder=5)

ax2.set_xlabel(r'Redshift $z$')
ax2.set_ylabel(r'$\Delta H/H_{\Lambda\mathrm{CDM}}$ [%]')
ax2.set_ylim(-22, 22)
ax2.grid(True, ls='--', alpha=0.3, lw=0.7)

# Legend below the figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',
           bbox_to_anchor=(0.5, -0.04), ncol=3,
           framealpha=0.95, edgecolor='#cccccc',
           fontsize=10.5, handlelength=2.2)
fig.subplots_adjust(bottom=0.14)

for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"Hz_comparison.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
