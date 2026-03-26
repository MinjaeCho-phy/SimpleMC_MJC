#!/usr/bin/env python3
"""
Distance modulus mu(z) comparison: LCDM best-fit vs DFT models best-fit,
with 1σ confidence bands.
Dataset: HD + Union3 + FSC  (chains_20260309 / chains/DFT)

Best-fit values (posterior mean from Summary files):
  LCDM        : Om=0.3515, Obh2=0.0220, h=0.6640
  DFTvac      : h=0.6242, Ok=1.0, Oh=0.0
  DFTvac_noh  : h=0.6238, Ok=1.0  (Oh=0 fixed)
  DFT1_w1l2   : h=0.6210, Ok=1.0, Oh=0.0, OL=0.0, Oe=0.0  (w=1, lambda=2)

Background scatter:
  union3_individual_sn.txt : 2087 individual Union3 SNe Ia
    mu_apparent = mB + alpha*x1 - beta*c  (alpha=0.15, beta=3.1, SALT3 standard)
    extracted from rubind/union3_release (github.com/rubind/union3_release)

Fit data (used in chains):
  Union3 (lcparam_full.txt / mag_covmat.txt):
    22 UNITY1.5 spline nodes (Rubin et al. 2023, arXiv:2311.12098)
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np
import warnings
from scipy.integrate import quad, cumulative_trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.models.DFTVacuum import DFTVacuum
from simplemc.models.DFT1Cosmology import DFT1w1l2Cosmology

c_kms = 299792.458  # km/s

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

# ── Helpers ────────────────────────────────────────────────────────
def Hz(z, model):
    if hasattr(model, 'hub'):
        return float(model.hub(z))
    H0 = model.h * 100.0
    return H0 * np.sqrt(max(model.RHSquared_a(1.0 / (1.0 + z)), 0.0))

def dL_Mpc(z, model):
    if z < 1e-8:
        return 0.0
    dc, _ = quad(lambda zp: c_kms / Hz(zp, model), 0.0, z, limit=100)
    return (1.0 + z) * dc

def mu_th_arr(z_arr, model):
    return np.array([5.0 * np.log10(max(dL_Mpc(z, model), 1e-30) * 1e5)
                     for z in z_arr])

def mu_th_arr_fast(z_arr, model):
    """Fast version using cumulative_trapezoid — used for band computation."""
    z_max = float(np.max(z_arr)) * 1.001
    z_fine = np.linspace(1e-6, z_max, 800)
    H_fine = np.array([Hz(z, model) for z in z_fine])
    H_fine[0] = H_fine[1]
    dc_fine = cumulative_trapezoid(c_kms / H_fine, z_fine, initial=0)
    dc_arr = np.interp(np.asarray(z_arr), z_fine, dc_fine)
    dL_arr = (1.0 + np.asarray(z_arr)) * dc_arr
    return 5.0 * np.log10(np.maximum(dL_arr, 1e-30) * 1e5)

def marginalize_M(mu_data, mu_err, z_data, ref_model):
    mu_ref = np.array([5.0 * np.log10(max(dL_Mpc(z, ref_model), 1e-30) * 1e5)
                       for z in z_data])
    w = 1.0 / mu_err**2
    M = ((mu_data - mu_ref) * w).sum() / w.sum()
    return mu_data - M, M, mu_ref

# ── Theory curves ──────────────────────────────────────────────────
z_arr = np.linspace(0.01, 2.3, 300)
print("Computing mu(z) ...")
mu = {k: mu_th_arr(z_arr, m) for k, m in MODELS.items()}

# ── Load data ──────────────────────────────────────────────────────
data_dir = os.path.join(_ROOT, "simplemc/data")

# Union3 22 UNITY1.5 spline nodes (used in fitting)
da      = np.loadtxt(os.path.join(data_dir, "lcparam_full.txt"), skiprows=1, usecols=(1, 2, 4))
z_sn    = da[:, 0]
mb_sn   = da[:, 2]
N_sn    = len(mb_sn)
syscov  = np.loadtxt(os.path.join(data_dir, "mag_covmat.txt"), skiprows=1).reshape((N_sn, N_sn))
mb_err  = np.sqrt(np.diag(syscov))
xdiag   = 1.0 / np.diag(syscov)
mu_lcdm_sn = np.array([5.0 * np.log10(max(dL_Mpc(z, lcdm), 1e-30) * 1e5) for z in z_sn])
M_sn    = ((mb_sn - mu_lcdm_sn) * xdiag).sum() / xdiag.sum()
mb_shifted = mb_sn - M_sn
print(f"Union3 nodes: N={N_sn}, M={M_sn:.4f}")

# Union3 2087 individual SNe (background scatter)
u3      = np.loadtxt(os.path.join(data_dir, "union3_individual_sn.txt"))
z_u3    = u3[:, 0]
mu_u3   = u3[:, 1]
dmu_u3  = u3[:, 2]
mu_u3_shifted, M_u3, mu_lcdm_u3 = marginalize_M(mu_u3, dmu_u3, z_u3, lcdm)
print(f"Union3 individual: N={len(z_u3)}, M={M_u3:.4f}")

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

mu_bands = {}
for key in MODELS:
    print(f"  mu(z) bands [{key}] ...")
    samp = _load_resample(CHAIN_PATHS[key])
    curves = []
    for row in samp:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = MODEL_FACTORIES[key](row)
            curves.append(mu_th_arr_fast(z_arr, m))
        except Exception:
            pass
    mu_bands[key] = _sigma1_bands(np.array(curves))

# ── Plot ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 7))
gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# 1σ bands (behind everything)
for k in MODELS:
    lo, hi = mu_bands[k]
    ax1.fill_between(z_arr, lo, hi, color=COLORS[k], alpha=0.18, lw=0, zorder=2)

# Background scatter: Union3 individual SNe
ax1.errorbar(z_u3, mu_u3_shifted, yerr=dmu_u3,
             fmt='.', ms=3, color='#aaaaaa', elinewidth=0.35,
             alpha=0.65, zorder=1, rasterized=True,
             label=r'Union3 SNe Ia ($N=2087$)')

# Best-fit model curves
for k in MODELS:
    ax1.plot(z_arr, mu[k], color=COLORS[k], ls=LS[k], lw=2.2,
             label=LABELS[k], zorder=5)

# Union3 22 spline nodes on top
ax1.errorbar(z_sn, mb_shifted, yerr=mb_err,
             fmt='D', ms=5.5, color='#111111', elinewidth=1.3,
             capsize=3.5, capthick=1.3, zorder=10,
             label=r'Union3 UNITY1.5 nodes ($N=22$)')

ax1.set_ylabel(r'$\mu(z)$  [mag]')
ax1.set_xlim(0, 2.3)
ax1.grid(True, ls='--', alpha=0.3, lw=0.7)
plt.setp(ax1.get_xticklabels(), visible=False)

# Residuals panel: Δμ = μ_model - μ_LCDM
mu_lcdm_arr = mu["LCDM"]
mu_lcdm_at_u3 = np.array([5.0 * np.log10(max(dL_Mpc(z, lcdm), 1e-30) * 1e5) for z in z_u3])

ax2.axhline(0, color=COLORS["LCDM"], lw=1.5, zorder=5)

# LCDM band in residual panel
lo_lcdm, hi_lcdm = mu_bands["LCDM"]
ax2.fill_between(z_arr, lo_lcdm - mu_lcdm_arr, hi_lcdm - mu_lcdm_arr,
                 color=COLORS["LCDM"], alpha=0.18, lw=0, zorder=3)

ax2.errorbar(z_u3, mu_u3_shifted - mu_lcdm_at_u3, yerr=dmu_u3,
             fmt='.', ms=3, color='#aaaaaa', elinewidth=0.35,
             alpha=0.65, zorder=1, rasterized=True)
ax2.errorbar(z_sn, mb_shifted - mu_lcdm_sn, yerr=mb_err,
             fmt='D', ms=5.5, color='#111111', elinewidth=1.3,
             capsize=3.5, capthick=1.3, zorder=10)

for k in ["DFTvac", "DFTvac_noh", "DFT1_w1l2"]:
    lo, hi = mu_bands[k]
    ax2.fill_between(z_arr, lo - mu_lcdm_arr, hi - mu_lcdm_arr,
                     color=COLORS[k], alpha=0.18, lw=0, zorder=3)
    ax2.plot(z_arr, mu[k] - mu_lcdm_arr,
             color=COLORS[k], ls=LS[k], lw=2.0, zorder=5)

ax2.set_xlabel(r'Redshift $z$')
ax2.set_ylabel(r'$\Delta\mu$  [mag]')
ax2.set_ylim(-1.2, 1.2)
ax2.grid(True, ls='--', alpha=0.3, lw=0.7)

# Legend below the figure (2 rows × 3 cols)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',
           bbox_to_anchor=(0.5, -0.04), ncol=3,
           framealpha=0.95, edgecolor='#cccccc',
           fontsize=10.5, handlelength=2.2)
fig.subplots_adjust(bottom=0.14)

for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"mu_comparison.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
