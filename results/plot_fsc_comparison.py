#!/usr/bin/env python3
"""
Fine structure constant variation Δα/α comparison, with 1σ confidence bands.
Dataset: HD + Union3 + FSC  (chains_20260309 / chains/DFT)

Note: LCDM has alpha_fsc fixed (fixfsc=True), so no band is drawn for LCDM.
      The DFT φ-field couples to α, giving a redshift-dependent Δα/α.
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
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

# ── Best-fit parameters ───────────────────────────────────────────
# LCDM: hardcoded posterior means (h, Om, Obh2 only — FSC is fixed)
# DFT models: computed from chain weighted means, because FSC is extremely
# sensitive to Ok (even 1e-5 deviation from 1.0 causes visible FSC signal).
# Using Ok=1.0 exactly gives FSC=0 for all DFT models, which is inconsistent
# with the bands derived from the actual posterior samples.

def _chain_weighted_mean(path, cols):
    d = np.loadtxt(path)
    log_w = np.log(np.where(d[:, 0] > 0, d[:, 0], 1e-300))
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    return {name: float(np.average(d[:, 2 + i], weights=w))
            for i, name in enumerate(cols)}

base_gr   = os.path.join(_ROOT, "simplemc/chains/chains_20260309/GR_chains_20260309")
base_dft  = os.path.join(_ROOT, "simplemc/chains/chains_20260309/DFT_chains_20260309")
base_dft1 = os.path.join(_ROOT, "simplemc/chains/DFT")

bf_lcdm     = dict(Om=0.3515, Obh2=0.0220, h=0.6640)
bf_dftvac    = _chain_weighted_mean(
    os.path.join(base_dft,  "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok", "Oh"])
bf_dftvac_noh = _chain_weighted_mean(
    os.path.join(base_dft,  "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok"])
bf_dft1_w1l2 = _chain_weighted_mean(
    os.path.join(base_dft1, "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok", "Oh", "OL", "Oe"])

lcdm = LCDMCosmology(Om=bf_lcdm["Om"], Obh2=bf_lcdm["Obh2"], h=bf_lcdm["h"])
dftvac = DFTVacuum(h=bf_dftvac["h"], Ok=bf_dftvac["Ok"],
                   Oh=bf_dftvac["Oh"], ishzero=False)
dftvac_noh = DFTVacuum(h=bf_dftvac_noh["h"], Ok=bf_dftvac_noh["Ok"], ishzero=True)
dft1_w1l2 = DFT1w1l2Cosmology(h=bf_dft1_w1l2["h"], Ok=bf_dft1_w1l2["Ok"],
                                Oh=bf_dft1_w1l2["Oh"], OL=bf_dft1_w1l2["OL"],
                                Oe=bf_dft1_w1l2["Oe"])

MODELS = {"LCDM": lcdm, "DFTvac": dftvac, "DFTvac_noh": dftvac_noh, "DFT1_w1l2": dft1_w1l2}

# ── Theory curves ──────────────────────────────────────────────────
z_arr = np.linspace(0.1, 7.5, 500)

def fsc_theory(z_arr, model):
    return np.array([1e5 * model.fine_structure_constant(1.0 / (1.0 + z)) for z in z_arr])

print("Computing FSC theory curves ...")
fsc = {k: fsc_theory(z_arr, m) for k, m in MODELS.items()}

# ── Confidence bands from posterior chains ─────────────────────────
# LCDM has fixfsc=True so alpha_fsc is not sampled → no band for LCDM.
DFT_CHAIN_PATHS = {
    "DFTvac":     os.path.join(base_dft,  "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFTvac_noh": os.path.join(base_dft,  "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFT1_w1l2":  os.path.join(base_dft1, "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
}

DFT_FACTORIES = {
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

fsc_bands = {}
for key, factory in DFT_FACTORIES.items():
    print(f"  FSC bands [{key}] ...")
    samp = _load_resample(DFT_CHAIN_PATHS[key])
    curves = []
    for row in samp:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = factory(row)
            curves.append(fsc_theory(z_arr, m))
        except Exception:
            pass
    fsc_bands[key] = _sigma1_bands(np.array(curves))

# ── Load data ──────────────────────────────────────────────────────
data_dir = os.path.join(_ROOT, "simplemc/data")
raw = np.loadtxt(os.path.join(data_dir, "fine_structure.dat"), usecols=(0, 1, 2))
z_d  = raw[:, 0]
da_d = raw[:, 1]
e_d  = raw[:, 2]
print(f"FSC data: N={len(z_d)}, z=[{z_d.min():.2f}, {z_d.max():.2f}]")

# ── Plot ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 5))
ax  = fig.add_subplot(111)

# 1σ bands for DFT models (behind data and curves)
for k in DFT_FACTORIES:
    lo, hi = fsc_bands[k]
    ax.fill_between(z_arr, lo, hi, color=COLORS[k], alpha=0.18, lw=0, zorder=2)

# Data
ax.errorbar(z_d, da_d, yerr=e_d,
            fmt='o', ms=4, color='#333333', elinewidth=0.9,
            capsize=2.5, capthick=0.9, alpha=0.75, zorder=5,
            label=r'$\Delta\alpha/\alpha$ data ($N=199$)')

# Reference line
ax.axhline(0, color='#888888', lw=0.8, ls='--', zorder=2)

# Model curves
for k in MODELS:
    ax.plot(z_arr, fsc[k], color=COLORS[k], ls=LS[k], lw=2.2,
            label=LABELS[k], zorder=6)

ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'$\Delta\alpha/\alpha \times 10^{5}$')
ax.set_xlim(0, 7.5)
ax.set_ylim(-38, 38)

ax.legend(loc='upper left', framealpha=0.92, edgecolor='#cccccc')
ax.grid(True, ls='--', alpha=0.3, lw=0.7)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"fsc_comparison.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
