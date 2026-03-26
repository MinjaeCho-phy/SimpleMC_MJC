#!/usr/bin/env python3
"""
Combined 3-panel plot: HD + Union3 + FSC data with LCDM and DFT model predictions.
Best-fit parameters from HD+Union3+FSC chains (chains_20260309 / chains/DFT).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.models.DFTVacuum import DFTVacuum
from simplemc.models.DFT1Cosmology import DFT1w1l2Cosmology

c_kms = 299792.458  # km/s

# ─────────────────────────────────────────────────────────────────
# Best-fit parameters (posterior mean from Summary files)
# ─────────────────────────────────────────────────────────────────
BESTFITS = {
    "LCDM":       dict(Om=0.3515, Obh2=0.0220, h=0.6640),
    "DFTvac":     dict(h=0.6242, Ok=1.0, Oh=0.0),
    "DFTvac_noh": dict(h=0.6238, Ok=1.0),
    "DFT1_w1l2":  dict(h=0.6210, Ok=1.0, Oh=0.0, OL=0.0, Oe=0.0),
}

COLORS  = {"LCDM": "royalblue", "DFTvac": "tomato",
           "DFTvac_noh": "seagreen", "DFT1_w1l2": "darkorange"}
LS      = {"LCDM": "-", "DFTvac": "--", "DFTvac_noh": ":", "DFT1_w1l2": "-."}
LABELS  = {
    "LCDM":       r"$\Lambda$CDM  ($h=0.664$, $\Omega_m=0.352$)",
    "DFTvac":     r"DFTvac  ($h=0.624$, $\Omega_k=1.0$, $\Omega_\mathfrak{h}=0$)",
    "DFTvac_noh": r"DFTvac (no $\mathfrak{h}$)  ($h=0.624$, $\Omega_k=1.0$)",
    "DFT1_w1l2":  r"DFT1 ($w=1$, $\lambda=2$)  ($h=0.621$, $\Omega_k=1.0$)",
}

# ─────────────────────────────────────────────────────────────────
# Build model objects
# ─────────────────────────────────────────────────────────────────
lcdm = LCDMCosmology(
    Om=BESTFITS["LCDM"]["Om"], Obh2=BESTFITS["LCDM"]["Obh2"], h=BESTFITS["LCDM"]["h"],
)
dftvac = DFTVacuum(
    h=BESTFITS["DFTvac"]["h"], Ok=BESTFITS["DFTvac"]["Ok"],
    Oh=BESTFITS["DFTvac"]["Oh"], ishzero=False,
)
dftvac_noh = DFTVacuum(
    h=BESTFITS["DFTvac_noh"]["h"], Ok=BESTFITS["DFTvac_noh"]["Ok"], ishzero=True,
)
dft1_w1l2 = DFT1w1l2Cosmology(
    h=BESTFITS["DFT1_w1l2"]["h"], Ok=BESTFITS["DFT1_w1l2"]["Ok"],
    Oh=BESTFITS["DFT1_w1l2"]["Oh"], OL=BESTFITS["DFT1_w1l2"]["OL"],
    Oe=BESTFITS["DFT1_w1l2"]["Oe"],
)

MODELS = {"LCDM": lcdm, "DFTvac": dftvac,
          "DFTvac_noh": dftvac_noh, "DFT1_w1l2": dft1_w1l2}

# ─────────────────────────────────────────────────────────────────
# Theory helpers
# ─────────────────────────────────────────────────────────────────
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

def mu_z(z, model):
    return 5.0 * np.log10(max(dL_Mpc(z, model), 1e-30) * 1e5)

def fsc_z(z, model):
    """Δα/α × 10^5"""
    a = 1.0 / (1.0 + z)
    return 1e5 * model.fine_structure_constant(a)

# ─────────────────────────────────────────────────────────────────
# Compute theory curves
# ─────────────────────────────────────────────────────────────────
z_Hz  = np.linspace(0.0,  2.5, 300)
z_mu  = np.linspace(0.01, 2.3, 300)
z_fsc = np.linspace(0.1,  7.5, 300)

theory = {k: {} for k in MODELS}

print("Computing H(z) ...")
for k, m in MODELS.items():
    theory[k]["Hz"] = np.array([Hz(z, m) for z in z_Hz])

print("Computing mu(z) ...")
for k, m in MODELS.items():
    theory[k]["mu"] = np.array([mu_z(z, m) for z in z_mu])

print("Computing FSC ...")
for k, m in MODELS.items():
    theory[k]["fsc"] = np.array([fsc_z(z, m) for z in z_fsc])

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), "simplemc/data")

# --- HD ---
hd = np.loadtxt(os.path.join(data_dir, "HDiagramCompilacion-data_31.txt"))
z_hd, H_hd, e_hd = hd[:, 0], hd[:, 1], hd[:, 2]

# --- Union3 (22 bins) ---
da      = np.loadtxt(os.path.join(data_dir, "lcparam_full.txt"),
                     skiprows=1, usecols=(1, 2, 4))
z_sn    = da[:, 0]
mb_sn   = da[:, 2]
N_sn    = len(mb_sn)
syscov  = np.loadtxt(os.path.join(data_dir, "mag_covmat.txt"),
                     skiprows=1).reshape((N_sn, N_sn))
mb_err  = np.sqrt(np.diag(syscov))
# Marginalize M with LCDM as reference
mu_lcdm_sn = np.array([mu_z(z, lcdm) for z in z_sn])
xdiag      = 1.0 / np.diag(syscov)
M_ref      = ((mb_sn - mu_lcdm_sn) * xdiag).sum() / xdiag.sum()
mb_shifted = mb_sn - M_ref
print(f"Union3: N={N_sn}, M_ref={M_ref:.4f}")

# --- FSC ---
fsc_raw = np.loadtxt(os.path.join(data_dir, "fine_structure.dat"), usecols=(0, 1, 2))
z_fsc_d  = fsc_raw[:, 0]
da_fsc   = fsc_raw[:, 1]   # Δα/α × 10^5
e_fsc    = fsc_raw[:, 2]
print(f"FSC: N={len(z_fsc_d)}")

# ─────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 13),
                          gridspec_kw={"hspace": 0.38})
ax_H, ax_mu, ax_fsc = axes

# ── Panel 1: H(z) ──────────────────────────────────────────────
ax_H.errorbar(z_hd, H_hd, yerr=e_hd,
              fmt="o", color="dimgray", ms=4,
              elinewidth=1, capsize=3, alpha=0.8, label="HD data (31 pts)", zorder=5)
for k, m in MODELS.items():
    ax_H.plot(z_Hz, theory[k]["Hz"], color=COLORS[k], lw=2.0, ls=LS[k], label=LABELS[k])

ax_H.set_xlim(0, 2.5)
ax_H.set_ylim(50, 400)
ax_H.set_ylabel(r"$H(z)$  [km s$^{-1}$ Mpc$^{-1}$]", fontsize=12)
ax_H.set_title("(a) Hubble parameter", fontsize=11, loc="left")
ax_H.legend(fontsize=8.5, loc="upper left")
ax_H.grid(True, alpha=0.25)

# ── Panel 2: μ(z) ──────────────────────────────────────────────
ax_mu.errorbar(z_sn, mb_shifted, yerr=mb_err,
               fmt="s", color="dimgray", ms=5,
               elinewidth=1, capsize=3, alpha=0.8,
               label=r"Union3 UNITY1.5 spline nodes ($N=22$)", zorder=5)
for k, m in MODELS.items():
    # shift theory curves by same M_ref for consistency
    ax_mu.plot(z_mu, theory[k]["mu"], color=COLORS[k], lw=2.0, ls=LS[k], label=LABELS[k])

ax_mu.set_xlim(0, 2.3)
ax_mu.set_ylabel(r"$\mu(z)$  [mag]", fontsize=12)
ax_mu.set_title("(b) Distance modulus", fontsize=11, loc="left")
ax_mu.legend(fontsize=8.5, loc="upper left")
ax_mu.grid(True, alpha=0.25)

# ── Panel 3: FSC ───────────────────────────────────────────────
ax_fsc.errorbar(z_fsc_d, da_fsc, yerr=e_fsc,
                fmt="^", color="dimgray", ms=4,
                elinewidth=0.8, capsize=2, alpha=0.6,
                label=r"FSC data (199 pts)", zorder=5)
for k, m in MODELS.items():
    ax_fsc.plot(z_fsc, theory[k]["fsc"], color=COLORS[k], lw=2.0, ls=LS[k], label=LABELS[k])

ax_fsc.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax_fsc.set_xlim(0, 7.5)
ax_fsc.set_ylim(-35, 35)
ax_fsc.set_xlabel(r"Redshift $z$", fontsize=12)
ax_fsc.set_ylabel(r"$\Delta\alpha/\alpha \times 10^5$", fontsize=12)
ax_fsc.set_title("(c) Fine structure constant variation", fontsize=11, loc="left")
ax_fsc.legend(fontsize=8.5, loc="upper right")
ax_fsc.grid(True, alpha=0.25)

plt.suptitle("Dataset comparison: best-fit models vs HD + Union3 + FSC data",
             fontsize=12, y=1.01)

for ext in ("pdf", "png"):
    outfile = os.path.join(os.path.dirname(__file__), f"combined_comparison.{ext}")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved: {outfile}")
