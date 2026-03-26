#!/usr/bin/env python3
"""
Deceleration parameter q(z) comparison: GR models vs DFT models.

Numerically computed via the kinematic definition:
  q(z) = -1 + (1+z) * H'(z) / H(z)

GR models:  LCDM           (chains_20260309)
DFT vacuum: DFTvac, DFTvac_noh  (chains_20260309)
DFT w/Oe:  DFT_l0, DFT_w1l2, DFT1_w1l2  (chains/DFT)
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

from simplemc.models.DFTVacuum    import (DFTVacuum,
                                           solve_ode_numba as _solve_vac,
                                           RHS_numba       as _rhs_vac)
from simplemc.models.DFTCosmology  import DFTCosmology, DFTw1l2Cosmology
from simplemc.models.DFT1Cosmology import (DFT1Cosmology,
                                            solve_ode_numba as _solve_dft1,
                                            RHS_numba       as _rhs_dft1)

# ── Publication style ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         12,
    'axes.labelsize':    13,
    'legend.fontsize':   10,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'axes.linewidth':    1.0,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.top':         True,
    'ytick.right':       True,
})

COLORS = {
    "LCDM":       "#4575B4",
    "DFTvac":     "#D6604D",
    "DFTvac_noh": "#1A9850",
    "DFT_l0":     "#E08214",
    "DFT_w1l2":   "#8073AC",
    "DFT1_w1l2":  "#01665E",
}
LS = {
    "LCDM":       "-",
    "DFTvac":     "--",
    "DFTvac_noh": ":",
    "DFT_l0":     "-.",
    "DFT_w1l2":   (0, (3, 1, 1, 1)),
    "DFT1_w1l2":  (0, (5, 1)),
}
LABELS = {
    "LCDM":       r"$\Lambda$CDM",
    "DFTvac":     r"DFT$_\mathrm{vac}$",
    "DFTvac_noh": r"DFT$_\mathrm{vac}$ (no $\mathfrak{h}$)",
    "DFT_l0":     r"DFT ($l=0$, free $w$)",
    "DFT_w1l2":   r"DFT ($w=1,\,l=2$)",
    "DFT1_w1l2":  r"DFT$_1$ ($w=1,\,l=2$)",
}

# ── Chain paths ─────────────────────────────────────────────────────
base_gr   = os.path.join(_ROOT, "simplemc/chains/chains_20260309/GR_chains_20260309")
base_vac  = os.path.join(_ROOT, "simplemc/chains/chains_20260309/DFT_chains_20260309")
base_oe   = os.path.join(_ROOT, "simplemc/chains/DFT")

# ── Helpers ─────────────────────────────────────────────────────────
def _load_resample(path, n=500, seed=42):
    d = np.loadtxt(path)
    log_w = np.log(np.where(d[:, 0] > 0, d[:, 0], 1e-300))
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    rng = np.random.default_rng(seed)
    return d[rng.choice(len(w), size=n, p=w, replace=True)]

def _chain_weighted_mean(path, cols):
    d = np.loadtxt(path)
    log_w = np.log(np.where(d[:, 0] > 0, d[:, 0], 1e-300))
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    return {name: float(np.average(d[:, 2 + i], weights=w))
            for i, name in enumerate(cols)}

def _sigma1_bands(curves):
    return np.nanquantile(curves, 0.1587, axis=0), np.nanquantile(curves, 0.8413, axis=0)

# ── q: analytical (ΛCDM) and ODE-direct (DFT) ───────────────────────
# ΛCDM: q = -1 + [3Ωm(1+z)³ + 2Ωk(1+z)²] / [2 E²(z)]
def q_lcdm(z_arr, Om, Ok=0.0):
    OL = 1.0 - Om - Ok
    E2 = Om*(1+z_arr)**3 + Ok*(1+z_arr)**2 + OL
    return -1.0 + (3.0*Om*(1+z_arr)**3 + 2.0*Ok*(1+z_arr)**2) / (2.0*E2)

# DFT: solve ODE, evaluate RHS for exact dH/dz — no numerical gradient
_ODE_ZMAX  = 30.0
_ODE_STEPS = 10000

def q_dft_vac(h, Ok, Oh):
    y0 = np.array([0.0, h*100.0])
    z_ode, y_ode = _solve_vac(y0, 0.0, _ODE_ZMAX, _ODE_STEPS, h, Ok, Oh)
    H    = y_ode[:, 1]
    dHdz = np.array([_rhs_vac(z_ode[i], y_ode[i], h, Ok, Oh)[1]
                     for i in range(len(z_ode))])
    return z_ode, -1.0 + (1.0+z_ode)*dHdz/H

def q_dft1(h, Ok, Oh, OL, Oe, w, l):
    y0 = np.array([0.0, h*100.0])
    z_ode, y_ode = _solve_dft1(y0, 0.0, _ODE_ZMAX, _ODE_STEPS, h, Ok, Oh, OL, Oe, w, l)
    H    = y_ode[:, 1]
    dHdz = np.array([_rhs_dft1(z_ode[i], y_ode[i], h, Ok, Oh, OL, Oe, w, l)[1]
                     for i in range(len(z_ode))])
    return z_ode, -1.0 + (1.0+z_ode)*dHdz/H

# kept for 1σ band sampling (hub-based, fast but less precise)
def compute_q_from_hub(z_arr, H_arr):
    dHdz = np.gradient(H_arr, z_arr)
    return -1.0 + (1.0 + z_arr) * dHdz / H_arr

def H_DFT(z_arr, model):
    return np.array([float(model.hub(z)) for z in z_arr])

# ── Load best-fit parameters ─────────────────────────────────────────
bf_lcdm       = _chain_weighted_mean(
    os.path.join(base_gr,  "LCDM_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["Om", "Obh2", "h"])
bf_dftvac     = _chain_weighted_mean(
    os.path.join(base_vac, "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok", "Oh"])
bf_dftvac_noh = _chain_weighted_mean(
    os.path.join(base_vac, "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok"])
bf_dft_l0     = _chain_weighted_mean(
    os.path.join(base_oe,  "DFT_l0_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok", "Oh", "Oe", "w"])
bf_dft_w1l2   = _chain_weighted_mean(
    os.path.join(base_oe,  "DFT_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok", "Oh", "Oe"])
bf_dft1_w1l2  = _chain_weighted_mean(
    os.path.join(base_oe,  "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok", "Oh", "OL", "Oe"])

print("Best-fit parameters:")
for k, bf in [("LCDM", bf_lcdm), ("DFTvac", bf_dftvac), ("DFTvac_noh", bf_dftvac_noh),
              ("DFT_l0", bf_dft_l0), ("DFT_w1l2", bf_dft_w1l2), ("DFT1_w1l2", bf_dft1_w1l2)]:
    print(f"  {k}: {bf}")

# ── Instantiate best-fit DFT models ─────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    dftvac     = DFTVacuum(h=bf_dftvac["h"],     Ok=bf_dftvac["Ok"],
                           Oh=bf_dftvac["Oh"],   ishzero=False)
    dftvac_noh = DFTVacuum(h=bf_dftvac_noh["h"], Ok=bf_dftvac_noh["Ok"],
                           ishzero=True)
    dft_l0     = DFTCosmology(h=bf_dft_l0["h"],    Ok=bf_dft_l0["Ok"],
                              Oh=bf_dft_l0["Oh"],   Oe=bf_dft_l0["Oe"],
                              w=bf_dft_l0["w"],     l=0.0)
    dft_w1l2   = DFTw1l2Cosmology(h=bf_dft_w1l2["h"],  Ok=bf_dft_w1l2["Ok"],
                                   Oh=bf_dft_w1l2["Oh"], Oe=bf_dft_w1l2["Oe"])
    dft1_w1l2  = DFT1Cosmology(h=bf_dft1_w1l2["h"],  Ok=bf_dft1_w1l2["Ok"],
                                Oh=bf_dft1_w1l2["Oh"], OL=bf_dft1_w1l2["OL"],
                                Oe=bf_dft1_w1l2["Oe"], w=1.0, l=2.0)

# ── Log-spaced z grid (for bands and ΛCDM) ───────────────────────────
z_arr = np.geomspace(0.02, _ODE_ZMAX, 600)

# ── Best-fit q curves ────────────────────────────────────────────────
print("\nComputing best-fit q(z) curves ...")

ALL_KEYS = ["LCDM", "DFTvac", "DFTvac_noh", "DFT_l0", "DFT_w1l2", "DFT1_w1l2"]

# ΛCDM: analytical
q_bf = {"LCDM": (z_arr, q_lcdm(z_arr, bf_lcdm["Om"]))}

# DFT: ODE-direct (no numerical gradient)
p = bf_dftvac
q_bf["DFTvac"]     = q_dft_vac(p["h"], p["Ok"], p["Oh"])
p = bf_dftvac_noh
q_bf["DFTvac_noh"] = q_dft_vac(p["h"], p["Ok"], 0.0)
p = bf_dft_l0
q_bf["DFT_l0"]     = q_dft1(p["h"], p["Ok"], p["Oh"], 0.0,            p["Oe"], p["w"], 0.0)
p = bf_dft_w1l2
q_bf["DFT_w1l2"]   = q_dft1(p["h"], p["Ok"], p["Oh"], 0.0,            p["Oe"], 1.0,   2.0)
p = bf_dft1_w1l2
q_bf["DFT1_w1l2"]  = q_dft1(p["h"], p["Ok"], p["Oh"], p["OL"],        p["Oe"], 1.0,   2.0)

for k in ALL_KEYS:
    z_k, q_k = q_bf[k]
    print(f"  [{k}]  q(z=0.02) = {np.interp(0.02, z_k, q_k):.4f}   q(z=1) = {np.interp(1., z_k, q_k):.4f}")

# ── 1σ bands ─────────────────────────────────────────────────────────
print("\nComputing 1σ bands ...")
N_DRAW = 500

def _make_bands(path, factory, seed):
    samp = _load_resample(path, N_DRAW, seed=seed)
    curves = []
    for row in samp:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = factory(row)
            qc = compute_q_from_hub(z_arr, H_DFT(z_arr, m))
            qc = np.where(np.isfinite(qc) & (qc > -3) & (qc < 3), qc, np.nan)
            if np.isfinite(qc).sum() > 10:
                curves.append(qc)
        except Exception:
            pass
    return _sigma1_bands(np.array(curves))

# LCDM band (analytical q per sample)
lcdm_samp = _load_resample(
    os.path.join(base_gr, "LCDM_phy_HD+Union3+FSC_nested_multi_1.txt"), N_DRAW, seed=10)
curves_lcdm = []
for row in lcdm_samp:
    try:
        qc = q_lcdm(z_arr, row[2])   # row[2] = Om
        qc = np.where(np.isfinite(qc) & (qc > -3) & (qc < 3), qc, np.nan)
        if np.isfinite(qc).sum() > 10:
            curves_lcdm.append(qc)
    except Exception:
        pass
q_bands = {"LCDM": _sigma1_bands(np.array(curves_lcdm))}

# DFT vacuum bands
print("  q bands [DFTvac] ...")
q_bands["DFTvac"] = _make_bands(
    os.path.join(base_vac, "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    lambda row: DFTVacuum(h=row[2], Ok=row[3], Oh=row[4], ishzero=False), seed=20)
print("  q bands [DFTvac_noh] ...")
q_bands["DFTvac_noh"] = _make_bands(
    os.path.join(base_vac, "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
    lambda row: DFTVacuum(h=row[2], Ok=row[3], ishzero=True), seed=21)

# DFT Oe bands
print("  q bands [DFT_l0] ...")
q_bands["DFT_l0"] = _make_bands(
    os.path.join(base_oe, "DFT_l0_phy_HD+Union3+FSC_nested_multi_1.txt"),
    lambda row: DFTCosmology(h=row[2], Ok=row[3], Oh=row[4], Oe=row[5], w=row[6], l=0.0),
    seed=30)
print("  q bands [DFT_w1l2] ...")
q_bands["DFT_w1l2"] = _make_bands(
    os.path.join(base_oe, "DFT_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
    lambda row: DFTw1l2Cosmology(h=row[2], Ok=row[3], Oh=row[4], Oe=row[5]),
    seed=31)
print("  q bands [DFT1_w1l2] ...")
q_bands["DFT1_w1l2"] = _make_bands(
    os.path.join(base_oe, "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
    lambda row: DFT1Cosmology(h=row[2], Ok=row[3], Oh=row[4], OL=row[5], Oe=row[6],
                              w=1.0, l=2.0),
    seed=32)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.axhline(0.0, color='#888888', lw=0.9, ls='--', zorder=2, label=r'$q=0$')

for key in ALL_KEYS:
    lo, hi = q_bands[key]
    ax.fill_between(z_arr, lo, hi, color=COLORS[key], alpha=0.15, lw=0, zorder=2)

z_plot = np.geomspace(0.02, _ODE_ZMAX, 800)
for key in ALL_KEYS:
    z_k, q_k = q_bf[key]
    q_on_grid = np.interp(z_plot, z_k, q_k)
    mask = np.isfinite(q_on_grid) & (q_on_grid > -1.5) & (q_on_grid < 1.5)
    ax.plot(z_plot[mask], q_on_grid[mask], color=COLORS[key], ls=LS[key], lw=2.0,
            label=LABELS[key], zorder=5)

ax.set_xscale('log')
ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'$q(z)$')
ax.set_xlim(z_plot[0], z_plot[-1])

all_q = np.concatenate([np.interp(z_plot, q_bf[k][0], q_bf[k][1]) for k in ALL_KEYS])
all_q = all_q[np.isfinite(all_q) & (all_q > -1.5) & (all_q < 1.5)]
ax.set_ylim(max(float(np.nanmin(all_q)) - 0.15, -1.8),
            min(float(np.nanmax(all_q)) + 0.15,  1.2))

ax.legend(loc='upper left', framealpha=0.92, edgecolor='#cccccc', ncol=2)
ax.grid(True, ls='--', alpha=0.3, lw=0.7)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"q_comparison.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
