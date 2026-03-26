#!/usr/bin/env python3
"""
Deceleration parameter q(z) for DFT models.

Key design choices (rethought from scratch):
  1. dH/dz is taken DIRECTLY from the ODE RHS — no numerical gradient.
  2. ODE is solved to z_max=30 (previous code was limited to z=8 by the model).
  3. Log-spaced z grid via uniform ODE integration + post-interpolation.
  4. q = -1 + (1+z) * (dH/dz) / H,  where dH/dz = RHS[1] evaluated at each point.
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

# Import ODE kernels directly — bypasses the z=8 interpolator limit
from simplemc.models.DFTVacuum   import (solve_ode_numba as _solve_vac,
                                          RHS_numba       as _rhs_vac)
from simplemc.models.DFT1Cosmology import (solve_ode_numba as _solve_dft1,
                                            RHS_numba       as _rhs_dft1)

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 13, 'legend.fontsize': 10.5,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
})

COLORS = {
    "DFTvac":     "#D6604D",
    "DFTvac_noh": "#1A9850",
    "DFT_l0":     "#E08214",
    "DFT_w1l2":   "#8073AC",
    "DFT1_w1l2":  "#01665E",
}
LS = {
    "DFTvac":     "--",
    "DFTvac_noh": ":",
    "DFT_l0":     "-.",
    "DFT_w1l2":   (0, (3,1,1,1)),
    "DFT1_w1l2":  (0, (5,1)),
}
LABELS = {
    "DFTvac":     r"DFT$_\mathrm{vac}$",
    "DFTvac_noh": r"DFT$_\mathrm{vac}$ (no $\mathfrak{h}$)",
    "DFT_l0":     r"DFT ($l=0$, free $w$, $\Omega_\varepsilon\neq0$)",
    "DFT_w1l2":   r"DFT ($w=1,\,l=2$, $\Omega_\varepsilon\neq0$)",
    "DFT1_w1l2":  r"DFT$_1$ ($w=1,\,l=2$, $\Omega_\varepsilon\neq0$, $\Omega_\Lambda\neq0$)",
}
DFT_KEYS = ["DFTvac", "DFTvac_noh", "DFT_l0", "DFT_w1l2", "DFT1_w1l2"]

# ── Chain paths ──────────────────────────────────────────────────────
base_vac = os.path.join(_ROOT, "simplemc/chains/chains_20260309/DFT_chains_20260309")
base_oe  = os.path.join(_ROOT, "simplemc/chains/DFT")

def _wm(path, cols):
    d = np.loadtxt(path)
    lw = np.log(np.where(d[:,0]>0, d[:,0], 1e-300)); lw -= lw.max()
    w = np.exp(lw); w /= w.sum()
    return {n: float(np.average(d[:,2+i], weights=w)) for i,n in enumerate(cols)}

# ── Best-fit params ──────────────────────────────────────────────────
bf = {
    "DFTvac":     _wm(os.path.join(base_vac,"DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok","Oh"]),
    "DFTvac_noh": _wm(os.path.join(base_vac,"DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok"]),
    "DFT_l0":     _wm(os.path.join(base_oe,"DFT_l0_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok","Oh","Oe","w"]),
    "DFT_w1l2":   _wm(os.path.join(base_oe,"DFT_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok","Oh","Oe"]),
    "DFT1_w1l2":  _wm(os.path.join(base_oe,"DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok","Oh","OL","Oe"]),
}
print("Best-fit params:")
for k,v in bf.items(): print(f"  {k}: {v}")

# ── ODE → q(z) without any numerical gradient ────────────────────────
Z_MAX  = 30.0          # extend well past z=8 (previous model limit)
N_STEP = 10000         # RK4 steps

def _q_from_ode_vac(h, Ok, Oh):
    """DFTVacuum: solve ODE then evaluate RHS for dH/dz at each point."""
    y0 = np.array([0.0, h*100.0])
    z_arr, y_arr = _solve_vac(y0, 0.0, Z_MAX, N_STEP, h, Ok, Oh)
    H    = y_arr[:, 1]
    dHdz = np.array([_rhs_vac(z_arr[i], y_arr[i], h, Ok, Oh)[1]
                     for i in range(len(z_arr))])
    q = -1.0 + (1.0 + z_arr) * dHdz / H
    return z_arr, q

def _q_from_ode_dft1(h, Ok, Oh, OL, Oe, w, l):
    """DFT1Cosmology: same approach."""
    y0 = np.array([0.0, h*100.0])
    z_arr, y_arr = _solve_dft1(y0, 0.0, Z_MAX, N_STEP, h, Ok, Oh, OL, Oe, w, l)
    H    = y_arr[:, 1]
    dHdz = np.array([_rhs_dft1(z_arr[i], y_arr[i], h, Ok, Oh, OL, Oe, w, l)[1]
                     for i in range(len(z_arr))])
    q = -1.0 + (1.0 + z_arr) * dHdz / H
    return z_arr, q

# ── Compute ──────────────────────────────────────────────────────────
print("\nComputing q(z) directly from ODE RHS ...")

p = bf

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    z_vac, q_vac = _q_from_ode_vac(
        p["DFTvac"]["h"], p["DFTvac"]["Ok"], p["DFTvac"]["Oh"])

    z_vac_noh, q_vac_noh = _q_from_ode_vac(
        p["DFTvac_noh"]["h"], p["DFTvac_noh"]["Ok"], 0.0)

    z_l0, q_l0 = _q_from_ode_dft1(
        p["DFT_l0"]["h"], p["DFT_l0"]["Ok"], p["DFT_l0"]["Oh"],
        0.0, p["DFT_l0"]["Oe"], p["DFT_l0"]["w"], 0.0)

    z_w1l2, q_w1l2 = _q_from_ode_dft1(
        p["DFT_w1l2"]["h"], p["DFT_w1l2"]["Ok"], p["DFT_w1l2"]["Oh"],
        0.0, p["DFT_w1l2"]["Oe"], 1.0, 2.0)

    z_1w1l2, q_1w1l2 = _q_from_ode_dft1(
        p["DFT1_w1l2"]["h"], p["DFT1_w1l2"]["Ok"], p["DFT1_w1l2"]["Oh"],
        p["DFT1_w1l2"]["OL"], p["DFT1_w1l2"]["Oe"], 1.0, 2.0)

results = {
    "DFTvac":     (z_vac,     q_vac),
    "DFTvac_noh": (z_vac_noh, q_vac_noh),
    "DFT_l0":     (z_l0,      q_l0),
    "DFT_w1l2":   (z_w1l2,    q_w1l2),
    "DFT1_w1l2":  (z_1w1l2,   q_1w1l2),
}

for k,(z,q) in results.items():
    q0 = float(np.interp(0.0, z, q))
    print(f"  [{k}]  q(z=0) = {q0:.6f}   q(z=1) = {np.interp(1.,z,q):.4f}   q(z=10) = {np.interp(10.,z,q):.4f}")

# ── Log-spaced z grid for smooth plot ────────────────────────────────
z_plot = np.geomspace(0.01, Z_MAX, 800)

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.axhline(0.0, color='#aaaaaa', lw=0.9, ls='--', zorder=1)
ax.axhline(0.5, color='#cccccc', lw=0.7, ls=':',  zorder=1, label=r'$q=0,\,0.5$')

for key in DFT_KEYS:
    z_arr, q_arr = results[key]
    # interpolate onto log-spaced plot grid for smoothness
    q_plot = np.interp(z_plot, z_arr, q_arr)
    # mask unphysical values (ODE noise at boundaries)
    mask = np.isfinite(q_plot) & (q_plot > -1.5) & (q_plot < 1.5)
    ax.plot(z_plot[mask], q_plot[mask],
            color=COLORS[key], ls=LS[key], lw=2.0,
            label=LABELS[key], zorder=5)

ax.set_xscale('log')
ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'$q(z)$')
ax.set_xlim(0.01, Z_MAX)

# y limits from data
all_q = np.concatenate([np.interp(z_plot, r[0], r[1]) for r in results.values()])
all_q = all_q[np.isfinite(all_q) & (all_q > -1.5) & (all_q < 1.5)]
ax.set_ylim(all_q.min() - 0.08, all_q.max() + 0.08)

ax.legend(loc='lower right', framealpha=0.92, edgecolor='#cccccc')
ax.grid(True, ls='--', alpha=0.25, lw=0.6)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"q_dft.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
