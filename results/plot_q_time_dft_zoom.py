#!/usr/bin/env python3
"""
Deceleration parameter q vs cosmic time Δt [Gyr] — DFT models only, zoomed near today.

Shows the small but non-zero q deviations around the current epoch.
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np
from scipy.integrate import cumulative_trapezoid
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

KM_S_MPC_TO_GYR = 1.022e-3

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 13, 'legend.fontsize': 10,
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
    "DFT_w1l2":   (0, (3, 1, 1, 1)),
    "DFT1_w1l2":  (0, (5, 1)),
}
LABELS = {
    "DFTvac":     r"DFT$_\mathrm{vac}$",
    "DFTvac_noh": r"DFT$_\mathrm{vac}$ (no $\mathfrak{h}$)",
    "DFT_l0":     r"DFT ($l=0$, free $w$)",
    "DFT_w1l2":   r"DFT ($w=1,\,l=2$)",
    "DFT1_w1l2":  r"DFT$_1$ ($w=1,\,l=2$)",
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
    "DFTvac":     _wm(os.path.join(base_vac, "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok","Oh"]),
    "DFTvac_noh": _wm(os.path.join(base_vac, "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok"]),
    "DFT_l0":     _wm(os.path.join(base_oe,  "DFT_l0_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok","Oh","Oe","w"]),
    "DFT_w1l2":   _wm(os.path.join(base_oe,  "DFT_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok","Oh","Oe"]),
    "DFT1_w1l2":  _wm(os.path.join(base_oe,  "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
                      ["h","Ok","Oh","OL","Oe"]),
}

print("Best-fit params:")
for k, v in bf.items():
    print(f"  {k}: {v}")

# ── Instantiate models ───────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    models = {
        "DFTvac":     DFTVacuum(h=bf["DFTvac"]["h"],       Ok=bf["DFTvac"]["Ok"],
                                Oh=bf["DFTvac"]["Oh"],     ishzero=False),
        "DFTvac_noh": DFTVacuum(h=bf["DFTvac_noh"]["h"],   Ok=bf["DFTvac_noh"]["Ok"],
                                ishzero=True),
        "DFT_l0":     DFTCosmology(h=bf["DFT_l0"]["h"],    Ok=bf["DFT_l0"]["Ok"],
                                   Oh=bf["DFT_l0"]["Oh"],  Oe=bf["DFT_l0"]["Oe"],
                                   w=bf["DFT_l0"]["w"],    l=0.0),
        "DFT_w1l2":   DFTw1l2Cosmology(h=bf["DFT_w1l2"]["h"],  Ok=bf["DFT_w1l2"]["Ok"],
                                        Oh=bf["DFT_w1l2"]["Oh"], Oe=bf["DFT_w1l2"]["Oe"]),
        "DFT1_w1l2":  DFT1Cosmology(h=bf["DFT1_w1l2"]["h"],  Ok=bf["DFT1_w1l2"]["Ok"],
                                    Oh=bf["DFT1_w1l2"]["Oh"], OL=bf["DFT1_w1l2"]["OL"],
                                    Oe=bf["DFT1_w1l2"]["Oe"], w=1.0, l=2.0),
    }

# ── Δt computation (hub-based) ───────────────────────────────────────
def compute_dt(hub_kms, z_future_min=-0.90, z_past_max=30.0):
    z_all = np.concatenate([np.linspace(z_future_min, -1e-3, 400),
                            np.geomspace(1e-3, z_past_max, 1000)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        H_gyr = np.array([float(hub_kms(z)) for z in z_all]) * KM_S_MPC_TO_GYR
    cum  = cumulative_trapezoid(1.0/((1.0+z_all)*H_gyr), z_all, initial=0.0)
    cum0 = float(np.interp(0.0, z_all, cum))
    return z_all, -(cum - cum0)

# ── q: ODE-direct for past (z>0), hub+gradient for future (z<0) ──────
_ODE_ZMAX  = 30.0
_ODE_STEPS = 10000

def q_ode_vac(h, Ok, Oh):
    y0 = np.array([0.0, h*100.0])
    z_p, y_p = _solve_vac(y0, 0.0, _ODE_ZMAX, _ODE_STEPS, h, Ok, Oh)
    H    = y_p[:, 1]
    dHdz = np.array([_rhs_vac(z_p[i], y_p[i], h, Ok, Oh)[1] for i in range(len(z_p))])
    return z_p, -1.0 + (1.0+z_p)*dHdz/H

def q_ode_dft1(h, Ok, Oh, OL, Oe, w, l):
    y0 = np.array([0.0, h*100.0])
    z_p, y_p = _solve_dft1(y0, 0.0, _ODE_ZMAX, _ODE_STEPS, h, Ok, Oh, OL, Oe, w, l)
    H    = y_p[:, 1]
    dHdz = np.array([_rhs_dft1(z_p[i], y_p[i], h, Ok, Oh, OL, Oe, w, l)[1]
                     for i in range(len(z_p))])
    return z_p, -1.0 + (1.0+z_p)*dHdz/H

def make_result(hub_kms, q_ode_func, z_future_min=-0.90):
    z_dt, dt = compute_dt(hub_kms, z_future_min=z_future_min)
    z_qp, q_past = q_ode_func()
    # future: hub + numerical gradient
    z_qf = np.linspace(z_future_min, -1e-3, 300)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        H_f = np.array([float(hub_kms(z)) for z in z_qf])
    q_fut = -1.0 + (1.0+z_qf)*np.gradient(H_f, z_qf)/H_f
    z_q = np.concatenate([z_qf, z_qp])
    q   = np.concatenate([q_fut, q_past])
    return dt, np.interp(z_dt, z_q, q)

print("\nComputing q(Δt) ...")
bv, bvn = bf["DFTvac"], bf["DFTvac_noh"]
bl, bw, b1 = bf["DFT_l0"], bf["DFT_w1l2"], bf["DFT1_w1l2"]

results = {
    "DFTvac":     make_result(models["DFTvac"].hub,
                              lambda: q_ode_vac(bv["h"], bv["Ok"], bv["Oh"])),
    "DFTvac_noh": make_result(models["DFTvac_noh"].hub,
                              lambda: q_ode_vac(bvn["h"], bvn["Ok"], 0.0)),
    "DFT_l0":     make_result(models["DFT_l0"].hub,
                              lambda: q_ode_dft1(bl["h"], bl["Ok"], bl["Oh"],
                                                 0.0, bl["Oe"], bl["w"], 0.0)),
    "DFT_w1l2":   make_result(models["DFT_w1l2"].hub,
                              lambda: q_ode_dft1(bw["h"], bw["Ok"], bw["Oh"],
                                                 0.0, bw["Oe"], 1.0, 2.0)),
    "DFT1_w1l2":  make_result(models["DFT1_w1l2"].hub,
                              lambda: q_ode_dft1(b1["h"], b1["Ok"], b1["Oh"],
                                                 b1["OL"], b1["Oe"], 1.0, 2.0)),
}
for key, (dt_i, q_i) in results.items():
    q0 = float(np.interp(0.0, dt_i[::-1], q_i[::-1]))
    print(f"  [{key}]  q(today) = {q0:.6f}")

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.axhline(0.0, color='#aaaaaa', lw=0.9, ls='--', zorder=1)
ax.axvline(0.0, color='#aaaaaa', lw=0.9, ls=':',  zorder=1)

XLIM = (-8, 8)
for key in DFT_KEYS:
    dt_i, q_i = results[key]
    mask = (dt_i >= XLIM[0]) & (dt_i <= XLIM[1]) & np.isfinite(q_i)
    ax.plot(dt_i[mask], q_i[mask], color=COLORS[key], ls=LS[key], lw=2.0,
            label=LABELS[key], zorder=5)

ax.set_xlabel(r'$\Delta t$ [Gyr]  (today $= 0$, past $< 0$, future $> 0$)')
ax.set_ylabel(r'$q$')
ax.set_xlim(XLIM)

# Auto y-limits from the data in view
all_q_in_view = []
for key in DFT_KEYS:
    dt_i, q_i = results[key]
    mask = (dt_i >= XLIM[0]) & (dt_i <= XLIM[1]) & np.isfinite(q_i)
    all_q_in_view.extend(q_i[mask].tolist())
all_q_in_view = np.array(all_q_in_view)
qlo = np.nanmin(all_q_in_view)
qhi = np.nanmax(all_q_in_view)
margin = (qhi - qlo) * 0.25 + 1e-4
ax.set_ylim(qlo - margin, qhi + margin)

# Past / Future labels
ybot = ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02
ax.text(-7.8, ybot, 'Past',   fontsize=10, color='#555555')
ax.text( 7.8, ybot, 'Future', fontsize=10, color='#555555', ha='right')

ax.legend(loc='upper right', framealpha=0.92, edgecolor='#cccccc')
ax.grid(True, ls='--', alpha=0.25, lw=0.6)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"q_time_dft_zoom.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
