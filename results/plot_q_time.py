#!/usr/bin/env python3
"""
Deceleration parameter q vs cosmic time Δt [Gyr].  Best-fit curves only.

x-axis: Δt = t(z) - t₀  (today = 0, past < 0, future > 0)
y-axis: q(z) = -1 + (1+z) H'(z)/H(z)

All hub functions are normalised to return H(z) in km/s/Mpc.
Cosmic time: Δt(z) = -∫₀ᶻ dz' / [(1+z') × H(z') × 1.022e-3]  [Gyr]
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

# 1 km/s/Mpc = 1.022e-3 Gyr^-1
KM_S_MPC_TO_GYR = 1.022e-3

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
    "LCDM":       "#222222",
    "w0waCDM":    "#4575B4",
    "DFTvac":     "#D6604D",
    "DFT_l0":     "#E08214",
    "DFT_w1l2":   "#8073AC",
}
LS = {
    "LCDM":       "-",
    "w0waCDM":    (0, (5, 2)),
    "DFTvac":     "--",
    "DFT_l0":     "-.",
    "DFT_w1l2":   (0, (3, 1, 1, 1)),
}
LABELS = {
    "LCDM":       r"$\Lambda$CDM",
    "w0waCDM":    r"$w_0 w_a$CDM",
    "DFTvac":     r"DFT$_\mathrm{vac}$",
    "DFT_l0":     r"DFT ($l=0$, free $w$)",
    "DFT_w1l2":   r"DFT ($w=1,\,l=2$)",
}

ALL_KEYS = ["LCDM", "w0waCDM", "DFTvac", "DFT_l0", "DFT_w1l2"]

# ── Chain paths ─────────────────────────────────────────────────────
base_gr  = os.path.join(_ROOT, "simplemc/chains/chains_20260309/GR_chains_20260309")
base_vac = os.path.join(_ROOT, "simplemc/chains/chains_20260309/DFT_chains_20260309")
base_oe  = os.path.join(_ROOT, "simplemc/chains/DFT")

def _chain_weighted_mean(path, cols):
    d = np.loadtxt(path)
    log_w = np.log(np.where(d[:, 0] > 0, d[:, 0], 1e-300))
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    return {name: float(np.average(d[:, 2+i], weights=w))
            for i, name in enumerate(cols)}

# ── Load best-fit params ─────────────────────────────────────────────
bf_lcdm      = _chain_weighted_mean(
    os.path.join(base_gr,  "LCDM_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["Om","Obh2","h"])
bf_w0wa      = _chain_weighted_mean(
    os.path.join(base_gr,  "owa0CDM_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["Om","Obh2","h","w","wa","Ok"])
bf_dftvac    = _chain_weighted_mean(
    os.path.join(base_vac, "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h","Ok","Oh"])
bf_dftvac_noh= _chain_weighted_mean(
    os.path.join(base_vac, "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h","Ok"])
bf_dft_l0    = _chain_weighted_mean(
    os.path.join(base_oe,  "DFT_l0_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h","Ok","Oh","Oe","w"])
bf_dft_w1l2  = _chain_weighted_mean(
    os.path.join(base_oe,  "DFT_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h","Ok","Oh","Oe"])
bf_dft1_w1l2 = _chain_weighted_mean(
    os.path.join(base_oe,  "DFT1_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h","Ok","Oh","OL","Oe"])

print("Best-fit parameters:")
for k, bf in [("LCDM",bf_lcdm),("w0waCDM",bf_w0wa),("DFTvac",bf_dftvac),
              ("DFTvac_noh",bf_dftvac_noh),("DFT_l0",bf_dft_l0),
              ("DFT_w1l2",bf_dft_w1l2),("DFT1_w1l2",bf_dft1_w1l2)]:
    print(f"  {k}: {bf}")

# ── Instantiate DFT models ───────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m_dftvac     = DFTVacuum(h=bf_dftvac["h"],     Ok=bf_dftvac["Ok"],
                             Oh=bf_dftvac["Oh"],   ishzero=False)
    m_dftvac_noh = DFTVacuum(h=bf_dftvac_noh["h"], Ok=bf_dftvac_noh["Ok"],
                             ishzero=True)
    m_dft_l0     = DFTCosmology(h=bf_dft_l0["h"],    Ok=bf_dft_l0["Ok"],
                                Oh=bf_dft_l0["Oh"],   Oe=bf_dft_l0["Oe"],
                                w=bf_dft_l0["w"],     l=0.0)
    m_dft_w1l2   = DFTw1l2Cosmology(h=bf_dft_w1l2["h"],  Ok=bf_dft_w1l2["Ok"],
                                    Oh=bf_dft_w1l2["Oh"], Oe=bf_dft_w1l2["Oe"])
    m_dft1_w1l2  = DFT1Cosmology(h=bf_dft1_w1l2["h"],  Ok=bf_dft1_w1l2["Ok"],
                                 Oh=bf_dft1_w1l2["Oh"], OL=bf_dft1_w1l2["OL"],
                                 Oe=bf_dft1_w1l2["Oe"], w=1.0, l=2.0)

# ── Hub functions: all return H(z) in km/s/Mpc ──────────────────────
# DFT models: model.hub(z) already returns km/s/Mpc
# ΛCDM: construct to also return km/s/Mpc

def make_lcdm_hub_kms(Om, h, Ok=0.0):
    H0 = h * 100.0  # km/s/Mpc
    OL = 1.0 - Om - Ok
    def hub(z):
        return H0 * float(np.sqrt(abs(Om*(1+z)**3 + Ok*(1+z)**2 + OL)))
    return hub

# ── Δt computation (hub-based, does not require dH/dz) ──────────────
def compute_dt(hub_kms, z_future_min=-0.90, z_past_max=1000.0):
    """Returns z_all, dt_all [Gyr].  dt<0 past, dt>0 future."""
    z_all = np.concatenate([np.linspace(z_future_min, -1e-3, 400),
                            np.geomspace(1e-3, z_past_max, 1000)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        H_gyr = np.array([float(hub_kms(z)) for z in z_all]) * KM_S_MPC_TO_GYR
    cum  = cumulative_trapezoid(1.0/((1.0+z_all)*H_gyr), z_all, initial=0.0)
    cum0 = float(np.interp(0.0, z_all, cum))
    return z_all, -(cum - cum0)

# ── q computation ────────────────────────────────────────────────────
# ΛCDM: analytical  q = -1 + [3Ωm(1+z)³ + 2Ωk(1+z)²] / [2 E²(z)]
def q_lcdm(z_arr, Om, Ok=0.0):
    OL = 1.0 - Om - Ok
    E2 = Om*(1+z_arr)**3 + Ok*(1+z_arr)**2 + OL
    return -1.0 + (3.0*Om*(1+z_arr)**3 + 2.0*Ok*(1+z_arr)**2) / (2.0*E2)

# w0waCDM: analytical  w(z) = w0 + wa·z/(1+z)
# q = -1 + [3Ωm(1+z)³ + 2Ωk(1+z)² + 3(1+w(z))·ΩΛ·f_de] / [2 E²(z)]
def q_w0wa(z_arr, Om, Ok, w0, wa):
    OL  = 1.0 - Om - Ok
    f_de = (1+z_arr)**(3*(1+w0+wa)) * np.exp(-3.0*wa*z_arr/(1+z_arr))
    E2   = Om*(1+z_arr)**3 + Ok*(1+z_arr)**2 + OL*f_de
    w_z  = w0 + wa*z_arr/(1+z_arr)
    return -1.0 + (3.0*Om*(1+z_arr)**3 + 2.0*Ok*(1+z_arr)**2
                   + 3.0*(1+w_z)*OL*f_de) / (2.0*E2)

def make_w0wa_hub_kms(Om, Ok, w0, wa, h):
    H0 = h * 100.0
    OL = 1.0 - Om - Ok
    def hub(z):
        f_de = (1+z)**(3*(1+w0+wa)) * np.exp(-3.0*wa*z/(1+z))
        return H0 * float(np.sqrt(abs(Om*(1+z)**3 + Ok*(1+z)**2 + OL*f_de)))
    return hub

# DFT past (z>0): ODE RHS directly — no numerical gradient
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

# DFT future (z<0): hub extrapolation + numerical gradient (q≈0 there)
def q_future_hub(hub_kms, z_future_min=-0.90):
    z_f = np.linspace(z_future_min, -1e-3, 300)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        H_f = np.array([float(hub_kms(z)) for z in z_f])
    dHdz = np.gradient(H_f, z_f)
    return z_f, -1.0 + (1.0+z_f)*dHdz/H_f

def make_dft_result(hub_kms, q_ode_func, z_future_min=-0.90):
    """Combine: ODE q for past + hub q for future; dt from hub for full range."""
    z_dt, dt = compute_dt(hub_kms, z_future_min=z_future_min, z_past_max=_ODE_ZMAX)
    z_qp, q_past   = q_ode_func()
    z_qf, q_future = q_future_hub(hub_kms, z_future_min)
    # Combined q(z) array: future + past
    z_q = np.concatenate([z_qf, z_qp])
    q   = np.concatenate([q_future, q_past])
    # Map q onto same z grid as dt via interpolation
    q_on_dt = np.interp(z_dt, z_q, q)
    return dt, q_on_dt, z_dt

# ── Compute curves ───────────────────────────────────────────────────
print("\nComputing best-fit q(Δt) curves ...")

# ΛCDM: analytical q, hub-based dt
p = bf_lcdm
lcdm_hub = make_lcdm_hub_kms(p["Om"], p["h"])
z_lcdm, dt_lcdm = compute_dt(lcdm_hub, z_future_min=-0.999, z_past_max=_ODE_ZMAX)
q_lcdm_arr = q_lcdm(z_lcdm, p["Om"])

# w0waCDM: analytical q, hub-based dt
p = bf_w0wa
w0wa_hub = make_w0wa_hub_kms(p["Om"], p["Ok"], p["w"], p["wa"], p["h"])
z_w0wa, dt_w0wa = compute_dt(w0wa_hub, z_future_min=-0.999, z_past_max=_ODE_ZMAX)
q_w0wa_arr = q_w0wa(z_w0wa, p["Om"], p["Ok"], p["w"], p["wa"])

# DFT: ODE-based q (past) + hub future
p = bf_dftvac
dt_vac, q_vac, z_vac = make_dft_result(
    m_dftvac.hub,
    lambda: q_ode_vac(p["h"], p["Ok"], p["Oh"]))

p = bf_dftvac_noh
dt_vacn, q_vacn, z_vacn = make_dft_result(
    m_dftvac_noh.hub,
    lambda: q_ode_vac(p["h"], p["Ok"], 0.0))

p = bf_dft_l0
dt_l0, q_l0, z_l0 = make_dft_result(
    m_dft_l0.hub,
    lambda: q_ode_dft1(p["h"], p["Ok"], p["Oh"], 0.0, p["Oe"], p["w"], 0.0))

p = bf_dft_w1l2
dt_w1l2, q_w1l2, z_w1l2 = make_dft_result(
    m_dft_w1l2.hub,
    lambda: q_ode_dft1(p["h"], p["Ok"], p["Oh"], 0.0, p["Oe"], 1.0, 2.0))

p = bf_dft1_w1l2
dt_1w1l2, q_1w1l2, z_1w1l2 = make_dft_result(
    m_dft1_w1l2.hub,
    lambda: q_ode_dft1(p["h"], p["Ok"], p["Oh"], p["OL"], p["Oe"], 1.0, 2.0))

results = {
    "LCDM":       (dt_lcdm,  q_lcdm_arr),
    "w0waCDM":    (dt_w0wa,  q_w0wa_arr),
    "DFTvac":     (dt_vac,   q_vac),
    "DFTvac_noh": (dt_vacn,  q_vacn),
    "DFT_l0":     (dt_l0,    q_l0),
    "DFT_w1l2":   (dt_w1l2,  q_w1l2),
    "DFT1_w1l2":  (dt_1w1l2, q_1w1l2),
}

for key, (dt_i, q_i) in results.items():
    t0 = float(-np.min(dt_i))
    print(f"  [{key}]  t₀ ≈ {t0:.2f} Gyr   q(today) = {float(np.interp(0., dt_i[::-1], q_i[::-1])):.6f}")

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

# Reference lines
ax.axhline(0.0, color='#888888', lw=0.9, ls='--', zorder=1)
ax.axvline(0.0, color='#888888', lw=0.9, ls=':',  zorder=1)

# Labels like the reference figure
ax.text(-13.5, 0.06,  'Deceleration', fontsize=10, color='#444444')
ax.text(-13.5, -0.06, 'Acceleration', fontsize=10, color='#444444', va='top')
ax.annotate('', xy=(-12.5,  0.20), xytext=(-12.5,  0.04),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
ax.annotate('', xy=(-12.5, -0.20), xytext=(-12.5, -0.04),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))

for key in ALL_KEYS:
    dt_i, q_i = results[key]
    mask = np.isfinite(q_i) & (q_i > -1.5) & (q_i < 1.5)
    ax.plot(dt_i[mask], q_i[mask], color=COLORS[key], ls=LS[key], lw=2.0,
            label=LABELS[key], zorder=5)

ax.set_xlabel(r'$\Delta t$ [Gyr]  (today $= 0$, past $< 0$, future $> 0$)')
ax.set_ylabel(r'$q$')
ax.set_xlim(-15, 13)
ax.set_ylim(-1.1, 1.1)

# Past / Future text at bottom
ybot = ax.get_ylim()[0] + 0.03
ax.text(-14.5, ybot, 'Past',   fontsize=10, color='#444444')
ax.text( 12.5, ybot, 'Future', fontsize=10, color='#444444', ha='right')

ax.legend(loc='lower right', framealpha=0.92, edgecolor='#cccccc', ncol=2)
ax.grid(True, ls='--', alpha=0.25, lw=0.6)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"q_time.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
