#!/usr/bin/env python3
"""
Effective equation of state w_eff(z) and w_DDE(z) comparison for DFT models.

w_eff is computed via the kinematic reconstruction (model-independent):
  w_eff = -1 + 2(1+z) H'(z) / (3 H(z))

w_DDE (dark energy EOS, matter subtracted) uses eq (3.40) of the paper:
  w_DDE = w_eff / (1 - Omega_m (H0/H)^2 (1+z)^3)
  where Omega_m is taken from the LCDM best-fit as an external prior.

NOTE: The DFT ODE does NOT include matter explicitly — H(z) comes entirely from
DFT dynamics (curvature Omega_k, H-flux Omega_h, dilaton phi).  The dilaton
kinetic energy is already encoded in H(z) via the DFT constraint; no separate
Omega_phi term is needed or correct in the denominator.

Dataset: HD + Union3 + FSC  (chains_20260309 / chains/DFT)
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np
from scipy.optimize import brentq
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simplemc.models.DFTVacuum import DFTVacuum

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
    "DFTvac":     "#D6604D",
    "DFTvac_noh": "#1A9850",
    "DFT1_w1l2":  "#E08214",
}
LS     = {"DFTvac": "--", "DFTvac_noh": ":", "DFT1_w1l2": "-."}
LABELS = {
    "DFTvac":     r"DFT$_\mathrm{vac}$",
    "DFTvac_noh": r"DFT$_\mathrm{vac}$ (no $\mathfrak{h}$)",
    "DFT1_w1l2":  r"$\Lambda$DFT",
}

# ── Best-fit parameters from posterior chains ──────────────────────
def _chain_weighted_mean(path, cols):
    d = np.loadtxt(path)
    log_w = np.log(np.where(d[:, 0] > 0, d[:, 0], 1e-300))
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    return {name: float(np.average(d[:, 2 + i], weights=w))
            for i, name in enumerate(cols)}

base_dft  = os.path.join(_ROOT, "simplemc/chains/chains_20260309/DFT_chains_20260309")

bf_dftvac     = _chain_weighted_mean(
    os.path.join(base_dft,  "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok", "Oh"])
bf_dftvac_noh = _chain_weighted_mean(
    os.path.join(base_dft,  "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["h", "Ok"])

print("Best-fit parameters (vacuum models only):")
for key, bf in [("DFTvac", bf_dftvac), ("DFTvac_noh", bf_dftvac_noh)]:
    print(f"  {key}: {bf}")

# ── Instantiate vacuum models ──────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    dftvac     = DFTVacuum(h=bf_dftvac["h"],     Ok=bf_dftvac["Ok"],
                           Oh=bf_dftvac["Oh"],   ishzero=False)
    dftvac_noh = DFTVacuum(h=bf_dftvac_noh["h"], Ok=bf_dftvac_noh["Ok"],
                           ishzero=True)

# ── Omega_m: borrow from ow0waCDM posterior (effective approach) ───
# DFT has no intrinsic dark sector. To define an effective w_DDE, we subtract
# the matter contribution using Om from the ow0waCDM fit, following eq (3.21).
#
# IMPORTANT: Do NOT subtract a dilaton term (Omega_phi) from the denominator.
# In this effective approach, ALL non-matter content (including the dilaton
# kinetic energy) is lumped into "dark energy". Subtracting Omega_phi
# separately would double-subtract it from the total energy budget.
# The DFT dynamics already encode the dilaton through H(z); no explicit
# Omega_phi term belongs in the denominator of wDDE.

base_gr = os.path.join(_ROOT, "simplemc/chains/chains_20260309/GR_chains_20260309")
bf_ow0wa = _chain_weighted_mean(
    os.path.join(base_gr, "owa0CDM_phy_HD+Union3+FSC_nested_multi_1.txt"),
    ["Om", "Obh2", "h", "w", "wa", "Ok"])
Om = bf_ow0wa["Om"]
print(f"\nOmega_m from ow0waCDM posterior mean: {Om:.4f}")

# ── Kinematic reconstruction ────────────────────────────────────────
# weff = -1 + 2(1+z) H'(z) / (3 H(z))    [no model-specific params needed]
# wDDE = weff / (1 - Om*(H0/H)^2*(1+z)^3)
#
# Using the kinematic form ensures we never need to reference Ok, Oh directly.
# All three DFT models have Ok ≈ 1.0 (attractor) and Oh ≈ 0 at the posterior
# mean, so tiny Ok deviations (< 1e-5) do not affect w_DDE numerically.

def weff_and_wDDE(z_arr, model, Om):
    H_arr = np.array([float(model.hub(z)) for z in z_arr])
    H0    = float(model.hub(0.0))
    dHdz  = np.gradient(H_arr, z_arr)
    weff  = -1.0 + 2.0 * (1.0 + z_arr) * dHdz / (3.0 * H_arr)
    Om_frac = Om * H0**2 * (1.0 + z_arr)**3 / H_arr**2
    with np.errstate(invalid='ignore', divide='ignore'):
        wDDE = np.where(np.abs(1.0 - Om_frac) > 1e-4,
                        weff / (1.0 - Om_frac), np.nan)
    return weff, wDDE


# ── Resampling helpers ──────────────────────────────────────────────
N_DRAW = 500

def _load_resample(path, n=N_DRAW, seed=42):
    d = np.loadtxt(path)
    log_w = np.log(np.where(d[:, 0] > 0, d[:, 0], 1e-300))
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    rng = np.random.default_rng(seed)
    return d[rng.choice(len(w), size=n, p=w, replace=True)]

def _sigma1_bands(curves):
    return np.nanquantile(curves, 0.1587, axis=0), np.nanquantile(curves, 0.8413, axis=0)

# ── 1σ bands: DFT chain (H(z)) × ow0waCDM chain (Ωm) ──────────────
# Pair sample i from DFT chain with sample i from ow0waCDM chain.
# DFT attractor keeps H(z) nearly constant; most band width comes from Ωm.
ow0wa_path = os.path.join(base_gr, "owa0CDM_phy_HD+Union3+FSC_nested_multi_1.txt")

DFT_CHAIN_PATHS = {
    "DFTvac":     os.path.join(base_dft, "DFTvac_phy_HD+Union3+FSC_nested_multi_1.txt"),
    "DFTvac_noh": os.path.join(base_dft, "DFTvac_noh_phy_HD+Union3+FSC_nested_multi_1.txt"),
}
DFT_FACTORIES = {
    "DFTvac":     lambda row: DFTVacuum(h=row[2], Ok=row[3], Oh=row[4], ishzero=False),
    "DFTvac_noh": lambda row: DFTVacuum(h=row[2], Ok=row[3], ishzero=True),
}

ow0wa_samp = _load_resample(ow0wa_path, seed=0)   # col 2 = Om

z_arr = np.linspace(0.01, 1.3, 400)

wDDE_bands = {}
for key, factory in DFT_FACTORIES.items():
    print(f"  w_DDE bands [{key}] ...")
    dft_samp = _load_resample(DFT_CHAIN_PATHS[key], seed=1)
    curves = []
    for i in range(N_DRAW):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = factory(dft_samp[i])
            Om_i = float(ow0wa_samp[i, 2])
            _, wd = weff_and_wDDE(z_arr, m, Om_i)
            # clip individual curves to avoid divergence contaminating quantiles
            wd = np.where(np.isfinite(wd) & (wd > -8) & (wd < 1), wd, np.nan)
            if np.isfinite(wd).sum() > 10:
                curves.append(wd)
        except Exception:
            pass
    wDDE_bands[key] = _sigma1_bands(np.array(curves))

# ── Compute best-fit curves ────────────────────────────────────────

print("Computing weff and w_DDE curves (vacuum models) ...")
weff_curves, wDDE_curves = {}, {}
for key, model in [("DFTvac",     dftvac),
                   ("DFTvac_noh", dftvac_noh)]:
    we, wd = weff_and_wDDE(z_arr, model, Om)
    weff_curves[key] = we
    wDDE_curves[key] = wd

# ── Print z=0 values ───────────────────────────────────────────────
for key in weff_curves:
    print(f"  [{key}]  weff(0) = {weff_curves[key][0]:.4f}"
          f"   wDDE(0) = {wDDE_curves[key][0]:.4f}")

# ── Phantom crossing z* where w_DDE = -1 ──────────────────────────
def find_phantom_crossing(z_arr, w_arr):
    """Return z_star where w_DDE crosses -1, or None."""
    finite = np.isfinite(w_arr)
    zf, wf = z_arr[finite], w_arr[finite]
    # look for sign change in (w + 1)
    diff = wf + 1.0
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    crossings = []
    for idx in sign_changes:
        try:
            z_lo, z_hi = zf[idx], zf[idx + 1]
            z_star = brentq(lambda z: float(np.interp(z, zf, wf)) + 1.0, z_lo, z_hi)
            crossings.append(z_star)
        except ValueError:
            pass
    return crossings[0] if crossings else None

VAC_KEYS = ["DFTvac", "DFTvac_noh"]

phantom_z = {}
for key, w in wDDE_curves.items():
    if key not in VAC_KEYS:
        continue
    zc = find_phantom_crossing(z_arr, w)
    phantom_z[key] = zc
    label = f"{zc:.3f}" if zc is not None else "not found"
    print(f"  Phantom crossing z* [{key}]: {label}")

# ── Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5))

# Reference lines
ax.axhline(-1,    color='#888888', lw=0.9, ls='--', zorder=2, label=r'$w=-1$')
ax.axhline(-1./3, color='#aaaaaa', lw=0.7, ls=':',  zorder=2, label=r'$w=-1/3$')

# 1σ bands (behind curves)
for key in VAC_KEYS:
    lo, hi = wDDE_bands[key]
    ax.fill_between(z_arr, lo, hi, color=COLORS[key], alpha=0.18, lw=0, zorder=2)

# DFT vacuum w_DDE curves only
for key in VAC_KEYS:
    ax.plot(z_arr, wDDE_curves[key], color=COLORS[key], ls=LS[key], lw=2.2,
            label=LABELS[key], zorder=5)

# Phantom crossing annotations (deduplicated)
annotated_z = []
for key in VAC_KEYS:
    zc = phantom_z.get(key)
    if zc is None or not (z_arr[0] < zc < z_arr[-1]):
        continue
    if any(abs(zc - zp) < 0.01 for zp in annotated_z):
        continue  # skip duplicate crossing at same z
    ax.axvline(zc, color='#888888', lw=0.9, ls='--', alpha=0.50, zorder=3)
    ax.annotate(
        rf'$z_*={zc:.3f}$',
        xy=(zc, -1.0),
        xytext=(zc + 0.06, -1.0 + 0.25),
        fontsize=10, color='#333333',
        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.9),
    )
    annotated_z.append(zc)

ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'$w_{\rm DDE}(z)$')
ax.set_xlim(0, 1.3)

all_w = np.concatenate([wDDE_curves[k][np.isfinite(wDDE_curves[k]) & (wDDE_curves[k] > -8)]
                        for k in VAC_KEYS])
if len(all_w):
    ylo = max(float(np.nanmin(all_w)) - 0.3, -7)
    yhi = min(float(np.nanmax(all_w)) + 0.3,  0.1)
    ax.set_ylim(ylo, yhi)

ax.legend(loc='lower left', framealpha=0.92, edgecolor='#cccccc')
ax.grid(True, ls='--', alpha=0.3, lw=0.7)

param_str = (
    rf"$\Omega_m = {Om:.4f}$  (from $ow_0w_a$CDM)"  "\n"
    rf"$h_\mathrm{{DFT}} = {bf_dftvac['h']:.4f}$"
)
ax.text(0.97, 0.97, param_str, transform=ax.transAxes,
        fontsize=9.5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#cccccc', alpha=0.85))

fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(_HERE, f"wDDE_comparison.{ext}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
