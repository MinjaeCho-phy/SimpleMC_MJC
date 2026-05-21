"""DFTCosmology — unified DFT / LDFT / DFT2 cosmology.

A single coupled phi(z)/H(z) RK4 model on the O(D,D) DFT background. The
cosmological-constant sector is toggled by ``varyOL``:

  * ``varyOL=False`` (default) — Omega_Lambda = 0. This is the plain **DFT**
    family (formerly DFT_fix.py): DFTCosmology, DFTw1l2/DFTl3w1/DFTl2w/DFTl0.
  * ``varyOL=True``            — Omega_Lambda is a free parameter. This is the
    **LDFT** family (formerly LDFTCosmology.py): "L" = with a Lambda term.
    ``DFT2`` (l=0 + OL) is just DFTl0Cosmology(varyOL=True).

Everything else (curvature Ok, H-flux Oh, dilaton-coupled species Oe with DFT
EoS (w, lambda), the alpha_fsc nuisance) is shared. ``ishzero`` drops Oh;
``fixfsc`` holds alpha_fsc at the lab value.

F(z) (== inSqrt), Eq (III.6): the dilaton-coupled species enters with
coefficient **(6 + 3*lambda)** [the +3*lambda was previously missing], while
Lambda / curvature / H-flux keep coefficient 6:

    F(z) = 3/H0^2 + ((6+3l) Oe (1+z)^{6(w+1)/(l+2)} e^{4/(l+2) phi}
                     + 6 OL + 6 Ok (1+z)^2 + 6 Oh (1+z)^6) / H^2
    dphi/dz = -(3 - H0 sqrt(F)) / (2(1+z))
    dH/dz   = -H0^2 [ (3w Oe(...) + 2 Ok (1+z)^2 + 6 Oh (1+z)^6)/H
                       - H sqrt(F)/H0 ] / (1+z)

Recollapse handling: H <= 1e-3 or non-finite at any z <= z_max -> the point is
rejected via prior_loglike() = -1e30; internal arrays are frozen so observable
methods never crash. lambda = -2 is a true singularity of the dilaton EoM
(note 6+3l -> 0 there): lp2 clamped at +/-1e-4 in the integrator, and
prior_loglike rejects |l+2| < 1e-4.

When first deploying after editing the numba RHS, clear stale artifacts:
    rm -rf simplemc/models/__pycache__
"""
from simplemc.cosmo.BaseCosmology import BaseCosmology
from simplemc.cosmo.paramDefs import (h_par, Ok_par, dft_Oh_par, dft_OL_par,
                                       dft_Oe_par, dft_w_par, dft_l_par,
                                       alpha_fsc_par)

import numpy as np
from numba import njit


# Lab reference matching LCDMCosmology.fine_structure_constant so that
# gamma = alpha_fsc / ALPHA_LAB = 1 reproduces the fixed-fsc prediction.
ALPHA_LAB = 0.0072973525643


# ---------------------------------------------------------------------------
#  Numba-accelerated ODE core
#
#  Physical constants packed into a single float64 array `p`:
#     p[0]=H0  p[1]=H0sq  p[2]=Ok  p[3]=Oh  p[4]=OL
#     p[5]=Oe  p[6]=w     p[7]=alpha  p[8]=beta  p[9]=coef_e (= 6 + 3*lambda)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, error_model='numpy')
def _rhs(z, phi, H, p):
    """DFT coupled ODE right-hand side. Returns (dphi/dz, dH/dz)."""
    H0    = p[0]
    H0sq  = p[1]
    Ok    = p[2]
    Oh    = p[3]
    OL    = p[4]
    Oe    = p[5]
    w     = p[6]
    alpha = p[7]
    beta  = p[8]
    coef_e = p[9]   # F(z) matter coefficient = 6 + 3*lambda  (Eq III.6)

    if H <= 0.0:
        H = H0

    if phi > 50.0:
        phi_c = 50.0
    elif phi < -50.0:
        phi_c = -50.0
    else:
        phi_c = phi

    zp1    = 1.0 + z
    ee     = np.exp(beta * phi_c)
    Oe_z   = Oe * zp1**alpha * ee
    zp1_sq = zp1 * zp1
    zp1_6  = zp1_sq * zp1_sq * zp1_sq
    Hsq    = H * H

    inSqrt = 3.0 / H0sq + (coef_e * Oe_z + 6.0 * (OL + Ok * zp1_sq
                                                  + Oh * zp1_6)) / Hsq

    if inSqrt >= 0.0:
        s = np.sqrt(inSqrt)
        dphidz = -(3.0 - H0 * s) / (2.0 * zp1)
        dHdz = -H0sq * (
            (3.0 * w * Oe_z + 2.0 * Ok * zp1_sq + 6.0 * Oh * zp1_6) / H
            - H * s / H0
        ) / zp1
    else:
        dphidz = -1.0
        dHdz   = -1.0

    return dphidz, dHdz


@njit(cache=True, fastmath=True, error_model='numpy')
def _make_params(H0, Ok, Oh, OL, Oe, w, l):
    """Pack pre-computed constants into a flat array for _rhs.

    `lp2` is clamped at +/-1e-4 to keep the integrator finite at the
    lambda = -2 string-frame singularity. `prior_loglike()` independently
    rejects |l+2| < 1e-4, so the clamped values never affect physics.
    """
    lp2 = l + 2.0
    if abs(lp2) < 1e-4:
        lp2 = 1e-4 if lp2 >= 0 else -1e-4
    alpha = 6.0 * (w + 1.0) / lp2
    beta  = 4.0 / lp2
    p = np.empty(10)
    p[0] = H0
    p[1] = H0 * H0
    p[2] = Ok
    p[3] = Oh
    p[4] = OL
    p[5] = Oe
    p[6] = w
    p[7] = alpha
    p[8] = beta
    p[9] = 6.0 + 3.0 * l   # matter coefficient in F(z), Eq (III.6); l=-2 -> 0
    return p


# ---------------------------------------------------------------------------
#  Fixed-step RK4 solver  (with recollapse-aware early termination)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, error_model='numpy')
def _dft_solve_rk4(H0, Ok, Oh, OL, Oe, w, l, z_max, steps):
    """RK4 solve of the DFT coupled ODEs + cumulative Da(z), single pass.

    Returns H_arr, phi_arr, Da_arr, broken_idx. After a recollapse (H <= 1e-3
    or non-finite, or H >= 1e10), the tails are frozen (H_arr -> 1e-10,
    phi/Da -> last valid) and broken_idx is the z-grid index where it
    triggered (-1 if the run completed normally).
    """
    p  = _make_params(H0, Ok, Oh, OL, Oe, w, l)
    dz = z_max / (steps - 1)

    H_arr   = np.empty(steps)
    phi_arr = np.empty(steps)
    Da_arr  = np.zeros(steps)

    phi = 0.0
    H   = H0
    H_arr[0]   = H0
    phi_arr[0] = 0.0

    broken_idx = -1

    for i in range(steps - 1):
        z  = i * dz
        zh = z + 0.5 * dz

        dp1, dH1 = _rhs(z,      phi,               H,               p)
        dp2, dH2 = _rhs(zh,     phi + 0.5*dz*dp1,  H + 0.5*dz*dH1, p)
        dp3, dH3 = _rhs(zh,     phi + 0.5*dz*dp2,  H + 0.5*dz*dH2, p)
        dp4, dH4 = _rhs(z + dz, phi + dz*dp3,      H + dz*dH3,     p)

        phi_new = phi + (dz / 6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4)
        H_new   = H   + (dz / 6.0) * (dH1 + 2.0*dH2 + 2.0*dH3 + dH4)

        # NaN-safe recollapse / blow-up detection (NaN fails both bounds).
        if not (1e-3 < H_new < 1e10):
            broken_idx = i + 1
            H_arr[i+1:]   = 1e-10
            phi_arr[i+1:] = phi_arr[i]
            Da_arr[i+1:]  = Da_arr[i]
            return H_arr, phi_arr, Da_arr, broken_idx

        H_arr[i + 1]   = H_new
        phi_arr[i + 1] = phi_new
        Da_arr[i + 1]  = Da_arr[i] + 0.5 * dz * (H0 / H_arr[i] + H0 / H_new)

        phi = phi_new
        H   = H_new

    return H_arr, phi_arr, Da_arr, broken_idx


# ---------------------------------------------------------------------------
#  DFTCosmology  (base class for all DFT / LDFT variants)
# ---------------------------------------------------------------------------

class DFTCosmology(BaseCosmology):
    """Unified DFT cosmology. ``varyOL`` toggles the cosmological constant.

    `varyOL=False` (default) — Omega_Lambda = 0 (DFT family).
    `varyOL=True`            — Omega_Lambda free (LDFT family).
    `ishzero=True`           — Omega_h fixed to 0, dropped from free list.
    `fixfsc=True`            — alpha_fsc held at ALPHA_LAB, dropped from list.

    Recollapse / non-finite ODE states are rejected via prior_loglike()=-1e30;
    arrays are frozen so observable methods never crash.
    """

    _Z_MAX    = 8.0
    _RK_STEPS = 800

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value,
                 l=dft_l_par.value, alpha_fsc=alpha_fsc_par.value,
                 varyOL=False, ishzero=False, fixfsc=False):
        self.Ok = Ok
        self.ishzero = ishzero
        self.Oh = 0.0 if ishzero else Oh
        self.varyOL = varyOL
        self.OL = OL if varyOL else 0.0
        self.Oe = Oe
        self.w  = w
        self.l  = l
        self.fixfsc = fixfsc
        self.alpha_fsc = ALPHA_LAB if fixfsc else alpha_fsc

        self._H0      = h * 100.0
        self._H_arr   = None
        self._phi_arr = None
        self._Da_arr  = None
        self._broken  = False

        self.parameters = self._build_params()
        self._z_grid = np.linspace(0.0, self._Z_MAX, self._RK_STEPS)

        BaseCosmology.__init__(self, h)
        self.updateParams([])

    # Free-parameter list. Order matches the historical chains:
    # [h, Ok, (Oh), (OL), Oe, (w), (l), (alpha_fsc)]. Subclasses that fix
    # w/l override `_eos_params` to drop them.
    def _eos_params(self):
        return [dft_w_par, dft_l_par]

    def _build_params(self):
        base = [h_par, Ok_par]
        if not self.ishzero:
            base.append(dft_Oh_par)
        if self.varyOL:
            base.append(dft_OL_par)
        base.append(dft_Oe_par)
        base += self._eos_params()
        if not self.fixfsc:
            base.append(alpha_fsc_par)
        return base

    def freeParameters(self):
        return self.parameters

    # -- parameter update (no ODE solve) ------------------------------------
    def _set_params(self, pars):
        BaseCosmology.updateParams(self, pars)
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh" and not self.ishzero:
                self.Oh = p.value
            elif p.name == "OL" and self.varyOL:
                self.OL = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
            elif p.name == "l_dft":
                self.l = p.value
            elif p.name == "alpha_fsc" and not self.fixfsc:
                self.alpha_fsc = p.value
        if not self.varyOL:
            self.OL = 0.0

    def updateParams(self, pars):
        self._set_params(pars)
        self.initialize()
        return True

    # -- ODE solve + precompute all lookups ---------------------------------
    def initialize(self):
        H0 = self.h * 100.0
        H_arr, phi_arr, Da_arr, broken_idx = _dft_solve_rk4(
            H0, self.Ok, self.Oh, self.OL, self.Oe, self.w, self.l,
            self._Z_MAX, self._RK_STEPS
        )
        self._H0      = H0
        self._H_arr   = H_arr
        self._phi_arr = phi_arr
        self._Da_arr  = Da_arr
        self._broken  = (broken_idx >= 0)
        return True

    # -- fast lookups (np.interp on uniform grid) ---------------------------
    def hub(self, z):
        return np.interp(z, self._z_grid, self._H_arr)

    def fine_structure_constant(self, a):
        z = 1.0 / a - 1.0
        phi = np.clip(np.interp(z, self._z_grid, self._phi_arr), -50.0, 50.0)
        gamma = self.alpha_fsc / ALPHA_LAB
        return gamma * np.exp(2.0 * phi) - 1.0

    def prior_loglike(self):
        # Reject points where the ODE recollapsed or went non-finite.
        if self._broken:
            return -1e30
        # lambda = -2 is a true singularity of the dilaton EoM: hard cutoff
        # plus a soft Gaussian penalty on the singular neighborhood.
        abs_l2 = abs(self.l + 2.0)
        if abs_l2 < 1e-4:
            return -1e30
        sigma = 0.03
        if abs_l2 < 3.0 * sigma:
            return -0.5 * (sigma / abs_l2) ** 2
        return 0.0

    def RHSquared_a(self, a):
        z = 1.0 / a - 1.0
        H = np.interp(z, self._z_grid, self._H_arr)
        with np.errstate(over='ignore', invalid='ignore'):
            result = (H / self._H0) ** 2
        if not np.isfinite(result) or result <= 0:
            return 1e-30
        return result

    # -- overrides: precomputed distance functions --------------------------
    def Hinv_z(self, z):
        z = np.atleast_1d(z)
        H = np.interp(z, self._z_grid, self._H_arr)
        invH = self._H0 / np.where(H > 1e-5, H, 1e-5)
        return np.where(np.isfinite(invH), invH, 1e10)

    def Da_z(self, z):
        z = np.atleast_1d(z)
        r = np.interp(z, self._z_grid, self._Da_arr)
        if self.Curv == 0:
            return r
        elif self.Curv > 0:
            q = np.sqrt(self.Curv)
            return np.sinh(r * q) / q
        else:
            q = np.sqrt(-self.Curv)
            return np.sin(r * q) / q


# ---------------------------------------------------------------------------
#  Constrained subclasses (each works as DFT [varyOL=False] or LDFT [True])
# ---------------------------------------------------------------------------

class DFTw1l2Cosmology(DFTCosmology):
    """Fixed w=1, l=2. Free: h, Ok, (Oh), (OL), Oe (+ alpha_fsc)."""

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 varyOL=False, ishzero=False, fixfsc=False):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe,
                              w=1.0, l=2.0, alpha_fsc=alpha_fsc,
                              varyOL=varyOL, ishzero=ishzero, fixfsc=fixfsc)

    def _eos_params(self):
        return []   # w, l fixed

    def updateParams(self, pars):
        self._set_params(pars)
        self.w = 1.0
        self.l = 2.0
        self.initialize()
        return True


class DFTl3w1Cosmology(DFTCosmology):
    """Constraint l = 3w - 1. Free: h, Ok, (Oh), (OL), Oe, w (+ alpha_fsc)."""

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 varyOL=False, ishzero=False, fixfsc=False):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=w,
                              l=3.0*w - 1.0, alpha_fsc=alpha_fsc,
                              varyOL=varyOL, ishzero=ishzero, fixfsc=fixfsc)

    def _eos_params(self):
        return [dft_w_par]   # l = 3w-1 derived

    def updateParams(self, pars):
        self._set_params(pars)
        self.l = 3.0 * self.w - 1.0
        self.initialize()
        return True


class DFTl2wCosmology(DFTCosmology):
    """Constraint l = 2w. Free: h, Ok, (Oh), (OL), Oe, w (+ alpha_fsc)."""

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 varyOL=False, ishzero=False, fixfsc=False):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=w,
                              l=2.0*w, alpha_fsc=alpha_fsc,
                              varyOL=varyOL, ishzero=ishzero, fixfsc=fixfsc)

    def _eos_params(self):
        return [dft_w_par]   # l = 2w derived

    def updateParams(self, pars):
        self._set_params(pars)
        self.l = 2.0 * self.w
        self.initialize()
        return True


class DFTl0Cosmology(DFTCosmology):
    """Fixed l=0. Free: h, Ok, (Oh), (OL), Oe, w (+ alpha_fsc).

    With varyOL=True this is the former DFT2Cosmology (l=0 + Omega_Lambda).
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 varyOL=False, ishzero=False, fixfsc=False):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=w,
                              l=0.0, alpha_fsc=alpha_fsc,
                              varyOL=varyOL, ishzero=ishzero, fixfsc=fixfsc)

    def _eos_params(self):
        return [dft_w_par]   # l = 0 fixed

    def updateParams(self, pars):
        self._set_params(pars)
        self.l = 0.0
        self.initialize()
        return True
