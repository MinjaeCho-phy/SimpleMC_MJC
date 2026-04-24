from simplemc.cosmo.BaseCosmology import BaseCosmology
from simplemc.cosmo.paramDefs import (h_par, Ok_par, dft_Oh_par,
                                       dft_Oe_par, dft_w_par, dft_l_par,
                                       alpha_fsc_par)

import numpy as np
from numba import njit


# Lab reference for the fine-structure constant, matched to the value
# used in LCDMCosmology.fine_structure_constant and DFT1Vacuum so that
# gamma = alpha_fsc / ALPHA_LAB = 1 reproduces the fixed-fsc prediction.
ALPHA_LAB = 0.0072973525643


# ---------------------------------------------------------------------------
#  Numba-accelerated ODE core
#
#  Physical constants are packed into a single float64 array `p`:
#     p[0]=H0  p[1]=H0sq  p[2]=Ok  p[3]=Oh  p[4]=OL
#     p[5]=Oe  p[6]=w     p[7]=alpha  p[8]=beta
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _rhs(z, phi, H, p):
    """
    DFT coupled ODE right-hand side.
    Returns (dphi/dz, dH/dz) as two scalars (no array allocation).
    """
    H0    = p[0]
    H0sq  = p[1]
    Ok    = p[2]
    Oh    = p[3]
    OL    = p[4]
    Oe    = p[5]
    w     = p[6]
    alpha = p[7]
    beta  = p[8]

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

    inSqrt = 3.0 / H0sq + 6.0 * (Oe_z + OL + Ok * zp1_sq
                                   + Oh * zp1_6) / Hsq

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


@njit(cache=True, fastmath=True)
def _make_params(H0, Ok, Oh, OL, Oe, w, l):
    """Pack pre-computed constants into a flat array for _rhs."""
    lp2   = l + 2.0
    alpha = 6.0 * (w + 1.0) / lp2
    beta  = 4.0 / lp2
    p = np.empty(9)
    p[0] = H0
    p[1] = H0 * H0
    p[2] = Ok
    p[3] = Oh
    p[4] = OL
    p[5] = Oe
    p[6] = w
    p[7] = alpha
    p[8] = beta
    return p


# ---------------------------------------------------------------------------
#  Fixed-step RK4 solver  (primary — fastest for this ODE)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _dft_solve_rk4(H0, Ok, Oh, OL, Oe, w, l, z_max, steps):
    """
    Fixed-step RK4 solve of the DFT coupled ODEs + cumulative Da(z)
    integral, all in a single pass on a uniform z grid.

    Returns (H_arr, phi_arr, Da_arr) with `steps` elements each.
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

    for i in range(steps - 1):
        z  = i * dz
        zh = z + 0.5 * dz

        dp1, dH1 = _rhs(z,      phi,               H,               p)
        dp2, dH2 = _rhs(zh,     phi + 0.5*dz*dp1,  H + 0.5*dz*dH1, p)
        dp3, dH3 = _rhs(zh,     phi + 0.5*dz*dp2,  H + 0.5*dz*dH2, p)
        dp4, dH4 = _rhs(z + dz, phi + dz*dp3,      H + dz*dH3,     p)

        phi += (dz / 6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4)
        H   += (dz / 6.0) * (dH1 + 2.0*dH2 + 2.0*dH3 + dH4)

        H_arr[i + 1]   = H
        phi_arr[i + 1] = phi

        # Accumulate comoving distance: Da(z) = int_0^z H0/H(z') dz'
        Da_arr[i + 1] = Da_arr[i] + 0.5 * dz * (H0 / H_arr[i] + H0 / H)

    return H_arr, phi_arr, Da_arr


# ---------------------------------------------------------------------------
#  DFTCosmology  (base class for all DFT variants)
# ---------------------------------------------------------------------------

class DFTCosmology(BaseCosmology):
    """
    DFT cosmology (OL=0 by default). Solves the DFT Friedmann equations
    via fixed-step RK4 and interpolates H(z), phi(z), Da(z).

    Parameters
    ----------
    h  : Hubble constant / 100
    Ok : curvature density, Omega_k
    Oh : h-flux density, Omega_h
    OL : cosmological constant density, Omega_Lambda  (fixed to 0 here)
    Oe : epsilon-matter density, Omega_epsilon
    w  : first EoS parameter
    l  : second EoS parameter  (singular at l = -2)
    """

    _Z_MAX    = 8.0
    _RK_STEPS = 800

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value, l=dft_l_par.value,
                 alpha_fsc=alpha_fsc_par.value, fixfsc=True):
        self.Ok = Ok
        self.Oh = Oh
        self.OL = 0.0
        self.Oe = Oe
        self.w  = w
        self.l  = l

        # Calibration nuisance for FSC: gamma = alpha_fsc / ALPHA_LAB
        # absorbs universal FSC systematic, symmetric with GR treatment.
        self.fixfsc    = fixfsc
        self.alpha_fsc = alpha_fsc

        base_params = [h_par, Ok_par, dft_Oh_par, dft_Oe_par,
                       dft_w_par, dft_l_par]
        if not fixfsc:
            base_params = base_params + [alpha_fsc_par]
        self.parameters = base_params
        self._z_grid = np.linspace(0.0, self._Z_MAX, self._RK_STEPS)

        BaseCosmology.__init__(self, h)
        self.updateParams([])

    def freeParameters(self):
        return self.parameters

    # -- parameter update (no ODE solve) ------------------------------------
    def _set_params(self, pars):
        """Set parameter values without triggering initialize()."""
        BaseCosmology.updateParams(self, pars)
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
            elif p.name == "l_dft":
                self.l = p.value
            elif p.name == "alpha_fsc" and not self.fixfsc:
                self.alpha_fsc = p.value
        self.OL = 0.0

    def updateParams(self, pars):
        self._set_params(pars)
        self.initialize()
        return True

    # -- ODE solve + precompute all lookups ---------------------------------
    def initialize(self):
        H0 = self.h * 100.0
        H_arr, phi_arr, Da_arr = _dft_solve_rk4(
            H0, self.Ok, self.Oh, self.OL, self.Oe, self.w, self.l,
            self._Z_MAX, self._RK_STEPS
        )
        self._H0      = H0
        self._H_arr   = H_arr
        self._phi_arr = phi_arr
        self._Da_arr  = Da_arr
        return True

    # -- fast lookups (np.interp on uniform grid) ---------------------------
    def hub(self, z):
        return np.interp(z, self._z_grid, self._H_arr)

    def fine_structure_constant(self, a):
        z = 1.0 / a - 1.0
        phi = np.clip(np.interp(z, self._z_grid, self._phi_arr), -50, 50)
        gamma = self.alpha_fsc / ALPHA_LAB
        return gamma * np.exp(2.0 * phi) - 1.0

    def prior_loglike(self):
        abs_l2 = abs(self.l + 2.0)
        if abs_l2 < 1e-10:
            return -1e30
        sigma = 0.03
        if abs_l2 < 3.0 * sigma:
            return -0.5 * (sigma / abs_l2) ** 2
        return 0.0

    def RHSquared_a(self, a):
        z = 1.0 / a - 1.0
        H = np.interp(z, self._z_grid, self._H_arr)
        result = (H / self._H0) ** 2
        if not np.isfinite(result) or result <= 0:
            return 1e-30
        return result

    # -- overrides: precomputed distance functions --------------------------
    def Hinv_z(self, z):
        z = np.atleast_1d(z)
        H = np.interp(z, self._z_grid, self._H_arr)
        return self._H0 / H

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
#  Constrained subclasses
# ---------------------------------------------------------------------------

class DFTw1l2Cosmology(DFTCosmology):
    """DFT with fixed w=1, l=2, OL=0. Free: h, Ok, Oh, Oe.

    With ``fixfsc=False``, also marginalizes over alpha_fsc as a
    calibration nuisance (gamma factor on the FSC observable),
    symmetric with the GR LCDM_fsc treatment.
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value,
                 alpha_fsc=alpha_fsc_par.value, fixfsc=True):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, Oe=Oe, w=1.0, l=2.0,
                              alpha_fsc=alpha_fsc, fixfsc=fixfsc)
        base_params = [h_par, Ok_par, dft_Oh_par, dft_Oe_par]
        if not fixfsc:
            base_params = base_params + [alpha_fsc_par]
        self.parameters = base_params

    def updateParams(self, pars):
        self._set_params(pars)
        self.w = 1.0
        self.l = 2.0
        self.initialize()
        return True


class DFTl3w1Cosmology(DFTCosmology):
    """DFT with constraint l = 3w - 1, OL=0. Free: h, Ok, Oh, Oe, w.

    With ``fixfsc=False``, also marginalizes over alpha_fsc (gamma factor
    on the FSC observable), symmetric with LCDM_fsc.
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value,
                 alpha_fsc=alpha_fsc_par.value, fixfsc=True):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, Oe=Oe, w=w,
                              l=3.0*w - 1.0,
                              alpha_fsc=alpha_fsc, fixfsc=fixfsc)
        base_params = [h_par, Ok_par, dft_Oh_par, dft_Oe_par, dft_w_par]
        if not fixfsc:
            base_params = base_params + [alpha_fsc_par]
        self.parameters = base_params

    def updateParams(self, pars):
        self._set_params(pars)
        self.l = 3.0 * self.w - 1.0
        self.initialize()
        return True


class DFTl2wCosmology(DFTCosmology):
    """DFT with constraint l = 2w, OL=0. Free: h, Ok, Oh, Oe, w.

    With ``fixfsc=False``, also marginalizes over alpha_fsc (gamma factor
    on the FSC observable), symmetric with LCDM_fsc.
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value,
                 alpha_fsc=alpha_fsc_par.value, fixfsc=True):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, Oe=Oe, w=w, l=2.0*w,
                              alpha_fsc=alpha_fsc, fixfsc=fixfsc)
        base_params = [h_par, Ok_par, dft_Oh_par, dft_Oe_par, dft_w_par]
        if not fixfsc:
            base_params = base_params + [alpha_fsc_par]
        self.parameters = base_params

    def updateParams(self, pars):
        self._set_params(pars)
        self.l = 2.0 * self.w
        self.initialize()
        return True


class DFTl0Cosmology(DFTCosmology):
    """DFT with fixed l=0, OL=0. Free: h, Ok, Oh, Oe, w.

    With ``fixfsc=False``, also marginalizes over alpha_fsc (gamma factor
    on the FSC observable), symmetric with LCDM_fsc.
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value,
                 alpha_fsc=alpha_fsc_par.value, fixfsc=True):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, Oe=Oe, w=w, l=0.0,
                              alpha_fsc=alpha_fsc, fixfsc=fixfsc)
        base_params = [h_par, Ok_par, dft_Oh_par, dft_Oe_par, dft_w_par]
        if not fixfsc:
            base_params = base_params + [alpha_fsc_par]
        self.parameters = base_params

    def updateParams(self, pars):
        self._set_params(pars)
        self.l = 0.0
        self.initialize()
        return True
