"""DFTMatterRadiation — "full matter universe on the DFT critical line".

Two matter species, both sitting on the DFT critical line l = 3w - 1, evolve
on the same O(D,D) background as the generic DFTCosmology (curvature Ok,
H-flux Oh, and the intrinsic dilaton kinetic term 3/H0^2). The single generic
string species (Oe, w, l) of DFTCosmology is here replaced by two *fixed*
critical-line species:

    dust       : w = 1,   l = 2  ->  alpha = 6(w+1)/(l+2) = 3,  beta = 4/(l+2) = 1
                 rho_em(z) = Oem * (1+z)^3 * e^{ phi}
    radiation  : w = 1/3, l = 0  ->  alpha = 4,                 beta = 2
                 rho_er(z) = Oer * (1+z)^4 * e^{2 phi}

On the critical line the redshift exponents collapse to the *standard*
cosmological scalings (matter ~ (1+z)^3, radiation ~ (1+z)^4); the only DFT
fingerprint left in each species is the dilaton coupling e^{beta*phi}.

Coupled ODEs (OL = 0). F(z) == inSqrt; each matter species enters F with
coefficient (6 + 3*lambda_i) per Eq (III.6) [dust lambda=2 -> 12, radiation
lambda=0 -> 6], while curvature / H-flux keep 6:

    F(z) = 3/H0^2 + (12 rho_em + 6 rho_er + 6 Ok (1+z)^2 + 6 Oh (1+z)^6) / H^2
    dphi/dz = -(3 - H0 sqrt(F)) / (2(1+z))
    dH/dz   = -H0^2 [ (3 rho_em + 1 rho_er + 2 Ok (1+z)^2 + 6 Oh (1+z)^6)/H
                       - H sqrt(F)/H0 ] / (1+z)

(dH/dz species coefficients are 3*w_i: dust 3*1 = 3, radiation 3*(1/3) = 1.)
Components add linearly, as elsewhere in the DFT family; there is no Sum=1
closure — Oem, Oer are free amplitudes.

Free parameters: h, Ok, Oh, Oem, Oer (+ alpha_fsc unless fixfsc=True; Oh
dropped when ishzero=True). Recollapse / non-finite ODE states are rejected
via prior_loglike() = -1e30, exactly like DFTCosmology.
"""
from simplemc.cosmo.BaseCosmology import BaseCosmology
from simplemc.cosmo.paramDefs import (h_par, dftmr_Ok_par, dft_Oh_par,
                                       dft_Oem_par, dft_Oer_par,
                                       alpha_fsc_par)

import numpy as np
from numba import njit


# Lab reference matching LCDMCosmology.fine_structure_constant so that
# gamma = alpha_fsc / ALPHA_LAB = 1 reproduces the fixed-fsc prediction.
ALPHA_LAB = 0.0072973525643


# ---------------------------------------------------------------------------
#  Numba-accelerated ODE core
#
#  Constants packed into a flat float64 array `p`:
#     p[0]=H0  p[1]=H0sq  p[2]=Ok  p[3]=Oh  p[4]=Oem  p[5]=Oer
#  Species constants (alpha, beta, 3w) are fixed on the critical line and
#  hard-coded below: dust (3, 1, 3), radiation (4, 2, 1).
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, error_model='numpy')
def _rhs(z, phi, H, p):
    """DFT coupled ODE right-hand side. Returns (dphi/dz, dH/dz)."""
    H0   = p[0]
    H0sq = p[1]
    Ok   = p[2]
    Oh   = p[3]
    Oem  = p[4]
    Oer  = p[5]

    if H <= 0.0:
        H = H0

    if phi > 50.0:
        phi_c = 50.0
    elif phi < -50.0:
        phi_c = -50.0
    else:
        phi_c = phi

    zp1    = 1.0 + z
    zp1_sq = zp1 * zp1
    zp1_3  = zp1_sq * zp1
    zp1_4  = zp1_sq * zp1_sq
    zp1_6  = zp1_sq * zp1_sq * zp1_sq

    # critical-line species: dust ~ (1+z)^3 e^{phi}, radiation ~ (1+z)^4 e^{2phi}
    em_z = Oem * zp1_3 * np.exp(phi_c)
    er_z = Oer * zp1_4 * np.exp(2.0 * phi_c)

    Hsq = H * H
    # inSqrt = F(z), Eq (III.6): each matter species carries coefficient
    # (6 + 3*lambda_i) -- dust (lambda=2) -> 12, radiation (lambda=0) -> 6 --
    # while curvature / H-flux keep coefficient 6.
    inSqrt = 3.0 / H0sq + (12.0 * em_z + 6.0 * er_z
                           + 6.0 * Ok * zp1_sq + 6.0 * Oh * zp1_6) / Hsq

    if inSqrt >= 0.0:
        s = np.sqrt(inSqrt)
        dphidz = -(3.0 - H0 * s) / (2.0 * zp1)
        dHdz = -H0sq * (
            (3.0 * em_z + 1.0 * er_z + 2.0 * Ok * zp1_sq + 6.0 * Oh * zp1_6) / H
            - H * s / H0
        ) / zp1
    else:
        dphidz = -1.0
        dHdz   = -1.0

    return dphidz, dHdz


@njit(cache=True, fastmath=True, error_model='numpy')
def _solve_rk4(H0, Ok, Oh, Oem, Oer, z_max, steps):
    """RK4 solve of the coupled ODEs + cumulative Da(z), single pass.

    Returns H_arr, phi_arr, Da_arr, broken_idx. After a recollapse
    (H <= 1e-3 or non-finite, or H >= 1e10), the tails are frozen
    (H_arr -> 1e-10, phi/Da -> last valid) and broken_idx is the z-grid
    index where it triggered (-1 if the run completed normally).
    """
    p = np.empty(6)
    p[0] = H0
    p[1] = H0 * H0
    p[2] = Ok
    p[3] = Oh
    p[4] = Oem
    p[5] = Oer

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

        dp1, dH1 = _rhs(z,      phi,              H,              p)
        dp2, dH2 = _rhs(zh,     phi + 0.5*dz*dp1, H + 0.5*dz*dH1, p)
        dp3, dH3 = _rhs(zh,     phi + 0.5*dz*dp2, H + 0.5*dz*dH2, p)
        dp4, dH4 = _rhs(z + dz, phi + dz*dp3,     H + dz*dH3,     p)

        phi_new = phi + (dz / 6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4)
        H_new   = H   + (dz / 6.0) * (dH1 + 2.0*dH2 + 2.0*dH3 + dH4)

        # NaN-safe recollapse / blow-up detection (see DFT_fix notes).
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
#  DFTMatterRadiation
# ---------------------------------------------------------------------------

class DFTMatterRadiation(BaseCosmology):
    """Two critical-line species (dust Oem + radiation Oer) on the DFT bg.

    `fixfsc=False` (default) — alpha_fsc is a free parameter.
    `fixfsc=True`            — alpha_fsc held at ALPHA_LAB, dropped from
                               the free-parameter list.
    `ishzero=False` (default) — Oh is a free parameter.
    `ishzero=True`            — Oh fixed to 0 and dropped from the list.
    """

    _Z_MAX    = 8.0
    _RK_STEPS = 800

    def __init__(self, h=h_par.value, Ok=dftmr_Ok_par.value, Oh=dft_Oh_par.value,
                 Oem=dft_Oem_par.value, Oer=dft_Oer_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 ishzero=False, fixfsc=False):
        self.Ok = Ok
        self.ishzero = ishzero
        self.Oh = 0.0 if ishzero else Oh
        self.Oem = Oem
        self.Oer = Oer
        self.fixfsc = fixfsc
        self.alpha_fsc = ALPHA_LAB if fixfsc else alpha_fsc

        self._H0      = h * 100.0
        self._H_arr   = None
        self._phi_arr = None
        self._Da_arr  = None
        self._broken  = False

        base_params = [h_par, dftmr_Ok_par]
        if not ishzero:
            base_params.append(dft_Oh_par)
        base_params += [dft_Oem_par, dft_Oer_par]
        if not fixfsc:
            base_params.append(alpha_fsc_par)
        self.parameters = base_params
        self._z_grid = np.linspace(0.0, self._Z_MAX, self._RK_STEPS)

        BaseCosmology.__init__(self, h)
        self.updateParams([])

    def freeParameters(self):
        return self.parameters

    def _set_params(self, pars):
        BaseCosmology.updateParams(self, pars)
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh" and not self.ishzero:
                self.Oh = p.value
            elif p.name == "Oem":
                self.Oem = p.value
            elif p.name == "Oer":
                self.Oer = p.value
            elif p.name == "alpha_fsc" and not self.fixfsc:
                self.alpha_fsc = p.value

    def updateParams(self, pars):
        self._set_params(pars)
        self.initialize()
        return True

    def initialize(self):
        H0 = self.h * 100.0
        H_arr, phi_arr, Da_arr, broken_idx = _solve_rk4(
            H0, self.Ok, self.Oh, self.Oem, self.Oer,
            self._Z_MAX, self._RK_STEPS
        )
        self._H0      = H0
        self._H_arr   = H_arr
        self._phi_arr = phi_arr
        self._Da_arr  = Da_arr
        self._broken  = (broken_idx >= 0)
        return True

    def hub(self, z):
        return np.interp(z, self._z_grid, self._H_arr)

    def fine_structure_constant(self, a):
        z = 1.0 / a - 1.0
        phi = np.clip(np.interp(z, self._z_grid, self._phi_arr), -50.0, 50.0)
        gamma = self.alpha_fsc / ALPHA_LAB
        return gamma * np.exp(2.0 * phi) - 1.0

    def prior_loglike(self):
        # Reject points where the ODE recollapsed or went non-finite. Use a
        # large-but-recursion-safe penalty: -1e12 is below the worst physical
        # log-likelihood (FSC at its alpha prior edges reaches ~-1e11) yet small
        # enough that dynesty's information/logzvar recursion stays accurate
        # (a -1e30 sentinel triggers catastrophic cancellation -> logzerr -> 0
        # once a large fraction of the wide prior is broken).
        if self._broken:
            return -1e12
        return 0.0

    def RHSquared_a(self, a):
        z = 1.0 / a - 1.0
        H = np.interp(z, self._z_grid, self._H_arr)
        with np.errstate(over='ignore', invalid='ignore'):
            result = (H / self._H0) ** 2
        if not np.isfinite(result) or result <= 0:
            return 1e-30
        return result

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
