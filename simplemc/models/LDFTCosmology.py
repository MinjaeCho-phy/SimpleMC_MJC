from simplemc.cosmo.BaseCosmology import BaseCosmology
from simplemc.cosmo.paramDefs import h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par, dft_l_par

from scipy.interpolate import interp1d
import numpy as np
from numba import njit


@njit(cache=True)
def _ldft_rhs(z, y, h, Ok, Oh, OL, Oe, w, l):
    phi = y[0]
    H   = y[1]

    if H <= 0.0:
        H = h * 100.0

    OeEvol1 = 6.0 * (w + 1.0) / (l + 2.0)

    if phi > 50.0:
        phi_c = 50.0
    elif phi < -50.0:
        phi_c = -50.0
    else:
        phi_c = phi

    OeEvol2 = np.exp(4.0 * phi_c / (l + 2.0))

    inSqrt = 3.0 / (h * 100.0)**2.0 + (
        6.0 * Oe * (1.0 + z)**OeEvol1 * OeEvol2
        + 6.0 * OL
        + 6.0 * Ok * (1.0 + z)**2.0
        + 6.0 * Oh * (1.0 + z)**6.0
    ) / H**2.0

    if not inSqrt < 0.0:
        s = inSqrt**0.5
        dphidz = -1.0 * (3.0 - h * 100.0 * s) / (2.0 * (1.0 + z))
        dHdz   = -(h * 100.0)**2.0 * (
            (3.0 * w * Oe * (1.0 + z)**OeEvol1 * OeEvol2
             + 2.0 * Ok * (1.0 + z)**2.0
             + 6.0 * Oh * (1.0 + z)**6.0) / H
            - H / (h * 100.0) * s
        ) / (1.0 + z)
    else:
        dphidz = -1.0
        dHdz   = -1.0

    return np.array([dphidz, dHdz])


@njit(cache=True)
def _ldft_solve(y0, z_start, z_end, steps, h, Ok, Oh, OL, Oe, w, l):
    z_vals = np.linspace(z_start, z_end, steps)
    dz     = z_vals[1] - z_vals[0]
    y_vals = np.zeros((steps, 2))
    y_vals[0] = y0
    y = y0.copy()
    for i in range(steps - 1):
        z  = z_vals[i]
        k1 = _ldft_rhs(z,           y,             h, Ok, Oh, OL, Oe, w, l)
        k2 = _ldft_rhs(z + 0.5*dz,  y + 0.5*dz*k1, h, Ok, Oh, OL, Oe, w, l)
        k3 = _ldft_rhs(z + 0.5*dz,  y + 0.5*dz*k2, h, Ok, Oh, OL, Oe, w, l)
        k4 = _ldft_rhs(z + dz,       y +     dz*k3, h, Ok, Oh, OL, Oe, w, l)
        y  = y + (dz / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y_vals[i + 1] = y
    return z_vals, y_vals


class LDFTCosmology(BaseCosmology):
    """
    LDFT cosmology. OL (cosmological constant) is a free parameter.
    All parameters free: h, Ok, Oh, OL, Oe, w, l.

    Parameters
    ----------
    h  : Hubble constant / 100
    Ok : curvature density, Omega_k
    Oh : h-flux density, Omega_h
    OL : cosmological constant density, Omega_Lambda
    Oe : epsilon-matter density, Omega_epsilon
    w  : first EoS parameter
    l  : second EoS parameter  (singular at l = -2)
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value,
                 w=dft_w_par.value, l=dft_l_par.value):
        self.Ok = Ok
        self.Oh = Oh
        self.OL = OL
        self.Oe = Oe
        self.w  = w
        self.l  = l

        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par,
                           dft_Oe_par, dft_w_par, dft_l_par]
        self.rk_steps = 5000
        self.z_values = np.linspace(0.0, 8.0, 500)

        BaseCosmology.__init__(self, h)
        self.updateParams([])

    def freeParameters(self):
        return self.parameters

    def updateParams(self, pars):
        BaseCosmology.updateParams(self, pars)
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh":
                self.Oh = p.value
            elif p.name == "OL":
                self.OL = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
            elif p.name == "l_dft":
                self.l = p.value
        self.initialize()
        return True

    def initialize(self):
        y0 = np.array([0.0, self.h * 100.0])
        z_out, y_out = _ldft_solve(
            y0, 0.0, 8.0, self.rk_steps,
            self.h, self.Ok, self.Oh, self.OL, self.Oe, self.w, self.l
        )
        phi_fine = interp1d(z_out, y_out[:, 0], kind='cubic', fill_value='extrapolate')
        H_fine   = interp1d(z_out, y_out[:, 1], kind='cubic', fill_value='extrapolate')
        phi_c = phi_fine(self.z_values)
        H_c   = H_fine(self.z_values)
        self.phiinterp = interp1d(self.z_values, phi_c, fill_value='extrapolate')
        self.Hinterp   = interp1d(self.z_values, H_c,   fill_value='extrapolate')
        return True

    def hub(self, z):
        return self.Hinterp(z)

    def fine_structure_constant(self, a):
        phi = np.clip(self.phiinterp(1.0/a - 1.0), -50, 50)
        return np.exp(2.0 * phi) - 1.0

    def prior_loglike(self):
        # Soft barrier around the l = -2 singularity (both sides allowed).
        # sigma = 0.03: penalty starts at |l+2| < 0.09, significant below 0.01.
        abs_l2 = abs(self.l + 2.0)
        if abs_l2 < 1e-10:
            return -1e30
        sigma = 0.03
        if abs_l2 < 3.0 * sigma:
            return -0.5 * (sigma / abs_l2) ** 2
        return 0.0

    def RHSquared_a(self, a):
        H0 = self.h * 100.0
        H  = self.hub(1.0/a - 1.0)
        result = (H / H0)**2
        if not np.isfinite(result) or result <= 0:
            return 1e-30
        return result


class LDFTw1l2Cosmology(LDFTCosmology):
    """LDFT with fixed w=1, l=2. Free: h, Ok, Oh, OL, Oe."""

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value):
        LDFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=1.0, l=2.0)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par]

    def updateParams(self, pars):
        ok = LDFTCosmology.updateParams(self, pars)
        if not ok:
            return False
        self.w = 1.0
        self.l = 2.0
        self.initialize()
        return True


class LDFTl3w1Cosmology(LDFTCosmology):
    """LDFT with constraint l = 3w + 1. Free: h, Ok, Oh, OL, Oe, w."""

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value):
        LDFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=w, l=3.0*w + 1.0)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par]

    def updateParams(self, pars):
        ok = LDFTCosmology.updateParams(self, pars)
        if not ok:
            return False
        self.l = 3.0 * self.w + 1.0
        self.initialize()
        return True


class LDFTl2wCosmology(LDFTCosmology):
    """LDFT with constraint l = 2w. Free: h, Ok, Oh, OL, Oe, w."""

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value):
        LDFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=w, l=2.0*w)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par]

    def updateParams(self, pars):
        ok = LDFTCosmology.updateParams(self, pars)
        if not ok:
            return False
        self.l = 2.0 * self.w
        self.initialize()
        return True


class LDFTl0Cosmology(LDFTCosmology):
    """LDFT with fixed l=0. Free: h, Ok, Oh, OL, Oe, w."""

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 OL=dft_OL_par.value, Oe=dft_Oe_par.value, w=dft_w_par.value):
        LDFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, OL=OL, Oe=Oe, w=w, l=0.0)
        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par]

    def updateParams(self, pars):
        ok = LDFTCosmology.updateParams(self, pars)
        if not ok:
            return False
        self.l = 0.0
        self.initialize()
        return True
