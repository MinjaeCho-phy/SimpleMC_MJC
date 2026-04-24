from simplemc.cosmo.paramDefs import (h_par, Ok_par, dft_Oh_par,
                                       alpha_fsc_par)
from simplemc.models.LCDMCosmology import LCDMCosmology

from scipy.interpolate import interp1d
import numpy as np
import scipy.integrate as sp_int
from numba import njit


# Lab reference for the fine-structure constant, matched to the value
# used in LCDMCosmology.fine_structure_constant so that
# gamma = alpha_fsc / ALPHA_LAB = 1 reproduces the fixed-fsc prediction.
ALPHA_LAB = 0.0072973525643


@njit(cache=True)
def RHS_numba(z, y, h, Ok, Oh):
    # y[0] = phi is evolved via dphidz but does not appear in the vacuum RHS.
    H = y[1]

    if H <= 0.0:
        H = h * 100.0

    H_sq = H**2.0
    inSqrt = 3.0 / (h * 100.0)**2.0 + (
        6.0 * Ok * (1.0 + z)**2.0
        + 6.0 * Oh * (1.0 + z)**6.0
    ) / H_sq

    if not inSqrt < 0.0:
        dphidz = -1.0 * (3 - (h * 100.0) * inSqrt**0.5) / (2.0 * (1.0 + z))
        dHdz = -(h * 100.0)**2.0 * (
            (
                2.0 * Ok * (1.0 + z)**2.0
                + 6.0 * Oh * (1.0 + z)**6.0
            ) / H
            - H / (h * 100.0) * inSqrt**0.5
        ) / (1.0 + z)
    else:
        dphidz = -1.0
        dHdz   = -1.0

    return np.array([dphidz, dHdz])


@njit(cache=True)
def solve_ode_numba(y0, z_start, z_end, steps, h, Ok, Oh):
    z_vals = np.linspace(z_start, z_end, steps)
    dz = z_vals[1] - z_vals[0]

    y_vals = np.zeros((steps, 2))
    y_vals[0] = y0

    y = y0.copy()

    for i in range(steps - 1):
        z = z_vals[i]

        k1 = RHS_numba(z, y, h, Ok, Oh)
        k2 = RHS_numba(z + 0.5*dz, y + 0.5*dz*k1, h, Ok, Oh)
        k3 = RHS_numba(z + 0.5*dz, y + 0.5*dz*k2, h, Ok, Oh)
        k4 = RHS_numba(z + dz, y + dz*k3, h, Ok, Oh)

        y = y + (dz/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y_vals[i+1] = y

    return z_vals, y_vals


class DFTVacuum(LCDMCosmology):
    """
    DFT vacuum cosmology (OL=0, Oe=0). Solves the coupled phi(z)/H(z)
    ODEs and interpolates; FSC observable is

        Delta alpha / alpha (z) = gamma * exp(2 phi(z)) - 1

    where ``gamma = alpha_fsc / ALPHA_LAB``.

    Parameters
    ----------
    h : float
        Hubble parameter H/100.
    Ok : float
        Curvature density Omega_k.
    Oh : float
        H-flux density Omega_h (ignored when ``ishzero=True``).
    alpha_fsc : float
        Present-day fine-structure constant. Only enters as a free
        parameter when ``fixfsc=False``; otherwise held at ALPHA_LAB
        (gamma = 1), which recovers the original vacuum prediction.
    ishzero : bool
        If True, fix Omega_h = 0 and drop it from the free-parameter list.
    fixfsc : bool
        If True (default), gamma = 1 and alpha_fsc is not free. If False,
        alpha_fsc is marginalized over as a calibration nuisance,
        symmetric with the GR LCDM_fsc treatment.
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 ishzero=False, fixfsc=True):
        self.h  = h
        self.Ok = Ok
        self.ishzero = ishzero
        self.Oh = 0.0 if ishzero else Oh

        base_params = [h_par, Ok_par] if ishzero else [h_par, Ok_par, dft_Oh_par]
        if not fixfsc:
            base_params = base_params + [alpha_fsc_par]
        self.parameters = base_params

        # High resolution for initial ODE solve
        self.rk_steps = 1000
        # Output grid for interpolation.
        # z=8.0 is enough for current datasets; extend if needed.
        self.z_max = 8.0
        self.z_values = np.linspace(0.0, self.z_max, 500)

        # LCDMCosmology stores self.alpha_fsc and self.fixfsc; the gamma
        # factor in fine_structure_constant picks them up automatically.
        LCDMCosmology.__init__(self, mnu=0, alpha_fsc=alpha_fsc, fixfsc=fixfsc)
        self.updateParams([])

    def freeParameters(self):
        return self.parameters

    def updateParams(self, pars):
        """Update the cosmological parameters.

        LCDMCosmology.updateParams handles h (via BaseCosmology) and
        alpha_fsc (when free), so gamma propagates to self.alpha_fsc
        automatically whenever alpha_fsc_par is in self.parameters.
        """
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False

        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh" and not self.ishzero:
                self.Oh = p.value
        self.initialize()
        return True

    def RHS(self, z, y):
        return RHS_numba(z, y, self.h, self.Ok, self.Oh)

    def initialize(self):
        y0 = np.array([0.0, self.h * 100.0])

        # 1. Solve ODE
        z_out, y_out = solve_ode_numba(
            y0, 0.0, self.z_max, self.rk_steps,
            self.h, self.Ok, self.Oh
        )

        # 2. Extract H(z) on fine grid
        H_fine = y_out[:, 1]

        # 3. Comoving distance r(z) = Int_0^z dz'/E(z'), E = H/H0
        E_z = H_fine / (self.h * 100.0)
        integrand = 1.0 / E_z
        r_fine = sp_int.cumulative_trapezoid(integrand, z_out, initial=0)

        # 4. Interpolators
        self.Da_interp = interp1d(z_out, r_fine, kind='cubic', fill_value='extrapolate')
        self.H_interp  = interp1d(z_out, H_fine, kind='cubic', fill_value='extrapolate')

        phi_fine = y_out[:, 0]
        self.phiinterp = interp1d(z_out, phi_fine, kind='cubic', fill_value='extrapolate')

        return True

    def hub(self, z):
        return self.H_interp(z)

    def fine_structure_constant(self, a):
        z = 1.0/a - 1.0
        phi_val = self.phiinterp(z)
        # Clamp phi to prevent overflow at large |phi|.
        phi_clamped = np.clip(phi_val, -50, 50)
        gamma = self.alpha_fsc / ALPHA_LAB
        return gamma * np.exp(2.0 * phi_clamped) - 1.0

    def RHSquared_a(self, a):
        # Used by growth-factor calculation if standard logic is called.
        z = 1.0 / a - 1.0
        H = self.hub(z)
        H0 = self.h * 100.0
        return (H / H0) ** 2
