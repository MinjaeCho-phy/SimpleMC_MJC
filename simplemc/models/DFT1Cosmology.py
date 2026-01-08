from simplemc import logger
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par
from simplemc.models.LCDMCosmology import LCDMCosmology

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import warnings

class DFT1Cosmology(LCDMCosmology):
    def __init__(self, h = h_par.value, Ok = Ok_par.value, Oh = dft_Oh_par.value, OL = dft_OL_par.value, Oe = dft_Oe_par.value, w = dft_w_par.value):
        """
        Parameter info

        h  : Hubble Constant / 100
        Ok : Density Parameter for curvature, \Omega_k
        Oh : Density Parameter for h-flux, \Omega_{\mathfrak{h}}
        OL : Density Parameter for Cosmological Constant, \Omega_{\Lambda}
        Oe : Density Parameter for general matter with EoS parameters (w,l), \Omega_{\epsilon}
        w  : First Equation of State Parameter,  w = (pressure)/("energy" density)
        """
        """ This is the one mimicking 2308.07149 """
        """ Parameter setting """
        self.h  = h
        self.Ok = Ok
        self.Oh = Oh
        self.Oe = Oe
        self.OL = OL
        self.w  = w

        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par]
        
        self.z_values = np.linspace(0.0, 8.0, 500)
        LCDMCosmology.__init__(self, mnu=0)
        self.updateParams([])


    def freeParameters(self):
        return self.parameters

    def updateParams(self, pars):
        """Update the cosmological parameters."""
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
            
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
        self.initialize()
        return True

    def RHS(self, z, y):
        """
        Right-hand side of the differential equation for H(a).
        Parameters
        ----------
        z : float
            Red shift.
        y : array-like
            [phi, H].
        Returns
        -------
        float
            dHdz   = Derivative dH/dz.
            dphidz = Derivative dphi/dz.
        """
        
        phi, H = y
        if H <= 0.0:
            H = self.h * 100.0

        OeEvol1 = 3.0 * (self.w + 1.0)
        # Clamp phi to prevent overflow in exp()
        phi_clamped = np.clip(phi, -50, 50)
        OeEvol2 = np.exp(2.0 * phi_clamped) 
        
        inSqrt = 3.0 / (self.h * 100.0)**2.0 + (
            6.0 * self.Oe * (1.0 + z)**OeEvol1 * OeEvol2
            + 6.0 * self.OL
            + 6.0 * self.Ok * (1.0 + z)**2.0
            + 6.0 * self.Oh * (1.0 + z)**6.0
        ) / (H**2.0)
        
        if not inSqrt < 0.0:
            dphidz   = -1.0 * (3 - (self.h * 100.0) * inSqrt**0.5) / (2.0 * (1.0 + z))
            dHdz = -(self.h * 100.0)**2.0 * (
                (
                    3.0 * self.w * self.Oe * (1.0 + z)**OeEvol1 * OeEvol2
                    + 2.0 * self.Ok * (1.0 + z)**2.0
                    + 6.0 * self.Oh * (1.0 + z)**6.0
                ) / H
                - H / (self.h * 100.0) * inSqrt**0.5
            ) / (1.0 + z)
        else:
            dphidz = -1.0
            dHdz   = -1.0
        
        return [dphidz, dHdz]

    def initialize(self):
        
        y0 = [0.0, self.h * 100.0]
        sol = solve_ivp(self.RHS, (self.z_values[0], self.z_values[-1]), y0, method='RK45', dense_output=True)

        if sol.success:
            self.phiinterp = interp1d(self.z_values, sol.sol(self.z_values)[0], fill_value='extrapolate')
            self.Hinterp = interp1d(self.z_values, sol.sol(self.z_values)[1], fill_value='extrapolate')
            return True
        else:
            self.Hinterp = interp1d(self.z_values, np.ones_like(self.z_values) * self.h * 100.0, fill_value='extrapolate')
            self.phiinterp = interp1d(self.z_values, np.zeros_like(self.z_values), fill_value='extrapolate')
            return True

    def hub(self, z):
        return self.Hinterp(z)

    def fine_structure_constant(self, a):
        phi_val = self.phiinterp(1.0/a - 1.0)
        # Clamp phi to prevent overflow
        phi_clamped = np.clip(phi_val, -50, 50)
        return np.exp(2.0 * phi_clamped) - 1.0

    def RHSquared_a(self, a):
        z = 1/a - 1.0
        H0 = self.h * 100
        H = self.hub(z)
        result = (H / H0)**2
        if not np.isfinite(result) or result <= 0:
            return 1e-30
        return result
