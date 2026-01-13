from simplemc import logger
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par
from simplemc.models.LCDMCosmology import LCDMCosmology

#from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import scipy.integrate as sp_int
import scipy as sp
import warnings
from numba import njit, float64

@njit(cache=True)
def RHS_numba(z, y, h, Ok, Oh, OL, Oe, w):
    phi = y[0]
    H = y[1]
    
    if H <= 0.0:
        H = h * 100.0

    OeEvol1 = 3.0 * (w + 1.0)
    # Clamp phi to prevent overflow in exp()
    if phi > 50.0:
        phi_clamped = 50.0
    elif phi < -50.0:
        phi_clamped = -50.0
    else:
        phi_clamped = phi
        
    OeEvol2 = np.exp(2.0 * phi_clamped) 
    
    H_sq = H**2.0
    inSqrt = 3.0 / (h * 100.0)**2.0 + (
        6.0 * Oe * (1.0 + z)**OeEvol1 * OeEvol2
        + 6.0 * OL
        + 6.0 * Ok * (1.0 + z)**2.0
        + 6.0 * Oh * (1.0 + z)**6.0
    ) / H_sq
    
    if not inSqrt < 0.0:
        dphidz   = -1.0 * (3 - (h * 100.0) * inSqrt**0.5) / (2.0 * (1.0 + z))
        dHdz = -(h * 100.0)**2.0 * (
            (
                3.0 * w * Oe * (1.0 + z)**OeEvol1 * OeEvol2
                + 2.0 * Ok * (1.0 + z)**2.0
                + 6.0 * Oh * (1.0 + z)**6.0
            ) / H
            - H / (h * 100.0) * inSqrt**0.5
        ) / (1.0 + z)
    else:
        dphidz = -1.0
        dHdz   = -1.0
    
    return np.array([dphidz, dHdz])

@njit(cache=True)
def solve_ode_numba(y0, z_start, z_end, steps, h, Ok, Oh, OL, Oe, w):
    z_vals = np.linspace(z_start, z_end, steps)
    dz = z_vals[1] - z_vals[0]
    
    y_vals = np.zeros((steps, 2))
    y_vals[0] = y0
    
    y = y0.copy()
    
    for i in range(steps - 1):
        z = z_vals[i]
        
        k1 = RHS_numba(z, y, h, Ok, Oh, OL, Oe, w)
        k2 = RHS_numba(z + 0.5*dz, y + 0.5*dz*k1, h, Ok, Oh, OL, Oe, w)
        k3 = RHS_numba(z + 0.5*dz, y + 0.5*dz*k2, h, Ok, Oh, OL, Oe, w)
        k4 = RHS_numba(z + dz, y + dz*k3, h, Ok, Oh, OL, Oe, w)
        
        y = y + (dz/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y_vals[i+1] = y
        
    return z_vals, y_vals

class DFT2Cosmology(LCDMCosmology):
    def __init__(self, h = h_par.value, Ok = Ok_par.value, Oh = dft_Oh_par.value, OL = dft_OL_par.value, Oe = dft_Oe_par.value, w = dft_w_par.value):
        """
        DFT2Cosmology: Optimized version with precomputed distance integrals.
        """
        self.h  = h
        self.Ok = Ok
        self.Oh = Oh
        self.Oe = Oe
        self.OL = OL
        self.w  = w

        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par]
        
        # High resolution for initial ODE solve
        self.rk_steps = 1000 
        # Output grid for interpolation. 
        # Note: z=8.0 is usually enough for most datasets, but can be extended if needed.
        self.z_max = 8.0
        self.z_values = np.linspace(0.0, self.z_max, 500) 
        
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
        return RHS_numba(z, y, self.h, self.Ok, self.Oh, self.OL, self.Oe, self.w)

    def initialize(self):
        y0 = np.array([0.0, self.h * 100.0])
        
        # 1. Solve ODE
        z_out, y_out = solve_ode_numba(
            y0, 
            0.0, 
            self.z_max, 
            self.rk_steps,
            self.h, self.Ok, self.Oh, self.OL, self.Oe, self.w
        )
        
        # 2. Extract H(z) on fine grid
        H_fine = y_out[:, 1]
        
        # 3. Compute Comoving Distance Integral on fine grid
        # r(z) = integral(1/H(z)) dz  (roughly, need to check units)
        # BaseCosmology uses DistIntegrand_a = 1./sqrt(RHSquared_a)/a^2
        # Da_z = int(DistIntegrand_a) from 1/(1+z) to 1
        # Let's stick to z-integration for simplicity if possible, or mapping a -> z
        
        # H(z) / H0 = E(z) = sqrt(RHSquared_a(a))
        # RHSquared_a(a) = (H(z)/H0)^2
        # DistIntegrand_a = 1/E(z) * (1/a^2) = 1/E(z) * (1+z)^2
        # But we integrate da. da = -dz/(1+z)^2.
        # Int(DistIntegrand_a da) = Int (1/E(z) * (1+z)^2 * -dz/(1+z)^2) 
        #                         = Int (-1/E(z) dz)
        # So r(z) = Int_0^z (1/E(z') dz')
        
        # E(z) = H_fine / (self.h * 100)
        # Avoid division by zero at z=0 (H should be close to H0)
        E_z = H_fine / (self.h * 100.0)
        integrand = 1.0 / E_z
        
        # Cumulative trapezoidal integration
        r_fine = sp_int.cumulative_trapezoid(integrand, z_out, initial=0)
        
        # 4. Create Interpolators
        # We need Da_z(z) and H(z)
        
        self.Da_interp = interp1d(z_out, r_fine, kind='cubic', fill_value='extrapolate')
        self.H_interp = interp1d(z_out, H_fine, kind='cubic', fill_value='extrapolate')
        
        # Also need phi for fine structure constant if needed
        phi_fine = y_out[:, 0]
        # We need phi(z) where z = 1/a - 1
        self.phiinterp = interp1d(z_out, phi_fine, kind='cubic', fill_value='extrapolate')
        
        return True

    def hub(self, z):
        return self.H_interp(z)

    # Override Da_z to use precomputed integral
    def Da_z(self, z):
        # BaseCosmology.Da_z computes the comoving distance r.
        # Then applies curvature correction.
        
        r = self.Da_interp(z)
        
        if self.Curv == 0:
            return r
        elif (self.Curv > 0):
            q = sp.sqrt(self.Curv)
            return sp.sinh(r*q)/(q)
        else:
            q = sp.sqrt(-self.Curv)
            return sp.sin(r*q)/(q)

    def fine_structure_constant(self, a):
        z = 1.0/a - 1.0
        phi_val = self.phiinterp(z)
        # Clamp phi to prevent overflow
        phi_clamped = np.clip(phi_val, -50, 50)
        return np.exp(2.0 * phi_clamped) - 1.0

    def RHSquared_a(self, a):
        # Used by growth factor calculation if standard logic called
        z = 1/a - 1.0
        H = self.hub(z)
        H0 = self.h * 100
        return (H / H0)**2
