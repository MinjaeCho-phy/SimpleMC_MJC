from simplemc import logger
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par, dft_l_par
from simplemc.models.LCDMCosmology import LCDMCosmology

#from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import warnings
from numba import njit

@njit(cache=True)
def RHS_numba(z, y, h, Ok, Oh, OL, Oe, w, l):
    phi = y[0]
    H = y[1]
    
    if H <= 0.0:
        H = h * 100.0

    OeEvol1 = 6.0 * (w + 1.0) / (l + 2.0)
    # Clamp phi to prevent overflow in exp()
    if phi > 50.0:
        phi_clamped = 50.0
    elif phi < -50.0:
        phi_clamped = -50.0
    else:
        phi_clamped = phi
        
    OeEvol2 = np.exp(4.0 * phi_clamped / (l + 2.0)) 
    
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
def solve_ode_numba(y0, z_start, z_end, steps, h, Ok, Oh, OL, Oe, w, l):
    z_vals = np.linspace(z_start, z_end, steps)
    dz = z_vals[1] - z_vals[0]
    
    y_vals = np.zeros((steps, 2))
    y_vals[0] = y0
    
    y = y0.copy()
    
    for i in range(steps - 1):
        z = z_vals[i]
        
        k1 = RHS_numba(z, y, h, Ok, Oh, OL, Oe, w, l)
        k2 = RHS_numba(z + 0.5*dz, y + 0.5*dz*k1, h, Ok, Oh, OL, Oe, w, l)
        k3 = RHS_numba(z + 0.5*dz, y + 0.5*dz*k2, h, Ok, Oh, OL, Oe, w, l)
        k4 = RHS_numba(z + dz, y + dz*k3, h, Ok, Oh, OL, Oe, w, l)
        
        y = y + (dz/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y_vals[i+1] = y
        
    return z_vals, y_vals

class DFT1Cosmology(LCDMCosmology):
    def __init__(self, h = h_par.value, Ok = Ok_par.value, Oh = dft_Oh_par.value, OL = dft_OL_par.value, Oe = dft_Oe_par.value, w = dft_w_par.value, l = dft_l_par.value):
        """
        Parameter info

        h  : Hubble Constant / 100
        Ok : Density Parameter for curvature, \Omega_k
        Oh : Density Parameter for h-flux, \Omega_{\mathfrak{h}}
        OL : Density Parameter for Cosmological Constant, \Omega_{\Lambda}
        Oe : Density Parameter for general matter with EoS parameters (w,l), \Omega_{\epsilon}
        w  : First Equation of State Parameter,  w = (pressure)/("energy" density)
        l  : Second Equation of State Parameter
        """
        """ Parameter setting """
        self.h  = h
        self.Ok = Ok
        self.Oh = Oh
        self.Oe = Oe
        self.OL = OL
        self.w  = w
        self.l  = l

        self.parameters = [h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par, dft_l_par]
        
        # Increased steps for RK4 accuracy to match adaptive solver better if needed
        # 500 might be enough, but let's be safe with 1000 or keep 500 and verify
        self.rk_steps = 5000 
        self.z_values = np.linspace(0.0, 8.0, 500) # This is just for interpolation grid def if used elsewhere
        
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
            elif p.name == "l_dft":
                self.l = p.value
        self.initialize()
        return True

    def RHS(self, z, y):
        # Wrapper for backward compatibility if ever needed, though efficiency dictates using numba directly
        return RHS_numba(z, y, self.h, self.Ok, self.Oh, self.OL, self.Oe, self.w, self.l)

    def initialize(self):
        y0 = np.array([0.0, self.h * 100.0])
        
        # Use Numba solver
        # Note: z_values[0] is 0.0, z_values[-1] is 8.0
        z_out, y_out = solve_ode_numba(
            y0, 
            0.0, 
            8.0, 
            self.rk_steps,
            self.h, self.Ok, self.Oh, self.OL, self.Oe, self.w, self.l
        )
        
        
        # y_out shape is (steps, 2), where column 0 is phi, column 1 is H
        
        # To match the original behavior/accuracy (which used linear interpolation on 500 points),
        # we downsample our high-resolution RK4 solution to the original grid.
        # z_values has 500 points. rk_steps is 5000. 
        # So we take every (rk_steps / 500)th point?
        # Actually z_values is defined as linspace(0, 8, 500).
        # Our solver uses linspace(0, 8, rk_steps).
        # We need to ensure we pick the points corresponding to z_values.
        
        # We can just interpolate our fine grid onto the coarse grid, or if it aligns, slice it.
        # With 5000 steps, we have 5000 intervals, so 5001 points.
        # 500 points means 499 intervals?
        # Standard linspace(0, 8, 500) gives 500 points.
        # Solver with 5000 steps (intervals) gives 5001 points usually?
        # My solve_ode_numba returns `steps` points.
        # np.linspace(start, end, steps)
        # If I want to match linspace(0, 8, 500), I should probably just interpolate
        # the fine solution onto self.z_values to be robust.
        
        # Create temporary interpolator for the fine grid
        phi_fine = interp1d(z_out, y_out[:, 0], kind='cubic', fill_value='extrapolate')
        H_fine = interp1d(z_out, y_out[:, 1], kind='cubic', fill_value='extrapolate')
        
        # Evaluate on the standard coarse grid
        phi_coarse = phi_fine(self.z_values)
        H_coarse = H_fine(self.z_values)
        
        # Create the final interpolators used by the class (matching original "linear on 500 pts")
        self.phiinterp = interp1d(self.z_values, phi_coarse, fill_value='extrapolate')
        self.Hinterp = interp1d(self.z_values, H_coarse, fill_value='extrapolate')
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
