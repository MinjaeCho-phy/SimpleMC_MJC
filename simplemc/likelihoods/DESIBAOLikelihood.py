
import numpy as np
from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
import scipy.linalg as la
import numpy as sp



class DESIBAOLikelihood(BaseLikelihood):
    def __init__(self, name, values_filename, cov_filename, fidtheory):
        """
        This module calculates likelihood for the consensus DESI-BAO
        ----------
        name
        values_filename
        cov_filename
        fidtheory

        Returns
        -------

        """
        BaseLikelihood.__init__(self ,name)

        self.rd = fidtheory.rd
        print("Loading ", values_filename)
        da = sp.loadtxt(values_filename, usecols = (0 ,1 ,2))
        self.zs    = da[:, 0]
        self.DM_DH = da[:, 1]
        self.type  = da[:, 2]

        print("Loading covariance DESIBAO")
        cov = np.loadtxt(cov_filename)
        assert(len(cov) == len(self.zs))
        print("Adding marginalising constant")
        cov += 3**2
        self.icov = la.inv(cov)

    def loglike(self):
        # Using the fast vectorized Da_z 
        da_values = self.theory_.Da_z(self.zs)
        hi_values = self.theory_.Hinv_z(self.zs)
        pref = self.theory_.prefactor()

        tvec = np.zeros_like(self.zs)
  
        mask4 = (self.type == 4)
        if np.any(mask4):
            tvec[mask4] = pref * da_values[mask4]
        mask5 = (self.type == 5)
        if np.any(mask5):
            tvec[mask5] = pref * hi_values[mask5]
        mask3 = (self.type == 3)
        if np.any(mask3):
            dv = (da_values[mask3]**2 * self.zs[mask3] * hi_values[mask3])**(1./3.)
            tvec[mask3] = pref * dv
      
        delta = tvec - self.DM_DH
        # Using @ is shorthand for dot product in modern Python
        chi2 = delta @ self.icov @ delta

        return -0.5 * chi2