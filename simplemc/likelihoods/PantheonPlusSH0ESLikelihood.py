from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
import numpy as np
import scipy.linalg as la
from scipy.interpolate import interp1d
from simplemc import cdir
import pandas as pd
from simplemc.cosmo.paramDefs import MB_par

class PantheonPlusSH0ESLikelihood(BaseLikelihood):
    def __init__(self, name, values_filename, cov_filename):
        self.name_ = name
        BaseLikelihood.__init__(self, name)

        print("Loading", values_filename)
        data = pd.read_csv(values_filename, delim_whitespace=True)
        self.origlen = len(data)

        # --- SH0ES Logic: Include calibrators (z < 0.01) ---
        # Cobaya uses (zcmb > 0.01) | is_calibrator
        self.ww = ((data['zHD'] > 0.01) | (data['IS_CALIBRATOR'] == 1)).values
        
        self.zcmb = data['zHD'][self.ww].values
        self.mag = data['m_b_corr'][self.ww].values
        self.is_calibrator = data['IS_CALIBRATOR'][self.ww].values.astype(bool)
        self.ceph_dist = data['CEPH_DIST'][self.ww].values
        
        self.N = len(self.mag)

        # --- Load Covariance ---
        print("Loading covariance...")
        full_cov_raw = np.fromfile(cov_filename, sep='\n')
        full_cov = full_cov_raw.reshape(self.origlen, self.origlen)
        self.cov = full_cov[np.ix_(self.ww, self.ww)]

        # --- Cholesky ---
        self.L = la.cholesky(self.cov, lower=True)
        self.MB = MB_par.value
        self.varyMB = True
        print("Pantheon+SH0ES: N=%i (Calibrators: %i)" % (self.N, np.sum(self.is_calibrator)))

    def setVaryMB(self, T=True):
        self.varyMB = T

    def freeParameters(self):
        """Tell the sampler that MB is a free parameter if varyMB is True"""
        l = []
        if self.varyMB:
            MB_par.setValue(self.MB)
            l.append(MB_par)
        return l

    def updateParams(self, pars):
        """Update the value of MB at each MCMC step"""
        for p in pars:
            if p.name == "MB":
                self.MB = p.value
        return True


    def loglike(self):
        # Pass the current value of self.MB to the calculation
        # Note: We now use self.MB instead of passing it as an argument
        #from simplemc.cosmo.paramDefs import MB_par
        #self.MB = MB_par.value
        r_values = self.theory_.Da_z(self.zcmb)
        
        #theor_mu = 5.0 * np.log10(r_values * (1.0 + self.zcmb) + 1e-10) + 25.0        
        hub_dist = self.theory_.c_ / (self.theory_.h * 100.0)
        theor_mu = 5.0 * np.log10(r_values * (1.0 + self.zcmb) * hub_dist + 1e-10) + 25.0
        # SH0ES Calibration
        theor_mu[self.is_calibrator] = self.ceph_dist[self.is_calibrator]

        # Residuals using the current step's MB
        #tvec = self.mag - (self.MB + theor_mu)
        tvec = self.mag - (self.theory_.MB_pp(1.0)+theor_mu)

        y = la.solve_triangular(self.L, tvec, lower=True)
        chi2 = np.dot(y, y)
        
        return -0.5 * chi2

class PantheonPlusSH0ES(PantheonPlusSH0ESLikelihood):
    def __init__(self):
        PantheonPlusSH0ESLikelihood.__init__(self, "PantheonPlusSH0ES", 
                                      cdir + "/data/pantheon+_lcparam_full_long_zhel.txt",
                                      cdir + "/data/pantheon+_sys_full_long.txt")
