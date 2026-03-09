from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
import numpy as np
import scipy.linalg as la
from scipy.interpolate import interp1d
from simplemc.setup_logger import cdir


class UNION3Likelihood(BaseLikelihood):
    def __init__(self, name, values_filename, cov_filename, ninterp=150):
        """
        This module calculates likelihood for UNION3 datasets.
        Parameters
        ----------
        name: str
            name of the likelihood
        values_filename: str
            directory and name of the data file
        cov_filename: str
            directory and name of the covariance matrix file
        ninterp: int
        """
        # first read data file
        self.name_ = name
        BaseLikelihood.__init__(self, name)
        print("Loading", values_filename)
        da = np.loadtxt(values_filename, skiprows=1, usecols=(1, 2, 4))
        self.zcmb = da[:, 0]
        self.zhelio = da[:, 1]
        self.mag = da[:, 2]
        self.N = len(self.mag)
        self.syscov = np.loadtxt(cov_filename, skiprows=1).reshape((self.N, self.N))
        self.cov = np.copy(self.syscov)
#        self.cov[np.diag_indices_from(self.cov)] += self.dmag**2
        self.xdiag = 1/self.cov.diagonal()  # diagonal before marginalising constant
        # add marginalising over a constant
        self.cov += 3**2
        self.zmin = self.zcmb.min()
        self.zmax = self.zcmb.max()
        self.zmaxi = 1.1 ## we interpolate to 1.1 beyond that exact calc
        print("Union3 : zmin=%f zmax=%f N=%i" % (self.zmin, self.zmax, self.N))
        self.zinter = np.linspace(1e-3, self.zmaxi, ninterp)
        self.icov = la.inv(self.cov)

    def loglike(self):
        dist = self.theory_.distance_modulus(self.zcmb)
        tvec = self.mag - dist
        tvec -= (tvec * self.xdiag).sum() / (self.xdiag.sum())
        chi2 = tvec @ self.icov @ tvec
        return -0.5 * chi2


class UNION3(UNION3Likelihood):
    """
    Likelihood to binned UNION3 SNIa compilation.
    """
    def __init__(self):
        UNION3Likelihood.__init__(self, "Union3", cdir+"/data/lcparam_full.txt",
                                      cdir+"/data/mag_covmat.txt")


