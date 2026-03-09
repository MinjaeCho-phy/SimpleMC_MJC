from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
import numpy as np
import scipy.linalg as la
from scipy.interpolate import interp1d
from simplemc import cdir
import pandas as pd

class FastPantheonLikelihood(BaseLikelihood):
    def __init__(self, name, values_filename, cov_filename, ninterp=150):
        self.name_ = name
        BaseLikelihood.__init__(self, name)

        #Efficient Data Loading
        print("Loading", values_filename)
        data = pd.read_csv(values_filename, delim_whitespace=True)
        self.origlen = len(data)
        self.ww = (data['zHD'] > 0.01).values
        self.zcmb = data['zHD'][self.ww].values
        self.zhelio = data['zHEL'][self.ww].values
        self.mag = data['m_b_corr'][self.ww].values
        self.N = len(self.mag)

        print("Loading covariance from {}".format(cov_filename))
        # Loading the entire file at once into a 1D array, then reshaping to 2D
        # This replaces the nested for-loop which is the main bottleneck.
        full_cov_raw = np.fromfile(cov_filename, sep='\n')
        full_cov = full_cov_raw.reshape(self.origlen, self.origlen)

        self.cov = full_cov[np.ix_(self.ww, self.ww)]

        #  Cholesky Decomposition
        self.L = la.cholesky(self.cov, lower=True)
        self.log_det_cov = 2 * np.sum(np.log(np.diag(self.L)))

        self.xdiag = 1 / self.cov.diagonal()
        self.zmin = self.zcmb.min()
        self.zmax = self.zcmb.max()
        self.zmaxi = 1.1

        print("Pantheon SN: zmin=%f zmax=%f N=%i" % (self.zmin, self.zmax, self.N))
        self.zinter = np.linspace(1e-3, self.zmaxi, ninterp)

    def loglike(self):
        r_values = self.theory_.Da_z(self.zcmb)
        dist_mod = 5.0 * np.log10(r_values * (1.0 + self.zcmb)) + 25.0
        tvec = self.mag - dist_mod
        tvec -= (tvec * self.xdiag).sum() / self.xdiag.sum()
        y = la.solve_triangular(self.L, tvec, lower=True)
        chi2 = np.dot(y, y)
        return -0.5 * chi2

class FastPantheon(FastPantheonLikelihood):
    def __init__(self):
        FastPantheonLikelihood.__init__(self, "FastPantheon",
                                      cdir + "/data/pantheon+_lcparam_full_long_zhel.txt",
                                      cdir + "/data/pantheon+_sys_full_long.txt")