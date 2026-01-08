from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
from scipy import constants
import scipy.linalg as la
import numpy as np
from simplemc import cdir

class FineStructureConstantLikelihood(BaseLikelihood):
    def __init__(self, name, values_filename):

        """
        This module calculates likelihood for the DESI 2024 data release
        DESI 2024 III: Baryon Acoustic Oscillations from Galaxies and Quasars
        https://arxiv.org/abs/2404.03000
        Parameters
        ----------
        name
        values_filename
        cov_filename
        fidtheory

        Returns
        -------

        """
        BaseLikelihood.__init__(self,name)
        print("Loading ", values_filename)
        da = np.loadtxt(values_filename, usecols = (0,1,2))
        self.zs    = da[:, 0]
        self.fsc_data = da[:, 1]
        self.sigma = da[:, 2]

        print("Loading covariance FSC")
        cov = np.diag(np.square(self.sigma))
        assert(len(cov) == len(self.zs))
        vals, vecs = la.eig(cov)
        vals = sorted(np.real(vals))
        print("Eigenvalues of cov matrix:", vals[0:3],'...',vals[-1])
        print("Adding marginalising constant")
        #cov += 3**2
        self.icov  = la.inv(cov)



    def loglike(self):

        tvec = []
        for i, z in enumerate(self.zs):
            #tvec.append(self.theory_.fine_structure_constant(1.0/(1+z))) 
            tvec.append(100000*self.theory_.fine_structure_constant(1.0/(1+z)))
        tvec = np.array(tvec)
        tvec += 0
        delta = tvec - self.fsc_data
        return -np.dot(delta, np.dot(self.icov, delta))/2.0


class FSC(FineStructureConstantLikelihood):
    def __init__(self):
        #fidTheory = LCDMCosmology(obh2, Om, h, mnu)
        FineStructureConstantLikelihood.__init__(self, "FSC", cdir+"/data/fine_structure.dat")
