import os
import numpy as np
import pandas as pd
from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
from simplemc import cdir

class DESDovekieLikelihood(BaseLikelihood):
    def __init__(self, name, values_filename, cov_inv_filename):
        self.name_ = name
        BaseLikelihood.__init__(self, name)
        
        # Load and Filter
        full_df = pd.read_csv(values_filename)
        mask_logic = (full_df['IDSURVEY'] == 10) & (full_df['zHD'] > 0.01)
        mask = mask_logic.values
        
        df = full_df[mask].reset_index(drop=True)
        self.zcmb = df['zHD'].values
        self.zhel = df['zHEL'].values
        self.mag_obs = df['MU'].values
        self.N = len(self.mag_obs)
        
        # Load Inverse Matrix
        with np.load(cov_inv_filename) as d:
            packed_data = d['cov'] 
            N_full = 1820
            full_matrix = np.zeros((N_full, N_full))
            full_matrix[np.triu_indices(N_full)] = packed_data
            full_matrix = full_matrix + full_matrix.T - np.diag(np.diag(full_matrix))
            self.icov = full_matrix[np.ix_(mask, mask)]
            
        # Marginalization Vectors
        self.ones = np.ones(self.N)
        self.icov_ones = self.icov @ self.ones
        self.S3 = np.dot(self.ones, self.icov_ones)

    def loglike(self):
        # 1. Comoving distance at z_cmb
        r = self.theory_.Da_z(self.zcmb)
        
        # 2. Luminosity Distance dL = (1 + z_hel) * r
        dl = (1.0 + self.zhel) * r
        
        # 3. Distance Modulus (Shape only)
        mu_theory = 5.0 * np.log10(dl)
        
        # 4. Residuals
        delta = self.mag_obs - mu_theory
        
        # 5. S-Method Marginalization (Equivalent to M-shift)
        icov_delta = self.icov @ delta
        S1 = np.dot(delta, icov_delta)
        S2 = np.dot(delta, self.icov_ones)
        
        chi2 = S1 - (S2**2 / self.S3)
            
        return -0.5 * chi2

class DESDovekie(DESDovekieLikelihood):
    def __init__(self):
        data_path = os.path.join(cdir, "data", "DES-Dovekie_HD.csv")
        cov_path = os.path.join(cdir, "data", "covtot_inv_000.npz")
        DESDovekieLikelihood.__init__(self, "DESDovekie", data_path, cov_path)
