
import os
import time
import cProfile
import pstats
from simplemc.models.DFT1Cosmology import DFT1Cosmology
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par

def profile_dft1():
    print("Initializing Model...")
    model = DFT1Cosmology()
    
    # Define some parameters to update
    # We will invoke updateParams multiple times to simulate sampling
    
    start_time = time.time()
    n_iterations = 20
    
    print(f"Running {n_iterations} iterations...")
    
    for i in range(n_iterations):
        # vary parameters slightly
        h_val = 0.7 + 0.01 * (i % 5)
        Ok_val = 0.0 + 0.01 * (i % 3)
        Oh_val = 0.0 + 0.01 * (i % 3)
        Oe_val = 0.0 + 0.01 * (i % 3)
        OL_val = 0.7 - Ok_val - Oh_val - Oe_val
        w_val = -1.0 + 0.1 * (i % 3)

        pars = [
            Parameter(h_par.name, h_val, err=0.0, bounds=h_par.bounds, Ltxname=h_par.Ltxname),
            Parameter(Ok_par.name, Ok_val, err=0.0, bounds=Ok_par.bounds, Ltxname=Ok_par.Ltxname),
            Parameter(dft_Oh_par.name, Oh_val, err=0.0, bounds=dft_Oh_par.bounds, Ltxname=dft_Oh_par.Ltxname),
            Parameter(dft_OL_par.name, OL_val, err=0.0, bounds=dft_OL_par.bounds, Ltxname=dft_OL_par.Ltxname),
            Parameter(dft_Oe_par.name, Oe_val, err=0.0, bounds=dft_Oe_par.bounds, Ltxname=dft_Oe_par.Ltxname),
            Parameter(dft_w_par.name, w_val, err=0.0, bounds=dft_w_par.bounds, Ltxname=dft_w_par.Ltxname)
        ]
        
        model.updateParams(pars)
        
        # Access some derived quantities to ensure calculation is done
        _ = model.hub(0.5)
        _ = model.fine_structure_constant(0.5)
        
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f} seconds")
    print(f"Time per iteration: {(end_time - start_time) / n_iterations:.4f} seconds")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profile_dft1()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
