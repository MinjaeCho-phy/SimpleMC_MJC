
import time
import numpy as np
from simplemc.models.DFT1Cosmology import DFT1Cosmology
from simplemc.models.DFT2Cosmology import DFT2Cosmology
from simplemc.cosmo.paramDefs import h_par, Ok_par, dft_Oh_par, dft_OL_par, dft_Oe_par, dft_w_par

def test_accuracy():
    print("Initializing models...")
    
    # Common parameters
    h_val = 0.67
    Ok_val = 0.0
    Oh_val = 0.0
    OL_val = 0.68
    Oe_val = 0.32
    w_val = -1.0

    # Initialize DFT1 (Conservative)
    t0 = time.time()
    dft1 = DFT1Cosmology(h=h_val, Ok=Ok_val, Oh=Oh_val, OL=OL_val, Oe=Oe_val, w=w_val)
    t1 = time.time()
    print(f"DFT1 initialized in {t1-t0:.4f}s")
    
    # Initialize DFT2 (Optimized)
    t0 = time.time()
    dft2 = DFT2Cosmology(h=h_val, Ok=Ok_val, Oh=Oh_val, OL=OL_val, Oe=Oe_val, w=w_val)
    t1 = time.time()
    print(f"DFT2 initialized in {t1-t0:.4f}s")
    
    # Test Points
    z_list = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 8.0]
    
    print("\nComparing Da_z(z) values:")
    print(f"{'z':<5} | {'DFT1 (quad)':<15} | {'DFT2 (interp)':<15} | {'Diff (%)':<10}")
    print("-" * 55)
    
    for z in z_list:
        # DFT1
        t_start = time.time()
        val1 = dft1.Da_z(z)
        t_dft1 = time.time() - t_start
        
        # DFT2
        t_start = time.time()
        val2 = dft2.Da_z(z)
        t_dft2 = time.time() - t_start
        
        diff_pct = 100 * abs(val1 - val2) / val1 if val1 != 0 else 0
        
        print(f"{z:<5} | {val1:<15.6f} | {val2:<15.6f} | {diff_pct:<10.4f}")
        # print(f"    Time: DFT1={t_dft1:.6f}s, DFT2={t_dft2:.6f}s")

def test_performance():
    print("\nPerformance Benchmark (1000 calls):")
    
    # Re-init
    dft1 = DFT1Cosmology()
    dft2 = DFT2Cosmology()
    
    z_test = 1.0
    
    # DFT1
    start = time.time()
    for _ in range(10): # Conservative count as it is slow
        dft1.Da_z(z_test)
    end = time.time()
    t_dft1 = (end - start) / 10
    print(f"DFT1 avg time per call: {t_dft1*1000:.4f} ms")
    
    # DFT2
    start = time.time()
    for _ in range(1000):
        dft2.Da_z(z_test)
    end = time.time()
    t_dft2 = (end - start) / 1000
    print(f"DFT2 avg time per call: {t_dft2*1000:.4f} ms")
    
    speedup = t_dft1 / t_dft2 if t_dft2 > 0 else 0
    print(f"Speedup Factor: {speedup:.1f}x")

if __name__ == "__main__":
    test_accuracy()
    test_performance()
