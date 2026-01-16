import sys
import os
from simplemc.DriverMC import DriverMC

# Set environment variables for single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def run_remaining():
    config_file = 'baseConfig4.ini'
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        return

    print(f"\n{'='*50}")
    print(f"Running configuration: {config_file}")
    print(f"{'='*50}\n")
    
    try:
        analyzer = DriverMC(iniFile=config_file)
        analyzer.executer()
        print(f"\nSuccessfully completed: {config_file}")
    except Exception as e:
        print(f"\nError running {config_file}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_remaining()
