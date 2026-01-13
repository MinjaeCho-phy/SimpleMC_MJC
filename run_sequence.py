import sys
import os
from simplemc.DriverMC import DriverMC

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def run_sequence():
    config_files = ['baseConfig1.ini', 'baseConfig2.ini', 'baseConfig3.ini', 'baseConfig4.ini']
    
    print(f"Starting sequential run for: {config_files}")
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"Error: Configuration file '{config_file}' not found.")
            continue
            
        print(f"\n{'='*50}")
        print(f"Running configuration: {config_file}")
        print(f"{'='*50}\n")
        
        try:
            analyzer = DriverMC(iniFile=config_file)
            analyzer.executer()
            # analyzer.postprocess() # Uncomment if postprocessing is needed immediately
            print(f"\nSuccessfully completed: {config_file}")
        except Exception as e:
            print(f"\nError running {config_file}: {e}")
            # Decide whether to continue or stop. 
            # For a batch job, continuing might be better, or stopping if they are dependent.
            # Here we print the error and continue to the next one, but re-raising is also an option.
            # Let's print full traceback for debugging
            import traceback
            traceback.print_exc()
            
            print(f"Skipping {config_file} due to error.")

    print(f"\n{'='*50}")
    print("All requested runs completed.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    run_sequence()
