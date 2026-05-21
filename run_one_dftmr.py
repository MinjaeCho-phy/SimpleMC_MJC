"""Run a single DFTmr config (for parallel re-run). Usage: python run_one_dftmr.py <cfg.ini>"""
import sys
from run_sequence_20260520_dftmr import dump_prior_info  # also sets OMP threads + imports DriverMC
from simplemc.DriverMC import DriverMC

cfg = sys.argv[1]
a = DriverMC(iniFile=cfg)
dump_prior_info(a, cfg)
a.executer()
print("ONE_DONE", cfg, flush=True)
