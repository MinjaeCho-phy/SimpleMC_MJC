"""Run a single DFTmr config. Usage: python run_one_dftmr.py <cfg.ini>

NOTE: the `if __name__ == "__main__"` guard is REQUIRED. DriverMC.executer()
starts a multiprocessing pool; on macOS (spawn start method) the child
re-imports this module, so without the guard the child re-runs executer() and
recursively spawns until it raises the multiprocessing "freeze_support"
RuntimeError. Each config already parallelises internally via that pool, so run
configs sequentially (or via run_sequence_20260520_dftmr.py) rather than
launching several copies of this script at once.
"""
import sys
from run_sequence_20260520_dftmr import dump_prior_info  # sets OMP threads + imports DriverMC
from simplemc.DriverMC import DriverMC


def main():
    cfg = sys.argv[1]
    a = DriverMC(iniFile=cfg)
    dump_prior_info(a, cfg)
    a.executer()
    print("ONE_DONE", cfg, flush=True)


if __name__ == "__main__":
    main()
