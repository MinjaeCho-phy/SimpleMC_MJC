"""Sequential DFTmr nested runs for the 20260520 batch.

DFTmr = "full matter universe on the DFT critical line (l = 3w-1)":
two species, dust (w=1, l=2 -> (1+z)^3 e^phi) + radiation (w=1/3, l=0 ->
(1+z)^4 e^{2phi}), on the Ok+Oh DFT background. Free: h, Ok, Oh, Oem, Oer,
alpha_fsc. Ok uses its own WIDE prior (dftmr_Ok_par, 0..1.05); the model
recollapses for Ok~1 + appreciable matter and rejects those points via
prior_loglike = -1e12 (recursion-safe).

Same four dataset combinations and chains dir (simplemc/chains/20260520) as
the DFTvac batch, for a direct DFTvac-vs-DFTmr comparison; output roots are
DFTmr_phy_<datasets>_... so they never collide with the DFTvac chains:

    1. HD       + Union3       + FSC
    2. HD       + DESDovekie   + FSC
    3. HD       + FastPantheon + FSC
    4. RiessH0  + FastPantheon + FSC

Each run writes <outputpath>_prior.txt (ini [custom]/[nested] + free-parameter
table) just like the other 20260520 scripts.
"""
import os
import sys
import time
import traceback
import configparser
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from simplemc.DriverMC import DriverMC

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(SCRIPT_DIR, "configs_20260520")

CONFIG_FILES = [
    "cfg_DFTmr_HD_Union3_FSC.ini",
    "cfg_DFTmr_HD_DESDovekie_FSC.ini",
    "cfg_DFTmr_HD_FastPantheon_FSC.ini",
    "cfg_DFTmr_RiessH0_FastPantheon_FSC.ini",
]


def stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dump_prior_info(analyzer, cfg_path):
    """Write the prior/sampler configuration for this run to <outputpath>_prior.txt.

    Captures: ini source, [custom]/[nested] sections verbatim, and the free
    parameter table (name, initial value, error, bounds, latex) read from
    analyzer.pars_info — the actual params the sampler will vary.
    """
    out_file = f"{analyzer.outputpath}_prior.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    cp = configparser.ConfigParser()
    cp.read(cfg_path)

    def section_dump(name):
        if not cp.has_section(name):
            return f"[{name}] (not present)\n"
        lines = [f"[{name}]"]
        for k, v in cp.items(name):
            lines.append(f"{k} = {v}")
        return "\n".join(lines) + "\n"

    lines = []
    lines.append(f"# Prior / sampler configuration snapshot")
    lines.append(f"# Written: {stamp()}")
    lines.append(f"# Source ini : {cfg_path}")
    lines.append(f"# Output root: {analyzer.outputpath}")
    lines.append(f"# Model      : {analyzer.model}")
    lines.append("")
    lines.append("## ini [custom]")
    lines.append(section_dump("custom"))
    lines.append("## ini [nested]")
    lines.append(section_dump("nested"))
    lines.append("## Free parameters (analyzer.pars_info)")
    lines.append(f"{'name':<14} {'init':>14} {'error':>12} {'low':>14} {'high':>14}   latex")
    lines.append("-" * 90)
    for p in analyzer.pars_info:
        low, high = p.bounds[0], p.bounds[1]
        lines.append(
            f"{p.name:<14} {p.value:>14.6g} {p.error:>12.6g} "
            f"{low:>14.6g} {high:>14.6g}   {p.Ltxname}"
        )
    lines.append("")

    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    print(f"[{stamp()}] prior info -> {out_file}", flush=True)


def main():
    total = len(CONFIG_FILES)
    overall_t0 = time.time()
    print(f"[{stamp()}] === DFTmr sequential run for 20260520: {total} configs ===", flush=True)

    results = []
    for i, name in enumerate(CONFIG_FILES, 1):
        cfg = os.path.join(CFG_DIR, name)
        print("\n" + "=" * 70, flush=True)
        print(f"[{stamp()}] ({i}/{total}) {name}", flush=True)
        print("=" * 70, flush=True)

        if not os.path.exists(cfg):
            print(f"MISSING: {cfg}", flush=True)
            results.append((name, "missing", 0.0))
            continue

        t0 = time.time()
        try:
            analyzer = DriverMC(iniFile=cfg)
            dump_prior_info(analyzer, cfg)
            analyzer.executer()
            dt = time.time() - t0
            print(f"[{stamp()}] OK  {name}  (elapsed: {dt/60:.2f} min)", flush=True)
            results.append((name, "ok", dt))
        except Exception as e:
            dt = time.time() - t0
            print(f"[{stamp()}] FAIL {name}: {e}", flush=True)
            traceback.print_exc()
            results.append((name, "fail", dt))

    overall_dt = time.time() - overall_t0
    print("\n" + "=" * 70, flush=True)
    print(f"[{stamp()}] === DFTmr runs finished. Total elapsed: {overall_dt/60:.2f} min ===", flush=True)
    print("Summary:", flush=True)
    for name, status, dt in results:
        print(f"  [{status:>7}] {name}  ({dt/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
