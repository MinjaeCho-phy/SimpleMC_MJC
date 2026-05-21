"""DFTmr 4-dataset analysis: per-dataset corner + overlay + fitness table vs LCDM.

Reads DFTmr chains from 20260520 and LCDM baselines from 20260514. Matches
DFTmr<->LCDM by sorted dataset components.
"""
import os, re, glob, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import MCSamples, plots as gdplots
warnings.filterwarnings("ignore")

DFT_DIR = "/Users/mcho/Minjae/SimpleMC_MJC/simplemc/chains/20260520"
LCDM_DIR = "/Users/mcho/Minjae/SimpleMC_MJC/simplemc/chains/20260514"
OUT = "/Users/mcho/Minjae/SimpleMC_MJC/postprocess_20260520"
os.makedirs(OUT, exist_ok=True)

DATASETS = ["HD+Union3+FSC", "HD+DESDovekie+FSC", "HD+FastPantheon+FSC", "RiessH0+FastPantheon+FSC"]
SCIENCE = ["h", "Ok", "Oh", "Oem", "Oer"]
N_DATA = {"HD+Union3+FSC": 31+22+199, "HD+DESDovekie+FSC": 31+20+199,
          "HD+FastPantheon+FSC": 31+1590+199, "RiessH0+FastPantheon+FSC": 1+1590+199}
LATEX = {"h":"h","Ok":r"\Omega_k","Oh":r"\Omega_h\times10^{4}","Oem":r"\Omega_{em}","Oer":r"\Omega_{er}"}

def norm(ds): return tuple(sorted(ds.split("+")))
def load_paramnames(path):
    names, likes = [], []
    for ln in open(path):
        ln = ln.strip()
        if not ln: continue
        nm = re.split(r"\s+", ln, 1)[0]
        if nm.endswith("_like"): likes.append(nm)
        elif nm.startswith("theory_prior"): likes.append(nm)
        else: names.append(nm)
    return names, likes
def load(prefix):
    names, likes = load_paramnames(prefix + ".paramnames")
    d = np.loadtxt(prefix + "_1.txt")
    w = d[:, 0]; nll = d[:, 1]
    P = d[:, 2:2+len(names)]; L = d[:, 2+len(names): 2+len(names)+len(likes)]
    m = (w > 0) & np.all(np.isfinite(P), axis=1) & np.isfinite(w)
    return names, likes, P[m], L[m], w[m]/w[m].max(), nll[m]
def parse_sum(path, key):
    if not path or not os.path.exists(path): return None
    for ln in open(path):
        if ln.startswith(key+":"): return ln.split(":",1)[1].strip()
    return None
def wms(x, w):
    mu = np.average(x, weights=w); return mu, np.sqrt(np.average((x-mu)**2, weights=w))

lcdm = {}
for s in glob.glob(os.path.join(LCDM_DIR, "LCDM_phy_*_nested_multi_Summary.txt")):
    ds = re.search(r"LCDM_phy_(.+?)_nested_multi", os.path.basename(s)).group(1)
    lcdm[norm(ds)] = s

rows, entries = [], []
for ds in DATASETS:
    pre = os.path.join(DFT_DIR, f"DFTmr_phy_{ds}_nested_multi")
    if not os.path.exists(pre + "_1.txt"):
        print(f"  ! missing DFTmr chain for {ds}"); continue
    names, likes, P, L, w, nll = load(pre)
    ib = int(np.argmin(nll))
    chi2 = {likes[j].replace("_like","").strip(): 2*(-L[ib, j]) for j in range(len(likes)) if not likes[j].startswith("theory")}
    logz = parse_sum(pre+"_Summary.txt", "logz"); z_d = float(logz.split("+/-")[0]) if logz else np.nan
    lz_l = parse_sum(lcdm.get(norm(ds)), "logz"); mx_l = parse_sum(lcdm.get(norm(ds)), "maxlike")
    z_l = float(lz_l.split("+/-")[0]) if lz_l else np.nan
    chi2_dft = 2*nll[ib]; chi2_lcdm = 2*float(mx_l) if mx_l else np.nan
    cons = {n: wms(P[:, names.index(n)], w) for n in names}
    rows.append(dict(ds=ds, chi2=chi2, chi2_dft=chi2_dft, chi2_lcdm=chi2_lcdm,
                     z_d=z_d, z_l=z_l, cons=cons, k=len(names), N=N_DATA.get(ds, np.nan)))
    entries.append(dict(ds=ds, names=names, P=P, w=w, nll=nll))

# per-dataset corner
for e in entries:
    names=[n for n in SCIENCE if n in e["names"]]; cols=[e["names"].index(n) for n in names]
    samp=e["P"][:, cols].copy(); labs=[]
    for j,n in enumerate(names):
        if n=="Oh": samp[:,j]*=1e4
        labs.append(LATEX[n])
    lims={}
    for j,n in enumerate(names):
        mu,sd=wms(samp[:,j], e["w"])
        if sd>0: lims[n]=(mu-4*sd, mu+4*sd)
    mcs=MCSamples(samples=samp, weights=e["w"], loglikes=e["nll"], names=names, labels=labs,
                  label=f"DFTmr {e['ds']}", settings={"smooth_scale_2D":0.3,"smooth_scale_1D":0.3,"ignore_rows":0})
    g=gdplots.get_subplot_plotter(width_inch=8); g.settings.lab_fontsize=14; g.settings.axes_fontsize=10
    g.triangle_plot([mcs], names, filled=True, contour_colors=["#d62728"], title_limit=1, param_limits=lims)
    g.fig.suptitle(f"DFTmr · {e['ds']}", y=1.02, fontsize=12)
    p=os.path.join(OUT, f"DFTmr_corner_{e['ds'].replace('+','-')}.png")
    g.fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(g.fig); print("corner ->", p)

# overlay
COLORS=["#d62728","#1f77b4","#2ca02c","#9467bd"]; mcs_list=[]
for e in entries:
    names=[n for n in SCIENCE if n in e["names"]]; cols=[e["names"].index(n) for n in names]
    samp=e["P"][:, cols].copy()
    for j,n in enumerate(names):
        if n=="Oh": samp[:,j]*=1e4
    mcs_list.append(MCSamples(samples=samp, weights=e["w"], loglikes=e["nll"], names=names,
                              labels=[LATEX[n] for n in names], label=e["ds"],
                              settings={"smooth_scale_2D":0.3,"smooth_scale_1D":0.3,"ignore_rows":0}))
if mcs_list:
    names=[n for n in SCIENCE if n in entries[0]["names"]]
    g=gdplots.get_subplot_plotter(width_inch=10); g.settings.lab_fontsize=14; g.settings.axes_fontsize=10; g.settings.legend_fontsize=11
    g.triangle_plot(mcs_list, names, filled=[True]+[False]*(len(mcs_list)-1),
                    contour_colors=COLORS[:len(mcs_list)], legend_labels=[e["ds"] for e in entries])
    g.fig.suptitle("DFTmr · 4-dataset overlay", y=1.02, fontsize=13)
    p=os.path.join(OUT,"DFTmr_corner_overlay.png"); g.fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(g.fig); print("overlay ->", p)

# fitness table
def f(x,p=3): return "nan" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.{p}f}"
L=[]
L.append("\n# DFTmr fitness comparison (corrected inSqrt, vs LCDM)\n")
L.append("| dataset | -logL_min | chi2_best | chi2(LCDM) | dchi2 | lnZ(DFTmr) | lnZ(LCDM) | dlnZ | dAIC | dBIC |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    dchi2=r["chi2_dft"]-r["chi2_lcdm"]; dlnz=r["z_d"]-r["z_l"]
    dAIC=(r["chi2_dft"]+2*r["k"])-(r["chi2_lcdm"]+2*4); dBIC=(r["chi2_dft"]+r["k"]*np.log(r["N"]))-(r["chi2_lcdm"]+4*np.log(r["N"]))
    L.append(f"| {r['ds']} | {f(r['chi2_dft']/2)} | {f(r['chi2_dft'],2)} | {f(r['chi2_lcdm'],2)} | {dchi2:+.2f} | {f(r['z_d'],2)} | {f(r['z_l'],2)} | {dlnz:+.2f} | {dAIC:+.2f} | {dBIC:+.2f} |")
L.append("\n## per-likelihood chi2 at best fit\n| dataset | HD/H0 | SNe | FSC |\n|---|---|---|---|")
for r in rows:
    sk=[k for k in r["chi2"] if k not in ("HD","FSC") and not k.startswith("Hubble")]
    sne=r["chi2"].get(sk[0]) if sk else np.nan
    h0=[k for k in r["chi2"] if k.startswith("Hubble")]
    hd=f(r['chi2'].get('HD'),1) if 'HD' in r['chi2'] else (f"H0:{f(r['chi2'][h0[0]],1)}" if h0 else "-")
    L.append(f"| {r['ds']} | {hd} | {f(sne,1)} ({sk[0] if sk else '?'}) | {f(r['chi2'].get('FSC'),1)} |")
L.append("\n## parameter constraints (mean +/- std)\n| dataset | h | Ok | Oem | Oer |\n|---|---|---|---|---|")
for r in rows:
    c=r["cons"]; g=lambda n: f"{c[n][0]:.4g}±{c[n][1]:.2g}" if n in c else "-"
    L.append(f"| {r['ds']} | {g('h')} | {g('Ok')} | {g('Oem')} | {g('Oer')} |")
tab="\n".join(L); print(tab)
open(os.path.join(OUT,"DFTmr_fitness_table.md"),"w").write(tab+"\n")
print("\ntable ->", os.path.join(OUT,"DFTmr_fitness_table.md")); print("ALL_DONE")
