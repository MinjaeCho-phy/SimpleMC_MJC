import numpy as np
from getdist import MCSamples
import getdist.plots as gdplt
import matplotlib.pyplot as plt

chain_file = "/home/mcho/SimpleMC_MJC/simplemc/chains/LCDM_phy_PantheonPlusSH0ES+FSC_nested_multi_1.txt"

# Columns: weight, -2logL, Om, Obh2, MB, h, PantheonPlusSH0ES_like, FSC_like, theory_prior
data = np.loadtxt(chain_file)
weights = data[:, 0]
param_data = data[:, 2:6]  # Om, Obh2, MB, h

# Keep only samples with non-zero weight
mask = weights > 0
param_data = param_data[mask]
weights = weights[mask]

names = ["Om", "Obh2", "MB", "h"]
labels = [r"\Omega_m", r"\Omega_b h^2", r"M_B", r"h"]

samples = MCSamples(
    samples=param_data,
    weights=weights,
    names=names,
    labels=labels,
    settings={"smooth_scale_2D": 0.4, "smooth_scale_1D": 0.4}
)

g = gdplt.get_subplot_plotter()
g.triangle_plot(
    samples,
    filled=True,
    contour_colors=["steelblue"],
    title_limit=1,
)

plt.suptitle(r"$\Lambda$CDM  —  PantheonPlus+SH0ES + FSC", fontsize=13, y=1.01)

output_path = "/home/mcho/SimpleMC_MJC/simplemc/chains/LCDM_phy_PantheonPlusSH0ES+FSC_cornerplot.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
