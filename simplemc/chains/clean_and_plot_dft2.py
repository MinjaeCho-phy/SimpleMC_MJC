import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import re
import os

# Configuration
chain_file = '/home/minjae/.gemini/antigravity/scratch/DFT_cosmology/simplemc/chains/DFT2_temp_phy_Minjae+Pantheon+FSC_nested_multi_1.txt'
param_file = '/home/minjae/.gemini/antigravity/scratch/DFT_cosmology/simplemc/chains/DFT2_temp_phy_Minjae+Pantheon+FSC_nested_multi.paramnames'
output_plot = 'DFT2_temp_Minjae_corner.png'

# Parameter names to plot (indices based on paramnames file, starting 0 for first param)
# Param file content: h, Ok, Oh, OL, Oe, w_dft, ...
# Mapped to columns in chain file.
# Chain file cols: Weight, -LogLike, param1, param2, ...
target_params = ['h', 'Ok', 'Oh', 'Oe', 'OL', 'w_dft']

def clean_value(val_str):
    """Parse string that might be a complex number '(real+imagj)' or float."""
    val_str = val_str.strip()
    # Remove parens
    if val_str.startswith('(') and val_str.endswith(')'):
        val_str = val_str[1:-1]
    
    try:
        # Try converting to complex
        c = complex(val_str)
        return c.real # Return real part
    except ValueError:
        try:
            return float(val_str)
        except ValueError:
            return np.nan

def read_and_clean_chain(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split() # Split by whitespace
            # Complex numbers like (1+2j) might be split if there are spaces? 
            # The file preview showed "(8.39...+126...j)". No spaces inside parens shown in `head` output usually unless specified.
            # But let's handle the case just in point.
            # Actually, `head` output: 0.0 (8.3...+12...j) 0.57...
            # Standard split() should work if no spaces inside parens.
            
            clean_row = [clean_value(p) for p in parts]
            data.append(clean_row)
    return np.array(data)

# Read param names
with open(param_file, 'r') as f:
    lines = [line for line in f if line.strip()]
    param_names_raw = [line.split()[0] for line in lines]
    labels_raw = [line.split()[1] if len(line.split()) > 1 else line.split()[0] for line in lines]

print(f"Reading chain file: {chain_file}")
chain_data = read_and_clean_chain(chain_file)
print(f"Data shape: {chain_data.shape}")

# Columns: 0=Weight, 1=MinusLogLike, 2...=Params
weights = chain_data[:, 0]
loglikes = chain_data[:, 1]
samples = chain_data[:, 2:]

# Handle NaNs
if np.isnan(samples).any():
    print("Warning: NaNs found in samples. removing rows with NaNs.")
    # Remove rows with NaNs
    mask = ~np.isnan(samples).any(axis=1)
    samples = samples[mask]
    weights = weights[mask]
    loglikes = loglikes[mask]

# Columns: 0=Weight, 1=MinusLogLike, 2...=Params
# Check weights
if np.all(weights == 0):
    print("Warning: All weights are zero. Setting weights to 1.0 for visualization.")
    weights = np.ones_like(weights)
elif np.any(weights == 0):
    print(f"Warning: {np.sum(weights == 0)} weights are zero. Replacing them with min positive weight or 0.")
    # getdist handles 0 weights (ignores sample).

# Filter parameters
# We need to map 'target_params' to indices in 'samples'
# param_names_raw corresponds to samples columns
param_indices = []
plot_names = []
plot_labels = []

for p in target_params:
    if p in param_names_raw:
        idx = param_names_raw.index(p)
        param_indices.append(idx)
        plot_names.append(p)
        
        # Manual fix for broken label of Oe
        original_label = labels_raw[idx]
        if p == 'Oe' and original_label.strip() == '\\Omega_{':
             print("Fixing broken label for Oe")
             plot_labels.append(r'\Omega_{\varepsilon}')
        else:
             plot_labels.append(original_label)
    else:
        print(f"Warning: Parameter {p} not found in paramfile.")

print(f"Plotting parameters: {plot_names}")

# Print stats
print("Sample stats:")
for i, name in enumerate(plot_names):
    col = samples[:, param_indices[i]]
    print(f"  {name}: min={np.min(col)}, max={np.max(col)}, std={np.std(col)}")
    if np.std(col) == 0:
        print(f"    Warning: Parameter {name} is constant!")

# Create MCSamples
# Use settings to make it robust to edges
settings = {'smooth_scale_2D': 0.5, 'smooth_scale_1D': 0.5, 'fine_bins': 50}
mc_samples = MCSamples(samples=samples[:, param_indices], weights=weights, names=plot_names, labels=plot_labels, name_tag='DFT2 Temp', settings=settings)

# Plot
g = plots.get_subplot_plotter()
g.triangle_plot([mc_samples], filled=True, title_limit=1)

output_path = os.path.join(os.getcwd(), output_plot)
g.export(output_path)
print(f"Corner plot saved to: {output_path}")
