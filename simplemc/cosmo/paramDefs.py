##
# This file has parameter definitions for all
# parameters used in this code.
##
# Change here for bounds, or import and rewrite.
##
##
# The Parameter class is defined as
# Parameter(name, value, err=0.0, bounds=None, Ltxname=None)

from simplemc.cosmo.Parameter import Parameter


# Parameters are value, variation, bounds.
# Base parameters.
Om_par = Parameter("Om", 0.3038, 0.05, (0.1, 0.5), "\Omega_m")
Obh2_par = Parameter("Obh2", 0.02234, 0.001, (0.02, 0.025), "\Omega_{b}h^2")
h_par = Parameter("h", 0.6821, 0.05, (0.4, 0.9), "h")

# DE equation of state parameters.
w_par = Parameter("w", -1.0, 0.1, (-2.0, 0.0), "w")
w0_par = Parameter("w0", -1.0, 0.1, (-2.0, 0.0), "w_0")
wa_par = Parameter("wa", 0.0, 0.1, (-2.0, 2.0), "w_a")

# Neutrino mass and effective number.
mnu_par = Parameter("mnu", 0.06, 0.1, (0, 1.0), "\Sigma m_{\\nu}")
Nnu_par = Parameter("Nnu", 3.046, 0.5, (3.0, 3.1), "N_{\\rm eff}")

# Curvature and DE equation of state.
# Ok_par = Parameter("Ok", 0.0, 0.001, (-0.1, 0.1), "\Omega_k") # For GR
# Ok_par = Parameter("Ok", 1.0, 0.001, (-2.0, 2.0), "\Omega_k") # For DFT (old wide prior)
Ok_par = Parameter("Ok", 1.0, 0.001, (0.97, 1.03), "\Omega_k") # For DFT (tightened per FSC sensitivity, 2026-04-30)

# Sigma 8 parameter (required by BaseCosmology)
s8_par = Parameter("s8", 0.8, 0.01, (0.5, 1.0), "s8")

# This is the prefactor parameter c/rdH0 (required by BaseCosmology)
Pr_par = Parameter("Pr", 28.6, 4, (5, 70), "c/(H_0r_d)")

# fine structure constant
alpha_fsc_par = Parameter("alpha_fsc", 0.0072973525643, 0.0001, (0.005, 0.01), "\\alpha_{\\rm fsc}")

# Absolute Magnitude
MB_par = Parameter("MB", -19.25, 0.01, (-20,-18.5), "M_B")

# DFT parameters
# Tightened priors (2026-04-30): old wide priors caused ~93% of the prior
# volume to be non-physical (H -> 0 / NaN), making sampling effectively
# impossible. The narrowed bounds below follow the e^phi / FSC sensitivity
# argument from Luis, but with Oh slightly looser than his earlier 1e-6
# scale and a permissive box on w/lambda for the model-distinguishing pars.
# Old wide-prior values are kept commented for reproducibility.
# dft_Oh_par = Parameter("Oh", 0.1, 0.1, (0.0,10.0), "\Omega_{\mathfrak{h}}")
# dft_Oe_par = Parameter("Oe", 0.1, 0.1, (0.0,10.0), "\Omega_{\varepsilon}")
dft_Oh_par = Parameter("Oh", 0.0, 1e-4, (0.0, 0.001), "\Omega_{\mathfrak{h}}")
dft_OL_par = Parameter("OL", 0.01, 0.01, (-0.1,0.1), "\Omega_{\Lambda}")
dft_Oe_par = Parameter("Oe", 0.01, 0.01, (0.0, 0.1), "\Omega_{\\varepsilon}")
dft_w_par  = Parameter("w_dft", 0.3, 1.0, (-0.33,10.0), "w")
dft_l_par  = Parameter("l_dft", 0.0, 2.0, (-10.0,10.0), "\lambda")
