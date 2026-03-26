import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ── Best-fit 파라미터 로드 (체인에서 min -2logL 행) ───────────────────────
chain  = np.loadtxt('DFT_w1l2_phy_HD+Union3+FSC_nested_multi_1.txt')
bf     = chain[np.argmin(chain[:, 1])]
h_val  = float(bf[2])
Ok_val = float(bf[3])
Oh_val = float(bf[4])
Oe_val = float(bf[5])
print(f"Best-fit  -2logL = {bf[1]:.4f}")
print(f"  h={h_val:.6f},  Ok={Ok_val:.6f},  Oh={Oh_val:.3e},  Oe={Oe_val:.3e}")

w_val = 1.0
l_val = 2.0
H0    = h_val * 100.0

# w=1, l=2  →  n1=3, n2=1
n1 = 6.0 * (w_val + 1.0) / (l_val + 2.0)
n2 = 4.0 / (l_val + 2.0)

# ── RHS (scipy 형식: y=[phi, H]) ──────────────────────────────────────────
def rhs_ivp(z, y):
    phi, H = y
    # H 음수 방지: 물리적으로 H > 0 이어야 함
    if H <= 0.0:
        return [0.0, 0.0]
    # phi 클램핑은 exp overflow 방지용 (phi 자체는 자유롭게 진화)
    phi_c   = np.clip(phi, -300.0, 300.0)
    Oe_term = Oe_val * (1.0 + z)**n1 * np.exp(n2 * phi_c)
    S = (3.0 / H0**2
         + (6.0 * Oe_term
            + 6.0 * Ok_val * (1.0 + z)**2
            + 6.0 * Oh_val * (1.0 + z)**6) / H**2)
    if S < 0.0:
        return [0.0, 0.0]
    sqS    = np.sqrt(S)
    dphidz = -(3.0 - H0 * sqS) / (2.0 * (1.0 + z))
    dHdz   = -(H0**2 / (1.0 + z)) * (
        (3.0 * w_val * Oe_term
         + 2.0 * Ok_val * (1.0 + z)**2
         + 6.0 * Oh_val * (1.0 + z)**6) / H
        - H / H0 * sqS
    )
    return [dphidz, dHdz]

# H=0 에 도달하면 적분 중단하는 이벤트
def H_zero(z, y):
    return y[1]
H_zero.terminal  = True
H_zero.direction = -1

# ── scipy adaptive solver (RK45, tight tolerance) ────────────────────────
z_eval = np.concatenate([[0.0],
                         np.logspace(np.log10(1e-4), np.log10(1000.0), 2000)])

sol = solve_ivp(
    rhs_ivp,
    t_span=(0.0, 1000.0),
    y0=[0.0, H0],
    method='RK45',
    t_eval=z_eval,
    rtol=1e-10,
    atol=1e-12,
    events=H_zero,
    dense_output=False,
)

print(f"Solver status: {sol.status}  ({sol.message})")
print(f"Integration reached z = {sol.t[-1]:.2f}")

z_arr   = sol.t
phi_arr = sol.y[0]
H_arr   = sol.y[1]

print(f"\nPhi range: [{phi_arr.min():.6f}, {phi_arr.max():.6f}]")
print(f"H(0)/H0 = {H_arr[0]/H0:.8f},  H[-1]/H0 = {H_arr[-1]/H0:.4f}")

# ── 각 성분 밀도 (H0² 단위) ───────────────────────────────────────────────
phi_c   = np.clip(phi_arr, -300.0, 300.0)
rho_eps = Oe_val * (1.0 + z_arr)**n1 * np.exp(n2 * phi_c)
rho_h   = Oh_val * (1.0 + z_arr)**6
rho_k   = Ok_val * (1.0 + z_arr)**2

H2      = H_arr**2
Om_eps  = 6.0 * rho_eps / H2
Om_h    = 6.0 * rho_h   / H2
Om_k    = 6.0 * rho_k   / H2

print(f"z=0: Om_eps={Om_eps[0]:.6e}, Om_h={Om_h[0]:.6e}, Om_k={Om_k[0]:.6e}")

# ── DFT Friedmann 예산 항 ─────────────────────────────────────────────────
term_vac = np.full_like(z_arr, 3.0)
term_eps = 6.0 * rho_eps * (H0**2 / H2)
term_k   = 6.0 * rho_k   * (H0**2 / H2)
term_h   = 6.0 * rho_h   * (H0**2 / H2)

# ── Plot ──────────────────────────────────────────────────────────────────
x = np.log10(1.0 + z_arr)      # x축: log10(1+z)

fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True,
                         gridspec_kw={'height_ratios': [2, 2, 1]})
ax1, ax2, ax3 = axes

title = (r'DFT density evolution  ($w=1,\ l=2,\ \Omega_L=0$)'
         f'\nh={h_val:.4f}, Ok={Ok_val:.4f}, Oh={Oh_val:.2e}, Oe={Oe_val:.2e}')

# ─ 상단: 물리적 밀도 ρ_i(z) (log-log) ─
ax1.semilogy(x, rho_k,        lw=2, color='C2',
             label=r'$\rho_k = \Omega_k(1+z)^2$')
ax1.semilogy(x, rho_h   + 1e-80, lw=2, color='C1',
             label=r'$\rho_h = \Omega_h(1+z)^6$')
ax1.semilogy(x, rho_eps + 1e-80, lw=2, color='C0',
             label=r'$\rho_\varepsilon = \Omega_\varepsilon(1+z)^3 e^{\phi}$')
ax1.set_ylabel(r'$\rho_i(z)$  (units of $H_0^2$)', fontsize=11)
ax1.set_title(title, fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ─ 중간: DFT Friedmann 예산 ─
ax2.semilogy(x, term_vac,         lw=1.5, color='gray', ls='--',
             label=r'DFT vacuum  ($3$)')
ax2.semilogy(x, term_k,           lw=2,   color='C2',
             label=r'$6\rho_k H_0^2/H^2$')
ax2.semilogy(x, term_h   + 1e-80, lw=2,   color='C1',
             label=r'$6\rho_h H_0^2/H^2$')
ax2.semilogy(x, term_eps + 1e-80, lw=2,   color='C0',
             label=r'$6\rho_\varepsilon H_0^2/H^2$')
ax2.set_ylabel(r'DFT Friedmann budget terms', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# ─ 하단: scalar field φ(z) ─
ax3.plot(x, phi_arr, color='C3', lw=2, label=r'$\phi(z)$')
ax3.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
ax3.set_xlabel(r'$z$  (log scale)', fontsize=12)
ax3.set_ylabel(r'$\phi(z)$', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

z_ticks  = np.array([0, 0.5, 1, 2, 5, 10, 100, 1000])
x_ticks  = np.log10(1.0 + z_ticks)
for ax in axes:
    ax.set_xticks(x_ticks)
ax3.set_xticklabels([str(z) for z in z_ticks])

plt.tight_layout()
out = 'density_evolution_DFT_w1l2.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved -> {out}')
