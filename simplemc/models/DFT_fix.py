"""
DFTCosmology — corrected version (drop-in replacement for DFTCosmology.py).

Changes vs. the previous DFT_fix.py:
  1. `Da_z` override RESTORED (per DFTCosmology_optimization_notes.md
     §6 — recovers the ~84× total speed-up from the optimization round).
  2. `initialize()` ALWAYS stores arrays. Broken points (ODE recollapse
     / NaN) keep the early-terminated padded array instead of leaving
     `_H_arr = None` — eliminates the ValueError crash in `Hinv_z`.
  3. Broken-point rejection signal moved into `prior_loglike()`:
     `_broken=True` → `-1e30`. Sampler sees a clean prior reject;
     downstream observables never crash.
  4. `fixfsc` re-enabled with INVERTED default: now `fixfsc=False`
     (alpha_fsc free) is the default. Pass `fixfsc=True` to fix
     alpha_fsc at ALPHA_LAB and drop it from the free-parameter list.
     runbase.py exposes the fixed case via the `_fsc` suffix.
  5. `error_model='numpy'` on numba RHS — IEEE-style inf/NaN instead of
     ZeroDivisionError on the λ=-2 boundary; slight speed-up.

Physical content from DFT_fix preserved:
  - Recollapse handling: H ≤ 1e-3 or non-finite → in-loop early
    terminate (Da, phi frozen at last valid value).
  - λ ≈ -2 singularity: lp2 clamped at ±1e-4, prior_loglike rejects
    |λ+2| < 1e-4.
  - Hinv_z finite-clamps to bound 1/H near recollapse.

Cluster notes:
  - All hot paths inside `@njit(cache=True, fastmath=True,
    error_model='numpy')`. Per-process compile cost is paid once and
    cached to `__pycache__`. No shared mutable state → safe under MPI
    / multiprocessing.
  - When first deploying after replacing this file, clear stale numba
    artifacts:
        rm -rf simplemc/models/__pycache__
"""
from simplemc.cosmo.BaseCosmology import BaseCosmology
from simplemc.cosmo.paramDefs import (h_par, Ok_par, dft_Oh_par,
                                       dft_Oe_par, dft_w_par, dft_l_par,
                                       alpha_fsc_par)

import numpy as np
from numba import njit


# Lab reference matching LCDMCosmology.fine_structure_constant so that
# gamma = alpha_fsc / ALPHA_LAB = 1 reproduces the fixed-fsc prediction.
ALPHA_LAB = 0.0072973525643


# ---------------------------------------------------------------------------
#  Numba-accelerated ODE core
#
#  Physical constants packed into a single float64 array `p`:
#     p[0]=H0  p[1]=H0sq  p[2]=Ok  p[3]=Oh  p[4]=OL
#     p[5]=Oe  p[6]=w     p[7]=alpha  p[8]=beta
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, error_model='numpy')
def _rhs(z, phi, H, p):
    """DFT coupled ODE right-hand side. Returns (dphi/dz, dH/dz)."""
    H0    = p[0]
    H0sq  = p[1]
    Ok    = p[2]
    Oh    = p[3]
    OL    = p[4]
    Oe    = p[5]
    w     = p[6]
    alpha = p[7]
    beta  = p[8]

    if H <= 0.0:
        H = H0

    if phi > 50.0:
        phi_c = 50.0
    elif phi < -50.0:
        phi_c = -50.0
    else:
        phi_c = phi

    zp1    = 1.0 + z
    ee     = np.exp(beta * phi_c)
    Oe_z   = Oe * zp1**alpha * ee
    zp1_sq = zp1 * zp1
    zp1_6  = zp1_sq * zp1_sq * zp1_sq
    Hsq    = H * H

    inSqrt = 3.0 / H0sq + 6.0 * (Oe_z + OL + Ok * zp1_sq
                                   + Oh * zp1_6) / Hsq

    if inSqrt >= 0.0:
        s = np.sqrt(inSqrt)
        dphidz = -(3.0 - H0 * s) / (2.0 * zp1)
        dHdz = -H0sq * (
            (3.0 * w * Oe_z + 2.0 * Ok * zp1_sq + 6.0 * Oh * zp1_6) / H
            - H * s / H0
        ) / zp1
    else:
        dphidz = -1.0
        dHdz   = -1.0

    return dphidz, dHdz


@njit(cache=True, fastmath=True, error_model='numpy')
def _make_params(H0, Ok, Oh, OL, Oe, w, l):
    """Pack pre-computed constants into a flat array for _rhs.

    `lp2` is clamped at ±1e-4 to keep the integrator finite at the
    λ = -2 string-frame singularity. `prior_loglike()` independently
    rejects |λ+2| < 1e-4, so the clamped values never affect physics.
    """
    lp2 = l + 2.0
    if abs(lp2) < 1e-4:
        lp2 = 1e-4 if lp2 >= 0 else -1e-4
    alpha = 6.0 * (w + 1.0) / lp2
    beta  = 4.0 / lp2
    p = np.empty(9)
    p[0] = H0
    p[1] = H0 * H0
    p[2] = Ok
    p[3] = Oh
    p[4] = OL
    p[5] = Oe
    p[6] = w
    p[7] = alpha
    p[8] = beta
    return p


# ---------------------------------------------------------------------------
#  Fixed-step RK4 solver  (with recollapse-aware early termination)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, error_model='numpy')
def _dft_solve_rk4(H0, Ok, Oh, OL, Oe, w, l, z_max, steps):
    """RK4 solve of the DFT coupled ODEs + cumulative Da(z), single pass.

    Returns
    -------
    H_arr, phi_arr, Da_arr : float64[steps]
        ODE solution on uniform z grid. After a recollapse (H ≤ 1e-3
        or non-finite), the tail of each array is frozen — H_arr to
        1e-10, phi_arr/Da_arr to their last valid value. This keeps
        all downstream interpolation finite.
    broken_idx : int
        -1 if the integration completed normally;
        otherwise the z-grid index where early termination triggered.
    """
    p  = _make_params(H0, Ok, Oh, OL, Oe, w, l)
    dz = z_max / (steps - 1)

    H_arr   = np.empty(steps)
    phi_arr = np.empty(steps)
    Da_arr  = np.zeros(steps)

    phi = 0.0
    H   = H0
    H_arr[0]   = H0
    phi_arr[0] = 0.0

    broken_idx = -1

    for i in range(steps - 1):
        z  = i * dz
        zh = z + 0.5 * dz

        dp1, dH1 = _rhs(z,      phi,               H,               p)
        dp2, dH2 = _rhs(zh,     phi + 0.5*dz*dp1,  H + 0.5*dz*dH1, p)
        dp3, dH3 = _rhs(zh,     phi + 0.5*dz*dp2,  H + 0.5*dz*dH2, p)
        dp4, dH4 = _rhs(z + dz, phi + dz*dp3,      H + dz*dH3,     p)

        phi_new = phi + (dz / 6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4)
        H_new   = H   + (dz / 6.0) * (dH1 + 2.0*dH2 + 2.0*dH3 + dH4)

        # Recollapse / non-physical detection. In DFT, H → 0 marks the
        # bouncing point; integration past it is not meaningful. Upper
        # bound 1e10 (km/s/Mpc) catches divergent runs before the
        # squared value in RHSquared_a overflows float64. The
        # `not (lo < x < hi)` form is NaN-safe: NaN comparisons return
        # False for both bounds, so the negation triggers — important
        # because `fastmath=True` can elide an explicit `np.isfinite`.
        if not (1e-3 < H_new < 1e10):
            broken_idx = i + 1
            H_arr[i+1:]   = 1e-10
            phi_arr[i+1:] = phi_arr[i]
            Da_arr[i+1:]  = Da_arr[i]
            return H_arr, phi_arr, Da_arr, broken_idx

        H_arr[i + 1]   = H_new
        phi_arr[i + 1] = phi_new

        # Cumulative trapezoidal: Da(z) = ∫₀ᶻ H₀/H(z') dz'
        Da_arr[i + 1] = Da_arr[i] + 0.5 * dz * (H0 / H_arr[i] + H0 / H_new)

        phi = phi_new
        H   = H_new

    return H_arr, phi_arr, Da_arr, broken_idx


# ---------------------------------------------------------------------------
#  DFTCosmology  (base class for all DFT variants)
# ---------------------------------------------------------------------------

class DFTCosmology(BaseCosmology):
    """DFT cosmology (OL = 0). RK4 + interpolation.

    `fixfsc=False` (default) — alpha_fsc is a free parameter.
    `fixfsc=True`            — alpha_fsc held at ALPHA_LAB, dropped from
                               the free-parameter list.
    `ishzero=False` (default) — Omega_h is a free parameter.
    `ishzero=True`            — Omega_h fixed to 0, dropped from the
                                free-parameter list.

    Recollapse handling: if the ODE produces H ≤ 1e-3 or non-finite at
    any z ≤ z_max, that parameter set is rejected via
    `prior_loglike() = -1e30`. The internal arrays are still populated
    (with frozen tails) so observable methods never crash.
    """

    _Z_MAX    = 8.0
    _RK_STEPS = 800

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value, l=dft_l_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 ishzero=False, fixfsc=False):
        self.Ok = Ok
        self.ishzero = ishzero
        self.Oh = 0.0 if ishzero else Oh
        self.OL = 0.0
        self.Oe = Oe
        self.w  = w
        self.l  = l
        self.fixfsc = fixfsc
        self.alpha_fsc = ALPHA_LAB if fixfsc else alpha_fsc

        self._H0      = h * 100.0
        self._H_arr   = None
        self._phi_arr = None
        self._Da_arr  = None
        self._broken  = False

        base_params = [h_par, Ok_par]
        if not ishzero:
            base_params.append(dft_Oh_par)
        base_params += [dft_Oe_par, dft_w_par, dft_l_par]
        if not fixfsc:
            base_params.append(alpha_fsc_par)
        self.parameters = base_params
        self._z_grid = np.linspace(0.0, self._Z_MAX, self._RK_STEPS)

        BaseCosmology.__init__(self, h)
        self.updateParams([])

    def freeParameters(self):
        return self.parameters

    # -- parameter update (no ODE solve) ------------------------------------
    def _set_params(self, pars):
        BaseCosmology.updateParams(self, pars)
        for p in pars:
            if p.name == "Ok":
                self.Ok = p.value
            elif p.name == "Oh" and not self.ishzero:
                self.Oh = p.value
            elif p.name == "Oe":
                self.Oe = p.value
            elif p.name == "w_dft":
                self.w = p.value
            elif p.name == "l_dft":
                self.l = p.value
            elif p.name == "alpha_fsc" and not self.fixfsc:
                self.alpha_fsc = p.value
        self.OL = 0.0

    def updateParams(self, pars):
        self._set_params(pars)
        self.initialize()
        return True

    # -- ODE solve + precompute all lookups ---------------------------------
    def initialize(self):
        H0 = self.h * 100.0
        H_arr, phi_arr, Da_arr, broken_idx = _dft_solve_rk4(
            H0, self.Ok, self.Oh, self.OL, self.Oe, self.w, self.l,
            self._Z_MAX, self._RK_STEPS
        )
        self._H0      = H0
        self._H_arr   = H_arr
        self._phi_arr = phi_arr
        self._Da_arr  = Da_arr
        self._broken  = (broken_idx >= 0)
        return True

    # -- fast lookups (np.interp on uniform grid) ---------------------------
    def hub(self, z):
        return np.interp(z, self._z_grid, self._H_arr)

    def fine_structure_constant(self, a):
        z = 1.0 / a - 1.0
        phi = np.clip(np.interp(z, self._z_grid, self._phi_arr), -50.0, 50.0)
        gamma = self.alpha_fsc / ALPHA_LAB
        return gamma * np.exp(2.0 * phi) - 1.0

    def prior_loglike(self):
        # Reject points where the ODE recollapsed or went non-finite.
        if self._broken:
            return -1e30
        # λ = -2 is a true singularity of the dilaton EoM: hard cutoff
        # plus a soft Gaussian penalty on the singular neighborhood.
        abs_l2 = abs(self.l + 2.0)
        if abs_l2 < 1e-4:
            return -1e30
        sigma = 0.03
        if abs_l2 < 3.0 * sigma:
            return -0.5 * (sigma / abs_l2) ** 2
        return 0.0

    def RHSquared_a(self, a):
        z = 1.0 / a - 1.0
        H = np.interp(z, self._z_grid, self._H_arr)
        # Suppress noisy overflow warnings on cluster stderr; the
        # finite/positive guard below catches the bad cases anyway.
        with np.errstate(over='ignore', invalid='ignore'):
            result = (H / self._H0) ** 2
        if not np.isfinite(result) or result <= 0:
            return 1e-30
        return result

    # -- overrides: precomputed distance functions --------------------------
    def Hinv_z(self, z):
        z = np.atleast_1d(z)
        H = np.interp(z, self._z_grid, self._H_arr)
        # H clamped against 1/H blow-up at recollapse; then any
        # remaining inf/NaN goes to a finite sentinel so the chi² is
        # large but bounded (sampler rejects cleanly).
        invH = self._H0 / np.where(H > 1e-5, H, 1e-5)
        return np.where(np.isfinite(invH), invH, 1e10)

    def Da_z(self, z):
        z = np.atleast_1d(z)
        r = np.interp(z, self._z_grid, self._Da_arr)
        if self.Curv == 0:
            return r
        elif self.Curv > 0:
            q = np.sqrt(self.Curv)
            return np.sinh(r * q) / q
        else:
            q = np.sqrt(-self.Curv)
            return np.sin(r * q) / q


# ---------------------------------------------------------------------------
#  Constrained subclasses
# ---------------------------------------------------------------------------

class DFTw1l2Cosmology(DFTCosmology):
    """DFT with fixed w=1, l=2, OL=0. Free: h, Ok, Oh, Oe (+ alpha_fsc if fixfsc=False).

    With ``ishzero=True``, drops Omega_h from the free-parameter list.
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, alpha_fsc=alpha_fsc_par.value,
                 ishzero=False, fixfsc=False):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, Oe=Oe, w=1.0, l=2.0,
                              alpha_fsc=alpha_fsc,
                              ishzero=ishzero, fixfsc=fixfsc)
        base_params = [h_par, Ok_par]
        if not ishzero:
            base_params.append(dft_Oh_par)
        base_params.append(dft_Oe_par)
        if not fixfsc:
            base_params.append(alpha_fsc_par)
        self.parameters = base_params

    def updateParams(self, pars):
        self._set_params(pars)
        self.w = 1.0
        self.l = 2.0
        self.initialize()
        return True


class DFTl3w1Cosmology(DFTCosmology):
    """DFT with constraint l = 3w - 1, OL=0. Free: h, Ok, Oh, Oe, w (+ alpha_fsc if fixfsc=False).

    With ``ishzero=True``, drops Omega_h from the free-parameter list.
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 ishzero=False, fixfsc=False):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, Oe=Oe, w=w,
                              l=3.0*w - 1.0, alpha_fsc=alpha_fsc,
                              ishzero=ishzero, fixfsc=fixfsc)
        base_params = [h_par, Ok_par]
        if not ishzero:
            base_params.append(dft_Oh_par)
        base_params += [dft_Oe_par, dft_w_par]
        if not fixfsc:
            base_params.append(alpha_fsc_par)
        self.parameters = base_params

    def updateParams(self, pars):
        self._set_params(pars)
        self.l = 3.0 * self.w - 1.0
        self.initialize()
        return True


class DFTl2wCosmology(DFTCosmology):
    """DFT with constraint l = 2w, OL=0. Free: h, Ok, Oh, Oe, w (+ alpha_fsc if fixfsc=False).

    With ``ishzero=True``, drops Omega_h from the free-parameter list.
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 ishzero=False, fixfsc=False):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, Oe=Oe, w=w, l=2.0*w,
                              alpha_fsc=alpha_fsc,
                              ishzero=ishzero, fixfsc=fixfsc)
        base_params = [h_par, Ok_par]
        if not ishzero:
            base_params.append(dft_Oh_par)
        base_params += [dft_Oe_par, dft_w_par]
        if not fixfsc:
            base_params.append(alpha_fsc_par)
        self.parameters = base_params

    def updateParams(self, pars):
        self._set_params(pars)
        self.l = 2.0 * self.w
        self.initialize()
        return True


class DFTl0Cosmology(DFTCosmology):
    """DFT with fixed l=0, OL=0. Free: h, Ok, Oh, Oe, w (+ alpha_fsc if fixfsc=False).

    With ``ishzero=True``, drops Omega_h from the free-parameter list.
    """

    def __init__(self, h=h_par.value, Ok=Ok_par.value, Oh=dft_Oh_par.value,
                 Oe=dft_Oe_par.value, w=dft_w_par.value,
                 alpha_fsc=alpha_fsc_par.value,
                 ishzero=False, fixfsc=False):
        DFTCosmology.__init__(self, h=h, Ok=Ok, Oh=Oh, Oe=Oe, w=w, l=0.0,
                              alpha_fsc=alpha_fsc,
                              ishzero=ishzero, fixfsc=fixfsc)
        base_params = [h_par, Ok_par]
        if not ishzero:
            base_params.append(dft_Oh_par)
        base_params += [dft_Oe_par, dft_w_par]
        if not fixfsc:
            base_params.append(alpha_fsc_par)
        self.parameters = base_params

    def updateParams(self, pars):
        self._set_params(pars)
        self.l = 0.0
        self.initialize()
        return True
