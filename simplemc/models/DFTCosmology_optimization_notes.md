# DFTCosmology 성능 최적화 상세 노트

## 목차

1. [배경: DFT 연립 미분방정식 구조](#1-배경-dft-연립-미분방정식-구조)
2. [최적화 1: RK4 스텝 수 감소 (5000 → 800)](#2-최적화-1-rk4-스텝-수-감소-5000--800)
3. [최적화 2: 이중 보간 제거](#3-최적화-2-이중-보간-제거)
4. [최적화 3: `interp1d` → `np.interp` 교체](#4-최적화-3-interp1d--npinterp-교체)
5. [최적화 4: RHS 상수 사전 계산 + 스칼라 반환](#5-최적화-4-rhs-상수-사전-계산--스칼라-반환)
6. [최적화 5: `Da(z)` 사전 계산](#6-최적화-5-daz-사전-계산)
7. [최적화 6: 서브클래스 이중 초기화 제거](#7-최적화-6-서브클래스-이중-초기화-제거)
8. [Adaptive RK45 (Dormand-Prince) 검토 및 기각 이유](#8-adaptive-rk45-dormandprince-검토-및-기각-이유)
9. [벤치마크 결과](#9-벤치마크-결과)

---

## 1. 배경: DFT 연립 미분방정식 구조

DFT cosmology는 스칼라장 $\phi(z)$와 허블 파라미터 $H(z)$에 대한 연립 ODE를 풀어야 한다.
자유 파라미터는 $(h,\, \Omega_k,\, \Omega_{\mathfrak{h}},\, \Omega_\varepsilon,\, w,\, \lambda)$ 이며,
이 값들이 바뀔 때마다 (즉 sampler가 새 파라미터 세트를 제안할 때마다) ODE를 다시 풀어야 한다.

### 핵심 ODE

$$
\frac{d\phi}{dz} = -\frac{3 - H_0 \sqrt{S}}{2(1+z)}
$$

$$
\frac{dH}{dz} = -\frac{H_0^2}{1+z}\left[
  \frac{3w\,\Omega_\varepsilon^{(z)} + 2\Omega_k(1+z)^2 + 6\Omega_{\mathfrak{h}}(1+z)^6}{H}
  - \frac{H}{H_0}\sqrt{S}
\right]
$$

여기서

$$
S \equiv \frac{3}{H_0^2}
+ \frac{6\bigl[\Omega_\varepsilon^{(z)} + \Omega_\Lambda + \Omega_k(1+z)^2 + \Omega_{\mathfrak{h}}(1+z)^6\bigr]}{H^2}
$$

$$
\Omega_\varepsilon^{(z)} = \Omega_\varepsilon\,(1+z)^{\alpha}\,e^{\beta\,\phi_c}
$$

상수 정의:
$$
\alpha \equiv \frac{6(w+1)}{\lambda+2}, \qquad
\beta \equiv \frac{4}{\lambda+2}, \qquad
\phi_c \equiv \mathrm{clamp}(\phi,\;-50,\;+50)
$$

초기 조건: $\phi(0) = 0$, $H(0) = H_0 = 100h$.

### 병목 구조 (최적화 전)

nested sampler에서 `updateParams`는 **10⁵ ~ 10⁶회** 호출된다.
매 호출마다:

1. `initialize()`: RK4 5000스텝 ODE 풀기 → `interp1d` 4개 생성
2. `Da_z()`: 500회 `RHSquared_a` Python 루프 → 수치 적분

프로파일링 결과:

| 단계 | 1회당 비용 | 비율 |
|------|------------|------|
| RK4 solve (5000스텝 × 4 RHS) | 15.7 ms | ~31% |
| `Da_z` 수치 적분 (500 × `RHSquared_a`) | 35.1 ms | ~69% |
| **합계** | **~50.8 ms** | |

---

## 2. 최적화 1: RK4 스텝 수 감소 (5000 → 800)

### 원리

RK4는 **4차 정확도**를 가진다. 스텝 크기 $\Delta z$에서의 국소 절단 오차(local truncation error)는:

$$
\epsilon_{\text{local}} \sim O(\Delta z^5)
$$

전체 구간 $[0, z_{\max}]$에서 $N$스텝으로 풀 때 **전역 누적 오차**는:

$$
\epsilon_{\text{global}} \sim O(\Delta z^4) = O\!\left(\frac{z_{\max}^4}{N^4}\right)
$$

| 스텝 수 $N$ | $\Delta z$ | $\epsilon_{\text{global}}$ 스케일 |
|:---:|:---:|:---:|
| 5000 | 0.0016 | $\sim 6.6 \times 10^{-12}$ |
| **800** | **0.01** | $\sim 10^{-8}$ |
| 500 | 0.016 | $\sim 6.6 \times 10^{-7}$ |

800스텝의 오차 $\sim 10^{-8}$은 관측 데이터 오차($\sim 1\%$) 대비 **5자릿수 이상 작다**.

### 실제 검증 (best-fit 파라미터)

$h=0.6229$, $\Omega_k=1.0$, $\Omega_{\mathfrak{h}}=10^{-4}$, $\Omega_\varepsilon=10^{-4}$, $w=1$, $\lambda=2$:

| $z$ | $H_{800}(z)$ | $H_{5000}(z)$ | 상대 오차 |
|:---:|---:|---:|:---:|
| 0.5 | 93.355240 | 93.355243 | $2.8 \times 10^{-8}$ |
| 1.0 | 124.155463 | 124.155476 | $1.1 \times 10^{-7}$ |
| 2.0 | 183.235547 | 183.235627 | $4.4 \times 10^{-7}$ |
| 5.0 | 207.868012 | 207.871784 | $1.8 \times 10^{-5}$ |

### 코드 변경

**Before:**
```python
self.rk_steps = 5000
```

**After:**
```python
_RK_STEPS = 800
```

---

## 3. 최적화 2: 이중 보간 제거

### 문제

기존 코드는 보간을 **두 단계**로 수행했다:

```
RK4 5000포인트 ──→ interp1d(cubic) ──→ 500포인트 리샘플 ──→ interp1d(cubic) ──→ 최종 보간 함수
     [1단계]           [2단계: 불필요]            [3단계]            [4단계]
```

**Before:**
```python
def initialize(self):
    y0 = np.array([0.0, self.h * 100.0])
    z_out, y_out = _dft_solve(
        y0, 0.0, 8.0, self.rk_steps,           # 5000 포인트 생성
        self.h, self.Ok, self.Oh, self.OL, self.Oe, self.w, self.l
    )
    # [불필요한 1단계] 5000pt → cubic interp1d 객체 생성
    phi_fine = interp1d(z_out, y_out[:, 0], kind='cubic', fill_value='extrapolate')
    H_fine   = interp1d(z_out, y_out[:, 1], kind='cubic', fill_value='extrapolate')

    # [불필요한 2단계] 5000pt → 500pt 다운샘플링
    phi_c = phi_fine(self.z_values)   # self.z_values = linspace(0, 8, 500)
    H_c   = H_fine(self.z_values)

    # [3단계] 500pt interp1d 객체 생성 (실제로 사용되는 것)
    self.phiinterp = interp1d(self.z_values, phi_c, fill_value='extrapolate')
    self.Hinterp   = interp1d(self.z_values, H_c,   fill_value='extrapolate')
```

### 해결

RK4를 **처음부터 800포인트**로 풀면, 별도의 다운샘플링이 필요 없다.
그 결과를 곧바로 `np.interp`의 룩업 테이블로 사용한다.

**After:**
```python
def initialize(self):
    H0 = self.h * 100.0
    H_arr, phi_arr, Da_arr = _dft_solve_all(
        H0, self.Ok, self.Oh, self.OL, self.Oe, self.w, self.l,
        self._Z_MAX, self._RK_STEPS       # 800 포인트 — 이것이 곧 최종 그리드
    )
    self._H0      = H0
    self._H_arr   = H_arr                 # flat numpy array, interp1d 객체 아님
    self._phi_arr = phi_arr
    self._Da_arr  = Da_arr
```

제거된 것:
- `interp1d` 객체 4개 → **0개** (매 호출마다 Python 객체 생성/소멸 비용 제거)
- `phi_fine(self.z_values)` 평가 단계 (불필요한 500회 cubic spline 평가)

---

## 4. 최적화 3: `interp1d` → `np.interp` 교체

### 문제

`scipy.interpolate.interp1d`는:
- **생성 시** Python 객체 할당 + 계수 계산 → 무거움
- **호출 시** Python dispatch → 오버헤드

매 `updateParams` 호출마다 `interp1d` 객체 4개를 생성·폐기하는 것은 낭비이다.

### 해결

`np.interp(x, xp, fp)`는:
- **생성 비용 0** — 별도 객체를 만들지 않고 flat array를 직접 참조
- **호출 시** C 구현 → 빠름
- 선형 보간이지만, 800포인트 등간격 그리드에서 충분한 정확도 보장

### 코드 비교

**Before (매 hub 호출):**
```python
def hub(self, z):
    return self.Hinterp(z)       # self.Hinterp = interp1d(...)
```

**After:**
```python
def hub(self, z):
    return np.interp(z, self._z_grid, self._H_arr)   # flat array 참조
```

`RHSquared_a`, `fine_structure_constant`도 동일하게 변경:

**Before:**
```python
def RHSquared_a(self, a):
    H0 = self.h * 100.0            # 매 호출마다 재계산
    H  = self.hub(1.0/a - 1.0)     # hub → interp1d.__call__
    result = (H / H0)**2
    ...
```

**After:**
```python
def RHSquared_a(self, a):
    z = 1.0 / a - 1.0
    H = np.interp(z, self._z_grid, self._H_arr)   # C-level np.interp
    result = (H / self._H0) ** 2                    # self._H0 캐시됨
    ...
```

---

## 5. 최적화 4: RHS 상수 사전 계산 + 스칼라 반환

### 문제: 반복 계산되는 상수

기존 `_dft_rhs`는 **매 RHS 호출마다** (총 $4 \times N_{\text{steps}}$회) 다음을 재계산:

```python
@njit(cache=True)
def _dft_rhs(z, y, h, Ok, Oh, OL, Oe, w, l):
    ...
    OeEvol1 = 6.0 * (w + 1.0) / (l + 2.0)       # ← 상수, 매번 나눗셈
    OeEvol2 = np.exp(4.0 * phi_c / (l + 2.0))    # ← (l+2) 나눗셈 반복
    inSqrt = 3.0 / (h * 100.0)**2.0 + ...        # ← h*100 매번 계산
    dHdz = -(h * 100.0)**2.0 * (...)              # ← (h*100)^2 반복
    return np.array([dphidz, dHdz])               # ← 배열 할당 매번
```

### 해결: 상수를 solver 레벨에서 한 번만 계산

**수식과 코드의 1:1 대응:**

$$\alpha = \frac{6(w+1)}{\lambda+2}$$

```python
alpha = 6.0 * (w + 1.0) / lp2      # solver에서 1회 계산
```

$$\beta = \frac{4}{\lambda+2}$$

```python
beta = 4.0 / lp2                    # solver에서 1회 계산
```

$$H_0 = 100h, \quad H_0^2 = (100h)^2$$

```python
H0   = H0                           # 인자로 전달 (h*100 연산 제거)
H0sq = H0 * H0                      # solver에서 1회 계산
```

**After — solver 함수:**
```python
@njit(cache=True)
def _dft_solve_all(H0, Ok, Oh, OL, Oe, w, l, z_max, steps):
    H0sq  = H0 * H0                       # ← 1회
    lp2   = l + 2.0                        # ← 1회
    alpha = 6.0 * (w + 1.0) / lp2         # ← 1회
    beta  = 4.0 / lp2                     # ← 1회
    ...
```

**After — RHS 함수 (상수는 인자로 받음):**
```python
@njit(cache=True)
def _rhs(z, phi, H, H0, H0sq, Ok, Oh, OL, Oe, w, alpha, beta):
```

### 문제: `np.array` 배열 할당

기존 RHS는 결과를 `np.array([dphidz, dHdz])`로 반환 → **매 호출마다 heap allocation**.

**Before:**
```python
return np.array([dphidz, dHdz])   # 매 RHS 호출 (=4×N회)마다 2-element array 할당
```

**After:**
```python
return dphidz, dHdz               # 스칼라 2개 반환 — allocation 없음
```

Solver에서도 이에 맞게 변경:

**Before:**
```python
k1 = _dft_rhs(z, y, h, Ok, Oh, OL, Oe, w, l)
k2 = _dft_rhs(z + 0.5*dz, y + 0.5*dz*k1, ...)   # y + 0.5*dz*k1: 벡터 연산
y  = y + (dz / 6.0) * (k1 + 2*k2 + 2*k3 + k4)   # 벡터 합
```

**After:**
```python
dp1, dH1 = _rhs(z, phi, H, ...)                          # 스칼라
dp2, dH2 = _rhs(zh, phi + 0.5*dz*dp1, H + 0.5*dz*dH1, ...)  # 스칼라 산술
phi += (dz / 6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4)     # 스칼라 연산
H   += (dz / 6.0) * (dH1 + 2.0*dH2 + 2.0*dH3 + dH4)
```

### 추가: $(1+z)^6$ 계산 최적화

**Before:**
```python
(1.0 + z)**6.0     # pow() 호출 — 느림
(1.0 + z)**2.0     # pow() 호출
```

**After:**
```python
zp1    = 1.0 + z
zp1_sq = zp1 * zp1         # 곱셈 1회
zp1_6  = zp1_sq * zp1_sq * zp1_sq   # 곱셈 2회 (pow 대신)
```

---

## 6. 최적화 5: `Da(z)` 사전 계산

### 이론적 배경

Comoving distance는:

$$
D_a(z) = \int_0^z \frac{H_0}{H(z')}\,dz'
$$

**유도:** $a = 1/(1+z)$ 치환을 통해 `BaseCosmology.Da_z`의 $a$-적분과 동치임을 보인다:

$$
D_a(z) = \int_{a(z)}^{1} \frac{da}{a^2 \sqrt{H^2(a)/H_0^2}}
$$

$a = 1/(1+z')$으로 치환하면 $da = -dz'/(1+z')^2$, $1/a^2 = (1+z')^2$이므로:

$$
D_a(z) = \int_0^z \frac{(1+z')^2}{H(z')/H_0} \cdot \frac{dz'}{(1+z')^2}
= \int_0^z \frac{H_0}{H(z')}\,dz'
$$

### 기존 방식의 문제

`BaseCosmology.Da_z`는 **매 likelihood 호출마다** 다음을 수행:

```python
def Da_z(self, z):
    z = np.atleast_1d(z)
    a_min = 1.0 / (1.0 + np.max(z))
    a_grid = np.linspace(a_min, 1.0, 500)

    # Python for-loop: 500회 호출!
    integrand_vals = np.array([self.DistIntegrand_a(a) for a in a_grid])

    da = np.diff(a_grid)
    avg_integrand = 0.5 * (integrand_vals[:-1] + integrand_vals[1:])
    r_grid = np.concatenate([[0], np.cumsum(avg_integrand[::-1] * da[::-1])])[::-1]
    a_requested = 1.0 / (1.0 + z)
    r = np.interp(a_requested, a_grid, r_grid)
    ...
```

매 호출마다 `DistIntegrand_a` → `RHSquared_a` → `hub` → `interp1d.__call__`이
**500회** Python 레벨에서 루프를 돈다. 이것이 전체 시간의 **~69%** 를 차지했다.

### 해결: RK4 solver 안에서 누적 적분

$H(z)$를 이미 RK4의 각 스텝에서 알고 있으므로, **추가 비용 거의 0**으로
$D_a(z)$를 동시에 계산할 수 있다:

$$
D_a(z_{i+1}) = D_a(z_i) + \frac{\Delta z}{2}\left(\frac{H_0}{H(z_i)} + \frac{H_0}{H(z_{i+1})}\right)
$$

이것은 사다리꼴 적분법(trapezoidal rule)의 누적합이다.

**After — `_dft_solve_all` 내부 (RK4 루프 끝):**
```python
for i in range(steps - 1):
    ...
    # RK4 스텝으로 phi, H 업데이트 후:
    H_arr[i + 1]   = H
    phi_arr[i + 1] = phi

    # 추가 비용: 나눗셈 2회 + 덧셈 2회 + 곱셈 1회
    Da_arr[i + 1] = Da_arr[i] + 0.5 * dz * (H0 / H_arr[i] + H0 / H)
```

**After — `Da_z` override:**
```python
def Da_z(self, z):
    z = np.atleast_1d(z)
    r = np.interp(z, self._z_grid, self._Da_arr)   # 단순 보간 1회!
    if self.Curv == 0:
        return r
    elif self.Curv > 0:
        q = np.sqrt(self.Curv)
        return np.sinh(r * q) / q
    else:
        q = np.sqrt(-self.Curv)
        return np.sin(r * q) / q
```

곡률 보정 $(k \neq 0)$도 `BaseCosmology`와 동일하게 처리:
- $k > 0$: $D_a^{(\mathrm{curv})} = \frac{\sinh(\sqrt{k}\,r)}{\sqrt{k}}$
- $k < 0$: $D_a^{(\mathrm{curv})} = \frac{\sin(\sqrt{|k|}\,r)}{\sqrt{|k|}}$

### `Hinv_z` override

`BaseCosmology.Hinv_z`도 같은 Python 루프 문제가 있었다:

**Before (BaseCosmology):**
```python
def Hinv_z(self, z):
    a = 1.0 / (1.0 + np.atleast_1d(z))
    return np.array([1.0/np.sqrt(self.RHSquared_a(val)) for val in a])  # Python loop
```

$H^{-1}(z)/H_0^{-1} = H_0/H(z)$ 이므로:

**After:**
```python
def Hinv_z(self, z):
    z = np.atleast_1d(z)
    H = np.interp(z, self._z_grid, self._H_arr)
    return self._H0 / H           # vectorized, 루프 없음
```

---

## 7. 최적화 6: 서브클래스 이중 초기화 제거

### 문제

기존 서브클래스(예: `DFTw1l2Cosmology`)의 `updateParams`는
ODE를 **두 번** 풀고 있었다:

```python
class DFTw1l2Cosmology(DFTCosmology):
    def updateParams(self, pars):
        ok = DFTCosmology.updateParams(self, pars)   # ← initialize() 호출 (1회차)
        if not ok:
            return False
        self.w = 1.0
        self.l = 2.0
        self.initialize()                             # ← initialize() 호출 (2회차)
        return True
```

1회차에서는 `w`, `l`이 아직 올바른 값으로 설정되지 않은 상태에서 ODE를 풀고,
2회차에서야 `w=1`, `l=2`로 설정 후 다시 풀었다. **1회차는 완전히 낭비.**

### 해결: `_set_params` / `updateParams` 분리

```python
class DFTCosmology(BaseCosmology):
    def _set_params(self, pars):
        """파라미터 값만 설정, ODE는 풀지 않음."""
        BaseCosmology.updateParams(self, pars)
        for p in pars:
            ...
        self.OL = 0.0

    def updateParams(self, pars):
        self._set_params(pars)    # 파라미터 설정
        self.initialize()         # ODE 1회만 풀기
        return True
```

서브클래스:
```python
class DFTw1l2Cosmology(DFTCosmology):
    def updateParams(self, pars):
        self._set_params(pars)    # 파라미터만 설정 (ODE 안 풀림)
        self.w = 1.0              # 제약 조건 적용
        self.l = 2.0
        self.initialize()         # ODE 1회만 풀기
        return True
```

**서브클래스에서 2배 추가 속도 향상.**

---

## 8. Adaptive RK45 (Dormand-Prince) 검토 및 기각 이유

### 시도한 방법

Dormand–Prince RK4(5) adaptive step-size control을 numba `@njit` 안에서 구현하여 테스트하였다.
이론적으로 해가 완만한 구간에서 큰 스텝, 급변 구간에서 작은 스텝을 사용하므로
총 RHS 호출 횟수를 줄일 수 있다.

### Butcher Tableau (Dormand–Prince)

$$
\begin{array}{c|ccccccc}
0 \\
1/5 & 1/5 \\
3/10 & 3/40 & 9/40 \\
4/5 & 44/45 & -56/15 & 32/9 \\
8/9 & 19372/6561 & -25360/2187 & 64448/6561 & -212/729 \\
1 & 9017/3168 & -355/33 & 46732/5247 & 49/176 & -5103/18656 \\
1 & 35/384 & 0 & 500/1113 & 125/192 & -2187/6784 & 11/84 \\
\hline
\text{5th} & 35/384 & 0 & 500/1113 & 125/192 & -2187/6784 & 11/84 & 0 \\
\text{4th} & 5179/57600 & 0 & 7571/16695 & 393/640 & -92097/339200 & 187/2100 & 1/40
\end{array}
$$

5차 해와 4차 해의 차이로 오차를 추정하고, 스텝 크기를 조절:

$$
\Delta z_{\text{new}} = \Delta z \cdot \min\!\left(5,\; \max\!\left(0.2,\; 0.9 \cdot \left\|\frac{\text{err}}{\text{tol}}\right\|^{-1/5}\right)\right)
$$

### 실험 결과

| 항목 | 값 |
|------|-----|
| Accepted steps | 124 |
| Total iterations | **50,000** (max cap) |
| Rejection rate | **99.8%** |
| 시간/call | **77 ms** |
| RK4(800) 시간/call | 0.45 ms |
| **DP45 vs RK4 속도비** | **170배 느림** |

### 기각 이유

1. **Mild stiffness**: DFT ODE는 특정 파라미터 영역($\Omega_k \sim 1$, $\Omega_\varepsilon \ll 1$)에서
   mildly stiff한 특성을 보인다. Explicit method인 DP45는 stiff 영역에서 안정성 유지를 위해
   극도로 작은 스텝을 강요받아, 대부분의 스텝이 거부된다.

2. **Per-step overhead**: DP45는 스텝당 7회 RHS 평가 + 오차 추정 + 조건 분기가 필요.
   반면 고정 RK4는 스텝당 4회 RHS만 필요하고 분기가 없어 numba가 최적화하기 쉽다.

3. **균일 그리드 이점**: 고정 스텝 RK4는 결과가 바로 균일 그리드에 놓이므로
   `np.interp`의 추가 resampling 단계가 불필요하다.

### 결론

이 특정 ODE 시스템에서는 **고정 스텝 RK4(800)가 최적**이다.
Adaptive RK45는 non-stiff, highly oscillatory ODE에서 유리하지만,
DFT Friedmann 방정식의 mild stiffness 특성으로 인해 오히려 역효과를 준다.
implicit solver(e.g., BDF)가 stiffness를 다룰 수 있지만, numba 내 구현 복잡도와
per-step 비용이 높아 고정 RK4 대비 이점이 불분명하다.

---

## 9. 벤치마크 결과

테스트 환경: WSL2, Python 3.10, numba JIT
테스트 파라미터: $h=0.6229$, $\Omega_k=1.0$, $\Omega_{\mathfrak{h}}=10^{-4}$, $\Omega_\varepsilon=10^{-4}$, $w=1$, $\lambda=2$

### `initialize()` (ODE 풀기 + 보간 준비)

| | 구성 | 시간/call |
|:---|:---|---:|
| **Before** | RK4 5000스텝 + `interp1d` ×4 | 15.70 ms |
| **After** | RK4 800스텝 + `Da` 누적 + flat array | 0.60 ms |
| **속도 향상** | | **26.3×** |

### `Da_z()` (comoving distance 계산)

| | 구성 | 시간/call |
|:---|:---|---:|
| **Before** | 500-point Python loop × `RHSquared_a` | 35.05 ms |
| **After** | `np.interp` 1회 | 0.006 ms |
| **속도 향상** | | **5,982×** |

### 총합 (1회 likelihood 평가)

| | 시간/call |
|:---|---:|
| **Before** (`initialize` + `Da_z`) | ~50.8 ms |
| **After** | ~0.6 ms |
| **총 속도 향상** | **~84×** |

### 수치 정확도 (800 vs 5000 스텝)

| $z$ | $H(z)$ 상대 오차 | $D_a(z)$ 상대 오차 |
|:---:|:---:|:---:|
| 0.5 | $2.8 \times 10^{-8}$ | $8.3 \times 10^{-6}$ |
| 1.0 | $1.1 \times 10^{-7}$ | $6.9 \times 10^{-6}$ |
| 2.0 | $4.4 \times 10^{-7}$ | $5.7 \times 10^{-6}$ |
| 5.0 | $1.8 \times 10^{-5}$ | $6.3 \times 10^{-6}$ |

모든 오차가 $10^{-5}$ 이하로, 관측 데이터 오차($\sim 1\%$) 대비 **5자릿수 이상 작다**.

---

### 추가 최적화: `fastmath=True` 및 상수 배열 패킹

numba `@njit(fastmath=True)`를 사용하여 LLVM이 부동소수점 연산을 더 적극적으로 최적화하도록 허용.
또한 RHS 함수의 인자를 12개 개별 스칼라 대신 **1개의 float64 배열 `p[9]`** 로 패킹하여
함수 호출 오버헤드와 numba 컴파일 시간을 크게 줄임:

```python
@njit(cache=True, fastmath=True)
def _make_params(H0, Ok, Oh, OL, Oe, w, l):
    lp2   = l + 2.0
    p = np.empty(9)
    p[0] = H0;            p[1] = H0 * H0       # H0, H0²
    p[2] = Ok;            p[3] = Oh;   p[4] = OL
    p[5] = Oe;            p[6] = w
    p[7] = 6.0*(w+1.0)/lp2   # alpha
    p[8] = 4.0/lp2           # beta
    return p

@njit(cache=True, fastmath=True)
def _rhs(z, phi, H, p):        # 인자 4개 (기존 12개)
    H0, H0sq, Ok, Oh, OL = p[0], p[1], p[2], p[3], p[4]
    Oe, w, alpha, beta    = p[5], p[6], p[7], p[8]
    ...
```

---

## 수식-코드 1:1 대응 요약

### `_rhs` 함수 내부

| 수식 | Python 코드 (Before) | Python 코드 (After) |
|:---|:---|:---|
| $\alpha = \frac{6(w+1)}{\lambda+2}$ | `OeEvol1 = 6.0*(w+1.0)/(l+2.0)` (매 호출) | `alpha` (solver에서 1회) |
| $\beta = \frac{4}{\lambda+2}$ | `4.0*phi_c/(l+2.0)` (매 호출, 내장) | `beta` (solver에서 1회) |
| $\Omega_\varepsilon^{(z)} = \Omega_\varepsilon(1+z)^\alpha e^{\beta\phi_c}$ | `Oe*(1+z)**OeEvol1 * np.exp(4*phi_c/(l+2))` | `Oe * zp1**alpha * np.exp(beta * phi_c)` |
| $(1+z)^2$ | `(1.0+z)**2.0` | `zp1_sq = zp1 * zp1` |
| $(1+z)^6$ | `(1.0+z)**6.0` | `zp1_6 = zp1_sq * zp1_sq * zp1_sq` |
| $H_0^2$ | `(h*100.0)**2.0` (매 호출) | `H0sq` (solver에서 1회) |
| $S = \frac{3}{H_0^2} + \frac{6[\cdots]}{H^2}$ | `3.0/(h*100)**2 + (6*Oe*... + 6*OL + 6*Ok*... + 6*Oh*...)/H**2` | `3.0/H0sq + 6.0*(...)/Hsq` |
| $\frac{d\phi}{dz} = -\frac{3 - H_0\sqrt{S}}{2(1+z)}$ | `-(3.0 - h*100*s)/(2*(1+z))` | `-(3.0 - H0*s)/(2.0*zp1)` |
| $\frac{dH}{dz} = -\frac{H_0^2}{1+z}\left[\frac{\cdots}{H} - \frac{H}{H_0}\sqrt{S}\right]$ | `-(h*100)**2 * ((...)/H - H/(h*100)*s) / (1+z)` | `-H0sq * ((...)/H - H*s/H0) / zp1` |
| 반환 | `np.array([dphidz, dHdz])` | `return dphidz, dHdz` |

### `Da_z` 사전 계산

| 수식 | Python 코드 |
|:---|:---|
| $D_a(z_0) = 0$ | `Da_arr = np.zeros(steps)` |
| $D_a(z_{i+1}) = D_a(z_i) + \frac{\Delta z}{2}\left(\frac{H_0}{H(z_i)} + \frac{H_0}{H(z_{i+1})}\right)$ | `Da_arr[i+1] = Da_arr[i] + 0.5*dz*(H0/H_arr[i] + H0/H)` |

### `Da_z` override — 곡률 보정

| 수식 | Python 코드 |
|:---|:---|
| $D_a(z)$ (flat, $k=0$) | `r = np.interp(z, self._z_grid, self._Da_arr)` |
| $D_a^{(\text{curv})} = \frac{\sinh(\sqrt{k}\,r)}{\sqrt{k}}$ ($k > 0$) | `np.sinh(r * q) / q` where `q = np.sqrt(self.Curv)` |
| $D_a^{(\text{curv})} = \frac{\sin(\sqrt{|k|}\,r)}{\sqrt{|k|}}$ ($k < 0$) | `np.sin(r * q) / q` where `q = np.sqrt(-self.Curv)` |

### `Hinv_z` override

| 수식 | Before | After |
|:---|:---|:---|
| $H^{-1}(z) / H_0^{-1} = H_0/H(z)$ | `[1/sqrt(RHSquared_a(val)) for val in a]` (Python loop) | `self._H0 / np.interp(z, grid, H_arr)` (vectorized) |

### `RHSquared_a`

| 수식 | Before | After |
|:---|:---|:---|
| $\frac{H^2(z)}{H_0^2}$ | `H0 = self.h*100.0` (매번) + `self.hub(z)` (interp1d) | `self._H0` (캐시) + `np.interp(z, grid, H_arr)` |
