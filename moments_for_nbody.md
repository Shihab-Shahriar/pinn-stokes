# Moments-based neighbourhood encoding for the n-body correction $m_t^{(n)}$

**Source.** ChatGPT conversation *"Compare Neighbor Representation"*,
<https://chatgpt.com/share/6a934876-af50-83e9-89bb-4845c5c673d7> (parsed 2026-08-29), which
discussed the slide deck `nemo_21_short.pdf` (the "21-coefficient" proposal with a pooled
neighbourhood direction `q`) and a screenshot proposing the descriptors `s_a, v_a, Q_a`.
Sections 1-3 record the chat's argument. Sections 4-8 are the design as adapted to the NeMO
paper's conventions (Eq. 15/17/19 block structure, target $i$ / source $j$, swap-even invariant
inputs) and to this repo; they are ours.

**Design decisions (v2, 2026-08-29).** Moments up to $l = 2$ only. Neighbourhood cutoff
$r_c = 8.0$ (measured from the pair midpoint; was 6.0). Eight radial bands of unit width,
$9$ components per band, $8 \times 9 = 72$ moment components per pair. Tensor bases extend
the paper's Eq. (17) so that $M_{ij} = M_{ji}^{T}$ holds by construction with the paper's
mechanism (rotation-invariant, pair-symmetric scalar coefficients on analytic bases); no
explicit symmetrisation pass and no positive-definiteness construction.

**One-sentence version.** Instead of one average direction $\mathbf q$ (or ten padded
per-neighbour scalar rows), describe the neighbourhood of a pair by its count ($s_a$), dipole
($\mathbf v_a$) and quadrupole ($Q_a$) in each of eight unit-width distance bands up to
$r_c = 8$; build pair-symmetric rotational invariants from them; let the existing MLP emit
scalar coefficients; multiply those onto analytic tensor bases built from $\hat{\mathbf z}$,
$\mathbf v_a$, $Q_a$ and $E(\cdot)$, exactly as the paper does with $I$, $\hat{\mathbf z}\hat{\mathbf z}^T$, $E(\hat{\mathbf z})$.

---

## 0. Notation (NeMO paper conventions)

| symbol | meaning |
|---|---|
| $i$, $j$ | target and source particle of the ordered pair; sphere radius $a = 1$ |
| $\mathbf r_{ij} = \mathbf x_i - \mathbf x_j$, $\;\ell = \lVert \mathbf r_{ij} \rVert$, $\;\hat{\mathbf z} = \mathbf r_{ij}/\ell$ | separation (source $\to$ target), distance, pair axis |
| $\mathbf m_{ij} = \tfrac12(\mathbf x_i + \mathbf x_j)$ | pair midpoint |
| $r_c = 8.0$ | neighbourhood radius about $\mathbf m_{ij}$ |
| $\mathcal N_{ij} = \{\,k \ne i,j : \lVert \mathbf x_k - \mathbf m_{ij} \rVert \le r_c\,\}$ | neighbourhood of the pair; $K = \lvert \mathcal N_{ij} \rvert$ is **not** capped |
| $\mathbf r_k = \mathbf x_k - \mathbf m_{ij}$, $\; r_k = \lVert \mathbf r_k \rVert$, $\; \hat{\mathbf r}_k = \mathbf r_k / r_k$ | neighbour vector from the midpoint (the paper's $\mathbf v_k$), distance, unit vector |
| $a = 1, \ldots, 8$ | radial **band** index (never a neighbour index) |
| $w_a(r)$ | band weight function, Section 3.1 |
| $s_a \in \mathbb R$, $\mathbf v_a \in \mathbb R^3$, $Q_a \in \mathbb R^{3\times3}$ | band moments, $l = 0, 1, 2$ |
| $\phi_{ij} = [\tilde d_{ij}, \tilde d_{ij}^2, \tilde d_{ij}^3, h_{ij}]$ | the paper's pair features ($\tilde d = \ell - \bar d_{\text{train}}$, $h$ = surface gap) |
| $E(\mathbf u) = [\varepsilon_{abc} u_c]_{ab}$ | skew matrix of $\mathbf u$; $E(\mathbf u)^T = -E(\mathbf u)$, $E(\mathbf u)\mathbf w = \mathbf u \times \mathbf w$ up to sign convention |
| $S(X) = X + X^{T}$, $\;\operatorname{Alt}(X) = X - X^{T}$ | symmetric / antisymmetric part (times 2) |
| $M_{ij} \in \mathbb R^{6\times6}$ | off-diagonal block: $[\mathbf U_i; \boldsymbol\Omega_i] = M_{ij}[\mathbf F_j; \mathbf T_j]$ |
| $I$ | $3\times3$ identity |

Paper Eq. (15):

$$
M_{ij} = m_t^{(2)}(X_i, X_j) + m_t^{(n)}(X_i, X_j, \mathcal N_{ij}),
$$

and this document is about the neighbourhood input and the tensor structure of $m_t^{(n)}$.

---

## 1. Baseline: the paper's five-scalar block (Eq. 17)

$$
M_{ij} =
\begin{bmatrix}
A_1 I + B_1 \hat{\mathbf z}\hat{\mathbf z}^{T} & C_1 E(\hat{\mathbf z}) \\
C_1 E(\hat{\mathbf z}) & A_2 I + B_2 \hat{\mathbf z}\hat{\mathbf z}^{T}
\end{bmatrix},
\qquad A_1, B_1, C_1, A_2, B_2 \ \text{rotation-invariant scalars}.
$$

In code: `L1 = d d^T`, `L2 = I - d d^T`, `L3 = [d]_x` in `src/model_archs.py`; $m_t^{(n)}$
currently reuses this exact form (`MultiBodyCorrection.predict_velocity`) with the five
scalars additionally depending on the neighbourhood. The block is therefore **axisymmetric
about $\hat{\mathbf z}$** no matter what the neighbourhood looks like.

Reciprocity in this form: swapping $i \leftrightarrow j$ sends $\hat{\mathbf z} \to -\hat{\mathbf z}$; $I$ and $\hat{\mathbf z}\hat{\mathbf z}^T$ are unchanged and symmetric, $E(\hat{\mathbf z})$ flips sign and is antisymmetric, so if the five scalars are the same for $(i,j)$ and $(j,i)$ then $M_{ji} = M_{ij}^T$ identically. This is the mechanism we keep (Section 5).

---

## 2. The 21-coefficient proposal and where it loses information (from the chat)

The slides build a third axis from the neighbourhood:

$$
\mathbf{pool} = \sum_{k} w_k\, \hat{\mathbf r}_k,
\qquad
\mathbf q = \mathbf{pool} - (\mathbf{pool}\cdot\hat{\mathbf z})\,\hat{\mathbf z},
\qquad
\hat{\mathbf p} = \hat{\mathbf z} \times \hat{\mathbf q},
$$

then use the frame $(\hat{\mathbf z}, \hat{\mathbf q}, \hat{\mathbf p})$ to span $6\ (TT) + 6\ (RR) + 9\ (RT) = 21$ coefficients.

Chat verdict:

1. Not a three-sphere-only formulation: the sum runs over any $K$ and is permutation invariant.
2. $21 = 6\cdot7/2$ is the number of independent entries of a symmetric $6\times6$ matrix; **the output space is not the weak point.**
3. The weak point is the encoder $\{\mathbf r_k\} \to \mathbf q$, a single **first ($l=1$) moment**:
   * $\mathbf r_1 = (0,+a,0)$, $\mathbf r_2 = (0,-a,0)$ give $\hat{\mathbf r}_1 + \hat{\mathbf r}_2 = 0 \Rightarrow \mathbf q = 0$: indistinguishable from *no neighbours*.
   * Many configurations share the same pooled vector but screen the pair very differently.
   * $\mathbf q \approx 0$ makes $\hat{\mathbf q} = \mathbf q/\lVert\mathbf q\rVert$ and $\hat{\mathbf p}$ unstable.

---

## 3. Band moments up to $l = 2$

### 3.1 Radial bands ($r_c = 8$, eight bands of width 1)

Band $a$ covers $[a-1, a]$, centre $r_a = a - \tfrac12$ ($0.5, 1.5, \ldots, 7.5$), with a tent weight

$$
w_a(r) = \max\bigl(0,\; 1 - \lvert r - r_a \rvert\bigr),
\qquad
w_1(r) = 1 \ \text{for } r \le r_1,
\qquad
w_8(r) = 1 \ \text{for } r_8 \le r \le r_c,
\qquad
w_a(r) = 0 \ \text{for } r > r_c .
$$

This is a partition of unity on $[0, r_c]$:

$$
\sum_{a=1}^{8} w_a(r) = 1 \quad \forall\, r \in [0, r_c],
$$

so a neighbour at $r_k = 3.0$ contributes $\tfrac12$ to band 3 and $\tfrac12$ to band 4; the
features are continuous in every $r_k$ (no jumps as a neighbour crosses a band edge) and
the hard cutoff at $r_c$ is the same convention as the rest of the near field. One shared
weight $w_a$ is used for all three moment orders (the chat's $f_a, g_a, h_a$ collapsed to one
soft histogram), so the radial dependence is left to the MLP, in the spirit of the paper's
polynomial distance features.

### 3.2 Definitions

$$
s_a = \sum_{k \in \mathcal N_{ij}} w_a(r_k)
\qquad\qquad (l = 0,\ 1\ \text{component: soft neighbour count in band } a)
$$

$$
\mathbf v_a = \sum_{k \in \mathcal N_{ij}} w_a(r_k)\, \hat{\mathbf r}_k
\qquad\qquad (l = 1,\ 3\ \text{components: dipole of band } a)
$$

$$
Q_a = \sum_{k \in \mathcal N_{ij}} w_a(r_k)\, \Bigl( \hat{\mathbf r}_k \hat{\mathbf r}_k^{T} - \tfrac13 I \Bigr)
\qquad (l = 2,\ 5\ \text{components: quadrupole of band } a)
$$

$Q_a$ is symmetric and traceless ($\operatorname{tr}(\hat{\mathbf r}\hat{\mathbf r}^T) = 1 = \operatorname{tr}\tfrac13 I$), hence 5 independent numbers. Per band $1 + 3 + 5 = 9$; over 8 bands the **moment descriptor is 72 numbers** for any $K$, with no ordering, padding or mask. Because $\sum_a w_a = 1$, $\sum_a s_a = K$.

### 3.3 Transformation laws

Rigid rotation of the configuration, $\mathbf x \to R\mathbf x$ with $R \in SO(3)$: $\mathbf r_k \to R\mathbf r_k$, $r_k \to r_k$, $\hat{\mathbf r}_k \to R\hat{\mathbf r}_k$, and $R$ factors out of every sum:

$$
s_a \to s_a, \qquad \mathbf v_a \to R\,\mathbf v_a, \qquad Q_a \to R\,Q_a\,R^{T}.
$$

Reflections ($R \in O(3)$, $\det R = -1$): identical laws, because $\hat{\mathbf r}_k$ is a polar vector (no cross products in the definitions). Label swap $i \leftrightarrow j$: $\mathbf m_{ij}$ is unchanged, so

$$
s_a,\ \mathbf v_a,\ Q_a \ \text{are swap-even}, \qquad \hat{\mathbf z} \to -\hat{\mathbf z} \ \text{is swap-odd}.
$$

Permutation of neighbours: invariant (plain sums).

### 3.4 What each order sees

$(-\hat{\mathbf r})(-\hat{\mathbf r})^T = \hat{\mathbf r}\hat{\mathbf r}^T$, so $Q_a$ records an **axis** even when opposite directions cancel in $\mathbf v_a$. For $\hat{\mathbf r}_1 = (0,1,0)$, $\hat{\mathbf r}_2 = (0,-1,0)$ in the same band:

$$
\mathbf v_a = 0,
\qquad
Q_a = \operatorname{diag}(0, 2, 0) - \tfrac23 I = \operatorname{diag}\bigl(-\tfrac23, \tfrac43, -\tfrac23\bigr) \ne 0 .
$$

| environment of the pair | $s_a$ | $\mathbf v_a$ | $Q_a$ |
|---|---:|---:|---:|
| no neighbours | 0 | 0 | 0 |
| isotropic shell | large | ~0 | ~0 |
| one neighbour above, one below | nonzero | 0 | nonzero |
| cluster on one side | nonzero | nonzero | usually nonzero |
| four neighbours in a cross around the pair | nonzero | ~0 | nonzero |

Multipole reading: $s$ = "how much", $\mathbf v$ = "which side", $Q$ = "which axis/plane". In spherical-harmonic terms these are the $l = 0, 1, 2$ terms ($2l + 1 = 1, 3, 5$ components) of the neighbour density in each band. **We stop at $l = 2$.**

---

## 4. Network input: pair-symmetric rotational invariants

The Cartesian components of $\mathbf v_a$, $Q_a$ are **not** fed to the MLP (that would make
$x, y, z$ special and break equivariance). Following the paper, the MLP input contains only
quantities that are rotation-invariant **and even under $i \leftrightarrow j$**; odd
quantities ($\hat{\mathbf z}\cdot\mathbf v_a$, $\hat{\mathbf z}^TQ_a\mathbf v_a$) enter only through even products. Per band $a$, nine invariants:

$$
\mathbf z_{ij} = \Bigl[\;
\phi_{ij};\;
\bigl\{\,
s_a,\;
\lVert \mathbf v_a \rVert^2,\;
(\hat{\mathbf z}\cdot\mathbf v_a)^2,\;
\hat{\mathbf z}^{T} Q_a \hat{\mathbf z},\;
\lVert Q_a \hat{\mathbf z} \rVert^2,\;
\operatorname{tr}(Q_a^2),\;
\operatorname{tr}(Q_a^3),\;
\mathbf v_a^{T} Q_a \mathbf v_a,\;
(\hat{\mathbf z}\cdot\mathbf v_a)(\hat{\mathbf z}^{T} Q_a \mathbf v_a)
\,\bigr\}_{a=1}^{8}
\;\Bigr] \in \mathbb R^{4 + 72 = 76}.
$$

Count check per band: $\mathbf v_a$ relative to $\hat{\mathbf z}$ has 2 even invariants, $Q_a$ relative to $\hat{\mathbf z}$ has 4, the $\mathbf v_a$-$Q_a$ relative azimuth 2, plus $s_a$: 9. Cross-band invariants ($\mathbf v_a\cdot\mathbf v_b$, $\operatorname{tr}(Q_aQ_b)$, $a \ne b$) are omitted; add them if needed.

$$
\bigl(c_1, \ldots, c_P\bigr) = \operatorname{MLP}(\mathbf z_{ij}),
\qquad P = 93 \ \text{(Section 5.2)},
$$

with the paper's $m_t^{(n)}$ architecture (6 layers, widths alternating 64/128) and a 93-wide output layer. All $c_\beta$ are rotation-invariant and swap-even by construction.

---

## 5. Tensor bases and reciprocity

### 5.1 The rule

Write $m_t^{(n)} = \begin{bmatrix} \Delta^{TT} & \Delta^{TR} \\ \Delta^{RT} & \Delta^{RR} \end{bmatrix}$ with each block a sum $\sum_\beta c_\beta\, T_\beta$ of a scalar times a basis tensor built from $\{I, \hat{\mathbf z}, \mathbf v_a, Q_a, E(\cdot)\}$. With swap-even coefficients, $M_{ji} = M_{ij}^{T}$ holds term by term iff every basis tensor satisfies

$$
T(-\hat{\mathbf z}) = T(\hat{\mathbf z})^{T}
\quad\Longleftrightarrow\quad
(\hat{\mathbf z}\text{-even and symmetric}) \ \text{or}\ (\hat{\mathbf z}\text{-odd and antisymmetric}),
$$

using $\Delta^{RT} = \Delta^{TR}$ with the same basis, as in Eq. (17). ($\mathbf v_a$, $Q_a$ are $\hat{\mathbf z}$-even, so "$\hat{\mathbf z}$-odd" just counts explicit factors of $\hat{\mathbf z}$.) Additionally, for $O(3)$ correctness, $\Delta^{TT}$ and $\Delta^{RR}$ must be true tensors (no $E(\cdot)$ / cross product, or an even number of them) and $\Delta^{TR}$, $\Delta^{RT}$ pseudotensors (exactly one), because torque and angular velocity are axial.

Consequences worth knowing: the symmetric $\hat{\mathbf z}$-odd tensor $S(\hat{\mathbf z}\mathbf v_a^T)$ is **forbidden** in $TT$ (it would violate reciprocity), unless multiplied by an odd invariant such as $(\hat{\mathbf z}\cdot\mathbf v_a)$; and $\operatorname{Alt}(Q_a E(\hat{\mathbf z})) = -E(Q_a\hat{\mathbf z})$ for traceless symmetric $Q_a$, so only one of them is needed.

### 5.2 The chosen basis

$TT$ and $RR$ (true tensors), $2 + 4\times8 = 34$ each:

$$
\bigl\{\; I,\;\; \hat{\mathbf z}\hat{\mathbf z}^{T} \;\bigr\}
\;\cup\;
\bigl\{\;
Q_a,\;\;
S(\hat{\mathbf z}\hat{\mathbf z}^{T} Q_a),\;\;
\mathbf v_a \mathbf v_a^{T},\;\;
\operatorname{Alt}(\hat{\mathbf z}\mathbf v_a^{T})
\;\bigr\}_{a=1}^{8}
$$

$TR = RT$ (pseudotensors), $1 + 3\times8 = 25$:

$$
\bigl\{\; E(\hat{\mathbf z}) \;\bigr\}
\;\cup\;
\bigl\{\;
E(Q_a \hat{\mathbf z}),\;\;
S\bigl(\hat{\mathbf z}\,(\hat{\mathbf z}\times\mathbf v_a)^{T}\bigr),\;\;
(\hat{\mathbf z}\cdot\mathbf v_a)\, E(\mathbf v_a)
\;\bigr\}_{a=1}^{8}
$$

Total $P = 34 + 34 + 25 = 93$ scalar coefficients (vs. 5 today). The first element of each set is the paper's basis; setting all other coefficients to zero recovers Eq. (17) exactly.

| basis tensor | $\hat{\mathbf z}$ parity | transpose | $O(3)$ type | block |
|---|---|---|---|---|
| $I$, $\hat{\mathbf z}\hat{\mathbf z}^T$, $Q_a$, $S(\hat{\mathbf z}\hat{\mathbf z}^TQ_a)$, $\mathbf v_a\mathbf v_a^T$ | even | symmetric | true | $TT$, $RR$ |
| $\operatorname{Alt}(\hat{\mathbf z}\mathbf v_a^T)$ | odd | antisymmetric | true | $TT$, $RR$ |
| $E(\hat{\mathbf z})$, $E(Q_a\hat{\mathbf z})$ | odd | antisymmetric | pseudo | $TR$, $RT$ |
| $S(\hat{\mathbf z}(\hat{\mathbf z}\times\mathbf v_a)^T)$ | even | symmetric | pseudo | $TR$, $RT$ |
| $(\hat{\mathbf z}\cdot\mathbf v_a)E(\mathbf v_a)$ | odd | antisymmetric | pseudo | $TR$, $RT$ |

Note that $\Delta^{TT}$ is no longer symmetric in general ($\operatorname{Alt}(\hat{\mathbf z}\mathbf v_a^T)$); only $M_{ij} = M_{ji}^T$ is required physically, and it holds.

### 5.3 Assembly and the residual form

$$
\Delta^{TT} = \sum_{\beta=1}^{34} c^{TT}_\beta(\mathbf z_{ij})\, T^{TT}_\beta,
\qquad
\Delta^{RR} = \sum_{\beta=1}^{34} c^{RR}_\beta(\mathbf z_{ij})\, T^{TT}_\beta,
\qquad
\Delta^{TR} = \Delta^{RT} = \sum_{\beta=1}^{25} c^{TR}_\beta(\mathbf z_{ij})\, T^{TR}_\beta,
$$

$$
M_{ij} = m_t^{(2)}(X_i, X_j) + m_t^{(n)}(X_i, X_j, \mathcal N_{ij}),
\qquad
m_t^{(n)} =
\begin{bmatrix} \Delta^{TT} & \Delta^{TR} \\ \Delta^{RT} & \Delta^{RR} \end{bmatrix},
$$

and the velocity is $[\mathbf U_i; \boldsymbol\Omega_i] \mathrel{+}= m_t^{(n)} [\mathbf F_j; \mathbf T_j]$ as today. Properties, all architectural:

* rotation/reflection equivariance: $M(RX) = \operatorname{diag}(R, \det R\cdot R)\, M(X)\, \operatorname{diag}(R, \det R\cdot R)^T$;
* reciprocity $M_{ji} = M_{ij}^T$, hence a symmetric near-field grand mobility, because the ordered pairs $(i,j)$ and $(j,i)$ see the same swap-even invariants and bases obeying Section 5.1 — the same mechanism as Eq. (17)/(19), no extra pass;
* permutation invariance over neighbours and no cap on $K$.

Alternative (from the chat) if one ever wants odd invariants inside the MLP: predict an unconstrained equivariant $A_{ij}$ and set $M_{ij} = \tfrac12(A_{ij} + A_{ji}^T)$. It is exact too but needs both orientations of a pair in the same evaluation; not needed with the basis above.

---

## 6. Where this sits

| level | encoder | output structure | status |
|---|---|---|---|
| 0 | $\ell$, $\hat{\mathbf z}$ (+ 10 padded scalar rows per neighbour for $m_t^{(n)}$) | 5 scalars on $I$, $\hat{\mathbf z}\hat{\mathbf z}^T$, $E(\hat{\mathbf z})$ | current NeMO |
| 1 | $\mathbf q = \sum_k w_k\hat{\mathbf r}_k$, frame $(\hat{\mathbf z}, \hat{\mathbf q}, \hat{\mathbf z}\times\hat{\mathbf q})$ | 21 coefficients | slide proposal; $l = 1$ only; unstable at $\mathbf q \approx 0$ |
| 2 | $\{s_a, \mathbf v_a, Q_a\}_{a=1}^{8}$, $l \le 2$, $r_c = 8$ | 76 invariants $\to$ MLP $\to$ 93 coefficients on the Section 5.2 bases | **this design** |

Higher angular orders and message passing are deliberately out of scope.

---

## 7. Reference implementation (numpy; verified for partition of unity, equivariance under $O(3)$, and $M_{ji} = M_{ij}^T$)

```python
import numpy as np
from numpy import einsum, eye
I3, NB, RC = eye(3), 8, 8.0

def E(u):                         # E(u)_{ab} = eps_{abc} u_c
    ux, uy, uz = u
    return np.array([[0, uz, -uy], [-uz, 0, ux], [uy, -ux, 0]])
def S(X):   return X + X.T
def Alt(X): return X - X.T

def band_weights(r):              # (K,) -> (K,NB), tent partition of unity on [0, RC]
    centres = np.arange(1, NB + 1) - 0.5                    # 0.5 ... 7.5
    w = np.clip(1.0 - np.abs(r[:, None] - centres[None, :]), 0.0, None)
    w[r <= centres[0], 0] = 1.0
    w[r >= centres[-1], -1] = 1.0
    w[r > RC, :] = 0.0
    return w

def moments(xi, xj, xk):          # xk: (K,3) neighbours within RC of the midpoint
    m = 0.5 * (xi + xj)
    rij = xi - xj; ell = np.linalg.norm(rij); z = rij / ell
    rk = xk - m; rn = np.linalg.norm(rk, axis=1); rh = rk / rn[:, None]
    W = band_weights(rn)                                        # (K,NB)
    s = W.sum(0)                                                # (NB,)      l=0
    v = einsum('ka,ki->ai', W, rh)                              # (NB,3)     l=1
    Q = einsum('ka,ki,kj->aij', W, rh, rh) - s[:, None, None] * I3[None] / 3   # (NB,3,3) l=2
    return z, s, v, Q

def invariants(z, s, v, Q):       # 9 per band, all rotation-invariant and swap-even
    zv  = v @ z
    Qz  = einsum('aij,j->ai', Q, z)
    return np.concatenate([
        s,
        (v * v).sum(1),
        zv ** 2,
        einsum('i,aij,j->a', z, Q, z),
        (Qz * Qz).sum(1),
        einsum('aij,aji->a', Q, Q),
        einsum('aij,ajk,aki->a', Q, Q, Q),
        einsum('ai,aij,aj->a', v, Q, v),
        zv * einsum('i,aij,aj->a', z, Q, v),
    ])                                                          # (72,)

def bases(z, v, Q):               # TT/RR bases (34,3,3) and TR/RT bases (25,3,3)
    zz = np.outer(z, z); zv = v @ z
    tt = [I3, zz]
    tr = [E(z)]
    for a in range(NB):
        tt += [Q[a], S(zz @ Q[a]), np.outer(v[a], v[a]), Alt(np.outer(z, v[a]))]
        tr += [E(Q[a] @ z), S(np.outer(z, np.cross(z, v[a]))), zv[a] * E(v[a])]
    return np.array(tt), np.array(tr)

def nbody_block(z, s, v, Q, mlp): # mlp: (76,) -> (93,)  (prepend the 4 pair features to invariants)
    c = mlp(np.concatenate([pair_features, invariants(z, s, v, Q)]))
    tt, tr = bases(z, v, Q)
    TT = einsum('b,bij->ij', c[:34],   tt)
    RR = einsum('b,bij->ij', c[34:68], tt)
    TR = einsum('b,bij->ij', c[68:93], tr)
    M = np.zeros((6, 6)); M[:3, :3] = TT; M[:3, 3:] = TR; M[3:, :3] = TR; M[3:, 3:] = RR
    return M
```

Checks run on this code (random configuration, random coefficient function of the invariants): $\sum_a w_a \equiv 1$ on $[0, 8]$; $\sum_a s_a = K$; $s_a, \mathbf v_a, Q_a$ unchanged and $\hat{\mathbf z}$ flipped under $i \leftrightarrow j$; $M_{ji} = M_{ij}^T$ to machine precision with a non-symmetric $TT$; $M(PX) = \operatorname{diag}(P, \det P\, P)\,M(X)\,\operatorname{diag}(P, \det P\, P)^T$ for a random rotation, a reflection, and a random improper rotation.

---

## 8. Mapping to this repo

**Current $m_t^{(n)}$ path** (`src/mob_op_nbody.py`, `src/gpu_nbody_mob.py`, `MultiBodyCorrection` in `src/model_archs.py`, paper Section 2.3.2):

* `DEFAULT_NEIGHBOR_CUTOFF = 6.0` is used both as the pair-distance gate for applying the correction and as the midpoint radius for neighbours; `_select_neighbor_indices` keeps the `max_neighbors = 10` closest to the midpoint, sorted by index.
* `_compute_neighbor_features`: 10 swap-even scalars per neighbour (`r_sum, r_diff, r_prod, |u|/l, (u/l)^2, rho/l, inv_sum, inv_prod, inv_absdiff, cos_k`), zero-padded to $10\times10$ plus a 10-mask, concatenated with raw positions; 114-dim input.
* `MultiBodyCorrection.predict_velocity`: 5 outputs on `L1/L2/L3(d_hat)`, times the source force.

**What changes with this design:**

| item | now | proposed |
|---|---|---|
| neighbourhood radius about $\mathbf m_{ij}$ | 6.0 | $r_c = 8.0$ (pair-distance gate for the correction is a separate parameter; keep the near-field split as is) |
| neighbour count | $\le 10$, top-$k$ + padding + mask | unbounded, segment-sum |
| per-pair neighbourhood descriptor | $10 \times 10 + 10$ | $8 \times 9 = 72$ moment components ($s_a, \mathbf v_a, Q_a$) |
| MLP input | 114 | $4 + 72 = 76$ swap-even invariants |
| MLP output | 5 | 93 |
| tensor bases | $I$, $\hat{\mathbf z}\hat{\mathbf z}^T$, $E(\hat{\mathbf z})$ | those + the 88 moment-built bases of Section 5.2 |
| block anisotropy | axisymmetric about $\hat{\mathbf z}$ | general, $O(3)$-equivariant |
| reciprocity | by Eq. (17) + even features | same mechanism, Section 5.1 rule |

**Why the current descriptor is weaker than it looks.** Each neighbour's 10 scalars fix its position only up to azimuth about $\hat{\mathbf z}$ and the reflection $u \to -u$; the **relative azimuth between neighbours is not represented at all** (two neighbours on the same side vs. opposite sides of the pair look identical), and the output block cannot express transverse anisotropy anyway. $Q_a$'s transverse components and the $\mathbf v_a$-based bases carry exactly that.

**Implementation notes.**

* GPU path: the moments are a segment-sum over the pair's neighbour list — the same gather as `_per_particle_topk`, with $72$ accumulators per pair and no top-$k$. The neighbour set must still be built over the **complete** edge list, never per pair-chunk (see `CLAUDE.md` / `artifacts/widebvh_far_field_report.md` §5); the moments of the ordered pairs $(i,j)$ and $(j,i)$ are identical and can be computed once per unordered pair.
* Sign convention: the code's `d_vec` points target $\to$ source, i.e. $-\hat{\mathbf z}$. The $\hat{\mathbf z}$-odd bases flip sign with the convention, which is absorbed by the coefficients as long as the same convention is used in training and inference.
* Training data (paper Section 3): regenerate the $m_t^{(n)}$ samples with the candidate shell sized for $r_c = 8$ about the midpoint and with the full neighbourhood inside $r_c$ rather than the $K$ smallest $q_k = r_{ki}r_{kj}$; the staged residual training (subtract analytic self + $m_s^{(2)}$ + $m_t^{(2)}$) is unchanged.
* Cost: the extra output width (128 $\to$ 93) adds ~$1.2\times10^4$ multiply-adds per pair to the paper's $4\times10^4$; assembling 93 bases is $93 \times 9$ FMAs per block, comparable.
