#pragma once

// SPDX-License-Identifier: Apache-2.0
//
// VSH near-pair correction for the second-kind BI sphere mobility solver
// (Yan et al. 2020 close-pair scheme, sphereSim.md sec. 4.2/5.4): for every
// ordered sphere pair with center distance < beta*(R_i+R_j) = 2*beta, replace
// the smooth 162-point quadrature of the pair interaction (already computed by
// the treecode/brute off-diagonal operator) with the EXACT operator on the
// band-limited density:
//
//   corr_i += E(d_ij)[Areal zeta_j]  -  smoothquad(d_ij)[w o zeta_j]
//
// The exact side synthesizes the source sphere's exterior Stokes field from
// its packed real VSH coefficients using the per-mode Lamb radial power laws
// fitted offline by scripts/gen_sphere_selfblocks.py (nearmodes.bin), and is
// exact for the band-limited density -- no quadrature error at any gap. The
// traction (target-normal) is sigma.n = -p n + (grad u + grad u^T) n with
// grad u by central finite differences of the mode-summed velocity (h from
// nearmodes.bin; truncation ~1e-8) and p from the analytic pressure mode sum.
// This avoids transcribing the paper's Appendix A del-u expressions entirely.
//
// Everything here is solver-level: the treecode engine is untouched. Included
// by sphere_bim.cuh only (after treecode.cuh, which provides vec3d and
// stokes::*; CUDA_CHECK/CUBLAS_CHECK come from sphere_bim.cuh).
//
// Determinism: pairs are grouped by TARGET sphere (one block per target with
// >= 1 near source); each block exclusively owns its 3B output rows, so the
// accumulation is atomic-free and run-to-run identical.

#include <unordered_map>

// defined later in sphere_bim.cuh (same translation unit)
static std::vector<double> sbimReadBin(const std::string &path,
                                       size_t expectCount);
static double *sbimUpload(const std::vector<double> &h);

namespace snear {

// compile-time caps (p = 8 data uses nmax = 8; table rows go to nmax+1)
constexpr int SN_NMAX_CAP = 12;             // max spherical-harmonic order
constexpr int SN_PTAB_CAP = ((SN_NMAX_CAP + 2) * (SN_NMAX_CAP + 3)) / 2;

__device__ inline int snPidx(int n, int m) { return (n * (n + 1)) / 2 + m; }

// P_n^m lookup with the m = -1 extension P_n^{-1} = -P_n^1/(n(n+1)) and
// out-of-range zeros (division-free derivative identities need m-1, m+1).
__device__ inline double snPget(const double *P, int n, int m)
{
  if (m > n || -m > n) return 0.0;
  if (m == -1) return -P[snPidx(n, 1)] / (double)(n * (n + 1));
  return P[snPidx(n, m)];
}

// Mode-summed exterior velocity (and optionally pressure) of one source
// sphere at point xl in SOURCE-LOCAL coordinates. sh_tab rows are the 12-wide
// nearmodes.bin records [kind,n,m,off,norm,caR,cbR,caG,cbG,cX,cP,*]; sh_coef
// is the sphere's packed real coefficient vector (Areal * zeta, weights
// included). Angular factors use the division-free identities (CS phase)
//   dP_n^m/dtheta   =  ( P_n^{m+1} - (n+m)(n-m+1) P_n^{m-1} ) / 2
//   m P_n^m / sin   = -( P_{n+1}^{m+1} + (n-m+1)(n-m+2) P_{n+1}^{m-1} ) / 2
// validated by the near_dP_dtheta_identity / near_mP_over_sin_identity gates.
template <bool WANT_P>
__device__ void snEvalUP(double xlx, double xly, double xlz,
                         const double *sh_tab, const double *sh_coef,
                         int nkept, int nmax, double u[3], double *pout)
{
  const double r2 = xlx * xlx + xly * xly + xlz * xlz;
  const double r = sqrt(r2);
  const double invr = 1.0 / r;
  double ct = xlz * invr;
  ct = fmin(1.0, fmax(-1.0, ct));
  const double sxy = sqrt(xlx * xlx + xly * xly);
  const double st = sxy * invr;
  double cp = 1.0, sp = 0.0;
  if (sxy > 0.0) { cp = xlx / sxy; sp = xly / sxy; }

  const int ntab = nmax + 1;
  double P[SN_PTAB_CAP];
  P[0] = 1.0;
  for (int m = 1; m <= ntab; ++m)
    P[snPidx(m, m)] = -(double)(2 * m - 1) * st * P[snPidx(m - 1, m - 1)];
  for (int m = 0; m < ntab; ++m)
    P[snPidx(m + 1, m)] = ct * (double)(2 * m + 1) * P[snPidx(m, m)];
  for (int n = 2; n <= ntab; ++n)
    for (int m = 0; m <= n - 2; ++m)
      P[snPidx(n, m)] = (ct * (double)(2 * n - 1) * P[snPidx(n - 1, m)]
                         - (double)(n + m - 1) * P[snPidx(n - 2, m)])
                        / (double)(n - m);

  double cmt[SN_NMAX_CAP + 1], smt[SN_NMAX_CAP + 1];
  cmt[0] = 1.0; smt[0] = 0.0;
  for (int m = 1; m <= nmax; ++m) {
    cmt[m] = cmt[m - 1] * cp - smt[m - 1] * sp;
    smt[m] = smt[m - 1] * cp + cmt[m - 1] * sp;
  }
  double rp[SN_NMAX_CAP + 3];
  rp[0] = 1.0;
  for (int k = 1; k <= nmax + 2; ++k) rp[k] = rp[k - 1] * invr;

  double ur = 0.0, ut = 0.0, uph = 0.0, pacc = 0.0;
  for (int t = 0; t < nkept; ++t) {
    const double *row = sh_tab + 12 * t;
    const int n = (int)row[1], m = (int)row[2], off = (int)row[3];
    const double norm = row[4];
    const double a1 = sh_coef[off];
    const double A = norm * P[snPidx(n, m)];
    const double D = norm * 0.5
        * (snPget(P, n, m + 1)
           - (double)((n + m) * (n - m + 1)) * snPget(P, n, m - 1));
    const double fR = row[5] * rp[n] + row[6] * rp[n + 2];
    const double fG = row[7] * rp[n] + row[8] * rp[n + 2];
    const double fX = row[9] * rp[n + 1];
    if (m == 0) {
      ur += a1 * fR * A;
      ut += a1 * fG * D;
      uph += a1 * fX * D;
      if (WANT_P) pacc += a1 * row[10] * rp[n + 1] * A;
    } else {
      const double a2 = sh_coef[off + 1];
      const double M = -norm * 0.5
          * (snPget(P, n + 1, m + 1)
             + (double)((n - m + 1) * (n - m + 2)) * snPget(P, n + 1, m - 1));
      const double cmv = cmt[m], smv = smt[m];
      const double al = fG * D, be = fX * M, ga = fX * D, de = fG * M;
      ur += 2.0 * fR * A * (a1 * cmv - a2 * smv);
      ut += 2.0 * ((a1 * al + a2 * be) * cmv - (a2 * al - a1 * be) * smv);
      uph += 2.0 * ((a1 * ga - a2 * de) * cmv - (a1 * de + a2 * ga) * smv);
      if (WANT_P)
        pacc += 2.0 * row[10] * rp[n + 1] * A * (a1 * cmv - a2 * smv);
    }
  }
  // spherical -> Cartesian
  const double erx = st * cp, ery = st * sp, erz = ct;
  const double etx = ct * cp, ety = ct * sp, etz = -st;
  const double epx = -sp, epy = cp;
  u[0] = ur * erx + ut * etx + uph * epx;
  u[1] = ur * ery + ut * ety + uph * epy;
  u[2] = ur * erz + ut * etz;
  if (WANT_P) *pout = pacc;
}

// Below this sin(theta) the analytic traction falls back to central FD: the
// spherical-frame gradient components use 1/sin combinations whose round-off
// amplifies like eps/sin^2 (the Cartesian tensor is finite, the frame is
// singular). Python pole scan: analytic-vs-FD rel err 3e-8 @ st=1e-2,
// 8e-8 @ 1e-3, 2e-7 @ 3e-4 -- comfortably below the 2.6e-6 mode-sum truth
// floor at the threshold.
constexpr double SN_POLE_ST_MIN = 3e-4;

// Analytic traction of the mode-summed exterior field at xl (source-local),
// target normal nrm: t = -p n + (L + L^T) n with the velocity gradient L
// assembled per mode in the spherical frame. Per mode (omega = e^{im phi}
// folded via E1/E2):
//   u_r = fR A, u_th = fG D - i fX M, u_ph = fX D + i fG M
//   dD/dth = D2 = -ct/st D - n(n+1) A + (m/st) M      (Legendre ODE)
//   dM/dth = (m/st) D - ct/st M
// phi-derivatives multiply by i m; the 1/sin combos are guarded by
// SN_POLE_ST_MIN in the caller. Radial derivatives are exact power laws.
// Row convention: L[a][b] = d_a u_b in the {e_r,e_th,e_ph} frame; the
// symmetrization makes the convention immaterial. Matches the Python
// modesum_grad_matrices mirror (gated via near_ref.bin).
__device__ void snEvalTractionAnalytic(double xlx, double xly, double xlz,
                                       const vec3d nrm, const double *sh_tab,
                                       const double *sh_coef, int nkept,
                                       int nmax, double t[3])
{
  const double r2 = xlx * xlx + xly * xly + xlz * xlz;
  const double r = sqrt(r2);
  const double invr = 1.0 / r;
  double ct = xlz * invr;
  ct = fmin(1.0, fmax(-1.0, ct));
  const double sxy = sqrt(xlx * xlx + xly * xly);
  const double st = sxy * invr;
  const double ist = 1.0 / st;
  const double ctist = ct * ist;
  double cp = 1.0, sp = 0.0;
  if (sxy > 0.0) { cp = xlx / sxy; sp = xly / sxy; }

  const int ntab = nmax + 1;
  double P[SN_PTAB_CAP];
  P[0] = 1.0;
  for (int m = 1; m <= ntab; ++m)
    P[snPidx(m, m)] = -(double)(2 * m - 1) * st * P[snPidx(m - 1, m - 1)];
  for (int m = 0; m < ntab; ++m)
    P[snPidx(m + 1, m)] = ct * (double)(2 * m + 1) * P[snPidx(m, m)];
  for (int n = 2; n <= ntab; ++n)
    for (int m = 0; m <= n - 2; ++m)
      P[snPidx(n, m)] = (ct * (double)(2 * n - 1) * P[snPidx(n - 1, m)]
                         - (double)(n + m - 1) * P[snPidx(n - 2, m)])
                        / (double)(n - m);
  double cmt[SN_NMAX_CAP + 1], smt[SN_NMAX_CAP + 1];
  cmt[0] = 1.0; smt[0] = 0.0;
  for (int m = 1; m <= nmax; ++m) {
    cmt[m] = cmt[m - 1] * cp - smt[m - 1] * sp;
    smt[m] = smt[m - 1] * cp + cmt[m - 1] * sp;
  }
  double rp[SN_NMAX_CAP + 4];
  rp[0] = 1.0;
  for (int k = 1; k <= nmax + 3; ++k) rp[k] = rp[k - 1] * invr;

  double ur = 0, ut = 0, uph = 0, pacc = 0;
  double drr = 0, drt = 0, drp = 0;   // d/dr sums
  double dtr = 0, dtt = 0, dtp = 0;   // d/dtheta sums
  double fpr = 0, fpt = 0, fpp = 0;   // (1/sin) d/dphi sums
  for (int tt = 0; tt < nkept; ++tt) {
    const double *row = sh_tab + 12 * tt;
    const int n = (int)row[1], m = (int)row[2], off = (int)row[3];
    const double norm = row[4];
    const double A = norm * P[snPidx(n, m)];
    const double D = norm * 0.5
        * (snPget(P, n, m + 1)
           - (double)((n + m) * (n - m + 1)) * snPget(P, n, m - 1));
    double M = 0.0, E1, E2;
    if (m == 0) {
      E1 = sh_coef[off];
      E2 = 0.0;
    } else {
      M = -norm * 0.5
          * (snPget(P, n + 1, m + 1)
             + (double)((n - m + 1) * (n - m + 2)) * snPget(P, n + 1, m - 1));
      const double a1 = sh_coef[off], a2 = sh_coef[off + 1];
      E1 = 2.0 * (a1 * cmt[m] - a2 * smt[m]);
      E2 = 2.0 * (a1 * smt[m] + a2 * cmt[m]);
    }
    const double mist = (double)m * ist;
    const double D2 = fma(-ctist, D, fma(mist, M, -(double)(n * (n + 1)) * A));
    const double dM = fma(mist, D, -ctist * M);
    const double mDs = mist * D, mMs = mist * M;
    const double fR = row[5] * rp[n] + row[6] * rp[n + 2];
    const double fG = row[7] * rp[n] + row[8] * rp[n + 2];
    const double fX = row[9] * rp[n + 1];
    const double dfR = -(double)n * row[5] * rp[n + 1]
        - (double)(n + 2) * row[6] * rp[n + 3];
    const double dfG = -(double)n * row[7] * rp[n + 1]
        - (double)(n + 2) * row[8] * rp[n + 3];
    const double dfX = -(double)(n + 1) * row[9] * rp[n + 2];
    ur += fR * A * E1;
    ut += (fG * D) * E1 - (-fX * M) * E2;
    uph += (fX * D) * E1 - (fG * M) * E2;
    pacc += row[10] * rp[n + 1] * A * E1;
    drr += dfR * A * E1;
    drt += (dfG * D) * E1 - (-dfX * M) * E2;
    drp += (dfX * D) * E1 - (dfG * M) * E2;
    dtr += fR * D * E1;
    dtt += (fG * D2) * E1 - (-fX * dM) * E2;
    dtp += (fX * D2) * E1 - (fG * dM) * E2;
    fpr += -(fR * M) * E2;
    fpt += (fX * mMs) * E1 - (fG * mDs) * E2;
    fpp += (-fG * mMs) * E1 - (fX * mDs) * E2;
  }

  double Ls[3][3];
  Ls[0][0] = drr;                 Ls[0][1] = drt;                 Ls[0][2] = drp;
  Ls[1][0] = invr * (dtr - ut);   Ls[1][1] = invr * (dtt + ur);   Ls[1][2] = invr * dtp;
  Ls[2][0] = invr * (fpr - uph);
  Ls[2][1] = invr * (fpt - ctist * uph);
  Ls[2][2] = invr * (fpp + ur + ctist * ut);

  // spherical -> Cartesian: Lc = E Ls E^T, basis columns e_r, e_th, e_ph
  const double E[3][3] = {{st * cp, ct * cp, -sp},
                          {st * sp, ct * sp, cp},
                          {ct, -st, 0.0}};
  double Lc[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int a = 0; a < 3; ++a) {
        const double ea = E[i][a];
        s += ea * (Ls[a][0] * E[j][0] + Ls[a][1] * E[j][1]
                   + Ls[a][2] * E[j][2]);
      }
      Lc[i][j] = s;
    }
  const double n3[3] = {nrm.x, nrm.y, nrm.z};
  for (int c = 0; c < 3; ++c)
    t[c] = -pacc * n3[c]
        + (Lc[c][0] + Lc[0][c]) * n3[0]
        + (Lc[c][1] + Lc[1][c]) * n3[1]
        + (Lc[c][2] + Lc[2][c]) * n3[2];
}

// One block per target sphere that has near sources; thread b < B owns target
// point b. KIND 0 = traction (K operator), 1 = Stokeslet (S operator).
// doSub == 0 skips the smooth-quadrature subtraction (selftest of the exact
// side alone). outAoS is accumulated (+=).
template <int KIND>
__global__ void snCorrKernel(const int *csrRow, const int *csrTgt,
                             const int *csrSrc, const vec3d *pts,
                             const vec3d *nrm, const vec3d *centers,
                             const double *coefs, const double *q,
                             const double *modeTab, int nkept, int nreal,
                             int B, int nmax, double hfd, int doSub,
                             double smoothPref, double *outAoS)
{
  extern __shared__ double sh[];
  double *sh_tab = sh;                       // 12 * nkept
  double *sh_coef = sh_tab + 12 * nkept;     // nreal
  double *sh_q = sh_coef + nreal;            // 3 * B

  const int row = blockIdx.x;
  const int i = csrTgt[row];
  for (int t = threadIdx.x; t < 12 * nkept; t += blockDim.x)
    sh_tab[t] = modeTab[t];

  const int b = threadIdx.x;
  vec3d Tp{0, 0, 0}, nA{0, 0, 0};
  if (b < B) {
    Tp = pts[(size_t)i * B + b];
    nA = nrm[(size_t)i * B + b];
  }
  double corr[3] = {0.0, 0.0, 0.0};

  for (int e = csrRow[row]; e < csrRow[row + 1]; ++e) {
    const int j = csrSrc[e];
    __syncthreads();                         // shared reuse across sources
    for (int t = threadIdx.x; t < nreal; t += blockDim.x)
      sh_coef[t] = coefs[(size_t)j * nreal + t];
    for (int t = threadIdx.x; t < 3 * B; t += blockDim.x)
      sh_q[t] = q[(size_t)3 * j * B + t];
    __syncthreads();
    if (b >= B) continue;

    const vec3d cj = centers[j];
    const double xl0 = Tp.x - cj.x, xl1 = Tp.y - cj.y, xl2 = Tp.z - cj.z;

    // exact side
    double tex[3];
    if (KIND == 0) {
      const double rl2 = xl0 * xl0 + xl1 * xl1 + xl2 * xl2;
      const double stl = sqrt((xl0 * xl0 + xl1 * xl1) / rl2);
      if (stl >= SN_POLE_ST_MIN) {
        snEvalTractionAnalytic(xl0, xl1, xl2, nA, sh_tab, sh_coef, nkept,
                               nmax, tex);
      } else {
        // near the source-frame pole the spherical gradient frame is
        // singular: central-FD fallback (rare; measure-zero geometry)
        double g[3][3], p0 = 0.0, uu[3];
        for (int ax = 0; ax < 3; ++ax) {
          double up[3], dn[3];
          const double ox = (ax == 0) ? hfd : 0.0;
          const double oy = (ax == 1) ? hfd : 0.0;
          const double oz = (ax == 2) ? hfd : 0.0;
          snEvalUP<false>(xl0 + ox, xl1 + oy, xl2 + oz, sh_tab, sh_coef,
                          nkept, nmax, up, nullptr);
          snEvalUP<false>(xl0 - ox, xl1 - oy, xl2 - oz, sh_tab, sh_coef,
                          nkept, nmax, dn, nullptr);
          const double ih = 1.0 / (2.0 * hfd);
          g[0][ax] = (up[0] - dn[0]) * ih;
          g[1][ax] = (up[1] - dn[1]) * ih;
          g[2][ax] = (up[2] - dn[2]) * ih;
        }
        snEvalUP<true>(xl0, xl1, xl2, sh_tab, sh_coef, nkept, nmax, uu, &p0);
        for (int c = 0; c < 3; ++c) {
          const double *n3 = &nA.x;
          tex[c] = -p0 * n3[c]
              + (g[c][0] + g[0][c]) * n3[0]
              + (g[c][1] + g[1][c]) * n3[1]
              + (g[c][2] + g[2][c]) * n3[2];
        }
      }
    } else {
      snEvalUP<false>(xl0, xl1, xl2, sh_tab, sh_coef, nkept, nmax, tex,
                      nullptr);
    }

    // smooth 162-point quadrature of the same pair (what the treecode/brute
    // path already added for it) -- subtract with the same p2p math.
    double sm[3] = {0.0, 0.0, 0.0};
    if (doSub) {
      for (int s = 0; s < B; ++s) {
        const vec3d src = pts[(size_t)j * B + s];
        const vec3d qs{sh_q[3 * s], sh_q[3 * s + 1], sh_q[3 * s + 2]};
        if (KIND == 0)
          stokes::traction_p2p(Tp, nA, src, qs, sm);
        else
          stokes::p2p(Tp, src, qs, sm);
      }
      sm[0] *= smoothPref; sm[1] *= smoothPref; sm[2] *= smoothPref;
    }
    corr[0] += tex[0] - sm[0];
    corr[1] += tex[1] - sm[1];
    corr[2] += tex[2] - sm[2];
  }

  if (b < B) {
    double *o = outAoS + (size_t)3 * ((size_t)i * B + b);
    o[0] += corr[0];
    o[1] += corr[1];
    o[2] += corr[2];
  }
}

// out[i] += a[i] -- folds the side-stream near-correction scratch buffer into
// the operator output once both streams have joined.
__global__ void snAddKernel(const double *a, double *out, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += a[i];
}

}  // namespace snear

// ==========================================================================
// host-side state + setup / apply / selftest
// ==========================================================================

struct SBimNear {
  int N = 0, B = 0;
  int nkept = 0, nreal = 0, nmax = 0;
  int nPairs = 0, nRows = 0;
  double beta = 2.0, hfd = 1e-5;
  int *d_csrRow = nullptr, *d_csrTgt = nullptr, *d_csrSrc = nullptr;
  double *d_Areal = nullptr;   // nreal x 3B col-major
  double *d_tab = nullptr;     // nkept x 12 row-major
  double *d_coefs = nullptr;   // nreal x N col-major
  vec3d *d_centers = nullptr;
  size_t shBytes = 0;
  int block = 0;
};

// Ordered near pairs (target i, source j), |c_i - c_j| < 2*beta, via a host
// cell list (O(N) at fixed density; also fine for the 2-sphere selftest).
static void sbimNearPairsCSR(const std::vector<double> &centers, int N,
                             double cutoff, std::vector<int> &csrRow,
                             std::vector<int> &csrTgt, std::vector<int> &csrSrc)
{
  const double inv = 1.0 / cutoff;
  auto cellOf = [&](int k, int c) {
    return (long long)std::floor(centers[(size_t)3 * k + c] * inv);
  };
  auto key = [](long long x, long long y, long long z) {
    return (x * 73856093LL) ^ (y * 19349663LL) ^ (z * 83492791LL);
  };
  std::unordered_multimap<long long, int> grid;
  grid.reserve((size_t)N * 2);
  for (int k = 0; k < N; ++k)
    grid.emplace(key(cellOf(k, 0), cellOf(k, 1), cellOf(k, 2)), k);
  const double cut2 = cutoff * cutoff;
  csrRow.clear(); csrTgt.clear(); csrSrc.clear();
  csrRow.push_back(0);
  for (int i = 0; i < N; ++i) {
    const long long cx = cellOf(i, 0), cy = cellOf(i, 1), cz = cellOf(i, 2);
    const size_t before = csrSrc.size();
    for (long long dx = -1; dx <= 1; ++dx)
      for (long long dy = -1; dy <= 1; ++dy)
        for (long long dz = -1; dz <= 1; ++dz) {
          auto range = grid.equal_range(key(cx + dx, cy + dy, cz + dz));
          for (auto it = range.first; it != range.second; ++it) {
            const int j = it->second;
            if (j == i) continue;
            // hash collisions are possible: re-verify the true distance
            const double ddx = centers[3 * i] - centers[3 * j];
            const double ddy = centers[3 * i + 1] - centers[3 * j + 1];
            const double ddz = centers[3 * i + 2] - centers[3 * j + 2];
            if (ddx * ddx + ddy * ddy + ddz * ddz < cut2)
              csrSrc.push_back(j);
          }
        }
    if (csrSrc.size() > before) {
      // dedupe (a source can be found via multiple colliding hash keys)
      std::sort(csrSrc.begin() + before, csrSrc.end());
      csrSrc.erase(std::unique(csrSrc.begin() + before, csrSrc.end()),
                   csrSrc.end());
      csrTgt.push_back(i);
      csrRow.push_back((int)csrSrc.size());
    }
  }
}

static void sbimNearInit(SBimNear *nr, const std::string &dir, int N,
                         const std::vector<double> &centers, double beta,
                         int B)
{
  nr->N = N;
  nr->B = B;
  nr->beta = beta;
  const std::vector<double> Areal = sbimReadBin(dir + "/Areal.bin", 0);
  const std::vector<double> tab = sbimReadBin(dir + "/nearmodes.bin", 0);
  if (Areal.size() % (size_t)(3 * B) || tab.size() % 12)
    throw std::runtime_error("near: bad Areal.bin/nearmodes.bin sizes");
  nr->nreal = (int)(Areal.size() / (size_t)(3 * B));
  nr->nkept = (int)(tab.size() / 12);
  nr->hfd = tab[11];
  nr->nmax = 0;
  for (int t = 0; t < nr->nkept; ++t)
    nr->nmax = std::max(nr->nmax, (int)tab[(size_t)12 * t + 1]);
  if (nr->nmax + 1 > snear::SN_NMAX_CAP)
    throw std::runtime_error("near: nmax exceeds SN_NMAX_CAP");

  std::vector<int> row, tgt, src;
  sbimNearPairsCSR(centers, N, 2.0 * beta, row, tgt, src);
  nr->nRows = (int)tgt.size();
  nr->nPairs = (int)src.size();
  std::printf("[near] beta=%.2f cutoff=%.2f ordered_pairs=%d target_rows=%d "
              "nreal=%d nkept=%d nmax=%d hfd=%.1e\n",
              beta, 2.0 * beta, nr->nPairs, nr->nRows, nr->nreal, nr->nkept,
              nr->nmax, nr->hfd);
  if (nr->nRows == 0) return;

  auto upInt = [](const std::vector<int> &h) {
    int *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
  };
  // csrRow was built rows-compacted: rebuild as [nRows+1]
  std::vector<int> rowc(nr->nRows + 1);
  for (int r = 0; r <= nr->nRows; ++r) rowc[r] = row[r];
  nr->d_csrRow = upInt(rowc);
  nr->d_csrTgt = upInt(tgt);
  nr->d_csrSrc = upInt(src);
  nr->d_Areal = sbimUpload(Areal);
  nr->d_tab = sbimUpload(tab);
  CUDA_CHECK(cudaMalloc(&nr->d_coefs,
                        (size_t)nr->nreal * N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&nr->d_centers, (size_t)N * sizeof(vec3d)));
  CUDA_CHECK(cudaMemcpy(nr->d_centers, centers.data(),
                        (size_t)3 * N * sizeof(double),
                        cudaMemcpyHostToDevice));
  nr->block = ((B + 31) / 32) * 32;
  nr->shBytes = (size_t)(12 * nr->nkept + nr->nreal + 3 * B) * sizeof(double);
}

static void sbimNearFree(SBimNear *nr)
{
  if (nr->d_csrRow) cudaFree(nr->d_csrRow);
  if (nr->d_csrTgt) cudaFree(nr->d_csrTgt);
  if (nr->d_csrSrc) cudaFree(nr->d_csrSrc);
  if (nr->d_Areal) cudaFree(nr->d_Areal);
  if (nr->d_tab) cudaFree(nr->d_tab);
  if (nr->d_coefs) cudaFree(nr->d_coefs);
  if (nr->d_centers) cudaFree(nr->d_centers);
  *nr = SBimNear{};
}

// kind: 0 traction, 1 Stokeslet. d_dens = RAW density (Areal has weights);
// d_q = weighted density (subtract side). Accumulates into d_outAoS. All work
// is issued on `stream` (default legacy stream when omitted), so the caller
// can overlap the correction with the treecode far field on a side stream.
static void sbimNearApply(SBimNear *nr, cublasHandle_t cublas, int kind,
                          const double *d_dens, const double *d_q,
                          const vec3d *d_pts, const vec3d *d_nrm,
                          double *d_outAoS, int doSub = 1,
                          cudaStream_t stream = 0)
{
  if (nr->nRows == 0) return;
  const double one = 1.0, zero = 0.0;
  cudaStream_t prev = 0;
  CUBLAS_CHECK(cublasGetStream(cublas, &prev));
  CUBLAS_CHECK(cublasSetStream(cublas, stream));
  CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nr->nreal, nr->N,
                           3 * nr->B, &one, nr->d_Areal, nr->nreal, d_dens,
                           3 * nr->B, &zero, nr->d_coefs, nr->nreal));
  CUBLAS_CHECK(cublasSetStream(cublas, prev));
  if (kind == 0)
    snear::snCorrKernel<0><<<nr->nRows, nr->block, nr->shBytes, stream>>>(
        nr->d_csrRow, nr->d_csrTgt, nr->d_csrSrc, d_pts, d_nrm, nr->d_centers,
        nr->d_coefs, d_q, nr->d_tab, nr->nkept, nr->nreal, nr->B, nr->nmax,
        nr->hfd, doSub, stokes::tractionPrefactor(), d_outAoS);
  else
    snear::snCorrKernel<1><<<nr->nRows, nr->block, nr->shBytes, stream>>>(
        nr->d_csrRow, nr->d_csrTgt, nr->d_csrSrc, d_pts, d_nrm, nr->d_centers,
        nr->d_coefs, d_q, nr->d_tab, nr->nkept, nr->nreal, nr->B, nr->nmax,
        nr->hfd, doSub, 1.0 / (8.0 * M_PI), d_outAoS);
  // NOTE: exact double 1/(8*pi), not (double)stokes::prefactor() (fp32-
  // rounded, ~6e-9 off): the subtracted smooth part must match the Python
  // reference operator; the treecode's own fp32-prefactor floor stays as the
  // (already accepted) ~1e-8 cross-path noise.
  CUDA_CHECK(cudaGetLastError());
}

// GPU selftest against near_ref.bin: for each reference displacement, build
// the 2-sphere config (source at origin with the reference density, target at
// d with zero density) and compare (a) the exact side alone and (b) the fused
// exact-minus-smooth correction, for both kernels, against the Python values.
static bool sbimNearSelftest(const std::string &dir)
{
  const std::vector<double> ref = sbimReadBin(dir + "/near_ref.bin", 0);
  const double hfd = ref[0];
  const int ncases = (int)ref[1];
  const int B = (int)ref[2];
  const int n3 = 3 * B;
  const std::vector<double> grid = sbimReadBin(dir + "/grid.bin",
                                               (size_t)n3);
  const std::vector<double> w = sbimReadBin(dir + "/weights.bin", (size_t)B);
  const double *dens = &ref[4];

  cublasHandle_t cb;
  CUBLAS_CHECK(cublasCreate(&cb));
  bool allOk = true;
  double worstExact = 0.0, worstCorr = 0.0, worstCorrOp = 0.0;
  for (int cs = 0; cs < ncases; ++cs) {
    const double *rec = &ref[4 + n3 + (size_t)cs * (3 + 4 * n3)];
    const double *d = rec;
    const double *tKx = rec + 3, *tKs = rec + 3 + n3;
    const double *uSx = rec + 3 + 2 * n3, *uSs = rec + 3 + 3 * n3;

    // sphere 0 = source (density dens) at origin; sphere 1 = target at d
    const std::vector<double> cent = {0, 0, 0, d[0], d[1], d[2]};
    SBimNear nr{};
    const double need = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    sbimNearInit(&nr, dir, 2, cent, 0.51 * need + 0.5, B);
    if (nr.hfd != hfd) std::printf("[near selftest] WARNING hfd mismatch\n");

    std::vector<double> hpts((size_t)2 * n3), hnrm((size_t)2 * n3),
        hdens((size_t)2 * n3, 0.0), hq((size_t)2 * n3, 0.0);
    for (int k = 0; k < 2; ++k)
      for (int b = 0; b < B; ++b)
        for (int c = 0; c < 3; ++c) {
          hpts[(size_t)3 * (k * B + b) + c] =
              grid[(size_t)3 * b + c] + cent[(size_t)3 * k + c];
          hnrm[(size_t)3 * (k * B + b) + c] = grid[(size_t)3 * b + c];
        }
    for (int t = 0; t < n3; ++t) {
      hdens[t] = dens[t];
      hq[t] = dens[t] * w[t / 3];
    }
    vec3d *d_p = nullptr, *d_n = nullptr;
    double *d_dn = nullptr, *d_qq = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p, (size_t)2 * B * sizeof(vec3d)));
    CUDA_CHECK(cudaMalloc(&d_n, (size_t)2 * B * sizeof(vec3d)));
    CUDA_CHECK(cudaMemcpy(d_p, hpts.data(), hpts.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n, hnrm.data(), hnrm.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    d_dn = sbimUpload(hdens);
    d_qq = sbimUpload(hq);
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)2 * n3 * sizeof(double)));

    auto run = [&](int kind, int doSub, std::vector<double> &out) {
      CUDA_CHECK(cudaMemset(d_out, 0, (size_t)2 * n3 * sizeof(double)));
      sbimNearApply(&nr, cb, kind, d_dn, d_qq, d_p, d_n, d_out, doSub);
      out.resize((size_t)2 * n3);
      CUDA_CHECK(cudaMemcpy(out.data(), d_out,
                            (size_t)2 * n3 * sizeof(double),
                            cudaMemcpyDeviceToHost));
    };
    auto maxAbs = [&](const double *v, int n) {
      double s = 0.0;
      for (int t = 0; t < n; ++t) s = std::max(s, std::abs(v[t]));
      return s;
    };
    // target sphere = index 1: its rows sit at offset n3
    std::vector<double> got;
    for (int kind = 0; kind < 2; ++kind) {
      const double *rx = (kind == 0) ? tKx : uSx;
      const double *rs = (kind == 0) ? tKs : uSs;
      run(kind, /*doSub=*/0, got);
      double e1 = 0.0;
      for (int t = 0; t < n3; ++t)
        e1 = std::max(e1, std::abs(got[n3 + t] - rx[t]));
      e1 /= maxAbs(rx, n3);
      run(kind, /*doSub=*/1, got);
      double eabs = 0.0, cscale = 0.0;
      for (int t = 0; t < n3; ++t) {
        const double cref = rx[t] - rs[t];
        eabs = std::max(eabs, std::abs(got[n3 + t] - cref));
        cscale = std::max(cscale, std::abs(cref));
      }
      // primary scale = operator magnitude (what GMRES sees); the correction
      // magnitude itself shrinks rapidly with distance, so corr-relative
      // error is only informative
      worstExact = std::max(worstExact, e1);
      worstCorr = std::max(worstCorr, eabs / cscale);
      worstCorrOp = std::max(worstCorrOp, eabs / maxAbs(rx, n3));
    }
    cudaFree(d_p); cudaFree(d_n); cudaFree(d_dn); cudaFree(d_qq);
    cudaFree(d_out);
    sbimNearFree(&nr);
  }
  cublasDestroy(cb);
  // vs-operator can be no better than the exact side (FD noise ~eps/h), so
  // it shares the 1e-9 gate
  const bool okE = worstExact < 1e-9, okC = worstCorr < 1e-6,
             okO = worstCorrOp < 1e-9;
  allOk = okE && okC && okO;
  std::printf("[near selftest] exact-side max rel err = %.3e (%s)  "
              "fused corr err: vs-corr = %.3e (%s), vs-operator = %.3e (%s)\n",
              worstExact, okE ? "PASS" : "FAIL", worstCorr,
              okC ? "PASS" : "FAIL", worstCorrOp, okO ? "PASS" : "FAIL");
  return allOk;
}
