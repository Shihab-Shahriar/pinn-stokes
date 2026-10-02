// SPDX-License-Identifier: Apache-2.0
//
// CPU MFS-GMRES ellipsoid mobility solver on the CPU treecode: the
// Broms-Barnett-Tornberg one-body-preconditioned MFS solve (GPU original in
// src/mfs_broms.cuh + src/test_mfs_ellipsoid.cu), ported from the pure-CPU
// pvfmm driver ~/programs/pvfmm/examples/src/mfs_ellipsoid.cpp with the
// evaluator swapped from pvfmm to tccpu::TreecodeCpu. P rigid ellipsoids with
// applied wrench (F,T), GMRES over the MFS densities gamma (n = 3*M*P
// unknowns), PETSc GMRES via the header-free interface in mfs_petsc.hpp
// (separate TU mfs_petsc_cpu.cpp; that PETSc build carries CUDA DT_NEEDEDs --
// run with the CUDA driver stubs on LD_LIBRARY_PATH on GPU-less nodes).
//
// Differences from the pvfmm driver:
//  - Evaluator: TreecodeCpu (BaryStokes PDEG=7) in PHYSICAL coordinates -- no
//    unit-cube rescale, no sfac; output already carries 1/(8 pi mu).
//  - Same-particle handling is selectable (-self):
//      subtract  full treecode field minus the analytic per-particle self
//                block R_k*(S_self*hat_ref_k)  (pvfmm-verbatim operator)
//      skip      in-traversal exclusion (GPU-parity operator): a node whose
//                subtree contains the target's own source bucket is never
//                MAC-accepted, and the own leaf is dropped. Requires object
//                source buckets.
//  - Source bucketizer selectable (-buckets object|grid); object = one BVH
//    leaf per particle (the GPU MFS operating point).
// The dense one-body algebra follows mfs_broms.cuh argument-for-argument
// (column-major, OpenBLAS/LAPACK instead of cuBLAS/cuSOLVER).
//
// Modes:
//   -mode selftest      identity-GMRES + dense identities + P=2 brute matvec
//                       (both -self modes)
//   -mode rhs-validate  compare clouds/lam0/u0 against the frozen export in
//                       mfs_ellipsoid_case + direct-sum spot checks
//   -mode solve         full solve; validates vs reference velocities when the
//                       CSV has them (42-col mobility layout)
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "treecode_cpu.h"

#include "mfs_case_io.hpp"
#include "mfs_petsc.hpp"

extern "C" {
void dgemm_(const char *TA, const char *TB, const int *M, const int *N,
            const int *K, const double *ALPHA, const double *A, const int *LDA,
            const double *B, const int *LDB, const double *BETA, double *C,
            const int *LDC);
void dpotrf_(const char *UPLO, const int *N, double *A, const int *LDA, int *INFO);
void dpotri_(const char *UPLO, const int *N, double *A, const int *LDA, int *INFO);
void dgesdd_(const char *JOBZ, const int *M, const int *N, double *A, const int *LDA,
             double *S, double *U, const int *LDU, double *VT, const int *LDVT,
             double *WORK, const int *LWORK, int *IWORK, int *INFO);
}

typedef std::vector<double> vec;

static void gemm(char ta, char tb, int m, int n, int k, double alpha,
                 const double *A, int lda, const double *B, int ldb,
                 double beta, double *C, int ldc)
{
  dgemm_(&ta, &tb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

// ==========================================================================
// ports of the mfs_broms.cuh device kernels (verbatim index arithmetic)
// ==========================================================================

// K (3p x 6) column-major (ld=3p): row block i = [I3, -skew(pts_i)].
static void buildK(const double *pts, double *K, int p)
{
  const size_t ld = (size_t)3 * p;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < p; ++i) {
    double px = pts[3 * i], py = pts[3 * i + 1], pz = pts[3 * i + 2];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        K[(size_t)b * ld + (3 * i + a)] = (a == b) ? 1.0 : 0.0;
    double S[3][3] = {{0, -pz, py}, {pz, 0, -px}, {-py, px, 0}};
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        K[(size_t)(3 + b) * ld + (3 * i + a)] = -S[a][b];
  }
}

// Dense self Stokeslet block S_self (3M x 3N), column-major (ld = 3M).
// Entry (3i+a, 3j+b) = pref*( delta_ab/r + r_a r_b / r^3 ), r = X_i - Y_j.
static void buildSelfBlock(const double *Xr, const double *Yr, double *S,
                           int M, int N, double pref)
{
  const int ld = 3 * M;
#pragma omp parallel for schedule(static)
  for (long idx = 0; idx < (long)M * N; ++idx) {
    int i = (int)(idx / N), j = (int)(idx % N);
    double r[3] = {Xr[3 * i] - Yr[3 * j], Xr[3 * i + 1] - Yr[3 * j + 1],
                   Xr[3 * i + 2] - Yr[3 * j + 2]};
    double r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    double ir = (r2 > 1e-28) ? 1.0 / std::sqrt(r2) : 0.0;
    double ir3 = ir * ir * ir;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        double val = pref * (ir * (a == b ? 1.0 : 0.0) + ir3 * r[a] * r[b]);
        S[(size_t)(3 * j + b) * ld + (3 * i + a)] = val;  // column-major
      }
  }
}

// Batched per-particle rotation of a stacked field (P*npts triples, particle k
// owns triples [k*npts,(k+1)*npts)). R is P*9 row-major. transpose=0 applies
// R_k (ref->global); transpose=1 applies R_k^T. Safe in place.
static void rotAll(const double *R, const double *in, double *out, int npts,
                   int P, int transpose)
{
#pragma omp parallel for schedule(static)
  for (long t = 0; t < (long)npts * P; ++t) {
    const double *Rk = R + (size_t)9 * (t / npts);
    double v0 = in[3 * t], v1 = in[3 * t + 1], v2 = in[3 * t + 2];
    if (!transpose) {
      out[3 * t]     = Rk[0] * v0 + Rk[1] * v1 + Rk[2] * v2;
      out[3 * t + 1] = Rk[3] * v0 + Rk[4] * v1 + Rk[5] * v2;
      out[3 * t + 2] = Rk[6] * v0 + Rk[7] * v1 + Rk[8] * v2;
    } else {
      out[3 * t]     = Rk[0] * v0 + Rk[3] * v1 + Rk[6] * v2;
      out[3 * t + 1] = Rk[1] * v0 + Rk[4] * v1 + Rk[7] * v2;
      out[3 * t + 2] = Rk[2] * v0 + Rk[5] * v1 + Rk[8] * v2;
    }
  }
}

// X_glob[k][i] = R_k * X_ref[i] + center_k.
static void tileRotTrans(const double *Xref, const double *R,
                         const double *centers, double *Xglob, int npts, int P)
{
#pragma omp parallel for schedule(static)
  for (long t = 0; t < (long)npts * P; ++t) {
    int k = (int)(t / npts);
    int i = (int)(t % npts);
    const double *Rk = R + (size_t)9 * k;
    double v0 = Xref[3 * i], v1 = Xref[3 * i + 1], v2 = Xref[3 * i + 2];
    Xglob[3 * t]     = Rk[0] * v0 + Rk[1] * v1 + Rk[2] * v2 + centers[3 * k];
    Xglob[3 * t + 1] = Rk[3] * v0 + Rk[4] * v1 + Rk[5] * v2 + centers[3 * k + 1];
    Xglob[3 * t + 2] = Rk[6] * v0 + Rk[7] * v1 + Rk[8] * v2 + centers[3 * k + 2];
  }
}

// ==========================================================================
// treecode wrapper: build tree + targets once, evaluate-many with new
// densities (upward + merged traversal per eval; geometry stays fixed)
// ==========================================================================
struct TreecodeEval {
  std::unique_ptr<tccpu::TreecodeCpu> tc;
  size_t n_trg = 0;
  vec trg_val;                       // AoS, physical units, 1/(8 pi mu) included
  double t_last = 0, t_total = 0;
  double t_up_last = 0, t_trav_last = 0;   // per-eval split (seconds)
  int count = 0;

  static_assert(sizeof(tccpu::vec3d) == 3 * sizeof(double),
                "vec3d must be 3 packed doubles");

  // src/trg: PHYSICAL AoS clouds. objectGroupSize = N per particle (object
  // buckets, the MFS operating point) or 0 for grid-hilbert; targetsPerOwner
  // = M per particle (enables skip mode).
  void init(const vec &src, const vec &trg, float mac, int maxLeaf,
            int objectGroupSize, size_t targetsPerOwner)
  {
    tccpu::TreecodeCpu::Config cfg;
    cfg.mac = mac;
    cfg.maxLeaf = maxLeaf;
    cfg.objectGroupSize = objectGroupSize;
    tc = std::make_unique<tccpu::TreecodeCpu>(cfg);
    double t0 = omp_get_wtime();
    tc->build(reinterpret_cast<const tccpu::vec3d *>(src.data()), src.size() / 3);
    tc->setTargets(reinterpret_cast<const tccpu::vec3d *>(trg.data()),
                   trg.size() / 3, targetsPerOwner);
    double t1 = omp_get_wtime();
    n_trg = trg.size() / 3;
    trg_val.assign(3 * n_trg, 0.0);
    const auto &st = tc->stats();
    std::printf("[treecode] build %.2f s (bucketize %.0f ms, bvh %.0f ms, "
                "targets %.0f ms): %u buckets, %u nodes, moments %.0f MB, "
                "%zu src, %zu trg, mac=%.3g %s rss=%.1f GB\n",
                t1 - t0, st.bucketMs, st.buildBvhMs, st.setTargetsMs,
                st.numBuckets, st.numNodes, st.nodeM2PMB,
                src.size() / 3, n_trg, (double)mac,
                objectGroupSize > 0 ? "object" : "grid-hilbert", mfs_rss_gb());
    std::fflush(stdout);
  }
  // Stokeslet field (mu=1, 1/8pi) at all targets, PHYSICAL coordinates.
  // skipOwn drops every source of the target's own particle in-traversal.
  const vec &eval(const vec &density, bool skipOwn)
  {
    double t0 = omp_get_wtime();
    tc->upward(reinterpret_cast<const tccpu::vec3d *>(density.data()));
    double t1 = omp_get_wtime();
    tc->evaluateAtTargets(trg_val.data(), skipOwn);
    double t2 = omp_get_wtime();
    t_up_last = t1 - t0;
    t_trav_last = t2 - t1;
    t_last = t2 - t0;
    t_total += t_last;
    ++count;
    return trg_val;
  }
};

// ==========================================================================
// MFS context (host port of MFSContext)
// ==========================================================================
struct MFS {
  int P = 0, M = 0, N = 0, tM = 0, tN = 0;
  long n = 0;
  double pref = 1.0 / (8.0 * M_PI);  // mu = 1
  // per-particle state (from CSV)
  vec centers, R, F, T;  // 3P, 9P (row-major), 3P, 3P
  // cached reference-frame operators (column-major)
  vec KN, Ginv, ImL, pinv, What, Sself;
  // clouds (physical, particle-major AoS)
  vec Xglob, Yglob;
  // scratch (persistent across matvecs)
  vec gammaRefAll, hatRefAll, hatGlob, selfRefAll;
  TreecodeEval tce;
  bool selfSkip = false;   // -self skip: in-traversal exclusion, no subtract
  // instrumentation
  int mvcount = 0;
  double t_dense = 0, t_rot = 0, t_tc_mv = 0, t_mv = 0;
  bool verbose_mv = true;
};

// One-time dense setup: port of mfsSetup (mfs_broms.cuh:877-1064).
static void denseSetup(MFS &m, const vec &Xref, const vec &Yref)
{
  const int M = m.M, N = m.N, tM = m.tM, tN = m.tN;
  double t0 = omp_get_wtime();

  // ---- K_N (3N x 6), K_M (3M x 6) ----
  m.KN.resize((size_t)tN * 6);
  vec KM((size_t)tM * 6);
  buildK(Yref.data(), m.KN.data(), N);
  buildK(Xref.data(), KM.data(), M);

  // ---- gram = K_N^T K_N (6x6); Ginv via Cholesky (SPD) ----
  m.Ginv.resize(36);
  gemm('T', 'N', 6, 6, tN, 1.0, m.KN.data(), tN, m.KN.data(), tN, 0.0, m.Ginv.data(), 6);
  int info = 0, six = 6;
  dpotrf_("L", &six, m.Ginv.data(), &six, &info);
  if (info != 0) { std::fprintf(stderr, "gram not SPD (potrf info=%d)\n", info); std::exit(1); }
  dpotri_("L", &six, m.Ginv.data(), &six, &info);
  if (info != 0) { std::fprintf(stderr, "gram inverse failed (potri info=%d)\n", info); std::exit(1); }
  for (int i = 0; i < 6; ++i)  // symmetrize: upper(j,i) <- lower(i,j)
    for (int j = 0; j < i; ++j) m.Ginv[j + 6 * i] = m.Ginv[i + 6 * j];
  double t1 = omp_get_wtime();

  // ---- Z = Ginv*K_N^T (6 x tN); L_proj = K_N*Z; ImL = I - L ----
  vec Z((size_t)6 * tN), Lproj((size_t)tN * tN);
  m.ImL.resize((size_t)tN * tN);
  gemm('N', 'T', 6, tN, 6, 1.0, m.Ginv.data(), 6, m.KN.data(), tN, 0.0, Z.data(), 6);
  gemm('N', 'N', tN, tN, 6, 1.0, m.KN.data(), tN, Z.data(), 6, 0.0, Lproj.data(), tN);
#pragma omp parallel for schedule(static)
  for (long idx = 0; idx < (long)tN * tN; ++idx) {
    int r = (int)(idx % tN), c = (int)(idx / tN);
    m.ImL[idx] = (r == c ? 1.0 : 0.0) - Lproj[idx];
  }
  double t2 = omp_get_wtime();

  // ---- S_self; L_r = K_M K_N^T; S_L = S_self*(I-L) + L_r ----
  m.Sself.resize((size_t)tM * tN);
  vec Lr((size_t)tM * tN), SL((size_t)tM * tN);
  buildSelfBlock(Xref.data(), Yref.data(), m.Sself.data(), M, N, m.pref);
  gemm('N', 'T', tM, tN, 6, 1.0, KM.data(), tM, m.KN.data(), tN, 0.0, Lr.data(), tM);
  SL = Lr;
  gemm('N', 'N', tM, tN, tN, 1.0, m.Sself.data(), tM, m.ImL.data(), tN, 1.0, SL.data(), tM);
  vec SLcheck = SL;  // dgesdd destroys its input; keep a copy for the sanity check
  double t3 = omp_get_wtime();

  // ---- SVD S_L = U S V^T (econ); pinv = V diag(1/s) U^T; W_hat = (I-L) pinv ----
  vec sig(tN), U((size_t)tM * tN), VT((size_t)tN * tN);
  std::vector<int> iwork(8 * (size_t)tN);
  int lwork = -1;
  double wq = 0;
  dgesdd_("S", &tM, &tN, SL.data(), &tM, sig.data(), U.data(), &tM, VT.data(), &tN,
          &wq, &lwork, iwork.data(), &info);
  lwork = (int)wq;
  vec work((size_t)lwork);
  dgesdd_("S", &tM, &tN, SL.data(), &tM, sig.data(), U.data(), &tM, VT.data(), &tN,
          work.data(), &lwork, iwork.data(), &info);
  if (info != 0) std::fprintf(stderr, "warning: gesdd info=%d\n", info);
  double t4 = omp_get_wtime();

  // Us = U * diag(1/sig)  (column scale)
#pragma omp parallel for schedule(static)
  for (int j = 0; j < tN; ++j) {
    double is = 1.0 / sig[j];
    for (int i = 0; i < tM; ++i) U[(size_t)j * tM + i] *= is;
  }
  m.pinv.resize((size_t)tN * tM);
  m.What.resize((size_t)tN * tM);
  gemm('T', 'T', tN, tM, tN, 1.0, VT.data(), tN, U.data(), tM, 0.0, m.pinv.data(), tN);
  gemm('N', 'N', tN, tM, tN, 1.0, m.ImL.data(), tN, m.pinv.data(), tN, 0.0, m.What.data(), tN);
  double t5 = omp_get_wtime();

  // ---- sanity: pinv is a right/left pseudo-inverse on range(S_L) ----
  {
    vec y(tN), Sy(tM), py(tN);
    for (int i = 0; i < tN; ++i) y[i] = drand48() - 0.5;
    gemm('N', 'N', tM, 1, tN, 1.0, SLcheck.data(), tM, y.data(), tN, 0.0, Sy.data(), tM);
    gemm('N', 'N', tN, 1, tM, 1.0, m.pinv.data(), tN, Sy.data(), tM, 0.0, py.data(), tN);
    // compare pinv*S_L*y against the projection of y (S_L has a 6-dim null
    // space complement; on ImL-projected vectors it must round-trip)
    vec yp(tN);
    gemm('N', 'N', tN, 1, tN, 1.0, m.ImL.data(), tN, y.data(), tN, 0.0, yp.data(), tN);
    vec Syp(tM), pyp(tN);
    gemm('N', 'N', tM, 1, tN, 1.0, SLcheck.data(), tM, yp.data(), tN, 0.0, Syp.data(), tM);
    gemm('N', 'N', tN, 1, tM, 1.0, m.pinv.data(), tN, Syp.data(), tM, 0.0, pyp.data(), tN);
    double num = 0, den = 0;
    for (int i = 0; i < tN; ++i) { double d = pyp[i] - yp[i]; num += d * d; den += yp[i] * yp[i]; }
    std::printf("[dense check] ||pinv*S_L*y - y||/||y|| (y in range(ImL)) = %.3e\n",
                std::sqrt(num / den));
    // ImL idempotency on the same vector
    vec y2(tN);
    gemm('N', 'N', tN, 1, tN, 1.0, m.ImL.data(), tN, yp.data(), tN, 0.0, y2.data(), tN);
    num = den = 0;
    for (int i = 0; i < tN; ++i) { double d = y2[i] - yp[i]; num += d * d; den += yp[i] * yp[i]; }
    std::printf("[dense check] ||ImL^2*y - ImL*y||/||ImL*y|| = %.3e\n", std::sqrt(num / den));
  }

  std::printf("[MFS setup] P=%d M=%d N=%d n=%ld total=%.2f s "
              "(K+gram=%.2f proj=%.2f self+SL=%.2f svd=%.2f pinv=%.2f)\n",
              m.P, M, N, m.n, t5 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4);
  std::fflush(stdout);
}

// Global clouds (physical frame; the treecode consumes them as-is).
static void buildClouds(MFS &m, const vec &Xref, const vec &Yref)
{
  m.Xglob.resize((size_t)m.tM * m.P);
  m.Yglob.resize((size_t)m.tN * m.P);
  tileRotTrans(Xref.data(), m.R.data(), m.centers.data(), m.Xglob.data(), m.M, m.P);
  tileRotTrans(Yref.data(), m.R.data(), m.centers.data(), m.Yglob.data(), m.N, m.P);

  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  for (const vec *c : {&m.Xglob, &m.Yglob})
    for (size_t i = 0; i < c->size(); i += 3)
      for (int d = 0; d < 3; ++d) {
        double v = (*c)[i + d];
        if (v < lo[d]) lo[d] = v;
        if (v > hi[d]) hi[d] = v;
      }
  std::printf("[clouds] bbox lo=(%.3f %.3f %.3f) hi=(%.3f %.3f %.3f)\n",
              lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
}

// lam0 (completion sources, global frame): port of mfsSolve:1101-1109.
static void buildLam0(MFS &m, vec &lam0)
{
  const int P = m.P, tN = m.tN;
  vec rhs6r((size_t)6 * P), vw((size_t)6 * P);
#pragma omp parallel for schedule(static)
  for (int k = 0; k < P; ++k) {  // rhs6r = [R_k^T*(-F_k); R_k^T*(-T_k)]
    const double *Rk = m.R.data() + (size_t)9 * k;
    double f[3] = {-m.F[3 * k], -m.F[3 * k + 1], -m.F[3 * k + 2]};
    double t[3] = {-m.T[3 * k], -m.T[3 * k + 1], -m.T[3 * k + 2]};
    rhs6r[6 * k + 0] = Rk[0] * f[0] + Rk[3] * f[1] + Rk[6] * f[2];
    rhs6r[6 * k + 1] = Rk[1] * f[0] + Rk[4] * f[1] + Rk[7] * f[2];
    rhs6r[6 * k + 2] = Rk[2] * f[0] + Rk[5] * f[1] + Rk[8] * f[2];
    rhs6r[6 * k + 3] = Rk[0] * t[0] + Rk[3] * t[1] + Rk[6] * t[2];
    rhs6r[6 * k + 4] = Rk[1] * t[0] + Rk[4] * t[1] + Rk[7] * t[2];
    rhs6r[6 * k + 5] = Rk[2] * t[0] + Rk[5] * t[1] + Rk[8] * t[2];
  }
  gemm('N', 'N', 6, P, 6, 1.0, m.Ginv.data(), 6, rhs6r.data(), 6, 0.0, vw.data(), 6);
  lam0.resize((size_t)tN * P);
  gemm('N', 'N', tN, P, 6, 1.0, m.KN.data(), tN, vw.data(), 6, 0.0, lam0.data(), tN);
  rotAll(m.R.data(), lam0.data(), lam0.data(), m.N, P, 0);  // ref -> global
}

// Matvec y = A x = x + S_offdiag*(W_hat_k x): port of MatMult_MFS. The
// off-diagonal Stokeslet is either treecode-full-minus-analytic-self
// (-self subtract, the pvfmm-verbatim operator) or the in-traversal
// exclusion (-self skip, the GPU-parity operator).
static void matvec(MFS &m, const double *x, double *y)
{
  const int P = m.P, tM = m.tM, tN = m.tN;
  double t0 = omp_get_wtime();

#pragma omp parallel for schedule(static)
  for (long i = 0; i < m.n; ++i) y[i] = x[i];  // diagonal block = I (eq 54)

  // hat_lam: gamma_ref -> W_hat -> hat_ref -> hat_glob
  rotAll(m.R.data(), x, m.gammaRefAll.data(), m.M, P, 1);
  double t1 = omp_get_wtime();
  gemm('N', 'N', tN, P, tM, 1.0, m.What.data(), tN, m.gammaRefAll.data(), tM,
       0.0, m.hatRefAll.data(), tN);
  double t2 = omp_get_wtime();
  rotAll(m.R.data(), m.hatRefAll.data(), m.hatGlob.data(), m.N, P, 0);
  double t3 = omp_get_wtime();

  // Stokeslet field via the treecode (physical coords, pref included)
  const vec &u = m.tce.eval(m.hatGlob, m.selfSkip);
  double t4 = omp_get_wtime();
#pragma omp parallel for schedule(static)
  for (long i = 0; i < m.n; ++i) y[i] += u[i];

  double t5 = omp_get_wtime(), t6 = t5, t7 = t5;
  if (!m.selfSkip) {
    // subtract the exact same-particle block: R_k * (S_self * hat_ref_k)
    gemm('N', 'N', tM, P, tN, 1.0, m.Sself.data(), tM, m.hatRefAll.data(), tN,
         0.0, m.selfRefAll.data(), tM);
    t6 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long t = 0; t < (long)m.M * P; ++t) {
      const double *Rk = m.R.data() + (size_t)9 * (t / m.M);
      const double *v = m.selfRefAll.data() + 3 * t;
      y[3 * t]     -= Rk[0] * v[0] + Rk[1] * v[1] + Rk[2] * v[2];
      y[3 * t + 1] -= Rk[3] * v[0] + Rk[4] * v[1] + Rk[5] * v[2];
      y[3 * t + 2] -= Rk[6] * v[0] + Rk[7] * v[1] + Rk[8] * v[2];
    }
    t7 = omp_get_wtime();
  }

  ++m.mvcount;
  m.t_mv += t7 - t0;
  m.t_tc_mv += t4 - t3;
  m.t_dense += (t2 - t1) + (t6 - t5);
  m.t_rot += (t1 - t0) + (t3 - t2) + (t5 - t4) + (t7 - t6);
  if (m.verbose_mv) {
    std::printf("[matvec %d] total=%.3f s (treecode=%.3f [up=%.3f trav=%.3f] "
                "dense=%.3f rot/axpy=%.3f)\n",
                m.mvcount, t7 - t0, t4 - t3, m.tce.t_up_last, m.tce.t_trav_last,
                (t2 - t1) + (t6 - t5), (t1 - t0) + (t3 - t2) + (t5 - t4) + (t7 - t6));
    std::fflush(stdout);
  }
}

// Recovery U[k] = -K_N^T pinv gamma_ref, rotated to global: port of
// mfsSolve:1177-1201 + recoverRotate. Uout is P x 6 [vx vy vz wx wy wz].
static void recover(MFS &m, const double *xsol, vec &Uout)
{
  const int P = m.P, tM = m.tM, tN = m.tN;
  rotAll(m.R.data(), xsol, m.gammaRefAll.data(), m.M, P, 1);
  gemm('N', 'N', tN, P, tM, 1.0, m.pinv.data(), tN, m.gammaRefAll.data(), tM,
       0.0, m.hatRefAll.data(), tN);  // lam_ref_all (reuses hatRefAll)
  vec Uref((size_t)6 * P);
  gemm('T', 'N', 6, P, tN, -1.0, m.KN.data(), tN, m.hatRefAll.data(), tN,
       0.0, Uref.data(), 6);
  Uout.assign((size_t)6 * P, 0.0);
#pragma omp parallel for schedule(static)
  for (int k = 0; k < P; ++k) {
    const double *Rk = m.R.data() + (size_t)9 * k;
    const double *v = Uref.data() + 6 * k;
    const double *w = v + 3;
    Uout[6 * k + 0] = Rk[0] * v[0] + Rk[1] * v[1] + Rk[2] * v[2];
    Uout[6 * k + 1] = Rk[3] * v[0] + Rk[4] * v[1] + Rk[5] * v[2];
    Uout[6 * k + 2] = Rk[6] * v[0] + Rk[7] * v[1] + Rk[8] * v[2];
    Uout[6 * k + 3] = Rk[0] * w[0] + Rk[1] * w[1] + Rk[2] * w[2];
    Uout[6 * k + 4] = Rk[3] * w[0] + Rk[4] * w[1] + Rk[5] * w[2];
    Uout[6 * k + 5] = Rk[6] * w[0] + Rk[7] * w[1] + Rk[8] * w[2];
  }
}

// Direct Stokeslet sum (skipSelf optional) at target t over all sources.
// Port of pairwiseStokeslet.
static void directAtTarget(const double *X, const double *Y, const double *F,
                           long t, long nS, int M, int N, double pref,
                           int skipSelf, double out[3])
{
  long tp = t / M;
  double tx = X[3 * t], ty = X[3 * t + 1], tz = X[3 * t + 2];
  double u0 = 0, u1 = 0, u2 = 0;
  for (long s = 0; s < nS; ++s) {
    if (skipSelf && (s / N) == tp) continue;
    double rx = tx - Y[3 * s];
    double ry = ty - Y[3 * s + 1];
    double rz = tz - Y[3 * s + 2];
    double r2 = rx * rx + ry * ry + rz * rz;
    if (r2 <= 1e-28) continue;
    double ir = 1.0 / std::sqrt(r2);
    double ir3 = ir / r2;
    double fx = F[3 * s], fy = F[3 * s + 1], fz = F[3 * s + 2];
    double rdf = rx * fx + ry * fy + rz * fz;
    u0 += fx * ir + rx * rdf * ir3;
    u1 += fy * ir + ry * rdf * ir3;
    u2 += fz * ir + rz * rdf * ir3;
  }
  out[0] = pref * u0;
  out[1] = pref * u1;
  out[2] = pref * u2;
}

// rel-L2 of (a - b) over the given component triples.
static double relL2(const double *a, const double *b, size_t n3)
{
  double num = 0, den = 0;
#pragma omp parallel for schedule(static) reduction(+ : num, den)
  for (long i = 0; i < (long)n3; ++i) {
    double d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return std::sqrt(num / den);
}

// Direct-sum spot check of a treecode full-field eval: sample nsamp targets
// with a fixed stride, brute-sum in physical coordinates, compare with the
// treecode output. Returns rel-L2 over the sampled triples.
static double directSpotCheck(const MFS &m, const vec &density, const vec &u_tc,
                              int nsamp)
{
  long nT = (long)m.tce.n_trg, nS = (long)m.Yglob.size() / 3;
  long step = std::max(1L, nT / std::max(1, nsamp));
  std::vector<long> tt;
  for (long t = 0; t < nT; t += step) tt.push_back(t);
  vec ud(3 * tt.size()), uf(3 * tt.size());
#pragma omp parallel for schedule(dynamic)
  for (long i = 0; i < (long)tt.size(); ++i) {
    directAtTarget(m.Xglob.data(), m.Yglob.data(), density.data(), tt[i], nS,
                   m.M, m.N, m.pref, 0, &ud[3 * i]);
    for (int d = 0; d < 3; ++d) uf[3 * i + d] = u_tc[3 * tt[i] + d];
  }
  return relL2(uf.data(), ud.data(), 3 * tt.size());
}

// ==========================================================================
// modes
// ==========================================================================

static void loadParticles(MFS &m, const std::string &csv, bool &hasRef,
                          std::vector<mfscase::Row> &rows, int maxP = 0)
{
  rows = mfscase::loadCSV(csv, hasRef);
  if (maxP > 0 && maxP < (int)rows.size()) {
    rows.resize(maxP);
    hasRef = false;  // reference velocities are mobility solutions of the FULL system
  }
  const int P = (int)rows.size();
  m.P = P;
  m.centers.resize((size_t)3 * P);
  m.F.resize((size_t)3 * P);
  m.T.resize((size_t)3 * P);
  m.R.resize((size_t)9 * P);
  for (int k = 0; k < P; ++k) {
    for (int d = 0; d < 3; ++d) {
      m.centers[3 * k + d] = rows[k].c[d];
      m.F[3 * k + d] = rows[k].f[d];
      m.T[3 * k + d] = rows[k].t[d];
    }
    for (int e = 0; e < 9; ++e) m.R[(size_t)9 * k + e] = rows[k].R[e];
  }
  // R_k orthogonality sanity
  double worst = 0;
  for (int k = 0; k < P; ++k) {
    const double *Rk = m.R.data() + (size_t)9 * k;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        double s = 0;
        for (int c = 0; c < 3; ++c) s += Rk[3 * a + c] * Rk[3 * b + c];
        worst = std::max(worst, std::fabs(s - (a == b ? 1.0 : 0.0)));
      }
  }
  std::printf("[csv] P=%d hasRef=%d max|R R^T - I|=%.2e\n", P, (int)hasRef, worst);
}

static void setupDims(MFS &m, int M, int N)
{
  m.M = M; m.N = N;
  m.tM = 3 * M; m.tN = 3 * N;
  m.n = 3L * M * m.P;
  m.gammaRefAll.resize((size_t)m.tM * m.P);
  m.hatRefAll.resize((size_t)m.tN * m.P);
  m.hatGlob.resize((size_t)m.tN * m.P);
  m.selfRefAll.resize((size_t)m.tM * m.P);
}

struct Options {
  std::string mode = "solve";
  std::string csv, colloc, source, caseDir, outCsv;
  std::string buckets = "object";     // object | grid
  std::string self = "subtract";      // subtract | skip
  double mac = 0.6;                   // PDEG7 operating point
  int maxLeaf = 256;                  // grid mode only
  int restart = 30, maxits = 500, samples = 2000, omp = 0;
  int warmup = 1, repeat = 3;         // -mode bench only
  int P = 0;  // truncate the CSV to the first P particles (0 = all)
  double rtol = 1e-7;
};

int main(int argc, char **argv)
{
  Options o;
  o.csv = "mfs/ellipsoid_mobility_p1000_delta0p2_nv34.csv";
  o.colloc = "mfs/points/b_ellipsoid_p1000.txt";
  o.source = "mfs/points/s_ellipsoid_p1000.txt";
  o.caseDir = "mfs_ellipsoid_case";

  std::vector<char *> petscArgv{argv[0]};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "-mode") o.mode = next();
    else if (a == "-csv") o.csv = next();
    else if (a == "-colloc") o.colloc = next();
    else if (a == "-source") o.source = next();
    else if (a == "-case") o.caseDir = next();
    else if (a == "-o") o.outCsv = next();
    else if (a == "-P") o.P = std::atoi(next().c_str());
    else if (a == "-mac") o.mac = std::atof(next().c_str());
    else if (a == "-maxleaf") o.maxLeaf = std::atoi(next().c_str());
    else if (a == "-buckets") o.buckets = next();
    else if (a == "-self") o.self = next();
    else if (a == "-rtol") o.rtol = std::atof(next().c_str());
    else if (a == "-restart") o.restart = std::atoi(next().c_str());
    else if (a == "-maxits") o.maxits = std::atoi(next().c_str());
    else if (a == "-samples") o.samples = std::atoi(next().c_str());
    else if (a == "-warmup") o.warmup = std::atoi(next().c_str());
    else if (a == "-repeat") o.repeat = std::atoi(next().c_str());
    else if (a == "-omp") o.omp = std::atoi(next().c_str());
    else petscArgv.push_back(argv[i]);  // forward unknown args to PETSc
  }
  if (o.omp > 0) omp_set_num_threads(o.omp);
  if (o.buckets != "object" && o.buckets != "grid") {
    std::fprintf(stderr, "-buckets must be object|grid\n");
    return 2;
  }
  if (o.self != "subtract" && o.self != "skip") {
    std::fprintf(stderr, "-self must be subtract|skip\n");
    return 2;
  }
  if (o.self == "skip" && o.buckets != "object") {
    std::fprintf(stderr, "-self skip requires -buckets object\n");
    return 2;
  }
  int pArgc = (int)petscArgv.size();
  char **pArgv = petscArgv.data();
  mfs_petsc_init(&pArgc, &pArgv);

  srand48(20260717);

  std::printf("mode=%s mac=%g buckets=%s self=%s rtol=%.1e restart=%d "
              "maxits=%d threads=%d\n",
              o.mode.c_str(), o.mac, o.buckets.c_str(), o.self.c_str(), o.rtol,
              o.restart, o.maxits, omp_get_max_threads());

  // ---- reference clouds + particles (all modes need them) ----
  MFS m;
  int M = 0, N = 0;
  vec Xref = mfscase::loadPointsASCII(o.colloc, M);
  vec Yref = mfscase::loadPointsASCII(o.source, N);
  bool hasRef = false;
  std::vector<mfscase::Row> rows;
  loadParticles(m, o.csv, hasRef, rows, o.P);
  m.selfSkip = (o.self == "skip");
  const int objGS = (o.buckets == "object") ? N : 0;

  int status = 0;

  if (o.mode == "selftest") {
    // ---- 1: identity-matvec GMRES must converge in 1 iteration ----
    {
      long n = 1000;
      vec b(n), x(n);
      for (long i = 0; i < n; ++i) b[i] = drand48();
      MFSSolveParams p; p.n = n;
      MFSSolveStats st;
      MFSMatVec ident = [n](const double *xx, double *yy) {
        std::memcpy(yy, xx, (size_t)n * sizeof(double));
      };
      int rc = mfs_petsc_gmres_solve(p, ident, b.data(), x.data(), st);
      double e = relL2(x.data(), b.data(), n);
      std::printf("[selftest] identity GMRES: rc=%d its=%d rnorm=%.2e relerr=%.2e (%s)\n",
                  rc, st.iterations, st.final_rnorm, e, st.reason.c_str());
      if (rc || st.iterations > 1 || e > 1e-12) status = 1;
    }
    // ---- 2: dense identities + P=2 brute-force matvec check, BOTH self
    //         modes on the same clouds/tree ----
    {
      MFS s;
      s.P = 2;
      s.centers.assign(m.centers.begin(), m.centers.begin() + 6);
      s.F.assign(m.F.begin(), m.F.begin() + 6);
      s.T.assign(m.T.begin(), m.T.begin() + 6);
      s.R.assign(m.R.begin(), m.R.begin() + 18);
      setupDims(s, M, N);
      denseSetup(s, Xref, Yref);
      buildClouds(s, Xref, Yref);
      // mac 0.3 at P=2 pushes the (single) other-particle leaf to exact P2P,
      // so this gate tests operator wiring + self handling at round-off.
      // (At the run mac, random gamma excites S_L's smallest singular values,
      // |hat| >> |gamma|, and far-field truncation amplified by self-field
      // cancellation dominates -- the far field is judged on PHYSICAL
      // strengths in rhs-validate instead.)
      const float macP2 = std::min((float)o.mac, 0.3f);
      std::printf("[selftest] P=2 check at mac=%.2f (exact-P2P regime)\n", macP2);
      s.tce.init(s.Yglob, s.Xglob, macP2, o.maxLeaf, objGS, (size_t)M);
      s.verbose_mv = false;

      vec x(s.n), yb(s.n);
      for (long i = 0; i < s.n; ++i) x[i] = drand48() - 0.5;

      // Isolate the treecode's own error first: full field (self included)
      // vs direct full sum, same strengths (needs one matvec to fill hatGlob).
      vec y(s.n);
      s.selfSkip = false;
      matvec(s, x.data(), y.data());
      long nT = (long)s.tM * s.P / 3, nS = (long)s.tN * s.P / 3;
      {
        const vec &uf = s.tce.eval(s.hatGlob, false);
        vec ud(3 * nT);
#pragma omp parallel for schedule(static)
        for (long t = 0; t < nT; ++t)
          directAtTarget(s.Xglob.data(), s.Yglob.data(), s.hatGlob.data(), t, nS,
                         s.M, s.N, s.pref, /*skipSelf=*/0, &ud[3 * t]);
        double hatrms = 0;
        for (double v : s.hatGlob) hatrms += v * v;
        std::printf("[selftest] P=2 treecode full field vs direct full sum: rel-L2 = %.3e "
                    "(|hat|_rms=%.2e from |x|_rms~0.29)\n",
                    relL2(uf.data(), ud.data(), 3 * nT),
                    std::sqrt(hatrms / s.hatGlob.size()));
      }
      // brute: y = x + off-diagonal Stokeslet of hatGlob
#pragma omp parallel for schedule(static)
      for (long t = 0; t < nT; ++t) {
        double u[3];
        directAtTarget(s.Xglob.data(), s.Yglob.data(), s.hatGlob.data(), t, nS,
                       s.M, s.N, s.pref, /*skipSelf=*/1, u);
        for (int d = 0; d < 3; ++d) yb[3 * t + d] = x[3 * t + d] + u[d];
      }
      // Field-only error (strip the common x term); frame/sign bugs give O(1).
      for (const char *mode : {"subtract", "skip"}) {
        const bool skip = (std::string(mode) == "skip");
        if (skip && objGS == 0) continue;   // skip needs object buckets
        s.selfSkip = skip;
        vec ym(s.n);
        matvec(s, x.data(), ym.data());
        vec fy(s.n), fyb(s.n);
        for (long i = 0; i < s.n; ++i) { fy[i] = ym[i] - x[i]; fyb[i] = yb[i] - x[i]; }
        double e = relL2(ym.data(), yb.data(), s.n);
        double ef = relL2(fy.data(), fyb.data(), s.n);
        std::printf("[selftest] P=2 matvec (-self %s vs brute skip-self): "
                    "rel-L2 = %.3e (field-only %.3e)\n", mode, e, ef);
        if (ef > 1e-2) status = 1;
      }
    }
    std::printf("[selftest] %s\n", status == 0 ? "PASS" : "FAIL");
  } else if (o.mode == "rhs-validate") {
    setupDims(m, M, N);
    denseSetup(m, Xref, Yref);
    buildClouds(m, Xref, Yref);

    // ---- clouds vs export ----
    size_t nt = 0, ns = 0, nst = 0, nvt = 0;
    std::vector<int32_t> tid;
    vec Xexp = mfscase::loadSoACoords(o.caseDir + "/colloc_targets.bin", nt, &tid);
    vec Yexp = mfscase::loadSoACoords(o.caseDir + "/proxy_sources.bin", ns, nullptr);
    vec Lexp = mfscase::loadSoAField(o.caseDir + "/proxy_strengths.bin", nst);
    vec Uexp = mfscase::loadSoAField(o.caseDir + "/velocity_treecode.bin", nvt);
    if (nt != (size_t)m.P * m.M || ns != (size_t)m.P * m.N || nst != ns || nvt != nt) {
      std::fprintf(stderr, "export size mismatch: nt=%zu ns=%zu nst=%zu nvt=%zu\n",
                   nt, ns, nst, nvt);
      return 1;
    }
    double dx = 0, dy = 0;
    for (size_t i = 0; i < Xexp.size(); ++i) dx = std::max(dx, std::fabs(Xexp[i] - m.Xglob[i]));
    for (size_t i = 0; i < Yexp.size(); ++i) dy = std::max(dy, std::fabs(Yexp[i] - m.Yglob[i]));
    bool idOk = tid.front() == 0 && tid.back() == m.P - 1 &&
                tid[(size_t)m.M] == 1;  // ids follow t/M
    std::printf("[rhs-validate] clouds vs export: max|dX|=%.3e max|dY|=%.3e ids=%s\n",
                dx, dy, idOk ? "ok" : "MISMATCH");

    // ---- lam0 vs export (validates the whole dense chain) ----
    vec lam0;
    buildLam0(m, lam0);
    std::printf("[rhs-validate] lam0 vs proxy_strengths.bin: rel-L2 = %.3e\n",
                relL2(lam0.data(), Lexp.data(), lam0.size()));

    // ---- u0 via treecode vs the GPU treecode RHS export ----
    m.tce.init(m.Yglob, m.Xglob, (float)o.mac, o.maxLeaf, objGS, (size_t)M);
    const vec &u0 = m.tce.eval(Lexp, false);  // strengths from the FILE
    std::printf("[rhs-validate] u0(treecode, %.3f s) vs velocity_treecode.bin "
                "(GPU treecode @ its own mac): rel-L2 = %.3e\n",
                m.tce.t_last, relL2(u0.data(), Uexp.data(), u0.size()));

    // ---- direct-sum spot check: the CPU treecode's own field error ----
    if (o.samples > 0) {
      double t0 = omp_get_wtime();
      double e = directSpotCheck(m, Lexp, u0, o.samples);
      std::printf("[rhs-validate] treecode vs direct sum on %d sampled targets: "
                  "rel-L2 = %.3e (%.1f s)\n", o.samples, e, omp_get_wtime() - t0);
    }

    // ---- off-diagonal matvec spot check (the GMRES operator's Stokeslet)
    // in BOTH self modes vs brute skip-self, physical frame, export
    // strengths. subtract measures the self-cancellation error; skip is the
    // in-traversal operator. ----
    if (o.samples > 0) {
      rotAll(m.R.data(), Lexp.data(), m.hatRefAll.data(), m.N, m.P, 1);
      gemm('N', 'N', m.tM, m.P, m.tN, 1.0, m.Sself.data(), m.tM,
           m.hatRefAll.data(), m.tN, 0.0, m.selfRefAll.data(), m.tM);
      long nT = (long)m.tce.n_trg, nS = (long)m.Yglob.size() / 3;
      long step = std::max(1L, nT / o.samples);
      std::vector<long> tt;
      for (long t = 0; t < nT; t += step) tt.push_back(t);
      vec uo(3 * tt.size()), ud(3 * tt.size());
      double t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
      for (long i = 0; i < (long)tt.size(); ++i) {
        long t = tt[i];
        const double *Rk = m.R.data() + (size_t)9 * (t / m.M);
        const double *v = m.selfRefAll.data() + 3 * t;
        double self[3] = {Rk[0] * v[0] + Rk[1] * v[1] + Rk[2] * v[2],
                          Rk[3] * v[0] + Rk[4] * v[1] + Rk[5] * v[2],
                          Rk[6] * v[0] + Rk[7] * v[1] + Rk[8] * v[2]};
        for (int d = 0; d < 3; ++d) uo[3 * i + d] = u0[3 * t + d] - self[d];
        directAtTarget(m.Xglob.data(), m.Yglob.data(), Lexp.data(), t, nS, m.M,
                       m.N, m.pref, /*skipSelf=*/1, &ud[3 * i]);
      }
      std::printf("[rhs-validate] off-diag matvec (full - self) vs direct skip-self "
                  "on %zu targets: rel-L2 = %.3e (%.1f s)\n",
                  tt.size(), relL2(uo.data(), ud.data(), 3 * tt.size()),
                  omp_get_wtime() - t0);
      if (objGS > 0) {
        const vec &us = m.tce.eval(Lexp, true);   // in-traversal skip
        vec uo2(3 * tt.size());
        for (long i = 0; i < (long)tt.size(); ++i)
          for (int d = 0; d < 3; ++d) uo2[3 * i + d] = us[3 * tt[i] + d];
        std::printf("[rhs-validate] off-diag matvec (-self skip) vs direct skip-self "
                    "on %zu targets: rel-L2 = %.3e\n",
                    tt.size(), relL2(uo2.data(), ud.data(), 3 * tt.size()));
      }
    }
  } else if (o.mode == "bench") {
    // ---- operator-only profiling harness ------------------------------------
    // The GMRES matvec is >99% one treecode eval (see the solve timing line),
    // so this mode reproduces the MFS operator exactly -- same clouds, same
    // object buckets, same skip semantics, same fp64 density layout -- but
    // skips denseSetup/PETSc so that a `perf stat` window is almost entirely
    // the merged traversal loop. TC_CPU_COMPONENTS=all|m2p|p2p|trav switches
    // off one of the two evaluation bodies at compile time to decompose it.
    setupDims(m, M, N);
    buildClouds(m, Xref, Yref);
    m.tce.init(m.Yglob, m.Xglob, (float)o.mac, o.maxLeaf, objGS, (size_t)M);

    int comps = tccpu::TreecodeCpu::kAll;
    const char *cs = std::getenv("TC_CPU_COMPONENTS");
    std::string ctok = cs ? cs : "all";
    if (ctok == "all") comps = tccpu::TreecodeCpu::kAll;
    else if (ctok == "m2p") comps = tccpu::TreecodeCpu::kM2P;
    else if (ctok == "p2p") comps = tccpu::TreecodeCpu::kP2P;
    else if (ctok == "trav") comps = tccpu::TreecodeCpu::kTravOnly;
    else { std::fprintf(stderr, "TC_CPU_COMPONENTS=all|m2p|p2p|trav\n"); return 2; }
    m.tce.tc->setComponents(comps);

    // density in the same space as hat_lam (3N per particle), random values:
    // traversal work and instruction mix are value-independent.
    vec dens((size_t)m.tN * m.P);
    for (size_t i = 0; i < dens.size(); ++i) dens[i] = drand48() - 0.5;

    const int W = o.warmup, R = std::max(1, o.repeat);
    std::printf("[bench] components=%s warmup=%d repeat=%d self=%s threads=%d\n",
                ctok.c_str(), W, R, m.selfSkip ? "skip" : "full",
                omp_get_max_threads());
    for (int i = 0; i < W; ++i) m.tce.eval(dens, m.selfSkip);
    double tmin = 1e300, tsum = 0, upsum = 0, trsum = 0;
    for (int i = 0; i < R; ++i) {
      m.tce.eval(dens, m.selfSkip);
      const auto &st = m.tce.tc->stats();
      std::printf("[bench] eval %d: %.4f s (upward %.4f, traverse %.4f) "
                  "m2p=%lld p2p=%lld leaves=%lld\n",
                  i, m.tce.t_last, m.tce.t_up_last, m.tce.t_trav_last,
                  st.m2pNodes, st.p2pInteractions, st.p2pLeaves);
      tmin = std::min(tmin, m.tce.t_last);
      tsum += m.tce.t_last; upsum += m.tce.t_up_last; trsum += m.tce.t_trav_last;
    }
    const auto &st = m.tce.tc->stats();
    const double NT = (double)m.tce.n_trg;
    // per-target interaction counts; node visits follow from the binary tree
    // (every non-terminal visit pushes exactly 2 children, and with -self skip
    // exactly one leaf per target -- the own bucket -- terminates uncounted).
    const double accPT = (double)st.m2pNodes / NT;
    const double leafPT = (double)st.p2pLeaves / NT + (m.selfSkip ? 1.0 : 0.0);
    const double visitsPT = 2.0 * (accPT + leafPT) - 1.0;
    const double p2pPT = (double)st.p2pInteractions / NT;
    // arithmetic flops (FMA = 2), excluding the div+sqrt pair per interaction
    const double m2pFlops = (double)st.m2pNodes * bary_cpu::NP3 * 30.0;
    const double p2pFlops = (double)st.p2pInteractions * 27.0;
    // cold-traffic upper bound: every accepted node re-reads its 6 KB moment
    // block, every near interaction re-reads 6 fp64 source streams
    const double m2pBytes = (double)st.m2pNodes * (3.0 * bary_cpu::NP3 * 4 + 16);
    const double p2pBytes = (double)st.p2pInteractions * 48.0;
    const double travBytes = (double)NT * visitsPT * 24.0;
    const double tavg = tsum / R;
    std::printf("\n[bench] targets=%.0f  per-target: visits=%.1f accepted=%.1f "
                "nearleaves=%.1f nearsrc=%.0f\n", NT, visitsPT, accPT, leafPT, p2pPT);
    std::printf("[bench] wall: avg %.4f s  min %.4f s  (upward %.4f, traverse %.4f)\n",
                tavg, tmin, upsum / R, trsum / R);
    std::printf("[bench] arithmetic: M2P %.1f GFLOP + P2P %.1f GFLOP = %.1f GFLOP "
                "/ eval -> %.1f GFLOP/s (%.1f GFLOP/s at min)\n",
                m2pFlops * 1e-9, p2pFlops * 1e-9, (m2pFlops + p2pFlops) * 1e-9,
                (m2pFlops + p2pFlops) * 1e-9 / tavg,
                (m2pFlops + p2pFlops) * 1e-9 / tmin);
    std::printf("[bench] div+sqrt pairs: %.2f G/eval (%.2f G/s)\n",
                ((double)st.m2pNodes * bary_cpu::NP3 + (double)st.p2pInteractions) * 1e-9,
                ((double)st.m2pNodes * bary_cpu::NP3 + (double)st.p2pInteractions) * 1e-9 / tavg);
    std::printf("[bench] cold-traffic bound: M2P %.1f GB + P2P %.1f GB + walk %.1f GB "
                "= %.1f GB/eval (%.1f GB/s if nothing cached)\n",
                m2pBytes * 1e-9, p2pBytes * 1e-9, travBytes * 1e-9,
                (m2pBytes + p2pBytes + travBytes) * 1e-9,
                (m2pBytes + p2pBytes + travBytes) * 1e-9 / tavg);
    std::printf("[bench] working set: moments %.0f MB, sources %.0f MB, "
                "targets %.0f MB, out %.0f MB, rss %.1f GB\n",
                st.nodeM2PMB, 6.0 * 8.0 * m.Yglob.size() / 3.0 / 1e6,
                (8.0 * 3 + 4.0 * 3) * m.Xglob.size() / 3.0 / 1e6,
                8.0 * m.Xglob.size() / 1e6, mfs_rss_gb());
    std::fflush(stdout);
  } else {  // solve
    setupDims(m, M, N);
    std::printf("P=%d ellipsoids, M=%d collocation, N=%d source, n=%ld unknowns%s\n",
                m.P, M, N, m.n, hasRef ? "" : "  [perf-only: no reference velocities]");
    double tAll0 = omp_get_wtime();
    denseSetup(m, Xref, Yref);
    buildClouds(m, Xref, Yref);
    m.tce.init(m.Yglob, m.Xglob, (float)o.mac, o.maxLeaf, objGS, (size_t)M);
    double tSetup1 = omp_get_wtime();

    // ---- RHS: u0 = FULL Stokeslet field of lam0 (self included) ----
    vec lam0;
    buildLam0(m, lam0);
    const vec &u = m.tce.eval(lam0, false);
    vec b(m.n);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < m.n; ++i) b[i] = u[i];
    double tRhs1 = omp_get_wtime();
    std::printf("[rhs] lam0+eval done (treecode %.3f s)\n", m.tce.t_last);
    if (o.samples > 0) {
      double e = directSpotCheck(m, lam0, u, o.samples);
      std::printf("[rhs] treecode vs direct sum on %d sampled targets: rel-L2 = %.3e\n",
                  o.samples, e);
    }
    std::fflush(stdout);

    // ---- GMRES ----
    MFSSolveParams p;
    p.n = m.n; p.rtol = o.rtol; p.restart = o.restart; p.maxits = o.maxits;
    MFSSolveStats st;
    vec xsol(m.n);
    MFSMatVec A = [&](const double *xx, double *yy) { matvec(m, xx, yy); };
    int rc = mfs_petsc_gmres_solve(p, A, b.data(), xsol.data(), st);
    if (rc) { std::fprintf(stderr, "KSPSolve failed rc=%d\n", rc); return rc; }

    // ---- recovery ----
    double tRec0 = omp_get_wtime();
    vec U;
    recover(m, xsol.data(), U);
    double tRec1 = omp_get_wtime();

    std::printf("\n[MFS solve] its=%d final_rnorm=%.6e reason=%s\n",
                st.iterations, st.final_rnorm, st.reason.c_str());
    std::printf("[MFS solve] ksp_wall=%.3f s (matvecs=%d, avg %.3f s: treecode %.3f, "
                "dense %.3f, rot/axpy %.3f)\n",
                st.ksp_wall_sec, m.mvcount,
                m.mvcount ? m.t_mv / m.mvcount : 0.0,
                m.mvcount ? m.t_tc_mv / m.mvcount : 0.0,
                m.mvcount ? m.t_dense / m.mvcount : 0.0,
                m.mvcount ? m.t_rot / m.mvcount : 0.0);
    std::printf("[MFS solve] setup(dense+clouds+tree)=%.3f s rhs=%.3f s recover=%.3f s "
                "total=%.3f s peakRSS=%.1f GB\n", tSetup1 - tAll0, tRhs1 - tSetup1,
                tRec1 - tRec0, omp_get_wtime() - tAll0, mfs_rss_gb("VmHWM:"));

    if (!o.outCsv.empty()) {
      FILE *f = std::fopen(o.outCsv.c_str(), "w");
      std::fprintf(f, "particle,vx,vy,vz,wx,wy,wz\n");
      for (int k = 0; k < m.P; ++k)
        std::fprintf(f, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", k,
                     U[6 * k], U[6 * k + 1], U[6 * k + 2], U[6 * k + 3],
                     U[6 * k + 4], U[6 * k + 5]);
      std::fclose(f);
    }

    if (!hasRef) {
      std::printf("\nperf-only run (config CSV has no reference velocities)\n");
      for (int k = 0; k < m.P && k < 4; ++k)
        std::printf("  particle %d: v=(%+.4f %+.4f %+.4f)  w=(%+.4f %+.4f %+.4f)\n",
                    k, U[6 * k], U[6 * k + 1], U[6 * k + 2],
                    U[6 * k + 3], U[6 * k + 4], U[6 * k + 5]);
    } else {
      // ---- compare (verbatim port of test_mfs_ellipsoid.cu:202-234) ----
      double maxV = 0, maxW = 0, refVmax = 0, refWmax = 0;
      double nV2 = 0, nW2 = 0, dV2 = 0, dW2 = 0;
      int kV = 0, kW = 0;
      for (int k = 0; k < m.P; ++k) {
        for (int d = 0; d < 3; ++d) {
          double ev = U[6 * k + d] - rows[k].v[d];
          double ew = U[6 * k + 3 + d] - rows[k].w[d];
          if (std::fabs(ev) > maxV) { maxV = std::fabs(ev); kV = k; }
          if (std::fabs(ew) > maxW) { maxW = std::fabs(ew); kW = k; }
          refVmax = std::max(refVmax, std::fabs(rows[k].v[d]));
          refWmax = std::max(refWmax, std::fabs(rows[k].w[d]));
          dV2 += ev * ev; nV2 += rows[k].v[d] * rows[k].v[d];
          dW2 += ew * ew; nW2 += rows[k].w[d] * rows[k].w[d];
        }
      }
      std::printf("\n%4s | %12s %12s %12s | %12s %12s %12s\n", "k", "v_x(mine)",
                  "v_x(ref)", "dv_x", "w_x(mine)", "w_x(ref)", "dw_x");
      for (int k = 0; k < m.P && k < 6; ++k)
        std::printf("%4d | %12.6f %12.6f %12.3e | %12.6f %12.6f %12.3e\n", k,
                    U[6 * k], rows[k].v[0], U[6 * k] - rows[k].v[0],
                    U[6 * k + 3], rows[k].w[0], U[6 * k + 3] - rows[k].w[0]);
      std::printf("\nvelocity     : max|dv| = %.3e (particle %d), max|v_ref| = %.3e -> rel %.3e | rel-L2 %.3e\n",
                  maxV, kV, refVmax, maxV / refVmax, std::sqrt(dV2 / nV2));
      std::printf("angular vel  : max|dw| = %.3e (particle %d), max|w_ref| = %.3e -> rel %.3e | rel-L2 %.3e\n",
                  maxW, kW, refWmax, maxW / refWmax, std::sqrt(dW2 / nW2));
      bool pass = (maxV / refVmax < 1e-2) && (maxW / refWmax < 1e-2);
      std::printf("\n%s\n", pass ? "PASS (within 1% of reference)" : "MISMATCH");
      if (!pass) status = 1;
    }
  }

  std::fflush(stdout);
  mfs_petsc_finalize();
  return status;
}
