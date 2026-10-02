#pragma once

// SPDX-License-Identifier: Apache-2.0
//
// CUDA port of the Broms-Barnett-Tornberg (2024) Method of Fundamental
// Solutions for the Stokes *mobility* problem (mfs/broms_mfs.py).
//
// Given P rigid particles with applied forces/torques, solve for their
// rigid-body velocities. Uses the recompleted, one-body-preconditioned MFS
// formulation (paper eqs 42,48,50,51,53,54): a well-conditioned square system
// A gamma = u0 solved with PETSc GMRES (matrix-free MatShell). The off-diagonal
// (inter-particle) Stokeslet sums are done here as a brute-force O(P^2*M*N)
// loop -- the natural hook where a treecode will slot in later.
//
// fp64 end-to-end. Monodisperse: every particle is a rigid rotation/translation
// of particle 0; the one-body algebra (SVD pseudo-inverse etc.) is computed once
// in the reference frame and reused for all particles through rotations.
//
// Dynamic-simulation design: the solver is a PERSISTENT context (MFSContext).
//   * mfsSetup()  builds + CACHES everything that depends only on particle shape
//                 and point resolution (M,N): X_ref/Y_ref, K_N, Ginv, ImL, pinv,
//                 W_hat -- plus all per-step device buffers and the PETSc
//                 Mat/KSP/Vec objects -- ONCE.
//   * mfsSolve()  is called per timestep with the new particle state (centers +
//                 row-major orientations R + applied force/torque). It generates
//                 the global clouds on the GPU from the cached reference cloud,
//                 recomputes only lam0 / u0, reuses the cached dense ops + PETSc
//                 objects, solves, and recovers the rigid-body velocities.
//   * mfsDestroy() tears everything down.
// A thin solveMobility() wrapper keeps the one-shot host-vector API working.
//
// Everything is GPU-resident: the only host<->device traffic per step is the
// small (P*3 / P*9) state arrays in and the P*6 velocities out. The per-particle
// dense applies are batched into single cuBLAS GEMMs (width P).
//
// Library strategy: particle data stays device-resident; the one-time dense
// setup uses cuBLAS + cuSOLVER; the GMRES vectors are VECSEQCUDA so PETSc reads
// our device pointers directly (zero-copy matvec). Column-major throughout for
// cuBLAS/cuSOLVER; row-major math is realized via CUBLAS_OP_T.
//
// Verification: drivers reproduce mfs/validate_triangle.py (3-sphere triangle
// sweep) and a 40-sphere reference CSV.

#include <petscksp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --------------------------------------------------------------------------
// error checking
// --------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(e_));                                     \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t s_ = (call);                                                 \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,      \
                   (int)s_);                                                    \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUSOLVER_CHECK(call)                                                    \
  do {                                                                          \
    cusolverStatus_t s_ = (call);                                              \
    if (s_ != CUSOLVER_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__,    \
                   (int)s_);                                                    \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// GPU treecode acceleration of the Stokeslet sums (defines CUDA_CHECK above
// first so the treecode headers' #ifndef fallback resolves to ours). This pulls
// in cuBQL + the multipole modules' __device__ globals, so mfs_broms.cuh (and
// hence this header) must be included by exactly one driver TU.
#include "treecode.cuh"

template<class MP = mp::BaryStokes>
using MFSTreecode = Treecode<MP>;

static_assert(sizeof(vec3d) == 3 * sizeof(double),
              "MFS double AoS cloud layout must match cuBQL::vec3d");

// ==========================================================================
// device kernels
// ==========================================================================

// Brute-force pairwise Stokeslet (Oseen) sum -- THE treecode hook.
// For each target collocation point t: u += pref*( f/r + R(R.f)/r^3 ),
// R = X_t - Y_s, summed over source points s (optionally skipping the source's
// own particle for the off-diagonal matvec). Targets and sources are AoS
// [particle][point][xyz]; particle(t)=t/M, particle(s)=s/N. Adds into U.
__global__ void pairwiseStokeslet(const double *__restrict__ X,  // 3*nT
                                  const double *__restrict__ Y,  // 3*nS
                                  const double *__restrict__ F,  // 3*nS
                                  double *__restrict__ U,        // 3*nT (+=)
                                  int nT, int nS, int M, int N,
                                  double pref, int skipSelf)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nT) return;
  int tp = t / M;
  double tx = X[3 * t], ty = X[3 * t + 1], tz = X[3 * t + 2];
  double u0 = 0.0, u1 = 0.0, u2 = 0.0;
  for (int s = 0; s < nS; ++s) {
    if (skipSelf && (s / N) == tp) continue;
    double rx = tx - Y[3 * s];
    double ry = ty - Y[3 * s + 1];
    double rz = tz - Y[3 * s + 2];
    double r2 = rx * rx + ry * ry + rz * rz;
    if (r2 <= 1e-28) continue;  // matches python r_norm > 1e-14 guard
    double ir = 1.0 / sqrt(r2);
    double ir3 = ir / r2;
    double fx = F[3 * s], fy = F[3 * s + 1], fz = F[3 * s + 2];
    double rdf = rx * fx + ry * fy + rz * fz;
    u0 += fx * ir + rx * rdf * ir3;
    u1 += fy * ir + ry * rdf * ir3;
    u2 += fz * ir + rz * rdf * ir3;
  }
  U[3 * t]     += pref * u0;
  U[3 * t + 1] += pref * u1;
  U[3 * t + 2] += pref * u2;
}

__global__ void treecodeCompMajorToAoS(const double *__restrict__ Ucm,
                                       double *__restrict__ Uaos,
                                       int n, int addInto)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double ux = Ucm[(size_t)0 * n + i];
  double uy = Ucm[(size_t)1 * n + i];
  double uz = Ucm[(size_t)2 * n + i];
  if (addInto) {
    Uaos[3 * i]     += ux;
    Uaos[3 * i + 1] += uy;
    Uaos[3 * i + 2] += uz;
  } else {
    Uaos[3 * i]     = ux;
    Uaos[3 * i + 1] = uy;
    Uaos[3 * i + 2] = uz;
  }
}

// Dense self Stokeslet block S_self (3M x 3N), column-major (ld = 3M).
// Entry (3i+a, 3j+b) = pref*( delta_ab/r + r_a r_b / r^3 ), r = X_i - Y_j.
// Matches broms_mfs._stokeslet_block after its (M,N,3,3)->(3M,3N) reshape.
__global__ void buildSelfBlock(const double *__restrict__ Xr,  // 3*M (ref)
                               const double *__restrict__ Yr,  // 3*N (ref)
                               double *__restrict__ S, int M, int N, double pref)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int i = idx / N, j = idx % N;
  double r[3] = {Xr[3 * i] - Yr[3 * j], Xr[3 * i + 1] - Yr[3 * j + 1],
                 Xr[3 * i + 2] - Yr[3 * j + 2]};
  double r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
  double ir = (r2 > 1e-28) ? 1.0 / sqrt(r2) : 0.0;
  double ir3 = ir * ir * ir;
  const int ld = 3 * M;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      double val = pref * (ir * (a == b ? 1.0 : 0.0) + ir3 * r[a] * r[b]);
      S[(size_t)(3 * j + b) * ld + (3 * i + a)] = val;  // column-major
    }
}

// ImL = I - L  (n x n, column-major).
__global__ void makeImL(double *__restrict__ ImL, const double *__restrict__ L,
                        int n)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)n * n) return;
  int r = (int)(idx % n), c = (int)(idx / n);  // column-major
  ImL[idx] = (r == c ? 1.0 : 0.0) - L[idx];
}

__global__ void reciprocalKernel(double *__restrict__ s, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) s[i] = 1.0 / s[i];
}

// Apply a 3x3 rotation (row-major, 9 doubles) to each xyz triple of a stacked
// vector. transpose=0 applies R (reference->global); transpose=1 applies R^T
// (global->reference). Matches python _stack_to_global (@R^T == R*col) and
// _stack_to_ref (@R == R^T*col).
__global__ void applyRot(const double *__restrict__ R,
                         const double *__restrict__ in, double *__restrict__ out,
                         int npts, int transpose)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= npts) return;
  double v0 = in[3 * t], v1 = in[3 * t + 1], v2 = in[3 * t + 2];
  if (!transpose) {
    out[3 * t]     = R[0] * v0 + R[1] * v1 + R[2] * v2;
    out[3 * t + 1] = R[3] * v0 + R[4] * v1 + R[5] * v2;
    out[3 * t + 2] = R[6] * v0 + R[7] * v1 + R[8] * v2;
  } else {
    out[3 * t]     = R[0] * v0 + R[3] * v1 + R[6] * v2;
    out[3 * t + 1] = R[1] * v0 + R[4] * v1 + R[7] * v2;
    out[3 * t + 2] = R[2] * v0 + R[5] * v1 + R[8] * v2;
  }
}

// Batched applyRot: a stacked field of P*npts triples (AoS by particle, so
// particle k owns triples [k*npts, (k+1)*npts) ), each rotated by ITS OWN
// per-particle rotation R_k (R is P*9 row-major). transpose semantics match
// applyRot. Safe in place (out==in): each thread touches only its own triple.
// One launch replaces the per-particle applyRot loop.
__global__ void applyRotAll(const double *__restrict__ R,
                            const double *__restrict__ in,
                            double *__restrict__ out, int npts, int P,
                            int transpose)
{
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)npts * P;
  if (t >= total) return;
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

// Generate a global cloud from the cached reference cloud:
//   X_glob[k][i] = R_k * X_ref[i] + center_k   (R applied ref->global).
// Xref is shared across particles (3*npts AoS); R is P*9 row-major; centers P*3.
// Output is AoS by particle (global triple index t, particle k=t/npts). For
// R_k = I this reduces to a pure translation (tile-without-rotation).
__global__ void tileRotateTranslate(const double *__restrict__ Xref,
                                    const double *__restrict__ R,
                                    const double *__restrict__ centers,
                                    double *__restrict__ Xglob, int npts, int P)
{
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)npts * P;
  if (t >= total) return;
  int k = (int)(t / npts);
  int i = (int)(t % npts);
  const double *Rk = R + (size_t)9 * k;
  double v0 = Xref[3 * i], v1 = Xref[3 * i + 1], v2 = Xref[3 * i + 2];
  Xglob[3 * t]     = Rk[0] * v0 + Rk[1] * v1 + Rk[2] * v2 + centers[3 * k];
  Xglob[3 * t + 1] = Rk[3] * v0 + Rk[4] * v1 + Rk[5] * v2 + centers[3 * k + 1];
  Xglob[3 * t + 2] = Rk[6] * v0 + Rk[7] * v1 + Rk[8] * v2 + centers[3 * k + 2];
}

// Build K (3p x 6) column-major (ld=3p): row block i = [I3, -skew(pts_i)].
// One thread per point fills its 3 rows across all 6 columns. Matches the
// (formerly host) buildK / broms_mfs._build_K with center=0. (No primitive
// fits a skew-symmetric block fill, so this stays a custom kernel.)
__global__ void buildKKernel(const double *__restrict__ pts,
                             double *__restrict__ K, int p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p) return;
  const size_t ld = (size_t)3 * p;
  double px = pts[3 * i], py = pts[3 * i + 1], pz = pts[3 * i + 2];
  // I3 block (cols 0..2): write all 9 entries so no pre-zeroing is needed.
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      K[(size_t)b * ld + (3 * i + a)] = (a == b) ? 1.0 : 0.0;
  // -skew(p) block (cols 3..5). skew = [[0,-pz,py],[pz,0,-px],[-py,px,0]].
  double S[3][3] = {{0, -pz, py}, {pz, 0, -px}, {-py, px, 0}};
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      K[(size_t)(3 + b) * ld + (3 * i + a)] = -S[a][b];
}

// Build the reference-frame wrench rhs6r (column-major 6 x P, element (i,k) at
// i + 6*k): per particle, rhs6 = -[F_k; T_k], rotated to the reference frame by
// R_k^T (the linear and angular 3-blocks each). Matches the host lam0 loop's
// rhs6 / mat3Tvec stage.
__global__ void buildRhs6Ref(const double *__restrict__ F,
                             const double *__restrict__ T,
                             const double *__restrict__ R,
                             double *__restrict__ rhs6r, int P)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= P) return;
  const double *Rk = R + (size_t)9 * k;
  double f[3] = {-F[3 * k], -F[3 * k + 1], -F[3 * k + 2]};
  double t[3] = {-T[3 * k], -T[3 * k + 1], -T[3 * k + 2]};
  // R^T applied (mat3Tvec): out[a] = sum_b R[b*3+a] * v[b].
  rhs6r[6 * k + 0] = Rk[0] * f[0] + Rk[3] * f[1] + Rk[6] * f[2];
  rhs6r[6 * k + 1] = Rk[1] * f[0] + Rk[4] * f[1] + Rk[7] * f[2];
  rhs6r[6 * k + 2] = Rk[2] * f[0] + Rk[5] * f[1] + Rk[8] * f[2];
  rhs6r[6 * k + 3] = Rk[0] * t[0] + Rk[3] * t[1] + Rk[6] * t[2];
  rhs6r[6 * k + 4] = Rk[1] * t[0] + Rk[4] * t[1] + Rk[7] * t[2];
  rhs6r[6 * k + 5] = Rk[2] * t[0] + Rk[5] * t[1] + Rk[8] * t[2];
}

// Recovery rotate + pack: U_ref is column-major 6 x P (element (i,k) at i+6*k);
// rotate each particle's linear and angular 3-blocks to global (R, ref->global)
// and pack row-major into Uglob (P x 6, [vx vy vz wx wy wz]).
__global__ void recoverRotate(const double *__restrict__ Uref,
                              const double *__restrict__ R,
                              double *__restrict__ Uglob, int P)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= P) return;
  const double *Rk = R + (size_t)9 * k;
  double v[3] = {Uref[6 * k + 0], Uref[6 * k + 1], Uref[6 * k + 2]};
  double w[3] = {Uref[6 * k + 3], Uref[6 * k + 4], Uref[6 * k + 5]};
  // R applied (mat3vec): out[a] = sum_b R[a*3+b] * v[b].
  Uglob[6 * k + 0] = Rk[0] * v[0] + Rk[1] * v[1] + Rk[2] * v[2];
  Uglob[6 * k + 1] = Rk[3] * v[0] + Rk[4] * v[1] + Rk[5] * v[2];
  Uglob[6 * k + 2] = Rk[6] * v[0] + Rk[7] * v[1] + Rk[8] * v[2];
  Uglob[6 * k + 3] = Rk[0] * w[0] + Rk[1] * w[1] + Rk[2] * w[2];
  Uglob[6 * k + 4] = Rk[3] * w[0] + Rk[4] * w[1] + Rk[5] * w[2];
  Uglob[6 * k + 5] = Rk[6] * w[0] + Rk[7] * w[1] + Rk[8] * w[2];
}

// Symmetrize a tiny 6x6 column-major matrix by copying its lower triangle into
// the upper. cuSOLVER potri only fills one triangle; Ginv must be full for the
// downstream GEMMs. One thread, 6x6 -- one-time setup cost.
__global__ void symmetrize6(double *__restrict__ A)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < i; ++j)
        A[j + 6 * i] = A[i + 6 * j];  // upper(j,i) <- lower(i,j)
}

static inline int grid1d(int n, int block) { return (n + block - 1) / block; }
static inline long grid1d(long n, int block) { return (n + block - 1) / block; }

// ==========================================================================
// host helpers
// ==========================================================================

// Load whitespace-separated "x y z" fp64 points. Returns AoS (3*n), sets n.
static std::vector<double> loadPointsASCII(const std::string &path, int &n)
{
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<double> v;
  double x;
  while (in >> x) v.push_back(x);
  if (v.size() % 3 != 0) {
    std::fprintf(stderr, "%s: %zu doubles not divisible by 3\n", path.c_str(),
                 v.size());
    std::exit(1);
  }
  n = (int)(v.size() / 3);
  return v;
}

static inline void mat3vec(const double R[9], const double v[3], double out[3])
{
  out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
  out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
  out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}
static inline void mat3Tvec(const double R[9], const double v[3], double out[3])
{
  out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
  out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
  out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
}
static inline double det3(const double Q[9])
{
  return Q[0] * (Q[4] * Q[8] - Q[5] * Q[7]) -
         Q[1] * (Q[3] * Q[8] - Q[5] * Q[6]) +
         Q[2] * (Q[3] * Q[7] - Q[4] * Q[6]);
}

// Kabsch via cuSOLVER 3x3 SVD. ref/particle are AoS (3*npts), already centered.
// Returns R (row-major 9) such that particle_i ~= R * ref_i.
// (python: U,_,Vt = svd(ref^T@particle); Q=U@Vt; det fix; R = Q^T)
// Used by the rotation self-test; not on the per-step solve path (mfsSolve
// receives orientations R directly).
static void kabsch(cusolverDnHandle_t solver, const double *refp,
                   const double *part, int npts, double Rout[9])
{
  // H = ref^T @ particle (3x3), H[a][b] = sum_i ref[i][a]*part[i][b].
  double H[9] = {0};
  for (int i = 0; i < npts; ++i)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        H[a * 3 + b] += refp[3 * i + a] * part[3 * i + b];

  // upload H column-major (3x3): Hcm[b*3+a] = H[a][b]
  double Hcm[9];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) Hcm[b * 3 + a] = H[a * 3 + b];

  double *dA, *dU, *dS, *dVT, *dWork;
  int *dInfo;
  int lwork = 0;
  CUDA_CHECK(cudaMalloc(&dA, 9 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dU, 9 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dS, 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dVT, 9 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dA, Hcm, 9 * sizeof(double), cudaMemcpyHostToDevice));
  CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(solver, 3, 3, &lwork));
  CUDA_CHECK(cudaMalloc(&dWork, (size_t)lwork * sizeof(double)));
  CUSOLVER_CHECK(cusolverDnDgesvd(solver, 'A', 'A', 3, 3, dA, 3, dS, dU, 3, dVT,
                                  3, dWork, lwork, nullptr, dInfo));
  double Ucm[9], VTcm[9];
  CUDA_CHECK(cudaMemcpy(Ucm, dU, 9 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(VTcm, dVT, 9 * sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dU); cudaFree(dS); cudaFree(dVT);
  cudaFree(dInfo); cudaFree(dWork);

  // Ucm/VTcm are column-major: U[a][s]=Ucm[s*3+a], VT[s][b]=VTcm[b*3+s].
  // Q = U @ VT (math), Q[a][b] = sum_s U[a][s]*VT[s][b].
  double Q[9];
  auto computeQ = [&](double Qm[9]) {
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        double acc = 0;
        for (int s = 0; s < 3; ++s) acc += Ucm[s * 3 + a] * VTcm[b * 3 + s];
        Qm[a * 3 + b] = acc;
      }
  };
  computeQ(Q);
  if (det3(Q) < 0.0) {
    for (int a = 0; a < 3; ++a) Ucm[2 * 3 + a] = -Ucm[2 * 3 + a];  // flip col 2
    computeQ(Q);
  }
  // R = Q^T
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) Rout[a * 3 + b] = Q[b * 3 + a];
}

// ==========================================================================
// timing helpers
// ==========================================================================

static inline bool mfsEnvFlag(const char *name, bool defaultValue)
{
  const char *v = std::getenv(name);
  if (!v || !*v) return defaultValue;
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
      std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 ||
      std::strcmp(v, "OFF") == 0 || std::strcmp(v, "no") == 0 ||
      std::strcmp(v, "NO") == 0)
    return false;
  return true;
}

static inline bool mfsNsightCudaMemoryUsageActive()
{
  const char *cfg = std::getenv("QUADD_CUDA_CONFIG");
  if (!cfg || !*cfg) return false;
  std::ifstream in(cfg);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("CollectMemoryActivities") != std::string::npos &&
        line.find("true") != std::string::npos)
      return true;
  }
  return false;
}

static inline bool mfsSkipPetscTeardownForProfiler()
{
  const char *override = std::getenv("MFS_SKIP_PETSC_TEARDOWN");
  if (override && *override)
    return mfsEnvFlag("MFS_SKIP_PETSC_TEARDOWN", false);
  // Nsight Systems 2025.1.x with --cuda-memory-usage=true has been observed to
  // segfault in PETSc KSPDestroy/late PETSc teardown for VECSEQCUDA objects.
  return mfsNsightCudaMemoryUsageActive();
}

static inline double mfsElapsedMs(PetscLogDouble t0, PetscLogDouble t1)
{
  return 1000.0 * (double)(t1 - t0);
}

struct MFSCudaEventSet {
  std::vector<cudaEvent_t> ev;
  bool enabled = true;

  explicit MFSCudaEventSet(int n, bool enabledIn = true)
      : ev(enabledIn ? (size_t)n : 0), enabled(enabledIn)
  {
    if (!enabled) return;
    for (cudaEvent_t &e : ev) CUDA_CHECK(cudaEventCreate(&e));
  }

  ~MFSCudaEventSet()
  {
    for (cudaEvent_t e : ev) cudaEventDestroy(e);
  }

  void record(int i)
  {
    if (enabled) CUDA_CHECK(cudaEventRecord(ev[(size_t)i], 0));
  }

  void sync(int i)
  {
    if (enabled) CUDA_CHECK(cudaEventSynchronize(ev[(size_t)i]));
  }

  double elapsed(int i0, int i1) const
  {
    if (!enabled) return 0.0;
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev[(size_t)i0], ev[(size_t)i1]));
    return (double)ms;
  }
};

struct MFSTreeCallTiming {
  bool used = false;
  bool rebuildGeometry = false;
  double wallMs = 0.0;
  double forceConvertMs = 0.0;
  double applyOrReapplyMs = 0.0;
  double compToAosMs = 0.0;
  double cudaMs = 0.0;
  MFSTreecode<>::Stats stats{};
};

struct MFSMatvecTiming {
  double wallMs = 0.0;
  double diagCopyWallMs = 0.0;
  double denseMs = 0.0;
  double stokesMs = 0.0;
  double selfSubtractMs = 0.0;
  double cudaMs = 0.0;
  MFSTreeCallTiming tree{};
};

struct MFSSetupTiming {
  double totalWallMs = 0.0;
  double refUploadMs = 0.0;
  double buildKMs = 0.0;
  double gramInverseMs = 0.0;
  double projectionMs = 0.0;
  double selfBlockMs = 0.0;
  double svdMs = 0.0;
  double pinvMs = 0.0;
};

struct MFSStepTiming {
  double totalWallMs = 0.0;
  double uploadMs = 0.0;
  double cloudMs = 0.0;
  double lam0Ms = 0.0;
  double rhsMs = 0.0;
  double fillBMs = 0.0;
  double kspWallMs = 0.0;
  double recoverMs = 0.0;
  MFSTreeCallTiming rhsTree{};
};

static inline void mfsAccumMatvec(MFSMatvecTiming &dst,
                                  const MFSMatvecTiming &src)
{
  dst.wallMs += src.wallMs;
  dst.diagCopyWallMs += src.diagCopyWallMs;
  dst.denseMs += src.denseMs;
  dst.stokesMs += src.stokesMs;
  dst.selfSubtractMs += src.selfSubtractMs;
  dst.cudaMs += src.cudaMs;
  dst.tree.wallMs += src.tree.wallMs;
  dst.tree.forceConvertMs += src.tree.forceConvertMs;
  dst.tree.applyOrReapplyMs += src.tree.applyOrReapplyMs;
  dst.tree.compToAosMs += src.tree.compToAosMs;
  dst.tree.cudaMs += src.tree.cudaMs;
  dst.tree.used = dst.tree.used || src.tree.used;
  dst.tree.rebuildGeometry = dst.tree.rebuildGeometry || src.tree.rebuildGeometry;
  dst.tree.stats = src.tree.stats;  // keep the most recent counts/breakdown.
}

// ==========================================================================
// persistent solver context
// ==========================================================================
// All device memory and PETSc objects are owned here and allocated once in
// mfsSetup(); mfsSolve() overwrites only the per-step state in place.
struct MFSContext {
  // ---- problem dims / scalars (fixed for the context's lifetime) ----
  int P, M, N;          // particles, collocation pts, source pts
  int n;                // 3*M*P  (PETSc system size)
  int tM, tN;           // 3*M, 3*N
  double mu, pref;      // pref = 1/(8*pi*mu)

  // ---- handles (borrowed, not owned) ----
  cublasHandle_t cublas;
  cusolverDnHandle_t solver;

  // ---- SHAPE + RESOLUTION cacheable (computed once, reused every step) ----
  double *d_Xref;  // tM        reference collocation cloud (centered)
  double *d_Yref;  // tN        reference source cloud (centered)
  double *d_KN;    // tN x 6    column-major
  double *d_Ginv;  // 6  x 6    gram^{-1} (symmetric, full)
  double *d_ImL;   // tN x tN   I - L_proj
  double *d_pinv;  // tN x tM   S_L^+
  double *d_What;  // tN x tM   (I - L_proj) * pinv
  double *d_Sself; // tM x tN   reference-frame self Stokeslet block (RETAINED for
                   //           the treecode matvec's analytic self subtraction)

  // ---- treecode acceleration of the Stokeslet sums (optional) ----
  bool useTreecode = false;
  MFSTreecode<> *tc = nullptr;          // built per timestep in mfsSolve
  MFSTreecode<>::Config tcCfg;
  double *d_tcVelCM = nullptr;          // 3*(P*M), component-major treecode output
  double *d_Uself = nullptr;            // P*tM  matvec self block (global frame)
  double *d_selfRefAll = nullptr;       // tM x P  S_self_ref @ hat_ref_all

  // ---- PER-TIMESTEP state (overwritten in place each mfsSolve) ----
  double *d_centers;  // P x 3   row-major AoS
  double *d_R;        // P x 9   row-major rotations
  double *d_F;        // P x 3   applied force
  double *d_T;        // P x 3   applied torque
  double *d_Xglob;    // P*tM    generated global collocation cloud
  double *d_Yglob;    // P*tN    generated global source cloud
  double *d_lam0;     // P*tN    completion sources (global)
  double *d_u0;       // P*tM    RHS
  double *d_hatGlob;  // P*tN    matvec scratch (off-diag Stokeslet input)

  // ---- batched scratch (reused by matvec / lam0 / recovery) ----
  double *d_gammaRefAll;  // P*tM  per-particle R^T-rotated gamma  (== tM x P)
  double *d_hatRefAll;    // P*tN  W_hat @ gammaRefAll             (== tN x P)
  double *d_lamRefAll;    // P*tN  recovery: pinv @ gammaRefAll    (== tN x P)
  double *d_rhs6r;        // 6 x P column-major ref-frame wrench
  double *d_vw;           // 6 x P column-major
  double *d_Uref;         // 6 x P column-major (-K_N^T @ lamRefAll)
  double *d_Uglob;        // P x 6 row-major recovery output

  // ---- PETSc objects (created once, reused across solves) ----
  Mat A;     // MatShell, context = this
  KSP ksp;
  PC pc;
  Vec b;     // VECSEQCUDA, zero-copy; filled from d_u0 each step
  Vec xsol;  // VECSEQCUDA

  // ---- instrumentation ----
  bool timingEnabled = true;
  int stepIndex = 0;
  PetscLogDouble stepWallStart = 0.0;
  PetscLogDouble kspWallStart = 0.0;
  MFSSetupTiming setupTiming{};
  MFSStepTiming stepTiming{};
  MFSMatvecTiming lastMatvecTiming{};
  MFSMatvecTiming matvecTotalTiming{};
  int matvecCount = 0;
  int monitorLastMatvecCount = 0;
  double monitorLastMatvecWallMs = 0.0;
};

static inline const vec3d *asVec3dCloud(const double *p)
{
  return reinterpret_cast<const vec3d *>(p);
}

static PetscErrorCode MFSKSPMonitor(KSP, PetscInt it, PetscReal rnorm,
                                    void *ctx)
{
  MFSContext *m = static_cast<MFSContext *>(ctx);
  PetscFunctionBeginUser;
  if (!m || !m->timingEnabled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscLogDouble now = 0.0;
  PetscCall(PetscTime(&now));
  const double solveMs = mfsElapsedMs(m->kspWallStart, now);
  const MFSMatvecTiming &mv = m->lastMatvecTiming;
  //const int newMv = m->matvecCount - m->monitorLastMatvecCount;
  const double newMvWall = m->matvecTotalTiming.wallMs -
                           m->monitorLastMatvecWallMs;

  if (m->useTreecode && mv.tree.used) {
    PetscCall(PetscPrintf(
        PETSC_COMM_SELF,
        "[MFS GMRES] it=%" PetscInt_FMT
        " rnorm=%.6e total solve=%.3f ms matvec(s)=%.3f ms \n"
        "cuda=%.3f dense=%.3f stokes=%.3f self=%.3f \n"
        "tc(reapply=%.3f wall %.3f cuda, up=%.3f m2p=%.3f p2p=%.3f pairs=%lld p2p_int=%lld)\n\n",
        it, (double)rnorm, solveMs,newMvWall,
        mv.cudaMs, mv.denseMs, mv.stokesMs, mv.selfSubtractMs, 
        mv.tree.wallMs, mv.tree.cudaMs,
        mv.tree.stats.upwardMs, (double)mv.tree.stats.travMs,
        (double)mv.tree.stats.p2pMs, mv.tree.stats.nPairs,
        mv.tree.stats.totalP2P));
  } else {
    PetscCall(PetscPrintf(
        PETSC_COMM_SELF,
        "[MFS GMRES] it=%" PetscInt_FMT
        " rnorm=%.6e total solve=%.3f ms matvec(s)=%.3f ms \n"
        "cuda=%.3f dense=%.3f stokes=%.3f self=%.3f \n\n",
        it, (double)rnorm, solveMs, newMvWall,
        mv.cudaMs, mv.denseMs, mv.stokesMs, mv.selfSubtractMs));
  }

  m->monitorLastMatvecCount = m->matvecCount;
  m->monitorLastMatvecWallMs = m->matvecTotalTiming.wallMs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void applyTreecodeStokeslet(MFSContext *m, const double *d_forceAoS,
                                   double *d_outAoS, bool addInto,
                                   bool rebuildGeometry,
                                   bool skipSameParticle,
                                   MFSTreeCallTiming *timing = nullptr)
{
  const int nS = m->P * m->N;
  const int nT = m->P * m->M;

  PetscLogDouble wall0 = 0.0, wall1 = 0.0;
  if (timing) {
    *timing = MFSTreeCallTiming{};
    timing->used = true;
    timing->rebuildGeometry = rebuildGeometry;
    PetscCallAbort(PETSC_COMM_SELF, PetscTime(&wall0));
  }

  MFSCudaEventSet ev(4, timing != nullptr);
  ev.record(0);
  ev.record(1);
  m->tc->setSkipSameGroup(skipSameParticle);

  if (rebuildGeometry) {
    m->tc->apply(asVec3dCloud(m->d_Yglob), nS, asVec3dCloud(m->d_Xglob), nT,
                 asVec3dCloud(d_forceAoS), m->d_tcVelCM,
                 /*will_reuse_tree=*/true);
  } else {
    m->tc->reapply(asVec3dCloud(d_forceAoS), m->d_tcVelCM);
  }
  ev.record(2);

  treecodeCompMajorToAoS<<<grid1d(nT, 128), 128>>>(m->d_tcVelCM, d_outAoS, nT,
                                                   addInto ? 1 : 0);
  CUDA_CHECK(cudaGetLastError());
  ev.record(3);
  ev.sync(3);

  if (timing) {
    PetscCallAbort(PETSC_COMM_SELF, PetscTime(&wall1));
    timing->wallMs = mfsElapsedMs(wall0, wall1);
    timing->forceConvertMs = ev.elapsed(0, 1);
    timing->applyOrReapplyMs = ev.elapsed(1, 2);
    timing->compToAosMs = ev.elapsed(2, 3);
    timing->cudaMs = ev.elapsed(0, 3);
    timing->stats = m->tc->stats();
  }
}

// Batched dense apply for the matvec: hat_lam_global = R_k * W_hat * R_k^T *
// gamma[k], for all k. The per-particle W_hat apply is a SINGLE GEMM of width P;
// the two rotations are single batched-kernel launches.
static void denseApplyBatched(MFSContext *m, const double *d_gamma,
                              double *d_hatGlob)
{
  const double one = 1.0, zero = 0.0;
  const int P = m->P, tM = m->tM, tN = m->tN;
  // gamma_ref_all (tM x P) = blockdiag(R_k^T) * gamma
  applyRotAll<<<grid1d((long)m->M * P, 128), 128>>>(m->d_R, d_gamma,
                                                    m->d_gammaRefAll, m->M, P, 1);
  // hat_ref_all (tN x P) = W_hat (tN x tM) * gamma_ref_all (tM x P)  [one GEMM]
  CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_N, CUBLAS_OP_N, tN, P, tM, &one,
                           m->d_What, tN, m->d_gammaRefAll, tM, &zero,
                           m->d_hatRefAll, tN));
  // hat_global = blockdiag(R_k) * hat_ref_all
  applyRotAll<<<grid1d((long)m->N * P, 128), 128>>>(m->d_R, m->d_hatRefAll,
                                                    d_hatGlob, m->N, P, 0);
}

// PETSc matvec callback: y = A x = x + offdiag Stokeslet(hat_lam(x)).
static PetscErrorCode MatMult_MFS(Mat A, Vec x, Vec y)
{
  MFSContext *m;
  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &m));

  MFSMatvecTiming mt{};
  PetscLogDouble wall0 = 0.0, wall1 = 0.0, copy0 = 0.0, copy1 = 0.0;
  if (m->timingEnabled) PetscCall(PetscTime(&wall0));

  if (m->timingEnabled) PetscCall(PetscTime(&copy0));
  PetscCall(VecCopy(x, y));  // diagonal block = I (eq 54)
  if (m->timingEnabled) {
    PetscCall(PetscTime(&copy1));
    mt.diagCopyWallMs = mfsElapsedMs(copy0, copy1);
  }

  const PetscScalar *dx;
  PetscScalar *dy;
  PetscCall(VecCUDAGetArrayRead(x, &dx));
  PetscCall(VecCUDAGetArray(y, &dy));

  MFSCudaEventSet ev(4, m->timingEnabled);
  ev.record(0);

  // hat_lam: d_hatGlob (global) and, as a side effect, d_hatRefAll (tN x P,
  // reference frame) which the self-block subtraction below reuses.
  denseApplyBatched(m, dx, m->d_hatGlob);
  ev.record(1);

  const int nT = m->P * m->M, nS = m->P * m->N;
  if (m->useTreecode) {
    // y = x + S_off * hat. The treecode is told the source/target particle
    // grouping, so same-particle sources are excluded during traversal/P2P
    // instead of computing an approximate full field and subtracting self later.
    applyTreecodeStokeslet(m, m->d_hatGlob, dy, /*addInto=*/true,
                           /*rebuildGeometry=*/
                           std::getenv("MFS_TC_FRESH_MATVEC") != nullptr,
                           /*skipSameParticle=*/true,
                           m->timingEnabled ? &mt.tree : nullptr);
    ev.record(2);
    ev.record(3);
  } else {
    pairwiseStokeslet<<<grid1d(nT, 128), 128>>>(m->d_Xglob, m->d_Yglob,
                                                m->d_hatGlob, dy, nT, nS, m->M,
                                                m->N, m->pref, /*skipSelf=*/1);
    ev.record(2);
    ev.record(3);
  }
  CUDA_CHECK(cudaGetLastError());
  if (m->timingEnabled)
    ev.sync(3);
  else
    CUDA_CHECK(cudaDeviceSynchronize());

  if (m->timingEnabled) {
    mt.denseMs = ev.elapsed(0, 1);
    mt.stokesMs = ev.elapsed(1, 2);
    mt.selfSubtractMs = m->useTreecode ? ev.elapsed(2, 3) : 0.0;
    mt.cudaMs = ev.elapsed(0, 3);
  }

  // Optional first-matvec A/B against brute (MFS_TC_CHECK): isolates the
  // skip-same-group reapply path, which the RHS check does not exercise.
  if (m->useTreecode && m->matvecCount == 0 && std::getenv("MFS_TC_CHECK")) {
    const size_t nsz = (size_t)m->n;
    double *d_b;
    CUDA_CHECK(cudaMalloc(&d_b, nsz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, dx, nsz * sizeof(double), cudaMemcpyDeviceToDevice));
    pairwiseStokeslet<<<grid1d(nT, 128), 128>>>(m->d_Xglob, m->d_Yglob,
                                                m->d_hatGlob, d_b, nT, nS, m->M,
                                                m->N, m->pref, /*skipSelf=*/1);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> tree(nsz), brute(nsz), xin(nsz);
    CUDA_CHECK(cudaMemcpy(tree.data(), dy, nsz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(brute.data(), d_b, nsz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xin.data(), dx, nsz * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_b);
    // Two normalizations: full y = x + S_off (what GMRES sees), and the pure
    // off-diagonal Stokeslet VELOCITY (the project-standard treecode relL2 --
    // same numerator, but the identity part no longer pads the denominator).
    double num = 0.0, den = 0.0, denOff = 0.0;
    for (size_t i = 0; i < nsz; ++i) {
      const double dd = tree[i] - brute[i];
      const double off = brute[i] - xin[i];
      num += dd * dd;
      den += brute[i] * brute[i];
      denOff += off * off;
    }
    std::fprintf(stderr,
                 "[MFS_TC_CHECK] matvec1 treecode-vs-brute rel-L2 = %.3e "
                 "(velocity-only rel-L2 = %.3e)\n",
                 std::sqrt(num / den), std::sqrt(num / denOff));
  }

  PetscCall(VecCUDARestoreArrayRead(x, &dx));
  PetscCall(VecCUDARestoreArray(y, &dy));

  if (m->timingEnabled) {
    PetscCall(PetscTime(&wall1));
    mt.wallMs = mfsElapsedMs(wall0, wall1);
    m->lastMatvecTiming = mt;
    mfsAccumMatvec(m->matvecTotalTiming, mt);
    ++m->matvecCount;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================
// setup: build + cache reference-frame state, allocate per-step buffers,
//        create PETSc objects -- ONCE.
// h_Xref0 / h_Yref0 are particle-0's clouds centered at the origin (AoS).
// ==========================================================================
static MFSContext *mfsSetup(cublasHandle_t cublas, cusolverDnHandle_t solver,
                            int P, int M, int N, const double *h_Xref0,
                            const double *h_Yref0, double mu)
{
  PetscLogDouble setupWall0 = 0.0, setupWall1 = 0.0;
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&setupWall0));

  MFSContext *m = new MFSContext();
  m->P = P; m->M = M; m->N = N; m->n = 3 * M * P;
  m->tM = 3 * M; m->tN = 3 * N;
  m->mu = mu; m->pref = 1.0 / (8.0 * M_PI * mu);
  m->cublas = cublas; m->solver = solver;
  m->timingEnabled = mfsEnvFlag("MFS_TIMING", true);
  const int tM = m->tM, tN = m->tN;
  const double one = 1.0, zero = 0.0;

  MFSCudaEventSet setupEv(8, m->timingEnabled);
  setupEv.record(0);

  // ---- reference clouds ----
  CUDA_CHECK(cudaMalloc(&m->d_Xref, (size_t)tM * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_Yref, (size_t)tN * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(m->d_Xref, h_Xref0, (size_t)tM * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m->d_Yref, h_Yref0, (size_t)tN * sizeof(double), cudaMemcpyHostToDevice));
  setupEv.record(1);

  // ---- K_N (3N x 6), K_M (3M x 6) on device ----
  double *d_KM;
  CUDA_CHECK(cudaMalloc(&m->d_KN, (size_t)tN * 6 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_KM, (size_t)tM * 6 * sizeof(double)));
  buildKKernel<<<grid1d(N, 128), 128>>>(m->d_Yref, m->d_KN, N);
  buildKKernel<<<grid1d(M, 128), 128>>>(m->d_Xref, d_KM, M);
  setupEv.record(2);

  // ---- gram = K_N^T K_N (6x6); Ginv via Cholesky (SPD) ----
  double *d_gram;
  CUDA_CHECK(cudaMalloc(&d_gram, 36 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_Ginv, 36 * sizeof(double)));
  CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 6, 6, tN, &one,
                           m->d_KN, tN, m->d_KN, tN, &zero, d_gram, 6));
  CUDA_CHECK(cudaMemcpy(m->d_Ginv, d_gram, 36 * sizeof(double), cudaMemcpyDeviceToDevice));
  int lwP = 0, lwI = 0;
  CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, 6, m->d_Ginv, 6, &lwP));
  CUSOLVER_CHECK(cusolverDnDpotri_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, 6, m->d_Ginv, 6, &lwI));
  int lwChol = lwP > lwI ? lwP : lwI;
  double *d_cholWork;
  int *d_info;
  CUDA_CHECK(cudaMalloc(&d_cholWork, (size_t)lwChol * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, 6, m->d_Ginv, 6, d_cholWork, lwChol, d_info));
  int info = 0;
  CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info != 0) { std::fprintf(stderr, "gram not SPD (potrf info=%d)\n", info); std::exit(1); }
  CUSOLVER_CHECK(cusolverDnDpotri(solver, CUBLAS_FILL_MODE_LOWER, 6, m->d_Ginv, 6, d_cholWork, lwChol, d_info));
  CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info != 0) { std::fprintf(stderr, "gram inverse failed (potri info=%d)\n", info); std::exit(1); }
  symmetrize6<<<1, 1>>>(m->d_Ginv);
  setupEv.record(3);

  // ---- Z = Ginv * K_N^T (6 x 3N); L_proj = K_N * Z (3N x 3N); ImL = I-L ----
  double *d_Z, *d_Lproj;
  CUDA_CHECK(cudaMalloc(&d_Z, (size_t)6 * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_Lproj, (size_t)tN * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_ImL, (size_t)tN * tN * sizeof(double)));
  CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, 6, tN, 6, &one,
                           m->d_Ginv, 6, m->d_KN, tN, &zero, d_Z, 6));
  CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, tN, tN, 6, &one,
                           m->d_KN, tN, d_Z, 6, &zero, d_Lproj, tN));
  makeImL<<<grid1d(tN * tN, 256), 256>>>(m->d_ImL, d_Lproj, tN);
  setupEv.record(4);

  // ---- S_self (3M x 3N); L_r = K_M K_N^T; S_L = S_self*(I-L) + L_r ----
  // S_self is RETAINED in the context (m->d_Sself): the treecode matvec reuses
  // it to subtract the exact same-particle (diagonal) block analytically.
  double *d_Lr, *d_SL;
  CUDA_CHECK(cudaMalloc(&m->d_Sself, (size_t)tM * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_Lr, (size_t)tM * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_SL, (size_t)tM * tN * sizeof(double)));
  buildSelfBlock<<<grid1d(M * N, 128), 128>>>(m->d_Xref, m->d_Yref, m->d_Sself, M, N, m->pref);
  CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, tM, tN, 6, &one,
                           d_KM, tM, m->d_KN, tN, &zero, d_Lr, tM));
  CUDA_CHECK(cudaMemcpy(d_SL, d_Lr, (size_t)tM * tN * sizeof(double), cudaMemcpyDeviceToDevice));
  CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, tM, tN, tN, &one,
                           m->d_Sself, tM, m->d_ImL, tN, &one, d_SL, tM));
  setupEv.record(5);

  // ---- SVD S_L = U S V^T (econ); pinv = V diag(1/s) U^T; W_hat = (I-L) pinv ----
  double *d_U, *d_S, *d_VT, *d_svdWork;
  int *d_svdInfo, lwSvd = 0;
  CUDA_CHECK(cudaMalloc(&d_U, (size_t)tM * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_S, (size_t)tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_VT, (size_t)tN * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_svdInfo, sizeof(int)));
  CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(solver, tM, tN, &lwSvd));
  CUDA_CHECK(cudaMalloc(&d_svdWork, (size_t)lwSvd * sizeof(double)));
  CUSOLVER_CHECK(cusolverDnDgesvd(solver, 'S', 'S', tM, tN, d_SL, tM, d_S, d_U,
                                  tM, d_VT, tN, d_svdWork, lwSvd, nullptr, d_svdInfo));
  CUDA_CHECK(cudaMemcpy(&info, d_svdInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (info != 0) std::fprintf(stderr, "warning: gesvd info=%d\n", info);
  setupEv.record(6);

  double *d_invS, *d_Us;
  CUDA_CHECK(cudaMalloc(&d_invS, (size_t)tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_Us, (size_t)tM * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_pinv, (size_t)tN * tM * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_What, (size_t)tN * tM * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_invS, d_S, (size_t)tN * sizeof(double), cudaMemcpyDeviceToDevice));
  reciprocalKernel<<<grid1d(tN, 128), 128>>>(d_invS, tN);
  CUBLAS_CHECK(cublasDdgmm(cublas, CUBLAS_SIDE_RIGHT, tM, tN, d_U, tM, d_invS, 1, d_Us, tM));
  CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_T, tN, tM, tN, &one,
                           d_VT, tN, d_Us, tM, &zero, m->d_pinv, tN));
  CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, tN, tM, tN, &one,
                           m->d_ImL, tN, m->d_pinv, tN, &zero, m->d_What, tN));
  setupEv.record(7);
  setupEv.sync(7);
  m->setupTiming.refUploadMs = setupEv.elapsed(0, 1);
  m->setupTiming.buildKMs = setupEv.elapsed(1, 2);
  m->setupTiming.gramInverseMs = setupEv.elapsed(2, 3);
  m->setupTiming.projectionMs = setupEv.elapsed(3, 4);
  m->setupTiming.selfBlockMs = setupEv.elapsed(4, 5);
  m->setupTiming.svdMs = setupEv.elapsed(5, 6);
  m->setupTiming.pinvMs = setupEv.elapsed(6, 7);

  // ---- free setup scratch (everything not in the cached set; d_Sself is KEPT) ----
  cudaFree(d_KM); cudaFree(d_gram); cudaFree(d_cholWork); cudaFree(d_info);
  cudaFree(d_Z); cudaFree(d_Lproj); cudaFree(d_Lr);
  cudaFree(d_SL); cudaFree(d_U); cudaFree(d_S); cudaFree(d_VT);
  cudaFree(d_svdWork); cudaFree(d_svdInfo); cudaFree(d_invS); cudaFree(d_Us);

  // ---- allocate per-step + batched-scratch buffers ----
  CUDA_CHECK(cudaMalloc(&m->d_centers, (size_t)P * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_R, (size_t)P * 9 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_F, (size_t)P * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_T, (size_t)P * 3 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_Xglob, (size_t)P * tM * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_Yglob, (size_t)P * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_lam0, (size_t)P * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_u0, (size_t)P * tM * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_hatGlob, (size_t)P * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_gammaRefAll, (size_t)P * tM * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_hatRefAll, (size_t)P * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_lamRefAll, (size_t)P * tN * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_rhs6r, (size_t)6 * P * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_vw, (size_t)6 * P * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_Uref, (size_t)6 * P * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_Uglob, (size_t)P * 6 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_tcVelCM, (size_t)P * tM * sizeof(double)));
  // treecode matvec self-subtraction scratch (small; allocated unconditionally).
  CUDA_CHECK(cudaMalloc(&m->d_Uself, (size_t)P * tM * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_selfRefAll, (size_t)tM * P * sizeof(double)));

  // ---- create PETSc objects ONCE (reused every solve) ----
  PetscCallAbort(PETSC_COMM_SELF, MatCreateShell(PETSC_COMM_SELF, m->n, m->n, m->n, m->n, m, &m->A));
  PetscCallAbort(PETSC_COMM_SELF, MatShellSetOperation(m->A, MATOP_MULT, (void (*)(void))MatMult_MFS));
  PetscCallAbort(PETSC_COMM_SELF, VecCreateSeqCUDA(PETSC_COMM_SELF, m->n, &m->b));
  PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(m->b, &m->xsol));
  PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &m->ksp));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(m->ksp, m->A, m->A));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetType(m->ksp, KSPGMRES));
  PetscCallAbort(PETSC_COMM_SELF, KSPGMRESSetRestart(m->ksp, 30));
  PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(m->ksp, &m->pc));
  PetscCallAbort(PETSC_COMM_SELF, PCSetType(m->pc, PCNONE));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(m->ksp, 1e-8, 0.0, PETSC_DEFAULT, 500));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetFromOptions(m->ksp));
  if (m->timingEnabled)
    PetscCallAbort(PETSC_COMM_SELF,
                   KSPMonitorSet(m->ksp, MFSKSPMonitor, m, nullptr));

  CUDA_CHECK(cudaDeviceSynchronize());
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&setupWall1));
  m->setupTiming.totalWallMs = mfsElapsedMs(setupWall0, setupWall1);
  if (m->timingEnabled) {
    PetscCallAbort(PETSC_COMM_SELF,
                   PetscPrintf(PETSC_COMM_SELF,
                               "[MFS setup] P=%d M=%d N=%d n=%d total_wall=%.3f ms "
                               "gpu(ref_upload=%.3f build_K=%.3f gram_inv=%.3f "
                               "projection=%.3f self_block=%.3f svd=%.3f pinv=%.3f)\n\n",
                               P, M, N, m->n, m->setupTiming.totalWallMs,
                               m->setupTiming.refUploadMs,
                               m->setupTiming.buildKMs,
                               m->setupTiming.gramInverseMs,
                               m->setupTiming.projectionMs,
                               m->setupTiming.selfBlockMs,
                               m->setupTiming.svdMs,
                               m->setupTiming.pinvMs));
  }
  return m;
}

// ==========================================================================
// solve one timestep: state = centers (P*3) + row-major R (P*9) + wrench F,T.
// Uout returned as P*6 row-major [vx vy vz wx wy wz].
// ==========================================================================
static void mfsSolve(MFSContext *m, const double *h_centers, const double *h_R,
                     const double *h_F, const double *h_T,
                     std::vector<double> &Uout)
{
  const double one = 1.0, zero = 0.0, negone = -1.0;
  const int P = m->P, M = m->M, N = m->N, tM = m->tM, tN = m->tN;

  ++m->stepIndex;
  m->stepTiming = MFSStepTiming{};
  m->lastMatvecTiming = MFSMatvecTiming{};
  m->matvecTotalTiming = MFSMatvecTiming{};
  m->matvecCount = 0;
  m->monitorLastMatvecCount = 0;
  m->monitorLastMatvecWallMs = 0.0;
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&m->stepWallStart));

  MFSCudaEventSet preSolveEv(5, m->timingEnabled);
  preSolveEv.record(0);

  // ---- upload per-step state (the only host->device traffic per step) ----
  CUDA_CHECK(cudaMemcpy(m->d_centers, h_centers, (size_t)P * 3 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m->d_R, h_R, (size_t)P * 9 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m->d_F, h_F, (size_t)P * 3 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m->d_T, h_T, (size_t)P * 3 * sizeof(double), cudaMemcpyHostToDevice));
  preSolveEv.record(1);

  // ---- generate global clouds on the GPU from the cached reference cloud ----
  tileRotateTranslate<<<grid1d((long)M * P, 128), 128>>>(m->d_Xref, m->d_R, m->d_centers, m->d_Xglob, M, P);
  tileRotateTranslate<<<grid1d((long)N * P, 128), 128>>>(m->d_Yref, m->d_R, m->d_centers, m->d_Yglob, N, P);
  preSolveEv.record(2);

  // ---- lam0_global (eqs 43-44), batched ----
  // rhs6r (6 x P) = R_k^T * (-[F_k; T_k]); vw = Ginv*rhs6r; lam0_ref = K_N*vw;
  // then rotate each particle's lam0 to global.
  buildRhs6Ref<<<grid1d(P, 128), 128>>>(m->d_F, m->d_T, m->d_R, m->d_rhs6r, P);
  CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_N, CUBLAS_OP_N, 6, P, 6, &one,
                           m->d_Ginv, 6, m->d_rhs6r, 6, &zero, m->d_vw, 6));
  CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_N, CUBLAS_OP_N, tN, P, 6, &one,
                           m->d_KN, tN, m->d_vw, 6, &zero, m->d_lam0, tN));
  applyRotAll<<<grid1d((long)N * P, 128), 128>>>(m->d_R, m->d_lam0, m->d_lam0, N, P, 0);
  preSolveEv.record(3);

  // ---- u0 = Stokeslet over ALL particles incl. self (eq 50, skipSelf=0) ----
  CUDA_CHECK(cudaMemset(m->d_u0, 0, (size_t)P * tM * sizeof(double)));
  if (m->useTreecode) {
    // Build source/target geometry once for this timestep, compute the RHS,
    // and keep the reusable tree + P2P list for PETSc MatMult reapply calls.
    applyTreecodeStokeslet(m, m->d_lam0, m->d_u0, /*addInto=*/false,
                           /*rebuildGeometry=*/true,
                           /*skipSameParticle=*/false,
                           m->timingEnabled ? &m->stepTiming.rhsTree : nullptr);
    // full field == skipSelf=0
    // Optional A/B against brute (set MFS_TC_CHECK): isolates the distinct-
    // target / per-source-force / permutation path before the matvec.
    if (std::getenv("MFS_TC_CHECK")) {
      const size_t n = (size_t)P * tM;
      std::vector<double> tree(n), brute(n);
      CUDA_CHECK(cudaMemcpy(tree.data(), m->d_u0, n * sizeof(double), cudaMemcpyDeviceToHost));
      double *d_b;
      CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
      CUDA_CHECK(cudaMemset(d_b, 0, n * sizeof(double)));
      pairwiseStokeslet<<<grid1d(P * M, 128), 128>>>(m->d_Xglob, m->d_Yglob, m->d_lam0,
                                                     d_b, P * M, P * N, M, N, m->pref, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(brute.data(), d_b, n * sizeof(double), cudaMemcpyDeviceToHost));
      cudaFree(d_b);
      double num = 0, den = 0;
      for (size_t i = 0; i < n; ++i) { double d = tree[i] - brute[i]; num += d * d; den += brute[i] * brute[i]; }
      const auto &st = m->tc->stats();
      std::fprintf(stderr,
                   "[MFS_TC_CHECK] RHS treecode-vs-brute rel-L2 = %.3e  "
                   "(near_pairs=%lld direct_p2p=%lld)\n",
                   std::sqrt(num / den), st.nPairs, st.totalP2P);
    }
  } else {
    pairwiseStokeslet<<<grid1d(P * M, 128), 128>>>(m->d_Xglob, m->d_Yglob, m->d_lam0,
                                                   m->d_u0, P * M, P * N, M, N,
                                                   m->pref, /*skipSelf=*/0);
  }
  CUDA_CHECK(cudaGetLastError());
  preSolveEv.record(4);
  preSolveEv.sync(4);
  m->stepTiming.uploadMs = preSolveEv.elapsed(0, 1);
  m->stepTiming.cloudMs = preSolveEv.elapsed(1, 2);
  m->stepTiming.lam0Ms = preSolveEv.elapsed(2, 3);
  m->stepTiming.rhsMs = preSolveEv.elapsed(3, 4);

  // ---- fill b from d_u0 (device-to-device; reuse the same Vec) ----
  MFSCudaEventSet fillBEv(2, m->timingEnabled);
  fillBEv.record(0);
  {
    PetscScalar *db;
    PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArray(m->b, &db));
    CUDA_CHECK(cudaMemcpy(db, m->d_u0, (size_t)m->n * sizeof(double), cudaMemcpyDeviceToDevice));
    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArray(m->b, &db));
  }
  fillBEv.record(1);
  fillBEv.sync(1);
  m->stepTiming.fillBMs = fillBEv.elapsed(0, 1);

  // ---- solve (reuse ksp) ----
  PetscLogDouble kspWall1 = 0.0;
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&m->kspWallStart));
  PetscCallAbort(PETSC_COMM_SELF, KSPSolve(m->ksp, m->b, m->xsol));
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&kspWall1));
  m->stepTiming.kspWallMs = mfsElapsedMs(m->kspWallStart, kspWall1);

  // ---- recover U[k] = -K_N^T pinv gamma_ref, rotated to global (eq 51) ----
  // batched: gamma_ref_all -> lam_ref_all = pinv*gamma_ref_all ->
  //          U_ref_all = -K_N^T*lam_ref_all -> rotate -> single D2H.
  MFSCudaEventSet recoverEv(2, m->timingEnabled);
  recoverEv.record(0);
  {
    const PetscScalar *dg;
    PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayRead(m->xsol, &dg));
    applyRotAll<<<grid1d((long)M * P, 128), 128>>>(m->d_R, dg, m->d_gammaRefAll, M, P, 1);
    CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_N, CUBLAS_OP_N, tN, P, tM, &one,
                             m->d_pinv, tN, m->d_gammaRefAll, tM, &zero, m->d_lamRefAll, tN));
    CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_T, CUBLAS_OP_N, 6, P, tN, &negone,
                             m->d_KN, tN, m->d_lamRefAll, tN, &zero, m->d_Uref, 6));
    recoverRotate<<<grid1d(P, 128), 128>>>(m->d_Uref, m->d_R, m->d_Uglob, P);
    recoverEv.record(1);
    if (m->timingEnabled) {
      recoverEv.sync(1);
      m->stepTiming.recoverMs = recoverEv.elapsed(0, 1);
    } else {
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    Uout.assign((size_t)P * 6, 0.0);
    CUDA_CHECK(cudaMemcpy(Uout.data(), m->d_Uglob, (size_t)P * 6 * sizeof(double), cudaMemcpyDeviceToHost));
    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(m->xsol, &dg));
  }

  PetscLogDouble stepWall1 = 0.0;
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&stepWall1));
  m->stepTiming.totalWallMs = mfsElapsedMs(m->stepWallStart, stepWall1);

  PetscInt its = 0;
  PetscReal rnorm = 0.0;
  KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
  const char *reasonString = nullptr;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(m->ksp, &its));
  PetscCallAbort(PETSC_COMM_SELF, KSPGetResidualNorm(m->ksp, &rnorm));
  PetscCallAbort(PETSC_COMM_SELF, KSPGetConvergedReason(m->ksp, &reason));
  PetscCallAbort(PETSC_COMM_SELF,
                 KSPGetConvergedReasonString(m->ksp, &reasonString));

  if (m->timingEnabled) {
    PetscCallAbort(PETSC_COMM_SELF,
                   PetscPrintf(PETSC_COMM_SELF,
                               "[MFS timestep] step=%d total_wall=%.3f ms "
                               "cuda(upload=%.3f clouds=%.3f lam0=%.3f rhs=%.3f "
                               "fill_b=%.3f recover=%.3f) ksp_wall=%.3f ms "
                               "its=%" PetscInt_FMT " final_rnorm=%.6e reason=%d(%s)\n",
                               m->stepIndex, m->stepTiming.totalWallMs,
                               m->stepTiming.uploadMs, m->stepTiming.cloudMs,
                               m->stepTiming.lam0Ms, m->stepTiming.rhsMs,
                               m->stepTiming.fillBMs, m->stepTiming.recoverMs,
                               m->stepTiming.kspWallMs, its, (double)rnorm,
                               (int)reason,
                               reasonString ? reasonString : "unknown"));
    if (m->useTreecode && m->stepTiming.rhsTree.used) {
      const auto &rt = m->stepTiming.rhsTree;
      const auto &st = rt.stats;
      PetscCallAbort(PETSC_COMM_SELF,
                     PetscPrintf(PETSC_COMM_SELF,
                                 "[MFS treecode build+rhs] step=%d wall=%.3f ms cuda=%.3f ms "
                                 "force=%.3f apply=%.3f cm2aos=%.3f "
                                 "src_bucket=%.3f tgt_bucket=%.3f bvh=%.3f "
                                 "prep_force=%.3f upward=%.3f m2p=%.3f p2p=%.3f "
                                 "src_buckets=%u tgt_buckets=%u nodes=%u pairs=%lld p2p_int=%lld\n",
                                 m->stepIndex, rt.wallMs, rt.cudaMs,
                                 rt.forceConvertMs, rt.applyOrReapplyMs,
                                 rt.compToAosMs, st.bucketMs, st.targetBucketMs,
                                 st.buildBvhMs, st.prepForcesMs, st.upwardMs,
                                 (double)st.travMs, (double)st.p2pMs,
                                 st.numSourceBuckets, st.numTargetBuckets,
                                 st.numNodes, st.nPairs, st.totalP2P));
    }
    if (m->matvecCount > 0) {
      const double inv = 1.0 / (double)m->matvecCount;
      const MFSMatvecTiming &tot = m->matvecTotalTiming;
      PetscCallAbort(PETSC_COMM_SELF,
                     PetscPrintf(PETSC_COMM_SELF,
                                 "[MFS matvec total] step=%d count=%d wall=%.3f ms avg=%.3f ms "
                                 "cuda=%.3f ms avg_cuda=%.3f ms diag_copy=%.3f "
                                 "dense=%.3f stokes=%.3f self=%.3f\n",
                                 m->stepIndex, m->matvecCount, tot.wallMs,
                                 tot.wallMs * inv, tot.cudaMs,
                                 tot.cudaMs * inv, tot.diagCopyWallMs,
                                 tot.denseMs, tot.stokesMs,
                                 tot.selfSubtractMs));
      if (m->useTreecode && tot.tree.used) {
        PetscCallAbort(PETSC_COMM_SELF,
                       PetscPrintf(PETSC_COMM_SELF,
                                   "[MFS treecode reapply total] step=%d count=%d wall=%.3f ms avg=%.3f ms "
                                   "cuda=%.3f ms avg_cuda=%.3f ms force=%.3f "
                                   "reapply=%.3f cm2aos=%.3f last(up=%.3f m2p=%.3f p2p=%.3f pairs=%lld p2p_int=%lld)\n",
                                   m->stepIndex, m->matvecCount,
                                   tot.tree.wallMs, tot.tree.wallMs * inv,
                                   tot.tree.cudaMs, tot.tree.cudaMs * inv,
                                   tot.tree.forceConvertMs,
                                   tot.tree.applyOrReapplyMs,
                                   tot.tree.compToAosMs,
                                   tot.tree.stats.upwardMs,
                                   (double)tot.tree.stats.travMs,
                                   (double)tot.tree.stats.p2pMs,
                                   tot.tree.stats.nPairs,
                                   tot.tree.stats.totalP2P));
      }
    }
  }
}

// ==========================================================================
// teardown
// ==========================================================================
static void mfsDestroy(MFSContext *m)
{
  if (!m) return;
  const bool trace = mfsEnvFlag("MFS_DESTROY_TRACE", false);
  auto mark = [trace](const char *what) {
    if (trace) {
      std::fprintf(stderr, "[mfsDestroy] %s\n", what);
      std::fflush(stderr);
    }
  };

  mark("sync before PETSc teardown");
  CUDA_CHECK(cudaDeviceSynchronize());
  if (mfsSkipPetscTeardownForProfiler()) {
    mark("skip PETSc object teardown under Nsight CUDA memory tracking");
  } else {
    mark("destroy KSP");
    PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&m->ksp));
    mark("destroy Vec b");
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&m->b));
    mark("destroy Vec xsol");
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&m->xsol));
    mark("destroy Mat shell");
    PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&m->A));
  }

  mark("destroy treecode");
  delete m->tc;
  m->tc = nullptr;
  CUDA_CHECK(cudaDeviceSynchronize());

#define MFS_CUDA_FREE(ptr)                                                     \
  do {                                                                         \
    cudaError_t _mfs_free_err = cudaFree(ptr);                                 \
    if (_mfs_free_err != cudaSuccess) {                                        \
      std::fprintf(stderr, "cudaFree(%s) failed: %s\n", #ptr,                 \
                   cudaGetErrorString(_mfs_free_err));                        \
      std::fflush(stderr);                                                     \
    }                                                                          \
    ptr = nullptr;                                                             \
  } while (0)

  mark("free MFS device buffers");
  MFS_CUDA_FREE(m->d_Xref); MFS_CUDA_FREE(m->d_Yref); MFS_CUDA_FREE(m->d_KN); MFS_CUDA_FREE(m->d_Ginv);
  MFS_CUDA_FREE(m->d_ImL); MFS_CUDA_FREE(m->d_pinv); MFS_CUDA_FREE(m->d_What); MFS_CUDA_FREE(m->d_Sself);
  MFS_CUDA_FREE(m->d_centers); MFS_CUDA_FREE(m->d_R); MFS_CUDA_FREE(m->d_F); MFS_CUDA_FREE(m->d_T);
  MFS_CUDA_FREE(m->d_Xglob); MFS_CUDA_FREE(m->d_Yglob); MFS_CUDA_FREE(m->d_lam0); MFS_CUDA_FREE(m->d_u0);
  MFS_CUDA_FREE(m->d_hatGlob); MFS_CUDA_FREE(m->d_gammaRefAll); MFS_CUDA_FREE(m->d_hatRefAll);
  MFS_CUDA_FREE(m->d_lamRefAll); MFS_CUDA_FREE(m->d_rhs6r); MFS_CUDA_FREE(m->d_vw);
  MFS_CUDA_FREE(m->d_Uref); MFS_CUDA_FREE(m->d_Uglob);
  MFS_CUDA_FREE(m->d_tcVelCM);
  MFS_CUDA_FREE(m->d_Uself); MFS_CUDA_FREE(m->d_selfRefAll);
#undef MFS_CUDA_FREE

  CUDA_CHECK(cudaDeviceSynchronize());
  mark("done");
  delete m;
}

// Enable the GPU treecode for the Stokeslet sums (RHS + matvec) and relax the
// GMRES tolerance to the fp32 treecode's accuracy floor (~1e-6). Call once after
// mfsSetup, before the first mfsSolve.
static void mfsSetTreecode(MFSContext *m, const MFSTreecode<>::Config &cfg,
                           double kspRtol = 1e-6)
{
  if (std::fabs(m->mu - stokes::MU) > 1e-12 * std::max(1.0, std::fabs(m->mu))) {
    std::fprintf(stderr,
                 "treecode MFS currently requires mu=%g because stokes_kernel.cuh "
                 "uses a compile-time prefactor; got mu=%g\n",
                 stokes::MU, m->mu);
    std::exit(1);
  }
  m->useTreecode = true;
  m->tcCfg = cfg;
  m->tcCfg.sourceGroupSize = m->N;
  m->tcCfg.targetGroupSize = m->M;
  m->tcCfg.skipSameGroup = false;
  // The treecode execution path is selected by the TC_PATH environment variable
  // inside Treecode::configure(); MFS no longer sets it on the Config.
  if (!m->tc)
    m->tc = new MFSTreecode<>(m->tcCfg);
  else
    m->tc->setConfig(m->tcCfg);
  PetscReal atol = 0.0, dtol = 0.0;
  PetscInt maxIts = 0;
  PetscCallAbort(PETSC_COMM_SELF,
                 KSPGetTolerances(m->ksp, nullptr, &atol, &dtol, &maxIts));
  PetscCallAbort(PETSC_COMM_SELF,
                 KSPSetTolerances(m->ksp, kspRtol, atol, dtol, maxIts));
}

// ==========================================================================
// backward-compatible one-shot wrapper (host-vector API)
// ==========================================================================
// Inputs (host): Xglob/Yglob global AoS clouds (3*P*M, 3*P*N), centers (P*3),
// F/T applied wrench (P*3 each), mu. Reference = particle 0. The existing
// drivers tile WITHOUT rotation, so every particle's orientation is identity
// (Kabsch on translation-only inputs would return I); we pass R = I and let the
// GPU regenerate the clouds from the reference cloud, reproducing the inputs.
static void solveMobility(cublasHandle_t cublas, cusolverDnHandle_t solver,
                          int P, int M, int N, const std::vector<double> &Xglob,
                          const std::vector<double> &Yglob,
                          const std::vector<double> &centers,
                          const std::vector<double> &Fapp,
                          const std::vector<double> &Tapp, double mu,
                          std::vector<double> &Uout)
{
  (void)Yglob;  // reference cloud derived from particle 0 of Xglob/Yglob below
  const int tM = 3 * M, tN = 3 * N;
  std::vector<double> Xref(tM), Yref(tN);
  for (int i = 0; i < M; ++i)
    for (int d = 0; d < 3; ++d) Xref[3 * i + d] = Xglob[3 * i + d] - centers[d];
  for (int j = 0; j < N; ++j)
    for (int d = 0; d < 3; ++d) Yref[3 * j + d] = Yglob[3 * j + d] - centers[d];

  std::vector<double> R((size_t)P * 9, 0.0);
  for (int k = 0; k < P; ++k) {
    R[(size_t)k * 9 + 0] = R[(size_t)k * 9 + 4] = R[(size_t)k * 9 + 8] = 1.0;
  }

  MFSContext *m = mfsSetup(cublas, solver, P, M, N, Xref.data(), Yref.data(), mu);
  mfsSolve(m, centers.data(), R.data(), Fapp.data(), Tapp.data(), Uout);
  mfsDestroy(m);
}
