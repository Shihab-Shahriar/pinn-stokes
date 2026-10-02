#pragma once

// SPDX-License-Identifier: Apache-2.0
//
// Second-kind boundary-integral mobility solver for suspensions of rigid unit
// spheres (Yan, Corona, Malhotra, Veerapaneni, Shelley, JCP 2020, eqs. 32-34),
// with the all-to-all operators accelerated by the particle-grouped treecode
// (TC_PATH=skel) in KAFMM style: unchanged Stokeslet source moments, a
// target-normal traction evaluation at per-particle skeleton points, and a
// traction-specific lift GEMM.
//
// Formulation (viscosity eta = 1, radius R = 1, |Gamma| = 4*pi, tau = 8*pi/3):
//   rho_j(x) = F_j/(4*pi) + (3/(8*pi)) T_j x (x - c_j)
//   (1/2 I + K + L)[zeta] = -(1/2 I + K)[rho]        (GMRES, matrix-free)
//   u = S[rho + zeta];  U_j = (1/(4*pi)) sum w u;  Omega_j = (3/(8*pi)) sum w (x-c_j) x u
// K is the TARGET-normal traction of the single layer, T_ijk = -3/(4*pi)
// r_i r_j r_k / r^5, discretized with the p=8 spherical-harmonic grid
// (B = 162 points/sphere, Gauss-Legendre x uniform, weighted density q = w*sigma).
//
// Same-sphere (singular) blocks are two shared dense 3B x 3B matrices K_self
// (principal value) and S_self, precomputed OFFLINE by
// scripts/gen_sphere_selfblocks.py and applied as one cuBLAS DGEMM per operator
// application; the treecode/brute path computes only the off-diagonal
// (other-sphere) smooth sums with skipSameGroup. All spheres share one grid
// orientation, so every per-sphere matrix is shared.
//
// Structure and conventions follow mfs_broms.cuh (persistent context, PETSc
// MatShell + VecSeqCUDA zero-copy GMRES, timing sets); that file is NOT
// included here. This header pulls in treecode.cuh, so it must be included by
// exactly one driver TU.

#include <petscksp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

#include "treecode.cuh"
#include "sphere_near_vsh.cuh"

using SBTreecode = Treecode<mp::BaryStokes>;

static_assert(sizeof(vec3d) == 3 * sizeof(double),
              "sphere-BIM double AoS layout must match cuBQL::vec3d");

// ==========================================================================
// device kernels
// ==========================================================================

// q[i] = w[(i/3) % B] * v[i]: quadrature-weight the density (AoS sphere-major).
__global__ void sbimWeightKernel(const double *v, const double *w, int B,
                                 int n, double *q)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  q[i] = w[(i / 3) % B] * v[i];
}

// dst AoS <- src component-major (treecode output layout), optional add.
__global__ void sbimCompMajorToAoSKernel(const double *src, double *dst,
                                         int nPts, int add)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * nPts) return;
  const int p = i / 3, c = i - 3 * p;
  const double v = src[(size_t)c * nPts + p];
  dst[i] = add ? dst[i] + v : v;
}

// Brute off-diagonal single-layer velocity: one thread per target, full fp64
// (sqrt) like treecode_validate.cuh::directSum64, skipping the target's own
// sphere (p/B == t/B). Writes (sets) AoS.
__global__ void sbimBruteStokesletKernel(const vec3d *pts, const double *q,
                                         int nPts, int B, double pref,
                                         double *outAoS)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nPts) return;
  const vec3d T = pts[t];
  const int own = t / B;
  double u0 = 0.0, u1 = 0.0, u2 = 0.0;
  for (int s = 0; s < nPts; ++s) {
    if (s / B == own) continue;
    const double Rx = T.x - pts[s].x;
    const double Ry = T.y - pts[s].y;
    const double Rz = T.z - pts[s].z;
    const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
    const double ir = 1.0 / sqrt(r2);
    const double ir3 = ir / r2;
    const double fx = q[3 * s], fy = q[3 * s + 1], fz = q[3 * s + 2];
    const double rdf = Rx * fx + Ry * fy + Rz * fz;
    u0 += fx * ir + Rx * rdf * ir3;
    u1 += fy * ir + Ry * rdf * ir3;
    u2 += fz * ir + Rz * rdf * ir3;
  }
  outAoS[3 * t]     = pref * u0;
  outAoS[3 * t + 1] = pref * u1;
  outAoS[3 * t + 2] = pref * u2;
}

// Brute off-diagonal single-layer traction (target normal): t += pref *
// R (R.q)(R.n)/r^5, pref = -3/(4*pi). Same layout/skip as the Stokeslet brute.
__global__ void sbimBruteTractionKernel(const vec3d *pts, const vec3d *nrm,
                                        const double *q, int nPts, int B,
                                        double pref, double *outAoS)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nPts) return;
  const vec3d T = pts[t];
  const vec3d nA = nrm[t];
  const int own = t / B;
  double t0 = 0.0, t1 = 0.0, t2 = 0.0;
  for (int s = 0; s < nPts; ++s) {
    if (s / B == own) continue;
    const double Rx = T.x - pts[s].x;
    const double Ry = T.y - pts[s].y;
    const double Rz = T.z - pts[s].z;
    const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
    const double ir = 1.0 / sqrt(r2);
    const double ir5 = (ir * ir) * (ir * ir) * ir;
    const double rdq = Rx * q[3 * s] + Ry * q[3 * s + 1] + Rz * q[3 * s + 2];
    const double rdn = Rx * nA.x + Ry * nA.y + Rz * nA.z;
    const double sfac = rdq * rdn * ir5;
    t0 += Rx * sfac;
    t1 += Ry * sfac;
    t2 += Rz * sfac;
  }
  outAoS[3 * t]     = pref * t0;
  outAoS[3 * t + 1] = pref * t1;
  outAoS[3 * t + 2] = pref * t2;
}

static inline int sbimGrid1d(int n, int block) { return (n + block - 1) / block; }

// ==========================================================================
// context
// ==========================================================================

struct SBimContext {
  int N = 0;      // spheres
  int B = 0;      // surface points per sphere
  int nPts = 0;   // N*B
  int n = 0;      // 3*N*B unknowns
  int n3B = 0;    // 3*B (self-block dimension)

  // reference grid (host copies kept for extraction/diagnostics)
  std::vector<double> h_grid;   // B x 3 (unit sphere, sphere-relative)
  std::vector<double> h_w;      // B
  std::vector<double> h_centers;  // N x 3

  // shared dense operators (col-major 3B x 3B) and small maps (col-major 3B x 6)
  double *d_Arhs = nullptr;   // 1/2 I + K_self
  double *d_Amat = nullptr;   // 1/2 I + K_self + L_self
  double *d_Sself = nullptr;  // S_self
  double *d_Rmat = nullptr;   // wrench -> rho
  double *d_Emat = nullptr;   // u -> [U; Omega] (via E^T)
  double *d_w = nullptr;      // B weights

  // geometry
  vec3d *d_pts = nullptr;     // nPts, sphere-major input order
  vec3d *d_nrm = nullptr;     // nPts outward normals (= x - c_j)

  // work buffers (all length n unless noted)
  double *d_FT = nullptr;     // 6 x N col-major wrench
  double *d_rho = nullptr;
  double *d_q = nullptr;      // weighted density
  double *d_tcCM = nullptr;   // treecode component-major output
  double *d_off = nullptr;    // off-diagonal operator result (AoS)
  double *d_sigma = nullptr;  // rho + zeta
  double *d_u = nullptr;      // surface velocity
  double *d_UOm = nullptr;    // 6 x N

  // treecode
  bool useTreecode = true;
  bool treeBuilt = false;
  SBTreecode *tc = nullptr;
  SBTreecode::Config tcCfg{};

  // VSH near-pair correction (Yan 2020; --near-beta, 0 disables). The
  // correction runs on its own non-blocking stream into d_nearScratch,
  // overlapping the treecode far field; one add kernel joins the streams.
  bool nearEnabled = false;
  SBimNear near{};
  cudaStream_t nearStream = nullptr;
  double *d_nearScratch = nullptr;   // length n
  double lastNearMs = 0.0;     // near part of the last sbimApplyOffdiag
  double matvecNearMs = 0.0;   // accumulated across GMRES matvecs

  // PETSc
  Mat A = nullptr;
  KSP ksp = nullptr;
  PC pc = nullptr;
  Vec b = nullptr, xsol = nullptr;

  cublasHandle_t cublas = nullptr;

  // timing (SBIM_TIMING, default on)
  bool timingEnabled = true;
  int matvecCount = 0;
  double matvecWallMs = 0.0;   // total across GMRES
  double matvecOffMs = 0.0;    // off-diagonal (treecode reapply / brute) part
  double rhsWallMs = 0.0;      // RHS incl. one-time tree build
  double rhsOffMs = 0.0;
  double treeBuildMs = 0.0;    // first traction apply (build+lift+traverse+eval)
  double kspWallMs = 0.0;
  double finalSMs = 0.0;
  double extractMs = 0.0;
};

// ==========================================================================
// data loading + operator assembly
// ==========================================================================

static std::vector<double> sbimReadBin(const std::string &path,
                                       size_t expectCount)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path +
                             " (run: python3 scripts/gen_sphere_selfblocks.py "
                             "--p 8 --out data/sphere_p8)");
  const size_t bytes = (size_t)f.tellg();
  if (expectCount && bytes != expectCount * sizeof(double))
    throw std::runtime_error(path + ": expected " +
                             std::to_string(expectCount) + " doubles, got " +
                             std::to_string(bytes / sizeof(double)));
  std::vector<double> v(bytes / sizeof(double));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(v.data()), (std::streamsize)bytes);
  if (!f) throw std::runtime_error("short read on " + path);
  return v;
}

static inline double sbimEps(int i, int j, int k)
{
  if (i == j || j == k || i == k) return 0.0;
  return ((j - i + 3) % 3 == 1) ? 1.0 : -1.0;  // even permutation of (0,1,2)
}

static double *sbimUpload(const std::vector<double> &h)
{
  double *d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  return d;
}

// Load the reference grid + self blocks, assemble the fused diagonal operators
// and the wrench/extraction maps, upload everything.
static void sbimLoadData(SBimContext *m, const std::string &dir)
{
  const std::vector<double> w = sbimReadBin(dir + "/weights.bin", 0);
  const int B = (int)w.size();
  const int n3 = 3 * B;
  const std::vector<double> grid = sbimReadBin(dir + "/grid.bin", (size_t)3 * B);
  const std::vector<double> K = sbimReadBin(dir + "/Kself.bin",
                                            (size_t)n3 * n3);
  const std::vector<double> S = sbimReadBin(dir + "/Sself.bin",
                                            (size_t)n3 * n3);
  const std::vector<double> L = sbimReadBin(dir + "/Lself.bin",
                                            (size_t)n3 * n3);
  m->B = B;
  m->n3B = n3;
  m->h_grid = grid;
  m->h_w = w;

  // A_rhs = 1/2 I + K_self ; A_mat = 1/2 I + K_self + L_self (col-major).
  std::vector<double> Arhs(K), Amat((size_t)n3 * n3);
  for (int i = 0; i < n3; ++i) Arhs[(size_t)i * n3 + i] += 0.5;
  for (size_t i = 0; i < Amat.size(); ++i) Amat[i] = Arhs[i] + L[i];
  m->d_Arhs = sbimUpload(Arhs);
  m->d_Amat = sbimUpload(Amat);
  m->d_Sself = sbimUpload(S);
  m->d_w = sbimUpload(w);

  // Rmat (3B x 6): rho = Rmat [F; T], rho_b = F/(4pi) + (3/8pi) T x x_b.
  // Emat (3B x 6): [U; Omega] = Emat^T u, U = (1/4pi) sum w u,
  // Omega = (3/8pi) sum w x_b x u_b.
  std::vector<double> R((size_t)n3 * 6, 0.0), E((size_t)n3 * 6, 0.0);
  const double c1 = 1.0 / (4.0 * M_PI), c2 = 3.0 / (8.0 * M_PI);
  for (int b = 0; b < B; ++b) {
    const double *x = &grid[(size_t)3 * b];
    for (int i = 0; i < 3; ++i) {
      R[(size_t)i * n3 + 3 * b + i] = c1;                // dF column i
      E[(size_t)i * n3 + 3 * b + i] = c1 * w[b];         // U column i
      // rho_i from T_j: c2 * eps(i,j,k) x_k  (column 3+j)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          R[(size_t)(3 + j) * n3 + 3 * b + i] += c2 * sbimEps(i, j, k) * x[k];
    }
    // Omega_j = c2 sum_b w_b eps(j,k,l) x_k u_l  -> E[3b+l, 3+j]
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          E[(size_t)(3 + j) * n3 + 3 * b + l] +=
              c2 * w[b] * sbimEps(j, k, l) * x[k];
  }
  m->d_Rmat = sbimUpload(R);
  m->d_Emat = sbimUpload(E);
}

// ==========================================================================
// off-diagonal operator applications
// ==========================================================================

// Off-diagonal traction (or Stokeslet) of the weighted density q into d_off
// (AoS, set). kernelKind selects the pair kernel; the treecode instance keeps
// its built tree across calls and kernel switches.
static void sbimApplyOffdiag(SBimContext *m, SBTreecode::KernelKind kind,
                             const double *d_dens, double *d_outAoS,
                             double *offMsOut)
{
  const int block = 128;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  if (m->timingEnabled) {
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
  }
  sbimWeightKernel<<<sbimGrid1d(m->n, block), block>>>(d_dens, m->d_w, m->B,
                                                       m->n, m->d_q);
  CUDA_CHECK(cudaGetLastError());

  // Launch the VSH near-pair correction FIRST, on its own non-blocking
  // stream, into the zeroed scratch buffer: it depends only on d_dens and
  // d_q, so it overlaps the (much longer) treecode far-field work below.
  // The result is folded in by one add kernel after the streams join --
  // numerically identical to the old in-place accumulation (same two-term
  // add per entry, same in-kernel summation order).
  cudaEvent_t evQ = nullptr, evNear = nullptr, n0 = nullptr, n1 = nullptr;
  m->lastNearMs = 0.0;
  const bool nearActive = m->nearEnabled && m->N > 1;
  if (nearActive) {
    CUDA_CHECK(cudaEventCreateWithFlags(&evQ, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evNear, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(evQ));                 // d_q ready (stream 0)
    CUDA_CHECK(cudaStreamWaitEvent(m->nearStream, evQ));
    if (m->timingEnabled) {
      CUDA_CHECK(cudaEventCreate(&n0));
      CUDA_CHECK(cudaEventCreate(&n1));
      CUDA_CHECK(cudaEventRecord(n0, m->nearStream));
    }
    CUDA_CHECK(cudaMemsetAsync(m->d_nearScratch, 0,
                               (size_t)m->n * sizeof(double), m->nearStream));
    sbimNearApply(&m->near, m->cublas,
                  kind == SBTreecode::KernelKind::Traction ? 0 : 1, d_dens,
                  m->d_q, m->d_pts, m->d_nrm, m->d_nearScratch, /*doSub=*/1,
                  m->nearStream);
    if (m->timingEnabled) CUDA_CHECK(cudaEventRecord(n1, m->nearStream));
    CUDA_CHECK(cudaEventRecord(evNear, m->nearStream));
  }

  if (m->useTreecode && m->N > 1) {
    m->tc->setKernel(kind);
    if (!m->treeBuilt) {
      m->tc->apply(m->d_pts, (size_t)m->nPts, m->d_pts, (size_t)m->nPts,
                   reinterpret_cast<const vec3d *>(m->d_q), m->d_tcCM,
                   /*will_reuse_tree=*/true);
      m->treeBuilt = true;
    } else {
      m->tc->reapply(reinterpret_cast<const vec3d *>(m->d_q), m->d_tcCM);
    }
    sbimCompMajorToAoSKernel<<<sbimGrid1d(m->n, block), block>>>(
        m->d_tcCM, d_outAoS, m->nPts, /*add=*/0);
    CUDA_CHECK(cudaGetLastError());
  } else if (m->N > 1) {
    if (kind == SBTreecode::KernelKind::Traction)
      sbimBruteTractionKernel<<<sbimGrid1d(m->nPts, block), block>>>(
          m->d_pts, m->d_nrm, m->d_q, m->nPts, m->B,
          stokes::tractionPrefactor(), d_outAoS);
    else
      sbimBruteStokesletKernel<<<sbimGrid1d(m->nPts, block), block>>>(
          m->d_pts, m->d_q, m->nPts, m->B, (double)stokes::prefactor(),
          d_outAoS);
    CUDA_CHECK(cudaGetLastError());
  } else {
    CUDA_CHECK(cudaMemset(d_outAoS, 0, (size_t)m->n * sizeof(double)));
  }

  // join: fold the side-stream correction into the operator output
  if (nearActive) {
    CUDA_CHECK(cudaStreamWaitEvent(0, evNear));
    snear::snAddKernel<<<sbimGrid1d(m->n, block), block>>>(m->d_nearScratch,
                                                           d_outAoS, m->n);
    CUDA_CHECK(cudaGetLastError());
  }

  if (m->timingEnabled) {
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));   // add kernel done => n0/n1 done
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (offMsOut) *offMsOut = (double)ms;
    if (nearActive) {
      float nms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&nms, n0, n1));
      m->lastNearMs = (double)nms;   // side-stream duration (overlapped)
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  } else if (offMsOut) {
    *offMsOut = 0.0;
  }
  if (nearActive) {
    cudaEventDestroy(evQ);
    cudaEventDestroy(evNear);
    if (n0) cudaEventDestroy(n0);
    if (n1) cudaEventDestroy(n1);
  }
}

// ==========================================================================
// PETSc matvec: y = (1/2 I + K + L) z
//             = K_offdiag[z]  (set)  +  (1/2 I + K_self + L_self) z  (DGEMM add)
// ==========================================================================
static PetscErrorCode MatMult_SBIM(Mat A, Vec x, Vec y)
{
  SBimContext *m;
  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &m));

  PetscLogDouble w0 = 0.0, w1 = 0.0;
  if (m->timingEnabled) PetscCall(PetscTime(&w0));

  const PetscScalar *dx;
  PetscScalar *dy;
  PetscCall(VecCUDAGetArrayRead(x, &dx));
  PetscCall(VecCUDAGetArray(y, &dy));

  double offMs = 0.0;
  sbimApplyOffdiag(m, SBTreecode::KernelKind::Traction, dx, dy, &offMs);

  const double one = 1.0;
  CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m->n3B, m->N,
                           m->n3B, &one, m->d_Amat, m->n3B, dx, m->n3B, &one,
                           dy, m->n3B));
  CUDA_CHECK(cudaDeviceSynchronize());

  PetscCall(VecCUDARestoreArrayRead(x, &dx));
  PetscCall(VecCUDARestoreArray(y, &dy));

  if (m->timingEnabled) {
    PetscCall(PetscTime(&w1));
    m->matvecWallMs += 1000.0 * (double)(w1 - w0);
    m->matvecOffMs += offMs;
    m->matvecNearMs += m->lastNearMs;
    ++m->matvecCount;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================
// setup / teardown
// ==========================================================================

static SBimContext *sbimSetup(const std::string &dataDir, int N,
                              const std::vector<double> &centers,
                              bool useTreecode, float mac, double kspRtol,
                              double nearBeta)
{
  SBimContext *m = new SBimContext();
  m->timingEnabled = [] {
    const char *e = std::getenv("SBIM_TIMING");
    return !(e && std::strcmp(e, "0") == 0);
  }();
  sbimLoadData(m, dataDir);
  m->N = N;
  m->nPts = N * m->B;
  m->n = 3 * m->nPts;
  m->h_centers = centers;
  m->useTreecode = useTreecode;

  CUBLAS_CHECK(cublasCreate(&m->cublas));

  // Keep the stream-ordered-allocator pool warm. cuBQL's warp refit does one
  // cudaMallocAsync/cudaFreeAsync pair per (re)apply, and the default pool
  // releases its memory back to the driver at every synchronization
  // (release threshold 0), so each matvec re-grows the pool -- and that
  // growth serializes against in-flight work on OTHER streams (measured: it
  // gated the refit launch behind the whole 57 ms side-stream near-correction
  // kernel, killing the intended overlap). With the threshold maxed the pool
  // retains the block and reuse is instant and stream-local.
  {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaMemPool_t pool = nullptr;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, dev));
    unsigned long long thresh = ~0ULL;
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                       &thresh));
  }

  // geometry: points = template + c_j, normals = template (unit sphere).
  {
    std::vector<double> pts((size_t)m->n), nrm((size_t)m->n);
    for (int k = 0; k < N; ++k)
      for (int b = 0; b < m->B; ++b)
        for (int c = 0; c < 3; ++c) {
          pts[(size_t)3 * (k * m->B + b) + c] =
              m->h_grid[(size_t)3 * b + c] + centers[(size_t)3 * k + c];
          nrm[(size_t)3 * (k * m->B + b) + c] = m->h_grid[(size_t)3 * b + c];
        }
    CUDA_CHECK(cudaMalloc(&m->d_pts, (size_t)m->nPts * sizeof(vec3d)));
    CUDA_CHECK(cudaMalloc(&m->d_nrm, (size_t)m->nPts * sizeof(vec3d)));
    CUDA_CHECK(cudaMemcpy(m->d_pts, pts.data(), (size_t)m->n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m->d_nrm, nrm.data(), (size_t)m->n * sizeof(double),
                          cudaMemcpyHostToDevice));
  }

  // VSH near-pair correction (paper's close-pair scheme; beta in (1.5, 2])
  if (nearBeta > 0.0 && N > 1) {
    sbimNearInit(&m->near, dataDir, N, centers, nearBeta, m->B);
    m->nearEnabled = (m->near.nRows > 0);
    if (m->nearEnabled) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&m->nearStream,
                                           cudaStreamNonBlocking));
      CUDA_CHECK(cudaMalloc(&m->d_nearScratch,
                            (size_t)m->n * sizeof(double)));
    }
  } else if (N > 1) {
    std::printf("[near] disabled (--near-beta=0): smooth quadrature only\n");
  }

  CUDA_CHECK(cudaMalloc(&m->d_FT, (size_t)6 * N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_rho, (size_t)m->n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_q, (size_t)m->n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_tcCM, (size_t)m->n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_off, (size_t)m->n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_sigma, (size_t)m->n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_u, (size_t)m->n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m->d_UOm, (size_t)6 * N * sizeof(double)));

  // treecode (skel path): env defaults, overridable by the caller's shell.
  if (useTreecode && N > 1) {
    setenv("TC_PATH", "skel", 0);
    setenv("TC_BUCKETIZER", "object", 0);
    setenv("TC_SKEL_P2P", "block", 0);
    // Retuned traction-skeleton default (rank 96, was 128): iso-accuracy once
    // the VSH near correction owns the near field; see sphere_mobility.cu.
    setenv("TC_TGT_SKEL_TRACTION", "96", 0);
    std::printf("[sbim] treecode env: TC_PATH=%s TC_BUCKETIZER=%s "
                "TC_SKEL_P2P=%s TC_TGT_SKEL=%s TC_TGT_SKEL_TRACTION=%s\n",
                std::getenv("TC_PATH"), std::getenv("TC_BUCKETIZER"),
                std::getenv("TC_SKEL_P2P"),
                std::getenv("TC_TGT_SKEL") ? std::getenv("TC_TGT_SKEL") : "(128)",
                std::getenv("TC_TGT_SKEL_TRACTION")
                    ? std::getenv("TC_TGT_SKEL_TRACTION") : "(=TC_TGT_SKEL)");
    m->tcCfg.mac = mac;
    m->tcCfg.sourceGroupSize = m->B;
    m->tcCfg.targetGroupSize = m->B;
    m->tcCfg.skipSameGroup = true;
    m->tc = new SBTreecode(m->tcCfg);
    m->tc->setTargetNormals(m->d_nrm, (size_t)m->nPts);
  }

  // PETSc objects (created once, reused across solves)
  PetscCallAbort(PETSC_COMM_SELF,
                 MatCreateShell(PETSC_COMM_SELF, m->n, m->n, m->n, m->n, m,
                                &m->A));
  PetscCallAbort(PETSC_COMM_SELF,
                 MatShellSetOperation(m->A, MATOP_MULT,
                                      (void (*)(void))MatMult_SBIM));
  PetscCallAbort(PETSC_COMM_SELF, VecCreateSeqCUDA(PETSC_COMM_SELF, m->n, &m->b));
  PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(m->b, &m->xsol));
  PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &m->ksp));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(m->ksp, m->A, m->A));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetType(m->ksp, KSPGMRES));
  // Large restart (effectively full GMRES): restart-30 stagnates badly at
  // N=10^4 (317 vs 60 iterations at rtol 1e-6) -- the uncorrected near-field
  // quadrature spreads the second-kind spectrum enough that the Krylov space
  // must be kept. PETSc allocates basis vectors lazily (chunks of 10), so a
  // big restart costs nothing while iteration counts stay moderate; override
  // with -ksp_gmres_restart.
  PetscCallAbort(PETSC_COMM_SELF, KSPGMRESSetRestart(m->ksp, 500));
  PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(m->ksp, &m->pc));
  PetscCallAbort(PETSC_COMM_SELF, PCSetType(m->pc, PCNONE));
  PetscCallAbort(PETSC_COMM_SELF,
                 KSPSetTolerances(m->ksp, kspRtol, 0.0, PETSC_DEFAULT, 500));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetFromOptions(m->ksp));
  return m;
}

static void sbimDestroy(SBimContext *m)
{
  if (!m) return;
  if (m->ksp) KSPDestroy(&m->ksp);
  if (m->A) MatDestroy(&m->A);
  if (m->b) VecDestroy(&m->b);
  if (m->xsol) VecDestroy(&m->xsol);
  sbimNearFree(&m->near);
  if (m->nearStream) cudaStreamDestroy(m->nearStream);
  if (m->d_nearScratch) cudaFree(m->d_nearScratch);
  delete m->tc;
  for (double *p : {m->d_Arhs, m->d_Amat, m->d_Sself, m->d_Rmat, m->d_Emat,
                    m->d_w, m->d_FT, m->d_rho, m->d_q, m->d_tcCM, m->d_off,
                    m->d_sigma, m->d_u, m->d_UOm})
    if (p) cudaFree(p);
  if (m->d_pts) cudaFree(m->d_pts);
  if (m->d_nrm) cudaFree(m->d_nrm);
  if (m->cublas) cublasDestroy(m->cublas);
  delete m;
}

// ==========================================================================
// one mobility solve: wrench (host, 6 per sphere, [F;T]) -> UOm (host, 6 per
// sphere, [U;Omega]). Also returns the surface velocity if uOut != nullptr.
// ==========================================================================
static void sbimSolve(SBimContext *m, const std::vector<double> &FT,
                      std::vector<double> &UOm, std::vector<double> *uOut)
{
  const double one = 1.0, zero = 0.0, negone = -1.0;
  PetscLogDouble w0 = 0.0, w1 = 0.0;
  m->matvecCount = 0;
  m->matvecWallMs = m->matvecOffMs = m->matvecNearMs = 0.0;

  // rho = Rmat [F;T]  (3B x N view of d_rho)
  CUDA_CHECK(cudaMemcpy(m->d_FT, FT.data(), (size_t)6 * m->N * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m->n3B, m->N, 6,
                           &one, m->d_Rmat, m->n3B, m->d_FT, 6, &zero,
                           m->d_rho, m->n3B));

  // RHS: b = -(A_rhs rho + K_offdiag[rho])   (one-time tree build lives here)
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&w0));
  sbimApplyOffdiag(m, SBTreecode::KernelKind::Traction, m->d_rho, m->d_off,
                   &m->rhsOffMs);
  m->treeBuildMs = m->rhsOffMs;
  {
    PetscScalar *db;
    PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArray(m->b, &db));
    CUDA_CHECK(cudaMemcpy(db, m->d_off, (size_t)m->n * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m->n3B, m->N,
                             m->n3B, &one, m->d_Arhs, m->n3B, m->d_rho, m->n3B,
                             &one, db, m->n3B));
    CUBLAS_CHECK(cublasDscal(m->cublas, m->n, &negone, db, 1));
    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArray(m->b, &db));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&w1));
  m->rhsWallMs = 1000.0 * (double)(w1 - w0);

  // GMRES
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&w0));
  PetscCallAbort(PETSC_COMM_SELF, KSPSolve(m->ksp, m->b, m->xsol));
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&w1));
  m->kspWallMs = 1000.0 * (double)(w1 - w0);
  {
    KSPConvergedReason reason;
    PetscInt its;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetConvergedReason(m->ksp, &reason));
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(m->ksp, &its));
    std::printf("[sbim] GMRES %s in %d iterations\n",
                reason > 0 ? "converged" : "DID NOT CONVERGE", (int)its);
  }

  // u = S_self (rho + zeta) + S_offdiag[rho + zeta]
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&w0));
  {
    const PetscScalar *dz;
    PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayRead(m->xsol, &dz));
    CUDA_CHECK(cudaMemcpy(m->d_sigma, m->d_rho, (size_t)m->n * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasDaxpy(m->cublas, m->n, &one, dz, 1, m->d_sigma, 1));
    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(m->xsol, &dz));
  }
  sbimApplyOffdiag(m, SBTreecode::KernelKind::Stokeslet, m->d_sigma, m->d_u,
                   nullptr);
  CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m->n3B, m->N,
                           m->n3B, &one, m->d_Sself, m->n3B, m->d_sigma,
                           m->n3B, &one, m->d_u, m->n3B));
  CUDA_CHECK(cudaDeviceSynchronize());
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&w1));
  m->finalSMs = 1000.0 * (double)(w1 - w0);

  // [U; Omega] = Emat^T u
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&w0));
  CUBLAS_CHECK(cublasDgemm(m->cublas, CUBLAS_OP_T, CUBLAS_OP_N, 6, m->N, m->n3B,
                           &one, m->d_Emat, m->n3B, m->d_u, m->n3B,
                           &zero, m->d_UOm, 6));
  UOm.resize((size_t)6 * m->N);
  CUDA_CHECK(cudaMemcpy(UOm.data(), m->d_UOm, (size_t)6 * m->N * sizeof(double),
                        cudaMemcpyDeviceToHost));
  if (uOut) {
    uOut->resize((size_t)m->n);
    CUDA_CHECK(cudaMemcpy(uOut->data(), m->d_u, (size_t)m->n * sizeof(double),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  PetscCallAbort(PETSC_COMM_SELF, PetscTime(&w1));
  m->extractMs = 1000.0 * (double)(w1 - w0);
}
