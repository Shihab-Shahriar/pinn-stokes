// SPDX-License-Identifier: Apache-2.0
//
// Standalone unit tests for the SphericalStokes multipole policy, transcribing
// the self-tests of ../stokes_fmm_moment_demo.py onto the actual device code:
//   (a) irregular solid-harmonic closed-form spot checks
//   (b) analytic Cartesian gradient (neighbour ladder) vs central finite
//       differences of the value table
//   (c) P2M + M2P vs a direct Stokeslet sum: box-relative invariance across
//       several centers, order sweep p=2..9 error-decay table, through both
//       the fp64 accumulators and the production fp32 NodeM2P store
//   (d) M2M (child -> parent translation) vs a direct P2M at the parent --
//       specifically guards the normalized channel-3 ratio coupling
//   (e) warp-cooperative m2pWarp vs the serial m2p reference
//
// This TU includes spherical_stokes.cuh directly (NOT treecode.cuh); it is the
// single TU of the sph_selftest executable, satisfying the ODR constraint on
// the policy's __device__ globals.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

#include "spherical_stokes.cuh"

using mp::SphericalStokes;
using mp::sph::dcplx;
using mp::sph::kdimOf;
using mp::sph::kOf;
using cuBQL::vec3f;
using cuBQL::vec3d;

constexpr int KM = SphericalStokes::K_MAX;         // 91
constexpr int PMAX = SphericalStokes::MAX_ORDER;   // 12
static const double INV8PI = 1.0 / (8.0 * M_PI);

#define CHECK_LAUNCH()                                                    \
  do {                                                                    \
    CUDA_CHECK(cudaGetLastError());                                       \
    CUDA_CHECK(cudaDeviceSynchronize());                                  \
  } while (0)

// ---------------------------------------------------------------------------
// Kernels wrapping the policy's device helpers.
// ---------------------------------------------------------------------------

// Per-coefficient irregular value + gradient at v (scale a=1), k = thread id.
template<int ORDER>
__global__ void kIrrEval(double3 v, int K, double *val2, double *grad6)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  const int n = mp::sph::c_kn_gs[k];
  const int m = mp::sph::c_km_gs[k];
  double ir, ir2;
  mp::sph::invR(v.x * v.x + v.y * v.y + v.z * v.z, ir, ir2);
  dcplx val, gx, gy, gz;
  mp::sph::irrValGrad<ORDER>(v.x, v.y, v.z, ir, ir2, n, m, val, gx, gy, gz);
  val2[2 * k + 0] = val.re;  val2[2 * k + 1] = val.im;
  grad6[6 * k + 0] = gx.re;  grad6[6 * k + 1] = gx.im;
  grad6[6 * k + 2] = gy.re;  grad6[6 * k + 3] = gy.im;
  grad6[6 * k + 4] = gz.re;  grad6[6 * k + 5] = gz.im;
}

// Fused P2M of N sources about (center, a) into fp64 moments (single thread;
// correctness harness, not a perf path). Accumulates in the K-sized local the
// production path uses, then stages out to the KM-strided host buffer
// (zero-padded beyond K).
template<int ORDER>
__global__ void kP2M(const double3 *src, const double3 *frc, int N,
                     double3 center, double a, double *MaccFlat)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  constexpr int K = kdimOf(ORDER);
  double Macc[4][K][2];
  for (int c = 0; c < 4; ++c)
    for (int k = 0; k < K; ++k)
      Macc[c][k][0] = Macc[c][k][1] = 0.0;
  const double invA = 1.0 / a;
  for (int i = 0; i < N; ++i) {
    const double S[3] = {(src[i].x - center.x) * invA,
                         (src[i].y - center.y) * invA,
                         (src[i].z - center.z) * invA};
    const double f[3] = {frc[i].x, frc[i].y, frc[i].z};
    mp::sph::p2mPoint<ORDER>(S, f, Macc);
  }
  auto out = reinterpret_cast<double(*)[KM][2]>(MaccFlat);
  for (int c = 0; c < 4; ++c)
    for (int k = 0; k < KM; ++k) {
      out[c][k][0] = (k < K) ? Macc[c][k][0] : 0.0;
      out[c][k][1] = (k < K) ? Macc[c][k][1] : 0.0;
    }
}

// Fused M2P from fp64 moments (single thread): the H-contraction over the
// production irrValGrad basis, but with fp64 moment inputs so truncation-only
// error is observable.
template<int ORDER>
__global__ void kM2P64(const double *MaccFlat, double3 center, double a,
                       double3 T, double *u_out)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  auto Macc = reinterpret_cast<const double(*)[KM][2]>(MaccFlat);
  constexpr int K = kdimOf(ORDER);
  const double invA = 1.0 / a;
  const double Dx = (T.x - center.x) * invA;
  const double Dy = (T.y - center.y) * invA;
  const double Dz = (T.z - center.z) * invA;
  double ir, ir2;
  mp::sph::invR(Dx * Dx + Dy * Dy + Dz * Dz, ir, ir2);
  double ux = 0.0, uy = 0.0, uz = 0.0;
  for (int k = 0; k < K; ++k) {
    const int n = mp::sph::c_kn_gs[k];
    const int m = mp::sph::c_km_gs[k];
    dcplx val, gx, gy, gz;
    mp::sph::irrValGrad<ORDER>(Dx, Dy, Dz, ir, ir2, n, m, val, gx, gy, gz);
    const dcplx M0 = {Macc[0][k][0], Macc[0][k][1]};
    const dcplx M1 = {Macc[1][k][0], Macc[1][k][1]};
    const dcplx M2 = {Macc[2][k][0], Macc[2][k][1]};
    const dcplx M3 = {Macc[3][k][0], Macc[3][k][1]};
    const dcplx H  = {M3.re - Dx * M0.re - Dy * M1.re - Dz * M2.re,
                      M3.im - Dx * M0.im - Dy * M1.im - Dz * M2.im};
    const double w = (m == 0) ? 1.0 : 2.0;
    ux += w * (M0.re * val.re - M0.im * val.im + H.re * gx.re - H.im * gx.im);
    uy += w * (M1.re * val.re - M1.im * val.im + H.re * gy.re - H.im * gy.im);
    uz += w * (M2.re * val.re - M2.im * val.im + H.re * gz.re - H.im * gz.im);
  }
  u_out[0] = ux * invA;  u_out[1] = uy * invA;  u_out[2] = uz * invA;
}

// Pack fp64 moments into the production fp32 NodeM2P.
__global__ void kPackNode(const double *MaccFlat, double a,
                          SphericalStokes::NodeM2P *node)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  auto Macc = reinterpret_cast<const double(*)[KM][2]>(MaccFlat);
  node->ownerMin = 0; node->ownerMax = 0;
  node->scale = (float)a; node->pad0 = 0.f;
  for (int k = 0; k < KM; ++k)
    for (int c = 0; c < 4; ++c) {
      node->M[k][c][0] = Macc[c][k][0];
      node->M[k][c][1] = Macc[c][k][1];
    }
}

template<int ORDER>
__global__ void kM2PSerial(const SphericalStokes::NodeM2P *node,
                           float3 center, float3 T, double *u_out)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const vec3d u = SphericalStokes::m2p<ORDER>(*node, vec3f(center.x, center.y, center.z),
                                              vec3f(T.x, T.y, T.z));
  u_out[0] = u.x;  u_out[1] = u.y;  u_out[2] = u.z;
}

template<int ORDER>
__global__ void kM2PWarp(const SphericalStokes::NodeM2P *node,
                         float3 center, float3 T, double *u_out)
{
  const int lane = threadIdx.x;
  const vec3d u = SphericalStokes::m2pWarp<ORDER>(*node, vec3f(center.x, center.y, center.z),
                                                  vec3f(T.x, T.y, T.z),
                                                  lane, 0xffffffffu);
  if (lane == 0) { u_out[0] = u.x;  u_out[1] = u.y;  u_out[2] = u.z; }
}

// Child -> parent moment translation (single thread), fp64 in/out; K-sized
// locals as in the production path, KM-strided staging for the host.
template<int ORDER>
__global__ void kM2M(const double *McFlat, double aChild, double3 cChild,
                     double3 cParent, double aParent, double *MpFlat)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  constexpr int K = kdimOf(ORDER);
  auto McIn = reinterpret_cast<const double(*)[KM][2]>(McFlat);
  double Mc[4][K][2], Mp[4][K][2];
  for (int c = 0; c < 4; ++c)
    for (int k = 0; k < K; ++k) {
      Mc[c][k][0] = McIn[c][k][0];
      Mc[c][k][1] = McIn[c][k][1];
      Mp[c][k][0] = Mp[c][k][1] = 0.0;
    }
  const double cc[3] = {cChild.x, cChild.y, cChild.z};
  const double pc[3] = {cParent.x, cParent.y, cParent.z};
  mp::sph::m2mTranslate<ORDER>(Mc, aChild, cc, pc, aParent, Mp);
  auto out = reinterpret_cast<double(*)[KM][2]>(MpFlat);
  for (int c = 0; c < 4; ++c)
    for (int k = 0; k < KM; ++k) {
      out[c][k][0] = (k < K) ? Mp[c][k][0] : 0.0;
      out[c][k][1] = (k < K) ? Mp[c][k][1] : 0.0;
    }
}

// ---------------------------------------------------------------------------
// Host-side order dispatch (runtime p -> compile-time template).
// ---------------------------------------------------------------------------
#define SPH_DISPATCH(p, CALL)                                             \
  switch (p) {                                                            \
    case 1: { constexpr int P = 1; CALL; } break;                         \
    case 2: { constexpr int P = 2; CALL; } break;                         \
    case 3: { constexpr int P = 3; CALL; } break;                         \
    case 4: { constexpr int P = 4; CALL; } break;                         \
    case 5: { constexpr int P = 5; CALL; } break;                         \
    case 6: { constexpr int P = 6; CALL; } break;                         \
    case 7: { constexpr int P = 7; CALL; } break;                         \
    case 8: { constexpr int P = 8; CALL; } break;                         \
    case 9: { constexpr int P = 9; CALL; } break;                         \
    case 10: { constexpr int P = 10; CALL; } break;                       \
    case 11: { constexpr int P = 11; CALL; } break;                       \
    case 12: { constexpr int P = 12; CALL; } break;                       \
    default: fprintf(stderr, "bad order %d\n", p); exit(2);               \
  }

// Label for the moment-storage error column. Storage is fp64; the column also
// carries fp32-geometry error from the fp32 centers/targets fed to m2p/m2pWarp
// (the diagnostic for the geometry floor).
static constexpr const char *STORE_LABEL = "fp64store";

// ---------------------------------------------------------------------------
// Host helpers.
// ---------------------------------------------------------------------------
static void directStokeslet(const std::vector<double3> &src,
                            const std::vector<double3> &frc,
                            double3 T, double u[3])
{
  u[0] = u[1] = u[2] = 0.0;
  for (size_t i = 0; i < src.size(); ++i) {
    const double rx = T.x - src[i].x, ry = T.y - src[i].y, rz = T.z - src[i].z;
    const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
    const double rdf = rx * frc[i].x + ry * frc[i].y + rz * frc[i].z;
    const double ir = 1.0 / r, ir3 = ir * ir * ir;
    u[0] += INV8PI * (frc[i].x * ir + rx * rdf * ir3);
    u[1] += INV8PI * (frc[i].y * ir + ry * rdf * ir3);
    u[2] += INV8PI * (frc[i].z * ir + rz * rdf * ir3);
  }
}

static double nodeScale(const std::vector<double3> &pts, double3 c)
{
  double r = 0.0;
  for (const auto &p : pts) {
    const double dx = p.x - c.x, dy = p.y - c.y, dz = p.z - c.z;
    r = std::max(r, std::sqrt(dx * dx + dy * dy + dz * dz));
  }
  return r > 0.0 ? r : 1.0;
}

static double relErr3(const double a[3], const double b[3])
{
  double dn = 0.0, nn = 0.0;
  for (int i = 0; i < 3; ++i) {
    dn += (a[i] - b[i]) * (a[i] - b[i]);
    nn += b[i] * b[i];
  }
  return std::sqrt(dn / nn);
}

template<class T>
static T *devAlloc(size_t n)
{
  T *p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

// ---------------------------------------------------------------------------
// (a) irregular closed forms; (b) ladder gradient vs finite differences.
// ---------------------------------------------------------------------------
static bool testIrregularBasis()
{
  bool ok = true;
  const double3 v = {1.3, -0.7, 2.1};
  const double r = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  constexpr int P = 8;
  const int K = kdimOf(P);

  double *dVal = devAlloc<double>(2 * KM), *dGrad = devAlloc<double>(6 * KM);
  std::vector<double> val(2 * KM), grad(6 * KM);

  auto evalAt = [&](double3 x, std::vector<double> &v2, std::vector<double> &g6) {
    kIrrEval<P><<<2, 32>>>(x, K, dVal, dGrad);
    CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(v2.data(), dVal, 2 * KM * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g6.data(), dGrad, 6 * KM * sizeof(double), cudaMemcpyDeviceToHost));
  };
  evalAt(v, val, grad);

  // (a) closed forms (demo _selftest): I[0,0]=1/r, I[1,0]=z/r^3,
  //     I[2,0]=(3z^2-r^2)/(2 r^5). All m=0 -> purely real.
  const double ref[3] = {1.0 / r, v.z / std::pow(r, 3),
                         (3.0 * v.z * v.z - r * r) / (2.0 * std::pow(r, 5))};
  const int kk[3] = {kOf(0, 0), kOf(1, 0), kOf(2, 0)};
  double worstA = 0.0;
  for (int i = 0; i < 3; ++i) {
    worstA = std::max(worstA, std::abs(val[2 * kk[i]] - ref[i]) / std::abs(ref[i]));
    worstA = std::max(worstA, std::abs(val[2 * kk[i] + 1]));
  }
  printf("[a] irregular closed forms: worst rel err %.3e  %s\n", worstA,
         worstA < 1e-11 ? "PASS" : "FAIL");
  ok = ok && (worstA < 1e-11);

  // (b) analytic ladder gradient vs central finite differences of val.
  const double h = 1e-6;
  double worstB = 0.0;
  std::vector<double> vp(2 * KM), vm(2 * KM), gtmp(6 * KM);
  for (int axis = 0; axis < 3; ++axis) {
    double3 xp = v, xm = v;
    (axis == 0 ? xp.x : axis == 1 ? xp.y : xp.z) += h;
    (axis == 0 ? xm.x : axis == 1 ? xm.y : xm.z) -= h;
    evalAt(xp, vp, gtmp);
    evalAt(xm, vm, gtmp);
    for (int k = 0; k < K; ++k)
      for (int ri = 0; ri < 2; ++ri) {
        const double fd = (vp[2 * k + ri] - vm[2 * k + ri]) / (2.0 * h);
        const double an = grad[6 * k + 2 * axis + ri];
        worstB = std::max(worstB, std::abs(an - fd));
      }
  }
  printf("[b] ladder gradient vs FD (p=%d): worst abs err %.3e  %s\n", P, worstB,
         worstB < 1e-6 ? "PASS" : "FAIL");
  ok = ok && (worstB < 1e-6);

  CUDA_CHECK(cudaFree(dVal));
  CUDA_CHECK(cudaFree(dGrad));
  return ok;
}

// ---------------------------------------------------------------------------
// (c) P2M + M2P vs direct Stokeslet + (e) m2pWarp vs serial m2p.
// ---------------------------------------------------------------------------
static bool testP2MM2P()
{
  bool ok = true;
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> uPos(-0.5, 0.5), uFrc(-1.0, 1.0);
  const int N = 200;
  std::vector<double3> src(N), frc(N);
  double3 mean = {0, 0, 0};
  for (int i = 0; i < N; ++i) {
    src[i] = {uPos(rng), uPos(rng), uPos(rng)};
    frc[i] = {uFrc(rng), uFrc(rng), uFrc(rng)};
    mean.x += src[i].x / N; mean.y += src[i].y / N; mean.z += src[i].z / N;
  }
  const double3 T = {3.0, -2.0, 4.0};
  double uDir[3];
  directStokeslet(src, frc, T, uDir);

  double3 *dSrc = devAlloc<double3>(N), *dFrc = devAlloc<double3>(N);
  CUDA_CHECK(cudaMemcpy(dSrc, src.data(), N * sizeof(double3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dFrc, frc.data(), N * sizeof(double3), cudaMemcpyHostToDevice));
  double *dMacc = devAlloc<double>(4 * KM * 2);
  double *dU = devAlloc<double>(3);
  auto *dNode = devAlloc<SphericalStokes::NodeM2P>(1);

  auto runPipeline = [&](double3 c, int p, double u64[3], double u32s[3],
                         double u32w[3]) {
    const double a = nodeScale(src, c);
    SPH_DISPATCH(p, (kP2M<P><<<1, 1>>>(dSrc, dFrc, N, c, a, dMacc)));
    CHECK_LAUNCH();
    SPH_DISPATCH(p, (kM2P64<P><<<1, 1>>>(dMacc, c, a, T, dU)));
    CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(u64, dU, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    kPackNode<<<1, 1>>>(dMacc, a, dNode);
    CHECK_LAUNCH();
    const float3 cf = {(float)c.x, (float)c.y, (float)c.z};
    const float3 Tf = {(float)T.x, (float)T.y, (float)T.z};
    SPH_DISPATCH(p, (kM2PSerial<P><<<1, 1>>>(dNode, cf, Tf, dU)));
    CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(u32s, dU, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    SPH_DISPATCH(p, (kM2PWarp<P><<<1, 32>>>(dNode, cf, Tf, dU)));
    CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(u32w, dU, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    // outputs are unscaled: apply the Stokeslet prefactor (exact fp64 here)
    for (int i = 0; i < 3; ++i) {
      u64[i] *= INV8PI; u32s[i] *= INV8PI; u32w[i] *= INV8PI;
    }
  };

  // Order sweep at the demo's box center (source mean).
  printf("[c] P2M+M2P vs direct (N=%d, target dist %.2f):\n", N,
         std::sqrt(std::pow(T.x - mean.x, 2) + std::pow(T.y - mean.y, 2) +
                   std::pow(T.z - mean.z, 2)));
  printf("     p |  fp64 relerr | %s relerr | warp-vs-serial\n", STORE_LABEL);
  double u64[3], u32s[3], u32w[3];
  double errP9 = 1.0, worstWS = 0.0, prevErr = 1.0;
  bool decays = true;
  for (int p = 2; p <= PMAX; ++p) {
    runPipeline(mean, p, u64, u32s, u32w);
    const double e64 = relErr3(u64, uDir);
    const double e32 = relErr3(u32s, uDir);
    const double ews = relErr3(u32w, u32s);            // (e) same moments
    worstWS = std::max(worstWS, ews);
    printf("     %d |  %.3e   | %.3e  | %.3e\n", p, e64, e32, ews);
    if (p >= 4 && e64 > prevErr * 2.0) decays = false; // fp64 must keep decaying
    prevErr = e64;
    if (p == PMAX) errP9 = e64;
  }
  const bool cPass = decays && errP9 < 1e-7;
  printf("[c] order sweep: p=%d fp64 relerr %.3e (<1e-7), decay %s  %s\n",
         PMAX, errP9, decays ? "ok" : "BROKEN", cPass ? "PASS" : "FAIL");
  ok = ok && cPass;

  // (e) warp vs serial on identical fp32 moments, all orders above.
  printf("[e] m2pWarp vs serial m2p: worst rel diff %.3e  %s\n", worstWS,
         worstWS < 1e-12 ? "PASS" : "FAIL");
  ok = ok && (worstWS < 1e-12);

  // Box-relative invariance (demo _selftest): different valid centers must
  // agree with direct at truncation level. p=9, three centers.
  double worstInv = 0.0, worstF32 = 0.0;
  const double3 centers[3] = {{0, 0, 0}, mean, {0.3, -0.2, 0.1}};
  for (const auto &c : centers) {
    runPipeline(c, PMAX, u64, u32s, u32w);
    worstInv = std::max(worstInv, relErr3(u64, uDir));
    worstF32 = std::max(worstF32, relErr3(u32s, uDir));
  }
  printf("[c] box-relative invariance (p=%d): fp64 worst %.3e (<1e-6), "
         "%s worst %.3e (<3e-6)  %s\n", PMAX, worstInv, STORE_LABEL, worstF32,
         (worstInv < 1e-6 && worstF32 < 3e-6) ? "PASS" : "FAIL");
  ok = ok && (worstInv < 1e-6) && (worstF32 < 3e-6);

  CUDA_CHECK(cudaFree(dSrc));  CUDA_CHECK(cudaFree(dFrc));
  CUDA_CHECK(cudaFree(dMacc)); CUDA_CHECK(cudaFree(dU));
  CUDA_CHECK(cudaFree(dNode));
  return ok;
}

// ---------------------------------------------------------------------------
// (d) M2M vs direct P2M at the parent (demo _test_m2m geometries).
// ---------------------------------------------------------------------------
static bool testM2M()
{
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> uPos(-0.4, 0.4), uFrc(-1.0, 1.0);
  const int N = 30, p = 8;
  const int K = kdimOf(p);
  const double3 pairs[2][2] = {{{1.4, -0.9, 0.6}, {0.2, 0.1, -0.3}},
                               {{-2.0, 0.5, 1.0}, {0.0, 0.0, 0.0}}};

  double3 *dSrc = devAlloc<double3>(N), *dFrc = devAlloc<double3>(N);
  double *dMc = devAlloc<double>(4 * KM * 2);
  double *dMpDirect = devAlloc<double>(4 * KM * 2);
  double *dMpM2M = devAlloc<double>(4 * KM * 2);
  std::vector<double> mpD(4 * KM * 2), mpT(4 * KM * 2);

  double worst = 0.0;
  for (int g = 0; g < 2; ++g) {
    const double3 cC = pairs[g][0], cP = pairs[g][1];
    std::vector<double3> src(N), frc(N);
    for (int i = 0; i < N; ++i) {
      src[i] = {cC.x + uPos(rng), cC.y + uPos(rng), cC.z + uPos(rng)};
      frc[i] = {uFrc(rng), uFrc(rng), uFrc(rng)};
    }
    CUDA_CHECK(cudaMemcpy(dSrc, src.data(), N * sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dFrc, frc.data(), N * sizeof(double3), cudaMemcpyHostToDevice));
    const double aC = nodeScale(src, cC), aP = nodeScale(src, cP);

    SPH_DISPATCH(p, (kP2M<P><<<1, 1>>>(dSrc, dFrc, N, cC, aC, dMc)));
    CHECK_LAUNCH();
    SPH_DISPATCH(p, (kP2M<P><<<1, 1>>>(dSrc, dFrc, N, cP, aP, dMpDirect)));
    CHECK_LAUNCH();
    SPH_DISPATCH(p, (kM2M<P><<<1, 1>>>(dMc, aC, cC, cP, aP, dMpM2M)));
    CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(mpD.data(), dMpDirect, mpD.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mpT.data(), dMpM2M, mpT.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (int c = 0; c < 4; ++c)
      for (int k = 0; k < K; ++k)
        for (int ri = 0; ri < 2; ++ri) {
          const size_t idx = ((size_t)c * KM + k) * 2 + ri;
          worst = std::max(worst, std::abs(mpT[idx] - mpD[idx]));
        }
  }
  printf("[d] M2M vs direct parent P2M (p=%d): worst abs moment err %.3e  %s\n",
         p, worst, worst < 1e-10 ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(dSrc));      CUDA_CHECK(cudaFree(dFrc));
  CUDA_CHECK(cudaFree(dMc));       CUDA_CHECK(cudaFree(dMpDirect));
  CUDA_CHECK(cudaFree(dMpM2M));
  return worst < 1e-10;
}

int main()
{
  SphericalStokes::setup(PMAX);
  bool ok = true;
  ok = testIrregularBasis() && ok;
  ok = testP2MM2P() && ok;
  ok = testM2M() && ok;
  printf(ok ? "sph_selftest: ALL PASS\n" : "sph_selftest: FAIL\n");
  return ok ? 0 : 1;
}
