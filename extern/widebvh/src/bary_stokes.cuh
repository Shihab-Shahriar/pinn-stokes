// SPDX-License-Identifier: Apache-2.0
//
// Multipole module: kernel-independent treecode (KITC) via barycentric Lagrange
// interpolation at Chebyshev points of the 2nd kind, specialized to the
// Stokeslet. Reference:
//   L. Wang, R. Krasny, S. Tlupova, "A kernel-independent treecode based on
//   barycentric Lagrange interpolation", 2019 (PDF in repo root).
//
// This is a *swappable* multipole module exposing the same surface as
// cartesian_stokes.cuh (Moments / MAX_ORDER / name / setup / upwardPass / m2p),
// so the generic treecode (traversal, MAC, near-field p2p) is reused unchanged;
// switch it in treecode.cu with one `using MP` line.
//
// Idea (kernel-independent): each cluster C is represented by a tensor-product
// grid of (PDEG+1)^3 Chebyshev "proxy" points s_k mapped to the node's bounding
// box, each carrying a 3-vector "modified weight" f_hat_k that replaces analytic
// multipole moments. Only kernel *evaluations* are needed -- no per-kernel
// analytic derivatives.
//   P2M  (leaf):  f_hat_k = sum_{j in C} L_k1(y1) L_k2(y2) L_k3(y3) f_j     (Alg.1)
//   M2M  (inner): re-interpolate each child's proxies onto the parent grid
//                 (exact for degree <= PDEG, so equals direct per-cluster P2M)
//   M2P  (eval):  u(x,C) = sum_k Stokeslet(x, s_k) . f_hat_k                 (eq.15)
//                 -- literally a direct Stokeslet sum over the proxy points.
//
// where the 1-D barycentric Lagrange basis (simple Chebyshev-2nd-kind weights
// w_k = (-1)^k delta_k, delta_k = 1/2 at the two endpoints else 1) is
//   L_k(y) = [w_k/(y - s_k)] / sum_j [w_j/(y - s_j)].
//
// Accuracy knob: PDEG (compile-time). The runtime `order` CLI arg is IGNORED by
// this module (kept only for interface compatibility with the driver's order
// dispatch). Bump PDEG below and recompile to trade speed/memory for accuracy.
//
// Precision: fp32 for proxy positions / modified weights / M2P math, matching
// cartesian_stokes.cuh; the per-target velocity is accumulated in fp64 by the
// caller (treecode.cu).
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda/refit_aggregate.h"
#include "refit_aggregate_warp.cuh"
#include "stokes_kernel.cuh"     // stokes::FP32_LEVEL, accumStokesFactored
#include "struct_recon.cuh"

// Defer to the includer's CUDA_CHECK (treecode.cu has a richer file:line one);
// fall back to a self-contained version so this header stands alone. #ifndef
// makes it clash-free regardless of include order.
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess)                                                \
      throw std::runtime_error(std::string("CUDA error: ") +             \
                               cudaGetErrorString(_e));                   \
  } while (0)
#endif

#ifndef BARY_STOKES_USE_SCALAR_REFIT
#define BARY_STOKES_USE_SCALAR_REFIT 0
#endif

namespace mp {

using cuBQL::vec3f;
using cuBQL::vec3d;
using cuBQL::bvh3f;

// ---------------------------------------------------------------------------
//  Barycentric-Lagrange internals. Wrapped in a nested namespace so the
//  __constant__/__device__ globals and helpers don't collide with the symbols
//  of cartesian_stokes.cuh (both headers are included by treecode.cu).
// ---------------------------------------------------------------------------
namespace bary {

// Polynomial interpolation degree (compile-time). NP = nodes per dimension.
// Accuracy knob -- measured rel-L2 vs direct sum on two_ball t=50, mac=0.5:
//   PDEG=3 -> 1.5e-4   PDEG=5 -> 1.8e-6   PDEG=7 -> 8e-8 (near the fp32 floor).
// M2M uses separable tensor-product contraction: O(3*(PDEG+1)^4/node).
// M2P is (PDEG+1)^3 evals/node (Stokeslet kernel, not separable).
//
// Overridable at build time with -DWIDEBVH_PDEG=<n> so one source tree can
// produce several degree variants (see add_widebvh_nemo_library in
// CMakeLists.txt). Every pre-existing target leaves it undefined and therefore
// keeps degree 7 exactly as before.
#ifndef WIDEBVH_PDEG
#define WIDEBVH_PDEG 7
#endif
constexpr int PDEG = WIDEBVH_PDEG;
constexpr int NP   = PDEG + 1;
constexpr int NP3  = NP * NP * NP;        // proxy points per cluster

// Chebyshev-2nd-kind nodes in [-1,1] and barycentric weights; set by setup().
__constant__ double c_cheb[NP];
__constant__ float  c_chebf[NP];   // fp32 copy (used by the fp32 upward pass)
__constant__ double c_baryw[NP];
#if WIDEBVH_FP32_LEVEL >= 3
__constant__ float  c_barywf[NP];  // fp32 copy for the fp32 upward pass
#endif

// Precomputed (k1,k2,k3) index tables for flat proxy index s = (k1*NP+k2)*NP+k3.
// Avoids integer % and / by NP (=7, non-power-of-2) in the hot M2P warp loop.
// Stored in device global (NOT __constant__) so divergent warp reads coalesce
// through L2/texture cache instead of serializing on the constant-memory bus.
__device__ unsigned char c_k1[NP3];
__device__ unsigned char c_k2[NP3];
__device__ unsigned char c_k3[NP3];

// Fully expanded per-axis Chebyshev offsets for flat proxy index s:
// c_chebxy[s] = (c_cheb[k1(s)], c_cheb[k2(s)]), c_chebz[s] = c_cheb[k3(s)].
// m2pWarp's per-lane s is divergent, and c_cheb[c_k1[s]] through __constant__
// serializes on distinct addresses (up to NP-way replay per load); these
// __device__ copies turn that into coalesced L1 loads (one LDG.128 + one
// LDG.64 per point). Values are the exact same doubles (bit-identical
// results). (A packed fp32 float4 variant was tried and measured ~1-2%
// SLOWER: the three F2F.F64 widening converts cost more than the 8B/point of
// L1 traffic saved.)
__device__ double2 c_chebxy[NP3];
__device__ double  c_chebz[NP3];

#if WIDEBVH_FP32_LEVEL >= 1
// fp32 twin of (c_chebxy, c_chebz) for the fp32 M2P: one float4 per proxy
// point (x, y, z, 0), i.e. a single LDG.128 and no F2F converts. (The float4
// experiment noted above lost 1-2% on the fp64 path because of the widening
// converts; on the fp32 path there is nothing to widen.)
__device__ float4 c_chebf4[NP3];
#endif

// Scene globals smuggled into combine() during the upward pass (combine()'s
// signature is fixed by refit_aggregate). Renamed vs. cartesian_stokes.cuh's
// d_pos_g/... to avoid duplicate-symbol clashes when both headers are included.
__device__ const vec3f *d_pos_gb          = nullptr;
__device__ const vec3d *d_force_gb        = nullptr;
__device__ const int   *d_owner_gb        = nullptr;
__device__ uint32_t    *d_owner_mask_gb   = nullptr;
__device__ int          d_owner_mask_words_gb = 0;
__device__ const int   *d_bucket_begin_gb = nullptr;
__device__ const int   *d_bucket_end_gb   = nullptr;
// Per-BVH-node total source-particle count, summed bottom-up during the refit
// (leaf = sum of its bucket spans; internal = sum of children). Written only when
// non-null (the treecode sets it via BaryStokes::setNodeCountSymbol when the
// P2P/multipole crossover TC_XOVER is enabled). Read at classify-accept time to
// route small nodes to exact P2P instead of the (PDEG+1)^3-point M2P.
__device__ int         *d_node_count_gb   = nullptr;

// Structured (monodisperse) source: when d_struct_tmpl_gb is non-null the P2M
// leaf loops reconstruct each source position from the shared template + the
// object's transform instead of reading d_pos_gb (which is empty in that mode).
// Set by BaryStokes::setStructuredSourceSymbols before the upward pass; null =>
// byte-identical legacy path. In object mode bucket id == object id, and the
// bucket's local index (pid - bucketBegin) is the template index.
__device__ const vec3d  *d_struct_tmpl_gb   = nullptr;   // template (nPts), ref frame
__device__ const double *d_struct_R_gb      = nullptr;   // nObj*9 row-major
__device__ const vec3d  *d_struct_center_gb = nullptr;   // nObj shifted centers

__device__ __host__ __forceinline__ int proxyIdx(int k1, int k2, int k3)
{
  return (k1 * NP + k2) * NP + k3;
}

// 1-D barycentric Lagrange basis L[0..PDEG] for coordinate y on an axis with box
// center c_d and half-extent h_d. Core of Algorithm 1. Two fp32 guards (without
// them the common tight-AABB leaves produce NaNs):
//   * collapsed axis (h_d == 0): a 1-particle / axis-flat bucket makes every node
//     s_k coincide -> div blowup. Collapse to L[0]=1 (degree-0 in that axis).
//   * exact hit (|y - s_k| <= tol): set L_k=1, rest 0. The tight box always has a
//     particle on each face, sitting on the endpoint nodes s_0,s_PDEG, so this
//     fires routinely.
__device__ __forceinline__ void
lagrange1d(double y, double c_d, double h_d, double L[NP])
{
  if (!(h_d > 1e-20)) {                 // collapsed axis
    L[0] = 1.0;
#pragma unroll
    for (int k = 1; k < NP; ++k) L[k] = 0.0;
    return;
  }
  const double tol = 1e-6 * h_d;
  double s[NP];
  int hit = -1;
#pragma unroll
  for (int k = 0; k < NP; ++k) {
    s[k] = c_d + h_d * c_cheb[k];
    if (fabs(y - s[k]) <= tol) hit = k;
  }
  if (hit >= 0) {
#pragma unroll
    for (int k = 0; k < NP; ++k) L[k] = (k == hit) ? 1.0 : 0.0;
    return;
  }
  double sum = 0.0;
  double dL[NP];
#pragma unroll
  for (int k = 0; k < NP; ++k) {
    const double ck = c_baryw[k] / (y - s[k]);
    dL[k] = ck;
    sum += ck;
  }
  const double inv = 1.0 / sum;
#pragma unroll
  for (int k = 0; k < NP; ++k) {
    L[k] = dL[k] * inv;
  }
}

#if WIDEBVH_FP32_LEVEL >= 3
// fp32 twin of lagrange1d for the WIDEBVH_FP32_LEVEL >= 3 upward pass, in
// CENTRED coordinates: the caller passes d = y - c_d (the fp32 difference of
// two fp32 coordinates a box apart is exact by Sterbenz), so the node offsets
// d - h_d*s_k carry ~ulp(h_d) instead of ~ulp(|c_d|) of error -- the same
// reason m2pWarp's fp32 body forms Tp - center first. Same two guards.
__device__ __forceinline__ void
lagrange1d_f(float d, float h_d, float L[NP])
{
  if (!(h_d > 1e-20f)) {                // collapsed axis
    L[0] = 1.f;
#pragma unroll
    for (int k = 1; k < NP; ++k) L[k] = 0.f;
    return;
  }
  const float tol = 1e-6f * h_d;
  float ds[NP];
  int hit = -1;
#pragma unroll
  for (int k = 0; k < NP; ++k) {
    ds[k] = fmaf(-h_d, c_chebf[k], d);   // y - s_k
    if (fabsf(ds[k]) <= tol) hit = k;
  }
  if (hit >= 0) {
#pragma unroll
    for (int k = 0; k < NP; ++k) L[k] = (k == hit) ? 1.f : 0.f;
    return;
  }
  float sum = 0.f;
  float dL[NP];
#pragma unroll
  for (int k = 0; k < NP; ++k) {
    const float ck = c_barywf[k] / ds[k];
    dL[k] = ck;
    sum += ck;
  }
  const float inv = 1.f / sum;
#pragma unroll
  for (int k = 0; k < NP; ++k) {
    L[k] = dL[k] * inv;
  }
}
#endif  // WIDEBVH_FP32_LEVEL >= 3

} // namespace bary

// ===========================================================================
//  The policy.
// ===========================================================================
struct BaryStokes {
  static constexpr int MAX_ORDER = 7;
  // 343 proxies per node: the warp-cooperative m2pWarp already saturates the
  // lanes, so the split-warpspec consumers keep the one-item-per-warp pop.
  static constexpr bool LANE_BATCH_M2P = false;
  // Keeps the fp32 target path (only SphericalStokes uses fp64-geometry M2P).
  static constexpr bool WANTS_FP64_TARGET = false;

  // Type of one node's far-field contribution as returned by m2pWarp: fp64 for
  // the production kernels, fp32 on the WIDEBVH_FP32_LEVEL >= 1 path where the
  // whole evaluation and the per-target accumulation stay in fp32 (see
  // stokes_kernel.cuh). The engine's traversal kernels declare their per-node
  // temporaries with this alias.
  using FarVec = std::conditional_t<(stokes::FP32_LEVEL >= 1), vec3f, vec3d>;

  // SoA node types. Traversal loads only NodeMAC (16 bytes) per node visit;
  // NodeM2P (box half-extents + proxy weights) is loaded only when MAC accepts.
  struct __align__(16) NodeMAC {
    float cx, cy, cz, halfDiag2;
  };

  struct NodeM2P {
    float hx, hy, hz, pad;
    int ownerMin, ownerMax, ownerPad0, ownerPad1;
    // fp32 moments: upward pass accumulates in fp64 and casts on the final
    // store; M2P evaluates in fp32 with fp64 accumulation (same contract as
    // the fp32 P2P in stokes_kernel.cuh). Halves the per-node moment bytes.
    float fhat[bary::NP3][3];
  };

  static const char *name() { return "barycentric-Lagrange(KITC)"; }

  static void setup(int order);

  // Point the refit's per-node source-count output at `p` (null disables the
  // write). Set by the treecode before the upward pass when TC_XOVER > 0 so the
  // P2P/multipole crossover has a per-node count to threshold on at traversal.
  static void setNodeCountSymbol(int *p)
  {
    CUDA_CHECK(cudaMemcpyToSymbol(bary::d_node_count_gb, &p, sizeof(p)));
  }

  // Point the P2M leaf loops at the structured source template + per-object
  // transforms (all null => legacy, byte-identical). Set by the treecode's
  // upward pass when TC_SRC_TEMPLATE is active.
  static void setStructuredSourceSymbols(const vec3d *tmpl, const double *R,
                                         const vec3d *center)
  {
    CUDA_CHECK(cudaMemcpyToSymbol(bary::d_struct_tmpl_gb,   &tmpl,   sizeof(tmpl)));
    CUDA_CHECK(cudaMemcpyToSymbol(bary::d_struct_R_gb,      &R,      sizeof(R)));
    CUDA_CHECK(cudaMemcpyToSymbol(bary::d_struct_center_gb, &center, sizeof(center)));
  }

  template<int ORDER>
  static __device__ inline vec3d m2p(const NodeM2P &m2p_data,
                                     vec3f center, vec3f Tp);

  template<int ORDER>
  static __device__ inline FarVec m2pWarp(const NodeM2P &m2p_data,
                                          vec3f center,
                                          vec3f Tp, int lane,
                                          unsigned int mask);

  // Traction far field from the SAME Stokeslet proxy moments (KAFMM-style
  // kernel aggregation): the fhat are kernel-independent equivalent strengths,
  // so the target-normal-contracted traction sum over the proxies approximates
  // the far-field traction operator. `nrm` is the outward normal at Tp.
  template<int ORDER>
  static __device__ inline vec3d tractionWarp(const NodeM2P &m2p_data,
                                              vec3f center, vec3f Tp,
                                              vec3d nrm, int lane,
                                              unsigned int mask);

  // Reduction-free body of tractionWarp: each lane accumulates its strided
  // share of the proxy sum into t0..t2. The skel traction eval kernel calls
  // this once per accepted node and shuffle-reduces ONCE after the node loop
  // instead of per node (the per-node reduce costs 45 SHFL + 45 DADD, ~11% of
  // that kernel's FP64 work; summation order changes at round-off only).
  template<int ORDER>
  static __device__ inline void tractionWarpAccum(const NodeM2P &m2p_data,
                                                  vec3f center, vec3f Tp,
                                                  vec3d nrm, int lane,
                                                  double &t0, double &t1,
                                                  double &t2);

  static void upwardPass(bvh3f bvh, NodeMAC *d_mac, NodeM2P *d_m2p,
                         const vec3f *d_pos, const vec3d *d_force,
                         const int *d_owner, uint32_t *d_ownerMask,
                         int ownerMaskWords,
                         const int *d_bucketBegin, const int *d_bucketEnd,
                         int order, cudaStream_t stream = 0);

  static __device__ void combine(bvh3f bvh, NodeMAC mac[], int nodeID);
};

__device__ BaryStokes::NodeM2P *d_m2p_gb = nullptr;

__device__ void (*BaryStokes_combine_fp)(bvh3f, BaryStokes::NodeMAC[], int)
    = &BaryStokes::combine;

// ===========================================================================
//  Definitions.
// ===========================================================================

inline void BaryStokes::setup(int /*order*/)
{
  double cheb[bary::NP];
  double w[bary::NP];
  for (int k = 0; k < bary::NP; ++k) {
    cheb[k] = std::cos(M_PI * (double)k / (double)bary::PDEG);
    const double dk = (k == 0 || k == bary::PDEG) ? 0.5 : 1.0;
    w[k] = ((k & 1) ? -1.0 : 1.0) * dk;
  }
  float chebf[bary::NP];
  for (int k = 0; k < bary::NP; ++k) chebf[k] = (float)cheb[k];
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_cheb,  cheb,  sizeof(cheb)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_chebf, chebf, sizeof(chebf)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_baryw, w,     sizeof(w)));
#if WIDEBVH_FP32_LEVEL >= 3
  float wf[bary::NP];
  for (int k = 0; k < bary::NP; ++k) wf[k] = (float)w[k];
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_barywf, wf, sizeof(wf)));
#endif

  unsigned char k1tab[bary::NP3], k2tab[bary::NP3], k3tab[bary::NP3];
  for (int s = 0; s < bary::NP3; ++s) {
    int tmp = s;
    k3tab[s] = tmp % bary::NP; tmp /= bary::NP;
    k2tab[s] = tmp % bary::NP; tmp /= bary::NP;
    k1tab[s] = tmp;
  }
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_k1, k1tab, sizeof(k1tab)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_k2, k2tab, sizeof(k2tab)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_k3, k3tab, sizeof(k3tab)));

  double2 chebxy[bary::NP3];
  double chebz[bary::NP3];
  for (int s = 0; s < bary::NP3; ++s) {
    chebxy[s] = make_double2(cheb[k1tab[s]], cheb[k2tab[s]]);
    chebz[s]  = cheb[k3tab[s]];
  }
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_chebxy, chebxy, sizeof(chebxy)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_chebz,  chebz,  sizeof(chebz)));

#if WIDEBVH_FP32_LEVEL >= 1
  float4 chebf4[bary::NP3];
  for (int s = 0; s < bary::NP3; ++s)
    chebf4[s] = make_float4(chebf[k1tab[s]], chebf[k2tab[s]], chebf[k3tab[s]], 0.f);
  CUDA_CHECK(cudaMemcpyToSymbol(bary::c_chebf4, chebf4, sizeof(chebf4)));
#endif
}

// Stokeslet far field: u = sum_k ( f_hat_k / r + R (R . f_hat_k) / r^3 ),
// R = Tp - s_k, over the (PDEG+1)^3 proxy points. Same kernel as
// stokes_kernel.cuh::p2p, unscaled (caller multiplies by the prefactor).
template<int ORDER>
__device__ inline vec3d
BaryStokes::m2p(const NodeM2P &m2p_data, vec3f center, vec3f Tp)
{
  (void)ORDER;
  using namespace bary;

  // fp64 geometry + rsqrtf-seeded fp64 Newton: R and r2 in fp64 (DFMAs),
  // 1/sqrt via fp32 rsqrtf seed refined by ONE fp64 Newton step (rel err
  // ~1.5*(2^-22)^2 ~ 1e-13; no fp64 rsqrt/sqrt/div software chains), fp64
  // downstream. Moments stay fp32 in memory, widened on load.
  double gx[NP], gy[NP], gz[NP];
#pragma unroll
  for (int k = 0; k < NP; ++k) {
    gx[k] = (double)center.x + (double)m2p_data.hx * c_cheb[k];
    gy[k] = (double)center.y + (double)m2p_data.hy * c_cheb[k];
    gz[k] = (double)center.z + (double)m2p_data.hz * c_cheb[k];
  }

  double u0 = 0.0, u1 = 0.0, u2 = 0.0;
  for (int k1 = 0; k1 < NP; ++k1) {
    const double Rx = (double)Tp.x - gx[k1];
    for (int k2 = 0; k2 < NP; ++k2) {
      const double Ry = (double)Tp.y - gy[k2];
      for (int k3 = 0; k3 < NP; ++k3) {
        const double Rz = (double)Tp.z - gz[k3];
        const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
        if (r2 == 0.0) continue;
        double ir = (double)rsqrtf((float)r2);
        ir = ir * fma(-0.5 * r2, ir * ir, 1.5);
        const double ir3 = (ir * ir) * ir;
        const int s = proxyIdx(k1, k2, k3);
        const double fx = m2p_data.fhat[s][0], fy = m2p_data.fhat[s][1], fz = m2p_data.fhat[s][2];
        const double rdf = Rx * fx + Ry * fy + Rz * fz;
        // Expanded form (this evaluator, unlike m2pWarp, already has ir3 in
        // hand). With RPY on, the same two coefficients as
        // stokes::accumStokesFactored, written against ir3 instead of ir:
        //   A = 1/r + c/r^3,  B = 1/r^3 - 3c/r^5.
        if constexpr (!stokes::RPY_ON) {
          u0 += fx * ir + Rx * rdf * ir3;
          u1 += fy * ir + Ry * rdf * ir3;
          u2 += fz * ir + Rz * rdf * ir3;
        } else {
          const double A    = fma(stokes::RPY_C, ir3, ir);
          const double B    = fma(-3.0 * stokes::RPY_C, ir3 * (ir * ir), ir3);
          const double Brdf = B * rdf;
          u0 += fx * A + Rx * Brdf;
          u1 += fy * A + Ry * Brdf;
          u2 += fz * A + Rz * Brdf;
        }
      }
    }
  }
  return vec3d(u0, u1, u2);
}

template<int ORDER>
__device__ inline BaryStokes::FarVec
BaryStokes::m2pWarp(const NodeM2P &m2p_data, vec3f center, vec3f Tp,
                    int lane, unsigned int mask)
{
  (void)ORDER;
  using namespace bary;

#if WIDEBVH_FP32_LEVEL >= 1
  // ALL-fp32 evaluation (WIDEBVH_FP32_LEVEL >= 1, see stokes_kernel.cuh).
  // Geometry: the target and the node centre are fp32 already, so the fp32
  // difference carries only their own quantization (~ulp of the box extent);
  // an accepted node sits at r > halfDiag/mac >> that, so the relative error in
  // R is ~1e-6 at NeMO's operating point. 1/r from rsqrtf alone (~2 ulp): a
  // Newton step would only be worth it if the fp32 arithmetic error were
  // visible against the expansion's truncation error, and it is not
  // (WIDEBVH_FP32_M2P_NR=1 adds one fp32 NR step for measurement).
  // Accumulation: fp32 per lane over its ~NP3/32 proxies, fp32 shuffle
  // reduce; the caller keeps accumulating in fp32 per target and widens once.
  const float dx = Tp.x - center.x;
  const float dy = Tp.y - center.y;
  const float dz = Tp.z - center.z;
  const float hx = m2p_data.hx, hy = m2p_data.hy, hz = m2p_data.hz;

  float u0 = 0.f, u1 = 0.f, u2 = 0.f;
  for (int s = lane; s < NP3; s += 32) {
    const float4 c = __ldg(&c_chebf4[s]);
    const float Rx = fmaf(-hx, c.x, dx);
    const float Ry = fmaf(-hy, c.y, dy);
    const float Rz = fmaf(-hz, c.z, dz);
    const float r2 = fmaf(Rx, Rx, fmaf(Ry, Ry, Rz * Rz));
    if (r2 == 0.f) continue;
    float ir = rsqrtf(r2);
#if defined(WIDEBVH_FP32_M2P_NR) && WIDEBVH_FP32_M2P_NR
    ir = ir * fmaf(-0.5f * r2, ir * ir, 1.5f);
#endif
    const float fx = m2p_data.fhat[s][0];
    const float fy = m2p_data.fhat[s][1];
    const float fz = m2p_data.fhat[s][2];
    const float rdf = fmaf(Rx, fx, fmaf(Ry, fy, Rz * fz));
    const float q   = rdf * (ir * ir);
    stokes::accumStokesFactored(ir, q, Rx, Ry, Rz, fx, fy, fz, u0, u1, u2);
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    u0 += __shfl_down_sync(mask, u0, offset);
    u1 += __shfl_down_sync(mask, u1, offset);
    u2 += __shfl_down_sync(mask, u2, offset);
  }
  return (lane == 0) ? vec3f(u0, u1, u2) : vec3f(0.f, 0.f, 0.f);
#else
  // fp64 geometry + rsqrtf-seeded fp64 Newton (see m2p()): no fp64
  // rsqrt/sqrt/div software chains; moments stay fp32, widened on load.
  const double dx = (double)Tp.x - (double)center.x;   // exact: fp32 inputs
  const double dy = (double)Tp.y - (double)center.y;
  const double dz = (double)Tp.z - (double)center.z;
  const double hx = m2p_data.hx, hy = m2p_data.hy, hz = m2p_data.hz;

  double u0 = 0.0, u1 = 0.0, u2 = 0.0;
  for (int s = lane; s < NP3; s += 32) {
    const double2 cxy = __ldg(&c_chebxy[s]);
    const double Rx = fma(-hx, cxy.x, dx);
    const double Ry = fma(-hy, cxy.y, dy);
    const double Rz = fma(-hz, __ldg(&c_chebz[s]), dz);
    const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
    if (r2 == 0.0) continue;
    double ir = (double)rsqrtf((float)r2);
    ir = ir * fma(-0.5 * r2, ir * ir, 1.5);
    const double fx = m2p_data.fhat[s][0];
    const double fy = m2p_data.fhat[s][1];
    const double fz = m2p_data.fhat[s][2];
    const double rdf = Rx * fx + Ry * fy + Rz * fz;
    // Factored form u_i += ir*(f_i + R_i*q), q=(R.f)/r^2 -- same algebra as
    // stokes::p2p (round-off-level difference only), ~3 fewer fp64 ops/point
    // than the expanded f_i*ir + R_i*rdf*ir3. Shared with p2p so that the
    // optional RPY regularization lands on the far field too: this loop IS the
    // far-field kernel evaluation, over the (PDEG+1)^3 proxy points.
    const double q = rdf * (ir * ir);
    stokes::accumStokesFactored(ir, q, Rx, Ry, Rz, fx, fy, fz, u0, u1, u2);
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    u0 += __shfl_down_sync(mask, u0, offset);
    u1 += __shfl_down_sync(mask, u1, offset);
    u2 += __shfl_down_sync(mask, u2, offset);
  }
  return (lane == 0) ? vec3d(u0, u1, u2) : vec3d(0.0, 0.0, 0.0);
#endif  // WIDEBVH_FP32_LEVEL >= 1
}

// Line-for-line sibling of m2pWarp with the Oseen accumulation replaced by the
// target-normal-contracted traction t_i += R_i*(R.f)(R.n)/r^5 (unscaled; the
// caller applies stokes::tractionPrefactor() = -3/(4*pi) once). Same fp64
// geometry, rsqrtf-seeded Newton 1/r, and shfl reduction.
template<int ORDER>
__device__ inline void
BaryStokes::tractionWarpAccum(const NodeM2P &m2p_data, vec3f center, vec3f Tp,
                              vec3d nrm, int lane,
                              double &t0, double &t1, double &t2)
{
  (void)ORDER;
  using namespace bary;

  const double dx = (double)Tp.x - (double)center.x;   // exact: fp32 inputs
  const double dy = (double)Tp.y - (double)center.y;
  const double dz = (double)Tp.z - (double)center.z;
  const double hx = m2p_data.hx, hy = m2p_data.hy, hz = m2p_data.hz;

  for (int s = lane; s < NP3; s += 32) {
    const double2 cxy = __ldg(&c_chebxy[s]);
    const double Rx = fma(-hx, cxy.x, dx);
    const double Ry = fma(-hy, cxy.y, dy);
    const double Rz = fma(-hz, __ldg(&c_chebz[s]), dz);
    const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
    if (r2 == 0.0) continue;
    double ir = (double)rsqrtf((float)r2);
    ir = ir * fma(-0.5 * r2, ir * ir, 1.5);
    const double fx = m2p_data.fhat[s][0];
    const double fy = m2p_data.fhat[s][1];
    const double fz = m2p_data.fhat[s][2];
    const double rdf = Rx * fx + Ry * fy + Rz * fz;
    const double rdn = Rx * nrm.x + Ry * nrm.y + Rz * nrm.z;
    const double ir2 = ir * ir;
    const double q = (rdf * rdn) * ((ir2 * ir2) * ir);   // (R.f)(R.n)/r^5
    t0 = fma(Rx, q, t0);
    t1 = fma(Ry, q, t1);
    t2 = fma(Rz, q, t2);
  }
}

template<int ORDER>
__device__ inline vec3d
BaryStokes::tractionWarp(const NodeM2P &m2p_data, vec3f center, vec3f Tp,
                         vec3d nrm, int lane, unsigned int mask)
{
  double t0 = 0.0, t1 = 0.0, t2 = 0.0;
  tractionWarpAccum<ORDER>(m2p_data, center, Tp, nrm, lane, t0, t1, t2);

  for (int offset = 16; offset > 0; offset >>= 1) {
    t0 += __shfl_down_sync(mask, t0, offset);
    t1 += __shfl_down_sync(mask, t1, offset);
    t2 += __shfl_down_sync(mask, t2, offset);
  }
  return (lane == 0) ? vec3d(t0, t1, t2) : vec3d(0.0, 0.0, 0.0);
}

__device__ void
BaryStokes::combine(bvh3f bvh, NodeMAC mac[], int nodeID)
{
  using namespace bary;
  const auto node = bvh.nodes[nodeID];
  const vec3f c  = node.bounds.center();
  const vec3f sz = node.bounds.size();

  NodeMAC nm;
  nm.cx = c.x; nm.cy = c.y; nm.cz = c.z;
  nm.halfDiag2 = 0.25f * cuBQL::sqrLength(sz);

  NodeM2P m2p;
  m2p.hx = 0.5f * sz.x;
  m2p.hy = 0.5f * sz.y;
  m2p.hz = 0.5f * sz.z;
  m2p.pad = 0.f;
  m2p.ownerMin = 0x7fffffff;
  m2p.ownerMax = -1;
  m2p.ownerPad0 = 0;
  m2p.ownerPad1 = 0;
  if (d_owner_mask_gb) {
    uint32_t *mask = d_owner_mask_gb + (size_t)nodeID * d_owner_mask_words_gb;
    for (int w = 0; w < d_owner_mask_words_gb; ++w) mask[w] = 0u;
  }
  // Accumulate moments in fp64; cast to the fp32 storage on the final store.
  double facc[NP3][3];
#pragma unroll 1
  for (int s = 0; s < NP3; ++s) {
    facc[s][0] = 0.0; facc[s][1] = 0.0; facc[s][2] = 0.0;
  }

  // Total source-particle count under this node (P2P/multipole crossover).
  int srcCount = 0;

  if (node.admin.count != 0) {
    const uint32_t off = node.admin.offset;
    for (uint32_t t = 0; t < node.admin.count; ++t) {
      const uint32_t bid = bvh.primIDs[off + t];
      const int begin = d_bucket_begin_gb[bid];
      const int end   = d_bucket_end_gb[bid];
      srcCount += end - begin;
      for (int pid = begin; pid < end; ++pid) {
        vec3f y;
        if (d_struct_tmpl_gb) {
          const vec3d p = reconstructSrc64(d_struct_R_gb + (size_t)9 * bid,
                                           d_struct_tmpl_gb[pid - begin],
                                           d_struct_center_gb[bid]);
          y = vec3f((float)p.x, (float)p.y, (float)p.z);
        } else {
          y = d_pos_gb[pid];
        }
        const vec3d f = d_force_gb[pid];
        if (d_owner_gb) {
          const int owner = d_owner_gb[pid];
          m2p.ownerMin = min(m2p.ownerMin, owner);
          m2p.ownerMax = max(m2p.ownerMax, owner);
          if (d_owner_mask_gb) {
            uint32_t *mask = d_owner_mask_gb + (size_t)nodeID * d_owner_mask_words_gb;
            mask[owner >> 5] |= (1u << (owner & 31));
          }
        }
        double Lx[NP], Ly[NP], Lz[NP];
        lagrange1d(y.x, nm.cx, m2p.hx, Lx);
        lagrange1d(y.y, nm.cy, m2p.hy, Ly);
        lagrange1d(y.z, nm.cz, m2p.hz, Lz);
#pragma unroll 1
        for (int k1 = 0; k1 < NP; ++k1) {
          const double wx = Lx[k1];
          for (int k2 = 0; k2 < NP; ++k2) {
            const double wxy = wx * Ly[k2];
            for (int k3 = 0; k3 < NP; ++k3) {
              const double w = (double)wxy * (double)Lz[k3];
              const int s = proxyIdx(k1, k2, k3);
              facc[s][0] += w * (double)f.x;
              facc[s][1] += w * (double)f.y;
              facc[s][2] += w * (double)f.z;
            }
          }
        }
      }
    }
  } else {
    for (int ch = 0; ch < 2; ++ch) {
      const int cid = node.admin.offset + ch;
      const NodeMAC &cmac = mac[cid];
      const NodeM2P &cm2p = d_m2p_gb[cid];
      if (d_node_count_gb) srcCount += d_node_count_gb[cid];
      m2p.ownerMin = min(m2p.ownerMin, cm2p.ownerMin);
      m2p.ownerMax = max(m2p.ownerMax, cm2p.ownerMax);
      if (d_owner_mask_gb) {
        uint32_t *mask = d_owner_mask_gb + (size_t)nodeID * d_owner_mask_words_gb;
        const uint32_t *childMask =
            d_owner_mask_gb + (size_t)cid * d_owner_mask_words_gb;
        for (int w = 0; w < d_owner_mask_words_gb; ++w) mask[w] |= childMask[w];
      }

      double Lmx[NP][NP], Lmy[NP][NP], Lmz[NP][NP];
#pragma unroll 1
      for (int q = 0; q < NP; ++q) {
        lagrange1d(cmac.cx + cm2p.hx * c_cheb[q], nm.cx, m2p.hx, Lmx[q]);
        lagrange1d(cmac.cy + cm2p.hy * c_cheb[q], nm.cy, m2p.hy, Lmy[q]);
        lagrange1d(cmac.cz + cm2p.hz * c_cheb[q], nm.cz, m2p.hz, Lmz[q]);
      }

      double tmp_a[NP3][3], tmp_b[NP3][3];
#pragma unroll 1
      for (int q1 = 0; q1 < NP; ++q1)
        for (int q2 = 0; q2 < NP; ++q2)
          for (int k3 = 0; k3 < NP; ++k3) {
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (int q3 = 0; q3 < NP; ++q3) {
              const double w = (double)Lmz[q3][k3];
              const int qs = proxyIdx(q1, q2, q3);
              sx += w * cm2p.fhat[qs][0];
              sy += w * cm2p.fhat[qs][1];
              sz += w * cm2p.fhat[qs][2];
            }
            const int s = proxyIdx(q1, q2, k3);
            tmp_a[s][0] = sx; tmp_a[s][1] = sy; tmp_a[s][2] = sz;
          }
#pragma unroll 1
      for (int q1 = 0; q1 < NP; ++q1)
        for (int k2 = 0; k2 < NP; ++k2)
          for (int k3 = 0; k3 < NP; ++k3) {
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (int q2 = 0; q2 < NP; ++q2) {
              const double w = (double)Lmy[q2][k2];
              const int s = proxyIdx(q1, q2, k3);
              sx += w * tmp_a[s][0];
              sy += w * tmp_a[s][1];
              sz += w * tmp_a[s][2];
            }
            const int s = proxyIdx(q1, k2, k3);
            tmp_b[s][0] = sx; tmp_b[s][1] = sy; tmp_b[s][2] = sz;
          }
#pragma unroll 1
      for (int k1 = 0; k1 < NP; ++k1)
        for (int k2 = 0; k2 < NP; ++k2)
          for (int k3 = 0; k3 < NP; ++k3) {
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (int q1 = 0; q1 < NP; ++q1) {
              const double w = (double)Lmx[q1][k1];
              const int s = proxyIdx(q1, k2, k3);
              sx += w * tmp_b[s][0];
              sy += w * tmp_b[s][1];
              sz += w * tmp_b[s][2];
            }
            const int s = proxyIdx(k1, k2, k3);
            facc[s][0] += sx;
            facc[s][1] += sy;
            facc[s][2] += sz;
          }
    }
  }
#pragma unroll 1
  for (int s = 0; s < NP3; ++s) {
    m2p.fhat[s][0] = (float)facc[s][0];
    m2p.fhat[s][1] = (float)facc[s][1];
    m2p.fhat[s][2] = (float)facc[s][2];
  }
  mac[nodeID] = nm;
  d_m2p_gb[nodeID] = m2p;
  if (d_node_count_gb) d_node_count_gb[nodeID] = srcCount;
}

struct BaryStokesWarpAggregate {
  using NodeMAC = BaryStokes::NodeMAC;
  using NodeM2P = BaryStokes::NodeM2P;

  // Accumulator / basis-weight type of the upward pass: fp64, or fp32 on the
  // WIDEBVH_FP32_LEVEL >= 3 path (see stokes_kernel.cuh). The moments are
  // stored fp32 either way; fp32 here trades the fp64 P2M/M2M arithmetic
  // (~5 fp64 ops per proxy per source, plus fp64 divides in lagrange1d) for
  // ~1e-6 relative error in the moments, and halves the per-warp shared
  // footprint (17 KB -> 8.5 KB at PDEG 7).
  using Acc = std::conditional_t<(stokes::FP32_LEVEL >= 3), float, double>;

  struct __align__(16) Shared {
    Acc fhat[bary::NP3][3];
    Acc slab_a[bary::NP * bary::NP][3];
    Acc slab_b[bary::NP * bary::NP][3];
    Acc Lmx[bary::NP][bary::NP];
    Acc Lmy[bary::NP][bary::NP];
    Acc Lmz[bary::NP][bary::NP];
    Acc Lx[bary::NP], Ly[bary::NP], Lz[bary::NP];
    Acc fx, fy, fz;
  };

  __device__ void operator()(bvh3f bvh, NodeMAC mac[], int nodeID,
                             int lane, unsigned int mask, Shared &sh) const
  {
    using namespace bary;
    const auto node = bvh.nodes[nodeID];
    const bool isLeaf = (node.admin.count != 0);
    const vec3f c = node.bounds.center();
    const vec3f sz = node.bounds.size();
    const float hx = 0.5f * sz.x, hy = 0.5f * sz.y, hz = 0.5f * sz.z;
    uint32_t *ownerMask =
        d_owner_mask_gb
            ? d_owner_mask_gb + (size_t)nodeID * d_owner_mask_words_gb
            : nullptr;

    if (lane == 0) {
      mac[nodeID] = {c.x, c.y, c.z, 0.25f * cuBQL::sqrLength(sz)};
      NodeM2P &m2p = d_m2p_gb[nodeID];
      m2p.hx = hx; m2p.hy = hy; m2p.hz = hz;
      m2p.pad = 0.f;
      m2p.ownerMin = 0x7fffffff;
      m2p.ownerMax = -1;
      m2p.ownerPad0 = 0;
      m2p.ownerPad1 = 0;
    }

    for (int s = lane; s < NP3; s += 32) {
      sh.fhat[s][0] = Acc(0);
      sh.fhat[s][1] = Acc(0);
      sh.fhat[s][2] = Acc(0);
    }
    if (ownerMask && isLeaf) {
      for (int w = lane; w < d_owner_mask_words_gb; w += 32) {
        ownerMask[w] = 0u;
      }
    }
    __syncwarp(mask);

    if (isLeaf) {
      int ownerMin = 0x7fffffff;
      int ownerMax = -1;
      int leafSrcCount = 0;   // total sources under this leaf (crossover count)
      const uint32_t off = node.admin.offset;
      for (uint32_t t = 0; t < node.admin.count; ++t) {
        const uint32_t bid = bvh.primIDs[off + t];
        const int begin = d_bucket_begin_gb[bid];
        const int end = d_bucket_end_gb[bid];
        leafSrcCount += end - begin;   // warp-uniform; lane 0 publishes below
        for (int pid = begin; pid < end; ++pid) {
          if (lane == 0) {
            vec3f y;
            if (d_struct_tmpl_gb) {
              // Structured source: reconstruct from template + object transform
              // (bid == object id; pid - begin == template index). fp64 recon then
              // fp32 cast, matching GatherShiftedFloat's quantization.
              const vec3d p = reconstructSrc64(d_struct_R_gb + (size_t)9 * bid,
                                               d_struct_tmpl_gb[pid - begin],
                                               d_struct_center_gb[bid]);
              y = vec3f((float)p.x, (float)p.y, (float)p.z);
            } else {
              y = d_pos_gb[pid];
            }
            const vec3d f = d_force_gb[pid];
            if (d_owner_gb) {
              const int owner = d_owner_gb[pid];
              ownerMin = min(ownerMin, owner);
              ownerMax = max(ownerMax, owner);
              if (ownerMask) ownerMask[owner >> 5] |= (1u << (owner & 31));
            }
#if WIDEBVH_FP32_LEVEL >= 3
            lagrange1d_f(y.x - c.x, hx, sh.Lx);
            lagrange1d_f(y.y - c.y, hy, sh.Ly);
            lagrange1d_f(y.z - c.z, hz, sh.Lz);
#else
            lagrange1d(y.x, c.x, hx, sh.Lx);
            lagrange1d(y.y, c.y, hy, sh.Ly);
            lagrange1d(y.z, c.z, hz, sh.Lz);
#endif
            sh.fx = (Acc)f.x;
            sh.fy = (Acc)f.y;
            sh.fz = (Acc)f.z;
          }
          __syncwarp(mask);

          for (int s = lane; s < NP3; s += 32) {
            int tmp = s;
            const int k3 = tmp % NP; tmp /= NP;
            const int k2 = tmp % NP; tmp /= NP;
            const int k1 = tmp;
            const Acc w = (Acc)sh.Lx[k1] * (Acc)sh.Ly[k2]
                        * (Acc)sh.Lz[k3];
            sh.fhat[s][0] += w * sh.fx;
            sh.fhat[s][1] += w * sh.fy;
            sh.fhat[s][2] += w * sh.fz;
          }
          __syncwarp(mask);
        }
      }
      if (lane == 0) {
        d_m2p_gb[nodeID].ownerMin = ownerMin;
        d_m2p_gb[nodeID].ownerMax = ownerMax;
        if (d_node_count_gb) d_node_count_gb[nodeID] = leafSrcCount;
      }
    } else {
      const int c0 = (int)node.admin.offset;
      const int c1 = c0 + 1;
      if (lane == 0) {
        d_m2p_gb[nodeID].ownerMin = min(d_m2p_gb[c0].ownerMin, d_m2p_gb[c1].ownerMin);
        d_m2p_gb[nodeID].ownerMax = max(d_m2p_gb[c0].ownerMax, d_m2p_gb[c1].ownerMax);
        if (d_node_count_gb)
          d_node_count_gb[nodeID] = d_node_count_gb[c0] + d_node_count_gb[c1];
      }
      if (ownerMask) {
        const uint32_t *m0 =
            d_owner_mask_gb + (size_t)c0 * d_owner_mask_words_gb;
        const uint32_t *m1 =
            d_owner_mask_gb + (size_t)c1 * d_owner_mask_words_gb;
        for (int w = lane; w < d_owner_mask_words_gb; w += 32) {
          ownerMask[w] = m0[w] | m1[w];
        }
      }

      for (int ch = 0; ch < 2; ++ch) {
        const int cid = (int)node.admin.offset + ch;
        const NodeMAC &cmac = mac[cid];
        const NodeM2P &cm2p = d_m2p_gb[cid];

        for (int q = lane; q < NP; q += 32) {
#if WIDEBVH_FP32_LEVEL >= 3
          // child proxy q relative to THIS node's centre: (child centre - centre)
          // is an fp32 difference of nearby coordinates, then the child offset.
          lagrange1d_f(fmaf(cm2p.hx, c_chebf[q], cmac.cx - c.x), hx, sh.Lmx[q]);
          lagrange1d_f(fmaf(cm2p.hy, c_chebf[q], cmac.cy - c.y), hy, sh.Lmy[q]);
          lagrange1d_f(fmaf(cm2p.hz, c_chebf[q], cmac.cz - c.z), hz, sh.Lmz[q]);
#else
          lagrange1d(cmac.cx + cm2p.hx * c_cheb[q], c.x, hx, sh.Lmx[q]);
          lagrange1d(cmac.cy + cm2p.hy * c_cheb[q], c.y, hy, sh.Lmy[q]);
          lagrange1d(cmac.cz + cm2p.hz * c_cheb[q], c.z, hz, sh.Lmz[q]);
#endif
        }
        __syncwarp(mask);

        for (int q1 = 0; q1 < NP; ++q1) {
          // Pass 1: contract q3 -> k3
          for (int s = lane; s < NP * NP; s += 32) {
            const int k3 = s % NP;
            const int q2 = s / NP;
            Acc sx = Acc(0), sy = Acc(0), szc = Acc(0);
            for (int q3 = 0; q3 < NP; ++q3) {
              const Acc w = (Acc)sh.Lmz[q3][k3];
              const int qs = proxyIdx(q1, q2, q3);
              sx += w * cm2p.fhat[qs][0];
              sy += w * cm2p.fhat[qs][1];
              szc += w * cm2p.fhat[qs][2];
            }
            sh.slab_a[s][0] = sx;
            sh.slab_a[s][1] = sy;
            sh.slab_a[s][2] = szc;
          }
          __syncwarp(mask);

          // Pass 2: contract q2 -> k2
          for (int s = lane; s < NP * NP; s += 32) {
            const int k3 = s % NP;
            const int k2 = s / NP;
            Acc sx = Acc(0), sy = Acc(0), szc = Acc(0);
            for (int q2 = 0; q2 < NP; ++q2) {
              const Acc w = (Acc)sh.Lmy[q2][k2];
              sx += w * sh.slab_a[q2 * NP + k3][0];
              sy += w * sh.slab_a[q2 * NP + k3][1];
              szc += w * sh.slab_a[q2 * NP + k3][2];
            }
            sh.slab_b[s][0] = sx;
            sh.slab_b[s][1] = sy;
            sh.slab_b[s][2] = szc;
          }
          __syncwarp(mask);

          // Pass 3: contract q1 -> k1, accumulate into fhat
          for (int s = lane; s < NP3; s += 32) {
            const int k3 = s % NP;
            const int k2 = (s / NP) % NP;
            const int k1 = s / (NP * NP);
            const Acc w = (Acc)sh.Lmx[q1][k1];
            sh.fhat[s][0] += w * sh.slab_b[k2 * NP + k3][0];
            sh.fhat[s][1] += w * sh.slab_b[k2 * NP + k3][1];
            sh.fhat[s][2] += w * sh.slab_b[k2 * NP + k3][2];
          }
          __syncwarp(mask);
        }
      }
    }

    for (int s = lane; s < NP3; s += 32) {
      d_m2p_gb[nodeID].fhat[s][0] = (float)sh.fhat[s][0];
      d_m2p_gb[nodeID].fhat[s][1] = (float)sh.fhat[s][1];
      d_m2p_gb[nodeID].fhat[s][2] = (float)sh.fhat[s][2];
    }
  }
};

inline void
BaryStokes::upwardPass(bvh3f bvh, NodeMAC *d_mac, NodeM2P *d_m2p,
                       const vec3f *d_pos, const vec3d *d_force,
                       const int *d_owner, uint32_t *d_ownerMask,
                       int ownerMaskWords,
                       const int *d_bucketBegin, const int *d_bucketEnd,
                       int /*order*/, cudaStream_t stream)
{
  CUDA_CHECK(cudaMemcpyToSymbol(bary::d_pos_gb,          &d_pos,         sizeof(d_pos)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::d_force_gb,        &d_force,       sizeof(d_force)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::d_owner_gb,        &d_owner,       sizeof(d_owner)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::d_owner_mask_gb,   &d_ownerMask,   sizeof(d_ownerMask)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::d_owner_mask_words_gb, &ownerMaskWords, sizeof(ownerMaskWords)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::d_bucket_begin_gb, &d_bucketBegin, sizeof(d_bucketBegin)));
  CUDA_CHECK(cudaMemcpyToSymbol(bary::d_bucket_end_gb,   &d_bucketEnd,   sizeof(d_bucketEnd)));
  CUDA_CHECK(cudaMemcpyToSymbol(mp::d_m2p_gb,            &d_m2p,         sizeof(d_m2p)));

#if BARY_STOKES_USE_SCALAR_REFIT
  void (*hostFp)(bvh3f, NodeMAC[], int) = nullptr;
  CUDA_CHECK(cudaMemcpyFromSymbol(&hostFp, BaryStokes_combine_fp, sizeof(hostFp)));
  cuBQL::cuda::refit_aggregate(bvh, d_mac, hostFp, stream);
#else
  BaryStokesWarpAggregate aggregate;
  tcgpu::refit_aggregate_warp(bvh, d_mac, aggregate, stream);
#endif
}

} // namespace mp
