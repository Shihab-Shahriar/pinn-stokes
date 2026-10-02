// SPDX-License-Identifier: Apache-2.0
//
// Multipole module: Cartesian Taylor expansion (order p <= 4) of the Stokeslet.
//
// This is the *swappable* unit. It bundles everything specific to the Cartesian
// expansion -- the per-node moment storage, the multi-index bookkeeping, the
// symmetry-compact 1/r derivatives, P2M+M2M (upward pass) and M2P (far field).
// The generic treecode (traversal, pipeline, near-field p2p in stokes_kernel.cuh)
// knows none of this; it only sees the policy surface:
//   - CartesianStokes::Moments        opaque per-node POD blob
//   - CartesianStokes::MAX_ORDER       sizing bound, for order validation
//   - CartesianStokes::name()          label for metric printing
//   - CartesianStokes::setup(order)    one-time host precompute (no-op here)
//   - CartesianStokes::upwardPass(...) P2M + M2M (wraps cuBQL refit_aggregate)
//   - CartesianStokes::m2p(...)        far-field evaluation (hot, inlined)
// A different multipole type (spherical harmonics, KIFMM, ...) is a new header
// exposing the same surface; switch it in treecode.cu with one `using` line.
//
// The math is a direct port of StokesKernel in ../kernel_sum.py:
//   _laplace_sym -> laplaceSym()   (symmetry-compact derivatives of 1/r)
//   m2p          -> m2p()
//   p2m / m2m    -> the combine() refit_aggregate callback
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda/refit_aggregate.h"

// For stokes::RPY_ON / RPY_C. treecode.cuh already includes this ahead of us,
// but m2p() reads the constants directly, so include it here rather than rely
// on include order (the header is #pragma once and pulls in nothing heavy).
#include "stokes_kernel.cuh"

// upwardPass() validates its CUDA calls with CUDA_CHECK. Defer to the includer's
// macro if it already has one (treecode.cu defines a richer file:line version);
// otherwise provide a minimal self-contained fallback so this header stands
// alone. The #ifndef guard makes this clash-free regardless of include order.
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess)                                                \
      throw std::runtime_error(std::string("CUDA error: ") +             \
                               cudaGetErrorString(_e));                   \
  } while (0)
#endif

namespace mp {

using cuBQL::vec3f;
using cuBQL::vec3d;
using cuBQL::bvh3f;

// ---- small integer helpers (Cartesian symmetry-compact bookkeeping) -------
//  Moments/derivatives use the multi-index order from kernel_sum.py:
//    n = 0: (0,0,0)
//    n = 1: (1,0,0), (0,1,0), (0,0,1)
//    n = 2: (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
//    n = 3: same lexicographic power order, 10 terms.
//  Orders 0..4 have 1/3/6/10/15 unique symmetric components.
__device__ __host__ inline int factSmall(int n)
{
  return (n <= 1) ? 1 : (n == 2) ? 2 : (n == 3) ? 6 : 24;
}

__device__ __host__ inline int binomSmall(int n, int k)
{
  return factSmall(n) / (factSmall(k) * factSmall(n - k));  // Todo: optimize, maybe cache
}

__device__ __host__ inline int momentOff(int n)
{
  return n * (n + 1) * (n + 2) / 6;
}

// linear index of multi-index (a,b,c) within derivative order n (c implied).
__device__ __host__ inline int idx(int n, int a, int b)
{
  return (n - a) * (n - a + 1) / 2 + (n - a - b);
}

__device__ __host__ inline int momentIdx(int n, int a, int b)
{
  return momentOff(n) + idx(n, a, b);
}

// ---- scene globals read by combine() during the upward pass ---------------
//  combine()'s signature is fixed by refit_aggregate, so the particle/bucket
//  arrays are smuggled in through these device globals, set by upwardPass()
//  with cudaMemcpyToSymbol before the refit. They are an implementation detail
//  of *this* module's P2M, so they live here next to their only reader.
__device__ const vec3f *d_pos_g          = nullptr;
__device__ const vec3d *d_force_g        = nullptr;
__device__ const int   *d_owner_g        = nullptr;
__device__ uint32_t    *d_owner_mask_g   = nullptr;
__device__ int          d_owner_mask_words_g = 0;
__device__ const int   *d_bucket_begin_g = nullptr;
__device__ const int   *d_bucket_end_g   = nullptr;
__device__ int          d_order_g        = 3;

// ===========================================================================
//  The policy.
// ===========================================================================
struct CartesianStokes {
  static constexpr int MAX_ORDER       = 4;
  static constexpr int MOMENT_TERMS    = 35;   // sum_{n=0}^4 (n+1)(n+2)/2
  // The m2p contraction is cheap (<= 35 slots) and fully register-resident, so
  // warp-cooperating on ONE node wastes lanes. Instead the split-warpspec
  // consumers batch-pop up to 32 ring items and run one serial m2p per lane.
  static constexpr bool LANE_BATCH_M2P = true;
  // Keeps the fp32 target path (only SphericalStokes uses fp64-geometry M2P).
  static constexpr bool WANTS_FP64_TARGET = false;
  // Per-node far-field contribution type (fp64 here; only BaryStokes has the
  // fp32 fast path, see WIDEBVH_FP32_LEVEL in stokes_kernel.cuh).
  using FarVec = vec3d;

  // How far above the moment order the 1/r derivative table has to reach. The
  // Oseen part of the kernel costs exactly one extra derivative (see m2p: the
  // R_j * T_{alpha+e_i} term). With RPY on, the finite-size correction is
  // -c * d_i d_j (1/r), which costs two. Off (the default, every non-NeMO
  // target) this is 1 and the table is the size it always was.
  static constexpr int DERIV_EXTRA = stokes::RPY_ON ? 2 : 1;

  template<int ORDER>
  struct Derivatives {
    static constexpr int order = ORDER + DERIV_EXTRA;
    static constexpr int count = (order + 1) * (order + 2) * (order + 3) / 6;

    double value[count];

    __device__ __forceinline__ double &at(int n, int a, int b)
    {
      return value[momentIdx(n, a, b)];
    }

    __device__ __forceinline__ const double &at(int n, int a, int b) const
    {
      return value[momentIdx(n, a, b)];
    }
  };

  // SoA node types. Traversal loads only NodeMAC (16 bytes) per node visit;
  // NodeM2P (multipole coefficients) is loaded only when MAC accepts.
  struct __align__(16) NodeMAC {
    float cx, cy, cz, halfDiag2;
  };

  struct NodeM2P {
    int ownerMin, ownerMax;
    // Node length scale (half-diagonal). Moments are stored SCALED:
    //   M[alpha] = sum_a f_a * ((y_a - c)/scale)^alpha
    // so |M| stays ~uniform across orders and tree levels (they live in fp32).
    // m2p compensates exactly by evaluating the 1/r derivative recurrence at
    // R/scale and multiplying the final velocity by 1/scale.
    float scale, pad0;
    float M[MOMENT_TERMS][3];
  };

  static const char *name() { return "Cartesian"; }
  static void setup(int /*order*/) {}  // analytic kernel: nothing to precompute

  template<int ORDER>
  static __device__ inline void laplaceSym(double Rx, double Ry, double Rz,
                                           Derivatives<ORDER> &T);

  template<int ORDER>
  static __device__ inline vec3d m2p(const NodeM2P &m2p_data,
                                     vec3f center, vec3f Tp);

  template<int ORDER>
  static __device__ inline vec3d m2pWarp(const NodeM2P &m2p_data,
                                         vec3f center,
                                         vec3f Tp, int lane,
                                         unsigned int mask);

  static void upwardPass(bvh3f bvh, NodeMAC *d_mac, NodeM2P *d_m2p,
                         const vec3f *d_pos, const vec3d *d_force,
                         const int *d_owner, uint32_t *d_ownerMask,
                         int ownerMaskWords,
                         const int *d_bucketBegin, const int *d_bucketEnd,
                         int order, cudaStream_t stream = 0);

  static __device__ void combine(bvh3f bvh, NodeMAC mac[], int nodeID);
};

__device__ CartesianStokes::NodeM2P *d_m2p_g = nullptr;

__device__ void (*CartesianStokes_combine_fp)(bvh3f, CartesianStokes::NodeMAC[], int)
    = &CartesianStokes::combine;

// ===========================================================================
//  Definitions.
// ===========================================================================

// Faithful to kernel_sum.py:_laplace_sym. With ORDER a compile-time constant
// the n/a/b loops fully unroll; every idx()/coef/branch then folds to a constant
// so each compact derivative entry is straight-line FMAs.
template<int ORDER>
__device__ inline void
CartesianStokes::laplaceSym(double Rx, double Ry, double Rz,
                            Derivatives<ORDER> &T)
{
  constexpr int DERIV_ORDER = Derivatives<ORDER>::order;
  // fp32 rsqrt seed + ONE fp64 Newton step (rel err ~1e-13); avoids fp64
  // division/sqrt software chains (same pattern as bary_stokes.cuh::m2p).
  const double r2 = Rx*Rx + Ry*Ry + Rz*Rz;
  double ir = (double)rsqrtf((float)r2);
  ir = ir * fma(-0.5 * r2, ir * ir, 1.5);
  const double invR2 = ir * ir;
  T.at(0, 0, 0) = ir;
  const double ir3 = ir * invR2;
  T.at(1, 1, 0) = -Rx * ir3;
  T.at(1, 0, 1) = -Ry * ir3;
  T.at(1, 0, 0) = -Rz * ir3;

  const double Rc[3] = {Rx, Ry, Rz};
  // recurrence:
  //   Tn(a,b,c) = -[ (2n-1)*sum_d alpha_d R_d T[n-1](alpha-e_d)
  //                 + (n-1)*sum_d alpha_d(alpha_d-1)T[n-2](alpha-2e_d) ] / (n r2)
  // with c implied by n-a-b in the compact index.
#pragma unroll
  for (int n = 2; n <= DERIV_ORDER; ++n) {
#pragma unroll
    for (int a = n; a >= 0; --a)
#pragma unroll
      for (int b = n - a; b >= 0; --b) {
        const int c = n - a - b;
        double t = 0.0;
        const double coef1 = (double)(2 * n - 1);
        const double coef2 = (double)(n - 1);
        if (a) t += coef1 * a * Rc[0] * T.at(n - 1, a - 1, b);
        if (b) t += coef1 * b * Rc[1] * T.at(n - 1, a, b - 1);
        if (c) t += coef1 * c * Rc[2] * T.at(n - 1, a, b);
        if (a > 1) t += coef2 * a * (a - 1) * T.at(n - 2, a - 2, b);
        if (b > 1) t += coef2 * b * (b - 1) * T.at(n - 2, a, b - 2);
        if (c > 1) t += coef2 * c * (c - 1) * T.at(n - 2, a, b);

        const double scale = -invR2 / static_cast<double>(n);
        T.at(n, a, b) = t * scale;
      }
  }
}

// Faithful to kernel_sum.py:m2p (R0 = target - center). fp32-storage/fp64-eval
// (the d1f592d treatment): moments stay fp32 in memory and are widened on
// load; geometry (R is exact in fp64: fp32 inputs), the 1/r derivative
// recurrence, and the contraction all run in fp64. ORDER is compile-time so
// the entire nest unrolls and `slot`/`coeff`/derivative indices fold away.
//
// RPY: the finite-size correction is a second derivative of the SAME Laplace
// kernel this expansion is already built on, so it costs no new machinery --
// only two more derivative orders. With G the Oseen tensor and c = 2a^2/3,
//
//   d_i d_j (1/r) = 3 R_i R_j / r^5 - d_ij / r^3
//   RPY_ij        = G_ij + c (d_ij/r^3 - 3 R_i R_j/r^5) = G_ij - c d_i d_j (1/r)
//
// so d^alpha RPY_ij = d^alpha G_ij - c * T_{alpha + e_i + e_j}. Valid for
// r >= 2a, which the near-field cutoff guarantees many times over. Moments and
// the upward pass are untouched: RPY changes the contraction, not the moments.
template<int ORDER>
__device__ inline vec3d
CartesianStokes::m2p(const NodeM2P &m2p_data, vec3f center, vec3f Tp)
{
  // Moments are node-scaled (see NodeM2P): evaluate the derivative recurrence
  // at R/scale and multiply the final velocity by 1/scale -- every g-term
  // below then carries exactly one net factor 1/scale (algebra telescopes).
  const double invS = 1.0 / (double)m2p_data.scale;
  // The RPY term sits two derivative orders above the Oseen terms, and the
  // recurrence runs in scaled coordinates, so it needs two extra factors of
  // 1/scale on top of the single one every other g-term telescopes to.
  [[maybe_unused]] const double RPY_C_S = stokes::RPY_C * invS * invS;
  const double Rx = ((double)Tp.x - (double)center.x) * invS;
  const double Ry = ((double)Tp.y - (double)center.y) * invS;
  const double Rz = ((double)Tp.z - (double)center.z) * invS;
  Derivatives<ORDER> L;
  laplaceSym<ORDER>(Rx, Ry, Rz, L);
  const double Rt[3] = {Rx, Ry, Rz};
  double uf[3] = {0.0, 0.0, 0.0};

  // Slot coefficient (-1)^n / (a! b! c!), lex order (a = n..0, b = n-a..0);
  // matches kernel_sum.py m2p's cn * mult with n! cancelled.
  constexpr double coeffTab[MOMENT_TERMS] = {
    1.0,
    -1.0, -1.0, -1.0,
    0.5, 1.0, 1.0, 0.5, 1.0, 0.5,
    -1.0/6.0, -0.5, -0.5, -0.5, -1.0, -0.5,
    -1.0/6.0, -0.5, -0.5, -1.0/6.0,
    1.0/24.0, 1.0/6.0, 1.0/6.0, 0.25, 0.5, 0.25, 1.0/6.0, 0.5, 0.5,
    1.0/6.0, 1.0/24.0, 1.0/6.0, 0.25, 1.0/6.0, 1.0/24.0
  };

  int slot = 0;
#pragma unroll
  for (int n = 0; n <= ORDER; ++n) {
#pragma unroll
    for (int a = n; a >= 0; --a)
#pragma unroll
      for (int b = n - a; b >= 0; --b, ++slot) {
        const int c2 = n - a - b;
        const int alpha[3] = {a, b, c2};
        const double coeff = coeffTab[slot];
        const float *Mc = m2p_data.M[slot];
        const double Talpha = L.at(n, a, b);

#pragma unroll
        for (int i = 0; i < 3; ++i) {
          const int ba = alpha[0] + (i == 0);
          const int bb = alpha[1] + (i == 1);
          const double Ti = L.at(n + 1, ba, bb);
          double acc = 0.0;
#pragma unroll
          for (int j = 0; j < 3; ++j) {
            double g = ((i == j) ? Talpha : 0.0) - Rt[j] * Ti;
            if (alpha[j]) {
              const int da = alpha[0] - (j == 0) + (i == 0);
              const int db = alpha[1] - (j == 1) + (i == 1);
              g -= (double)alpha[j] * L.at(n, da, db);
            }
            // -c * T_{alpha + e_i + e_j}; ba/bb already carry the +e_i.
            if constexpr (stokes::RPY_ON)
              g -= RPY_C_S * L.at(n + 2, ba + (j == 0), bb + (j == 1));
            acc = fma(g, (double)Mc[j], acc);
          }
          uf[i] = fma(coeff, acc, uf[i]);
        }
      }
  }

  return vec3d(uf[0] * invS, uf[1] * invS, uf[2] * invS);
}

template<int ORDER>
__device__ inline vec3d
CartesianStokes::m2pWarp(const NodeM2P &m2p_data, vec3f center,
                         vec3f Tp, int lane, unsigned int mask)
{
  (void)mask;
  if (lane != 0) return vec3d(0.0, 0.0, 0.0);
  return m2p<ORDER>(m2p_data, center, Tp);
}

__device__ void
CartesianStokes::combine(bvh3f bvh, NodeMAC mac[], int nodeID)
{
  const auto node = bvh.nodes[nodeID];
  const vec3f c = node.bounds.center();
  const vec3f sz = node.bounds.size();

  NodeMAC nm;
  nm.cx = c.x; nm.cy = c.y; nm.cz = c.z;
  nm.halfDiag2 = 0.25f * cuBQL::sqrLength(sz);

  // Moment scale (see NodeM2P); the floor guards degenerate (point) boxes --
  // their offsets are exactly 0, so any positive scale is consistent.
  const float nodeS = fmaxf(sqrtf(nm.halfDiag2), 1e-12f);
  const float invS  = 1.f / nodeS;

  NodeM2P m2p;
  m2p.ownerMin = 0x7fffffff;
  m2p.ownerMax = -1;
  m2p.scale = nodeS;
  m2p.pad0  = 0.f;
  if (d_owner_mask_g) {
    uint32_t *mask = d_owner_mask_g + (size_t)nodeID * d_owner_mask_words_g;
    for (int w = 0; w < d_owner_mask_words_g; ++w) mask[w] = 0u;
  }
  for (int s = 0; s < MOMENT_TERMS; ++s)
    for (int j = 0; j < 3; ++j)
      m2p.M[s][j] = 0.f;
  const int order = d_order_g;

  if (node.admin.count != 0) {
    const uint32_t off = node.admin.offset;
    for (uint32_t t = 0; t < node.admin.count; ++t) {
      const uint32_t bid = bvh.primIDs[off + t];
      const int begin = d_bucket_begin_g[bid];
      const int end   = d_bucket_end_g[bid];
      for (int pid = begin; pid < end; ++pid) {
        vec3f a = d_pos_g[pid] - c;
        a.x *= invS; a.y *= invS; a.z *= invS;   // scaled offset, |a| <= 1
        const vec3d f = d_force_g[pid];
        if (d_owner_g) {
          const int owner = d_owner_g[pid];
          m2p.ownerMin = min(m2p.ownerMin, owner);
          m2p.ownerMax = max(m2p.ownerMax, owner);
          if (d_owner_mask_g) {
            uint32_t *mask = d_owner_mask_g + (size_t)nodeID * d_owner_mask_words_g;
            mask[owner >> 5] |= (1u << (owner & 31));
          }
        }
        const float ax2 = a.x * a.x, ay2 = a.y * a.y, az2 = a.z * a.z;
        const float px[5] = {1.f, a.x, ax2, ax2 * a.x, ax2 * ax2};
        const float py[5] = {1.f, a.y, ay2, ay2 * a.y, ay2 * ay2};
        const float pz[5] = {1.f, a.z, az2, az2 * a.z, az2 * az2};
        for (int n = 0; n <= order; ++n)
          for (int ax = n; ax >= 0; --ax)
            for (int ay = n - ax; ay >= 0; --ay) {
              const int az = n - ax - ay;
              const int s = momentIdx(n, ax, ay);
              const float w = px[ax] * py[ay] * pz[az];
              for (int j = 0; j < 3; ++j)
                m2p.M[s][j] += w * (float)f[j];
            }
      }
    }
  } else {
    for (int ch = 0; ch < 2; ++ch) {
      const int cid = node.admin.offset + ch;
      const NodeMAC &cmac = mac[cid];
      const NodeM2P &cm2p = d_m2p_g[cid];
      m2p.ownerMin = min(m2p.ownerMin, cm2p.ownerMin);
      m2p.ownerMax = max(m2p.ownerMax, cm2p.ownerMax);
      if (d_owner_mask_g) {
        uint32_t *mask = d_owner_mask_g + (size_t)nodeID * d_owner_mask_words_g;
        const uint32_t *childMask =
            d_owner_mask_g + (size_t)cid * d_owner_mask_words_g;
        for (int w = 0; w < d_owner_mask_words_g; ++w) mask[w] |= childMask[w];
      }
      // Shift in PARENT-scaled coordinates; child moments are child-scaled, so
      // each child term of total order bn also carries (s_child/s_parent)^bn.
      const vec3f d = vec3f((cmac.cx - c.x) * invS,
                            (cmac.cy - c.y) * invS,
                            (cmac.cz - c.z) * invS);
      const float ratio = cm2p.scale * invS;
      float rpow[MAX_ORDER + 1];
      rpow[0] = 1.f;
      for (int k = 1; k <= order; ++k) rpow[k] = rpow[k - 1] * ratio;
      const float dx2 = d.x * d.x, dy2 = d.y * d.y, dz2 = d.z * d.z;
      const float px[5] = {1.f, d.x, dx2, dx2 * d.x, dx2 * dx2};
      const float py[5] = {1.f, d.y, dy2, dy2 * d.y, dy2 * dy2};
      const float pz[5] = {1.f, d.z, dz2, dz2 * d.z, dz2 * dz2};

      for (int n = 0; n <= order; ++n)
        for (int ax = n; ax >= 0; --ax)
          for (int ay = n - ax; ay >= 0; --ay) {
            const int az = n - ax - ay;
            const int s = momentIdx(n, ax, ay);
            for (int bx = 0; bx <= ax; ++bx)
              for (int by = 0; by <= ay; ++by)
                for (int bz = 0; bz <= az; ++bz) {
                  const int bn = bx + by + bz;
                  const int bs = momentIdx(bn, bx, by);
                  const float w =
                    (float)(binomSmall(ax, bx) *
                            binomSmall(ay, by) *
                            binomSmall(az, bz))
                    * px[ax - bx] * py[ay - by] * pz[az - bz] * rpow[bn];
                  for (int j = 0; j < 3; ++j)
                    m2p.M[s][j] += w * cm2p.M[bs][j];
                }
          }
    }
  }
  mac[nodeID] = nm;
  d_m2p_g[nodeID] = m2p;
}

inline void
CartesianStokes::upwardPass(bvh3f bvh, NodeMAC *d_mac, NodeM2P *d_m2p,
                            const vec3f *d_pos, const vec3d *d_force,
                            const int *d_owner, uint32_t *d_ownerMask,
                            int ownerMaskWords,
                            const int *d_bucketBegin, const int *d_bucketEnd,
                            int order, cudaStream_t stream)
{
  CUDA_CHECK(cudaMemcpyToSymbol(d_pos_g,          &d_pos,         sizeof(d_pos)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_force_g,        &d_force,       sizeof(d_force)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_owner_g,        &d_owner,       sizeof(d_owner)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_owner_mask_g,   &d_ownerMask,   sizeof(d_ownerMask)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_owner_mask_words_g, &ownerMaskWords, sizeof(ownerMaskWords)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_bucket_begin_g, &d_bucketBegin, sizeof(d_bucketBegin)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_bucket_end_g,   &d_bucketEnd,   sizeof(d_bucketEnd)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_order_g,        &order,         sizeof(order)));
  CUDA_CHECK(cudaMemcpyToSymbol(mp::d_m2p_g,      &d_m2p,         sizeof(d_m2p)));

  void (*hostFp)(bvh3f, NodeMAC[], int) = nullptr;
  CUDA_CHECK(cudaMemcpyFromSymbol(&hostFp, CartesianStokes_combine_fp, sizeof(hostFp)));
  cuBQL::cuda::refit_aggregate(bvh, d_mac, hostFp, stream);
}

} // namespace mp
