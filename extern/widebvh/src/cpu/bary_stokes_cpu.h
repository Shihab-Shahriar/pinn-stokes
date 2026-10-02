// SPDX-License-Identifier: Apache-2.0
//
// CPU port of the barycentric-Lagrange (KITC) Stokes multipole policy
// (src/bary_stokes.cuh). Kernel-independent Chebyshev proxy moments:
// P2M/M2M build per-node "modified weights" fhat at a (PDEG+1)^3 tensor grid
// of Chebyshev points; M2P is a direct Stokeslet sum over those proxies.
//
// Differences from the GPU policy (all deliberate, see the CPU-port plan):
//  * no owner/group-exclusion fields, no crossover counts, no structured
//    template -- this port serves the plain treecode only;
//  * NodeM2P stores fhat in SoA layout fhat[3][NP3] (component-major) so the
//    M2P proxy loop reads three contiguous fp32 streams (auto-vectorization);
//    the GPU stores AoS fhat[NP3][3] for coalescing instead;
//  * expanded per-axis Chebyshev tables chebX/Y/Z[NP3] replace the GPU's
//    c_chebxy/c_chebz/c_k1..k3 device tables (same idea: no /,% by NP in the
//    hot loop);
//  * 1/r is a plain 1.0/sqrt (GPU: rsqrtf seed + one fp64 Newton step).
// The math (lagrange1d guards, P2M outer product, 3-pass separable M2M,
// fp64 accumulation with fp32 moment storage) is ported verbatim.
#pragma once

#include "cpu_common.h"

#if (defined(TCCPU_FAST_RSQRT) || defined(TCCPU_M2P_FP32)) && defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace bary_cpu {

using tccpu::vec3f;
using tccpu::vec3d;
using tccpu::box3f;

// Polynomial interpolation degree (compile-time), same operating point as the
// GPU policy (bary_stokes.cuh:85-87). NP3 = proxy points per cluster.
#ifndef TCCPU_PDEG
#define TCCPU_PDEG 7
#endif
constexpr int PDEG = TCCPU_PDEG;
constexpr int NP   = PDEG + 1;
constexpr int NP3  = NP * NP * NP;

inline int proxyIdx(int k1, int k2, int k3) { return (k1 * NP + k2) * NP + k3; }

// Chebyshev-2nd-kind nodes in [-1,1], barycentric weights, and the fully
// expanded per-axis node tables indexed by the flat proxy index
// s = (k1*NP+k2)*NP+k3. Values match BaryStokes::setup() (bary_stokes.cuh:293).
struct Tables {
  double cheb[NP];
  double baryw[NP];
  double chebX[NP3], chebY[NP3], chebZ[NP3];
  float chebXf[NP3], chebYf[NP3], chebZf[NP3];  // fp32 twins (TCCPU_M2P_FP32)

  Tables()
  {
    for (int k = 0; k < NP; ++k) {
      cheb[k] = std::cos(M_PI * (double)k / (double)PDEG);
      const double dk = (k == 0 || k == PDEG) ? 0.5 : 1.0;
      baryw[k] = ((k & 1) ? -1.0 : 1.0) * dk;
    }
    for (int s = 0; s < NP3; ++s) {
      chebX[s] = cheb[(s / (NP * NP)) % NP];
      chebY[s] = cheb[(s / NP) % NP];
      chebZ[s] = cheb[s % NP];
      chebXf[s] = (float)chebX[s];
      chebYf[s] = (float)chebY[s];
      chebZf[s] = (float)chebZ[s];
    }
  }
};

inline const Tables &tables() { static const Tables t; return t; }

// SoA node types, same hot/cold split as the GPU (bary_stokes.cuh:204-205):
// traversal loads only NodeMAC (16 bytes) per node visit; NodeM2P is loaded
// only when the MAC accepts.
struct alignas(16) NodeMAC {
  float cx, cy, cz, halfDiag2;
};

struct alignas(64) NodeM2P {
  // fp32 moments in SoA: fhat[c][s] is component c of the proxy-s weight.
  // Each row is 2 KB and 64B-aligned -> three contiguous streams in m2pAccum.
  float fhat[3][NP3];
  float hx, hy, hz;   // box half-extents
};

// 1-D barycentric Lagrange basis L[0..PDEG] for coordinate y on an axis with
// box center c_d and half-extent h_d. Verbatim port of bary::lagrange1d
// (bary_stokes.cuh:154-189) including its two guards:
//  * collapsed axis (h_d ~ 0): degree-0 in that axis, L[0]=1;
//  * exact node hit (|y - s_k| <= tol): L_k=1, rest 0 (tight AABBs always have
//    a particle sitting on the endpoint nodes, so this fires routinely).
inline void lagrange1d(double y, double c_d, double h_d, double L[NP])
{
  const Tables &tb = tables();
  if (!(h_d > 1e-20)) {                 // collapsed axis
    L[0] = 1.0;
    for (int k = 1; k < NP; ++k) L[k] = 0.0;
    return;
  }
  const double tol = 1e-6 * h_d;
  double s[NP];
  int hit = -1;
  for (int k = 0; k < NP; ++k) {
    s[k] = c_d + h_d * tb.cheb[k];
    if (std::fabs(y - s[k]) <= tol) hit = k;
  }
  if (hit >= 0) {
    for (int k = 0; k < NP; ++k) L[k] = (k == hit) ? 1.0 : 0.0;
    return;
  }
  double sum = 0.0;
  double dL[NP];
  for (int k = 0; k < NP; ++k) {
    const double ck = tb.baryw[k] / (y - s[k]);
    dL[k] = ck;
    sum += ck;
  }
  const double inv = 1.0 / sum;
  for (int k = 0; k < NP; ++k) L[k] = dL[k] * inv;
}

// Node geometry from the (refit) BVH node bounds, exactly as combine() derives
// it (bary_stokes.cuh:481-492): fp32 center, squared half-diagonal, and fp32
// half-extents.
inline void initNodeGeometry(const box3f &bounds, NodeMAC &nm, NodeM2P &m2p)
{
  const vec3f c  = bounds.center();
  const vec3f sz = bounds.size();
  nm.cx = c.x; nm.cy = c.y; nm.cz = c.z;
  nm.halfDiag2 = 0.25f * (sz.x * sz.x + sz.y * sz.y + sz.z * sz.z);
  m2p.hx = 0.5f * sz.x;
  m2p.hy = 0.5f * sz.y;
  m2p.hz = 0.5f * sz.z;
}

// P2M: leaf branch of BaryStokes::combine (bary_stokes.cuh:512-558, minus
// owner/structured/crossover plumbing). For every source in the leaf's bucket
// spans: 1-D barycentric bases in x/y/z, tensor outer product, accumulate the
// force into fp64 facc, cast to the fp32 fhat on the final store. Positions
// are the fp32 bucket-ordered points (same as the GPU upward pass); forces are
// fp64 bucket-ordered.
inline void p2mLeaf(const box3f &bounds,
                    const uint32_t *primIDs, uint64_t off, uint32_t cnt,
                    const int *bucketBegin, const int *bucketEnd,
                    const vec3f *pos, const vec3d *force,
                    NodeMAC &nm, NodeM2P &m2p)
{
  initNodeGeometry(bounds, nm, m2p);

  double facc[3][NP3];
  for (int c = 0; c < 3; ++c)
    for (int s = 0; s < NP3; ++s) facc[c][s] = 0.0;

  for (uint32_t t = 0; t < cnt; ++t) {
    const uint32_t bid = primIDs[off + t];
    const int begin = bucketBegin[bid];
    const int end   = bucketEnd[bid];
    for (int pid = begin; pid < end; ++pid) {
      const vec3f y = pos[pid];
      const vec3d f = force[pid];
      double Lx[NP], Ly[NP], Lz[NP];
      lagrange1d(y.x, nm.cx, m2p.hx, Lx);
      lagrange1d(y.y, nm.cy, m2p.hy, Ly);
      lagrange1d(y.z, nm.cz, m2p.hz, Lz);
      for (int k1 = 0; k1 < NP; ++k1) {
        const double wx = Lx[k1];
        for (int k2 = 0; k2 < NP; ++k2) {
          const double wxy = wx * Ly[k2];
          const int base = (k1 * NP + k2) * NP;
          for (int k3 = 0; k3 < NP; ++k3) {
            const double w = wxy * Lz[k3];
            facc[0][base + k3] += w * f.x;
            facc[1][base + k3] += w * f.y;
            facc[2][base + k3] += w * f.z;
          }
        }
      }
    }
  }

  for (int c = 0; c < 3; ++c)
    for (int s = 0; s < NP3; ++s) m2p.fhat[c][s] = (float)facc[c][s];
}

// M2M: inner branch of BaryStokes::combine (bary_stokes.cuh:559-631). For each
// child, re-interpolate its proxy weights onto the parent grid via three
// separable contraction passes (q3->k3, q2->k2, q1->k1), O(3*NP^4) per child.
// Children must be finalized before the parent (descending-index sweep).
inline void m2mInner(const box3f &bounds, uint64_t childOff,
                     const NodeMAC *macArr, const NodeM2P *m2pArr,
                     NodeMAC &nm, NodeM2P &m2p)
{
  const Tables &tb = tables();
  initNodeGeometry(bounds, nm, m2p);

  double facc[3][NP3];
  for (int c = 0; c < 3; ++c)
    for (int s = 0; s < NP3; ++s) facc[c][s] = 0.0;

  for (int ch = 0; ch < 2; ++ch) {
    const uint64_t cid = childOff + (uint64_t)ch;
    const NodeMAC &cmac = macArr[cid];
    const NodeM2P &cm2p = m2pArr[cid];

    // Parent-grid bases of the child's proxy coordinates, one row per child
    // grid index q. Child proxy positions widen exactly like the GPU:
    // fp32 center/half widened to fp64, fp64 cheb node.
    double Lmx[NP][NP], Lmy[NP][NP], Lmz[NP][NP];
    for (int q = 0; q < NP; ++q) {
      lagrange1d((double)cmac.cx + (double)cm2p.hx * tb.cheb[q], nm.cx, m2p.hx, Lmx[q]);
      lagrange1d((double)cmac.cy + (double)cm2p.hy * tb.cheb[q], nm.cy, m2p.hy, Lmy[q]);
      lagrange1d((double)cmac.cz + (double)cm2p.hz * tb.cheb[q], nm.cz, m2p.hz, Lmz[q]);
    }

    double tmp_a[3][NP3], tmp_b[3][NP3];
    // pass 1: contract child q3 -> parent k3
    for (int q1 = 0; q1 < NP; ++q1)
      for (int q2 = 0; q2 < NP; ++q2)
        for (int k3 = 0; k3 < NP; ++k3) {
          double sx = 0.0, sy = 0.0, sz = 0.0;
          for (int q3 = 0; q3 < NP; ++q3) {
            const double w = Lmz[q3][k3];
            const int qs = proxyIdx(q1, q2, q3);
            sx += w * (double)cm2p.fhat[0][qs];
            sy += w * (double)cm2p.fhat[1][qs];
            sz += w * (double)cm2p.fhat[2][qs];
          }
          const int s = proxyIdx(q1, q2, k3);
          tmp_a[0][s] = sx; tmp_a[1][s] = sy; tmp_a[2][s] = sz;
        }
    // pass 2: contract child q2 -> parent k2
    for (int q1 = 0; q1 < NP; ++q1)
      for (int k2 = 0; k2 < NP; ++k2)
        for (int k3 = 0; k3 < NP; ++k3) {
          double sx = 0.0, sy = 0.0, sz = 0.0;
          for (int q2 = 0; q2 < NP; ++q2) {
            const double w = Lmy[q2][k2];
            const int s = proxyIdx(q1, q2, k3);
            sx += w * tmp_a[0][s];
            sy += w * tmp_a[1][s];
            sz += w * tmp_a[2][s];
          }
          const int s = proxyIdx(q1, k2, k3);
          tmp_b[0][s] = sx; tmp_b[1][s] = sy; tmp_b[2][s] = sz;
        }
    // pass 3: contract child q1 -> parent k1
    for (int k1 = 0; k1 < NP; ++k1)
      for (int k2 = 0; k2 < NP; ++k2)
        for (int k3 = 0; k3 < NP; ++k3) {
          double sx = 0.0, sy = 0.0, sz = 0.0;
          for (int q1 = 0; q1 < NP; ++q1) {
            const double w = Lmx[q1][k1];
            const int s = proxyIdx(q1, k2, k3);
            sx += w * tmp_b[0][s];
            sy += w * tmp_b[1][s];
            sz += w * tmp_b[2][s];
          }
          const int s = proxyIdx(k1, k2, k3);
          facc[0][s] += sx;
          facc[1][s] += sy;
          facc[2][s] += sz;
        }
  }

  for (int c = 0; c < 3; ++c)
    for (int s = 0; s < NP3; ++s) m2p.fhat[c][s] = (float)facc[c][s];
}

// M2P: Stokeslet far field u += sum_s [ fhat_s / r + R (R.fhat_s)/r^3 ] over
// the NP3 proxy points, in the factored form of m2pWarp (bary_stokes.cuh:
// 407-410): u_i += ir*(f_i + R_i*q), q = (R.f)/r^2. Serial per-target version
// of the (unused) BaryStokes::m2p (bary_stokes.cuh:332-373), restructured as a
// single flat proxy loop over contiguous streams so it auto-vectorizes.
// fp32 target/center/half-extents widened to fp64 (exact), fp32 moments
// widened on load, fp64 accumulation. UNSCALED like p2p64.
#if defined(TCCPU_M2P_FP32) && defined(__AVX512F__)
// -- experimental M2P: fp32 evaluation, 16 lanes ----------------------------
// The loop below is FMA-issue bound (~82% of the socket's fp64 FMA peak after
// the rsqrt swap), so the only remaining hardware lever is halving the vector
// count. Everything the far field touches is already fp32 (moments) or exactly
// representable geometry, so this evaluates R, r, and the Stokeslet in fp32 --
// 16 proxies per vector, no vcvtps2pd -- and only the per-node reduction is
// widened back to fp64. Costs accuracy: the fp32 accumulation of NP3 terms is
// the floor, not the fp32 moments. Kept as an explicit precision/speed A/B.
inline void m2pAccum(const NodeM2P &nd, const NodeMAC &nm, vec3f Tp,
                     double &u0, double &u1, double &u2)
{
  const Tables &tb = tables();
  const __m512 dx = _mm512_set1_ps(Tp.x - nm.cx);
  const __m512 dy = _mm512_set1_ps(Tp.y - nm.cy);
  const __m512 dz = _mm512_set1_ps(Tp.z - nm.cz);
  const __m512 hx = _mm512_set1_ps(nd.hx);
  const __m512 hy = _mm512_set1_ps(nd.hy);
  const __m512 hz = _mm512_set1_ps(nd.hz);
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 half = _mm512_set1_ps(0.5f);
  const __m512 three_half = _mm512_set1_ps(1.5f);
  const __m512 zero = _mm512_setzero_ps();
  __m512 a0 = zero, a1 = zero, a2 = zero;
  auto body = [&](int s, __mmask16 lane) {
    const __m512 Rx =
        _mm512_fnmadd_ps(hx, _mm512_maskz_loadu_ps(lane, tb.chebXf + s), dx);
    const __m512 Ry =
        _mm512_fnmadd_ps(hy, _mm512_maskz_loadu_ps(lane, tb.chebYf + s), dy);
    const __m512 Rz =
        _mm512_fnmadd_ps(hz, _mm512_maskz_loadu_ps(lane, tb.chebZf + s), dz);
    const __m512 r2 =
        _mm512_fmadd_ps(Rx, Rx, _mm512_fmadd_ps(Ry, Ry, _mm512_mul_ps(Rz, Rz)));
    const __mmask16 nz = _mm512_cmp_ps_mask(r2, zero, _CMP_NEQ_OQ);
    const __m512 r2s = _mm512_mask_blend_ps(nz, one, r2);
    // rsqrt14 + one Newton step is already exact to fp32 (14 -> 28 bits).
    __m512 y = _mm512_rsqrt14_ps(r2s);
    y = _mm512_mul_ps(y, _mm512_fnmadd_ps(_mm512_mul_ps(half, r2s),
                                          _mm512_mul_ps(y, y), three_half));
    const __m512 ir = _mm512_maskz_mov_ps(nz, y);
    const __m512 fx = _mm512_maskz_loadu_ps(lane, nd.fhat[0] + s);
    const __m512 fy = _mm512_maskz_loadu_ps(lane, nd.fhat[1] + s);
    const __m512 fz = _mm512_maskz_loadu_ps(lane, nd.fhat[2] + s);
    const __m512 q = _mm512_mul_ps(
        _mm512_fmadd_ps(Rx, fx, _mm512_fmadd_ps(Ry, fy, _mm512_mul_ps(Rz, fz))),
        _mm512_mul_ps(ir, ir));
    a0 = _mm512_fmadd_ps(ir, _mm512_fmadd_ps(Rx, q, fx), a0);
    a1 = _mm512_fmadd_ps(ir, _mm512_fmadd_ps(Ry, q, fy), a1);
    a2 = _mm512_fmadd_ps(ir, _mm512_fmadd_ps(Rz, q, fz), a2);
  };
  constexpr int kFull = NP3 & ~15;
  for (int s = 0; s < kFull; s += 16) body(s, (__mmask16)0xFFFF);
  if constexpr (NP3 & 15)
    body(kFull, (__mmask16)((1u << (NP3 & 15)) - 1u));
  u0 += (double)_mm512_reduce_add_ps(a0);
  u1 += (double)_mm512_reduce_add_ps(a1);
  u2 += (double)_mm512_reduce_add_ps(a2);
}
#elif defined(TCCPU_FAST_RSQRT) && defined(__AVX512F__)
// -- experimental M2P: vrsqrt14pd + Newton instead of vsqrtpd + vdivpd -------
// Profiling (agg-056, 96T, p10000) put the FP divide/sqrt unit at ~59%
// occupancy in the loop below -- 18 of the ~31 cycles each 8-proxy vector
// costs -- while the FMA ports sat at ~56%. vrsqrt14pd gives 14 correct bits
// off the FMA/vector pipes; TCCPU_RSQRT_NR Newton steps take that to ~28 bits
// (one step) or full fp64 (two). One step is the default because the moments
// themselves are fp32 (~6e-8 relative), so 4e-9 is well under the existing
// noise floor -- gated by the p1000 mobility solve and the direct spot check.
#ifndef TCCPU_RSQRT_NR
#define TCCPU_RSQRT_NR 1
#endif
inline void m2pAccum(const NodeM2P &nd, const NodeMAC &nm, vec3f Tp,
                     double &u0, double &u1, double &u2)
{
  const Tables &tb = tables();
  const __m512d dx = _mm512_set1_pd((double)Tp.x - (double)nm.cx);
  const __m512d dy = _mm512_set1_pd((double)Tp.y - (double)nm.cy);
  const __m512d dz = _mm512_set1_pd((double)Tp.z - (double)nm.cz);
  const __m512d hx = _mm512_set1_pd((double)nd.hx);
  const __m512d hy = _mm512_set1_pd((double)nd.hy);
  const __m512d hz = _mm512_set1_pd((double)nd.hz);
  const __m512d one = _mm512_set1_pd(1.0);
  const __m512d half = _mm512_set1_pd(0.5);
  const __m512d three_half = _mm512_set1_pd(1.5);
  const __m512d zero = _mm512_setzero_pd();
  __m512d a0 = zero, a1 = zero, a2 = zero;
  // NP3 is a multiple of 8 only for some PDEG (512 for 7, 216 for 5, but 343
  // for 6), so the tail runs the same body under a compile-time lane mask.
  auto body = [&](int s, __mmask8 lane) {
    const __m512d Rx =
        _mm512_fnmadd_pd(hx, _mm512_maskz_loadu_pd(lane, tb.chebX + s), dx);
    const __m512d Ry =
        _mm512_fnmadd_pd(hy, _mm512_maskz_loadu_pd(lane, tb.chebY + s), dy);
    const __m512d Rz =
        _mm512_fnmadd_pd(hz, _mm512_maskz_loadu_pd(lane, tb.chebZ + s), dz);
    const __m512d r2 =
        _mm512_fmadd_pd(Rx, Rx, _mm512_fmadd_pd(Ry, Ry, _mm512_mul_pd(Rz, Rz)));
    // r2 == 0 would make rsqrt14 return +Inf and the Newton step NaN, so the
    // zero lanes are evaluated at 1.0 and masked back to 0 (same contract as
    // the scalar ternary). Masked-off tail lanes also load fhat = 0, so they
    // contribute exactly nothing to the accumulators.
    const __mmask8 nz = _mm512_cmp_pd_mask(r2, zero, _CMP_NEQ_OQ);
    const __m512d r2s = _mm512_mask_blend_pd(nz, one, r2);
    __m512d y = _mm512_rsqrt14_pd(r2s);
    for (int it = 0; it < TCCPU_RSQRT_NR; ++it)  // y *= 1.5 - 0.5*r2*y*y
      y = _mm512_mul_pd(
          y, _mm512_fnmadd_pd(_mm512_mul_pd(half, r2s), _mm512_mul_pd(y, y),
                              three_half));
    const __m512d ir = _mm512_maskz_mov_pd(nz, y);
    const __m512d fx =
        _mm512_cvtps_pd(_mm256_maskz_loadu_ps(lane, nd.fhat[0] + s));
    const __m512d fy =
        _mm512_cvtps_pd(_mm256_maskz_loadu_ps(lane, nd.fhat[1] + s));
    const __m512d fz =
        _mm512_cvtps_pd(_mm256_maskz_loadu_ps(lane, nd.fhat[2] + s));
    const __m512d q = _mm512_mul_pd(
        _mm512_fmadd_pd(Rx, fx, _mm512_fmadd_pd(Ry, fy, _mm512_mul_pd(Rz, fz))),
        _mm512_mul_pd(ir, ir));
    a0 = _mm512_fmadd_pd(ir, _mm512_fmadd_pd(Rx, q, fx), a0);
    a1 = _mm512_fmadd_pd(ir, _mm512_fmadd_pd(Ry, q, fy), a1);
    a2 = _mm512_fmadd_pd(ir, _mm512_fmadd_pd(Rz, q, fz), a2);
  };
  constexpr int kFull = NP3 & ~7;
  for (int s = 0; s < kFull; s += 8) body(s, (__mmask8)0xFF);
  if constexpr (NP3 & 7)
    body(kFull, (__mmask8)((1u << (NP3 & 7)) - 1u));
  u0 += _mm512_reduce_add_pd(a0);
  u1 += _mm512_reduce_add_pd(a1);
  u2 += _mm512_reduce_add_pd(a2);
}
#else
inline void m2pAccum(const NodeM2P &nd, const NodeMAC &nm, vec3f Tp,
                     double &u0, double &u1, double &u2)
{
  const Tables &tb = tables();
  const double dx = (double)Tp.x - (double)nm.cx;
  const double dy = (double)Tp.y - (double)nm.cy;
  const double dz = (double)Tp.z - (double)nm.cz;
  const double hx = (double)nd.hx, hy = (double)nd.hy, hz = (double)nd.hz;
  const float *F0 = nd.fhat[0];
  const float *F1 = nd.fhat[1];
  const float *F2 = nd.fhat[2];
  double a0 = 0.0, a1 = 0.0, a2 = 0.0;
#pragma omp simd reduction(+ : a0, a1, a2)
  for (int s = 0; s < NP3; ++s) {
    // R = T - (center + h*cheb) = (T - center) - h*cheb
    const double Rx = std::fma(-hx, tb.chebX[s], dx);
    const double Ry = std::fma(-hy, tb.chebY[s], dy);
    const double Rz = std::fma(-hz, tb.chebZ[s], dz);
    const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
    const double ir = (r2 == 0.0) ? 0.0 : 1.0 / std::sqrt(r2);
    const double fx = (double)F0[s];
    const double fy = (double)F1[s];
    const double fz = (double)F2[s];
    const double q  = (Rx * fx + Ry * fy + Rz * fz) * (ir * ir);
    a0 += ir * (fx + Rx * q);
    a1 += ir * (fy + Ry * q);
    a2 += ir * (fz + Rz * q);
  }
  u0 += a0; u1 += a1; u2 += a2;
}
#endif  // TCCPU_FAST_RSQRT

} // namespace bary_cpu
