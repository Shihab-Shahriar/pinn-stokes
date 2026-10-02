// SPDX-License-Identifier: Apache-2.0
//
// Multipole module: FMM3D-style spherical-harmonic Stokes expansion (order <= 12).
//
// The Stokeslet is decomposed into FOUR scalar Laplace multipole channels
// (FMM3D's stokkernels trick, prototyped/tested in ../stokes_fmm_moment_demo.py):
//   channels 0..2 carry the force components, channel 3 carries dot(f, S).
// Moments are complex solid-harmonic coefficients in FLAT triangular storage
// (K = (p+1)(p+2)/2 coefficients per channel, k(n,m) = n(n+1)/2 + m, m >= 0;
// m < 0 is implied by conjugation). M2P fuses the four channels: at a target it
// forms H[k] = M3[k] - Dx*M0[k] - Dy*M1[k] - Dz*M2[k] and evaluates 3 scalar
// potentials + ONE gradient:
//   u_i = (1/a) * sum_k w_k * ( Re(M_i[k]*val[k]) + Re(H[k]*g_i[k]) ),
// w_k = (m==0 ? 1 : 2) (real-pair weight), val = irregular solid harmonic
// I_nm(D), g = its Cartesian D-gradient via the degree-(n+1) ladder.
//
// ALL-NORMALIZED convention (differs from the demo in two deliberate ways):
//  * prefactor: the engine multiplies m2p/p2p output by 1/(8*pi*mu) once
//    (stokes_kernel.cuh); the demo's INV4PI * 0.5 equals exactly 1/(8*pi), so
//    charges here are RAW: q0..q2 = f components, q3 = dot(f, S).
//  * node scale: everything runs in normalized coordinates S = (y-c)/a and
//    D = (x-c)/a with a = node half-diagonal (fp32 dynamic-range uniformity,
//    same treatment as cartesian_stokes.cuh). q3 uses NORMALIZED S (demo uses
//    physical s), which moves one factor a out of channel 3; the M2P then
//    carries a single uniform 1/a and the M2M channel-3 translation gains one
//    extra ratio factor (see m2mTranslate).
//
// Correctness traps (each guarded by sph_selftest.cu):
//  1. Phase conventions: the REGULAR basis (P2M and the M2M RD table) uses
//     w = x - i*y (demo conj_phase=True); the IRREGULAR basis (M2P) uses
//     w = x + i*y. Flipping either breaks the real-pair weight trick.
//  2. Recurrence asymmetry: the regular vertical recurrence multiplies the c2
//     term by r2 only; the irregular one multiplies the WHOLE bracket by 1/r2,
//     and its diagonal + first vertical step carry an extra 1/r2.
//  3. dminus sign rule: +1 for m==0 (with conj(I[n+1,1])), -1 for m >= 1.
//  4. M2M channel 3: M3'_parent = ratio * T(M3'_child) + delta_norm . (T M0..2)
//     -- one extra ratio on the translated channel, coupling per child with the
//     TRANSLATED channels.
//  5. M2M phase i^e with possibly negative e: reduce e mod 4, apply as swap/sign.
//
// Policy surface (same as cartesian_stokes.cuh / bary_stokes.cuh):
//   SphericalStokes::{NodeMAC, NodeM2P, MAX_ORDER, LANE_BATCH_M2P, name, setup,
//                     upwardPass, m2p<ORDER>, m2pWarp<ORDER>, combine}
// LANE_BATCH_M2P = false: m2pWarp is warp-cooperative (lane k owns coefficient
// k, k+32; per-lane irregular column walks in rolling registers, no shared
// memory, __shfl_down_sync reduce). fp64 moment storage, fp64 evaluation,
// rsqrtf-seeded fp64 Newton for 1/r.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda/refit_aggregate.h"

// Defer to the includer's CUDA_CHECK (treecode.cuh has a richer file:line one);
// fallback keeps this header self-contained for sph_selftest.cu.
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

namespace sph {

constexpr int MAX_P = 12;                             // max expansion order p
constexpr int NB    = MAX_P + 2;                      // table rows: degrees 0..p+1
constexpr int K_MAX = (MAX_P + 1) * (MAX_P + 2) / 2;  // 91 coeffs per channel

__device__ __host__ inline constexpr int kdimOf(int p) { return (p + 1) * (p + 2) / 2; }
__device__ __host__ inline constexpr int kOf(int n, int m) { return n * (n + 1) / 2 + m; }

// ---- minimal complex-double helpers (kept POD; nvcc folds these fully) ----
struct dcplx { double re, im; };
__device__ __forceinline__ dcplx cmul(dcplx a, dcplx b)
{ return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
__device__ __forceinline__ dcplx cscale(dcplx a, double s) { return {a.re * s, a.im * s}; }
__device__ __forceinline__ dcplx cconj(dcplx a) { return {a.re, -a.im}; }

// ---- precomputed recurrence/ladder/translation constants ------------------
// Indexed by lane-dependent (n,m) in the hot m2pWarp, so these live in device
// global memory read through __ldg (NOT __constant__: divergent lane addresses
// would serialize on the constant bus -- same rationale as bary_stokes.cuh's
// c_k1/c_k2/c_k3 tables). Filled by setup(); ~7 KB total, L2/tex resident.
__device__ unsigned char c_kn_gs[K_MAX];        // flat k -> degree n
__device__ unsigned char c_km_gs[K_MAX];        // flat k -> order m
__device__ double c_diagstep_gs[NB];            // -sqrt((2m-1)/(2m)), m>=1
__device__ double c_sq2mp1_gs[NB];              // sqrt(2m+1)
__device__ double c_c2_gs[NB * NB];             // sqrt((n+m-1)(n-m-1))
__device__ double c_invpair_gs[NB * NB];        // 1/sqrt((n-m)(n+m))
__device__ double c_cpl_gs[NB * NB];            // sqrt((n+m+1)(n+m+2))   ladder
__device__ double c_cmi_gs[NB * NB];            // sqrt((n-m+1)(n-m+2))   ladder
__device__ double c_cz_gs[NB * NB];             // sqrt((n+m+1)(n-m+1))   ladder
__device__ double c_A_gs[NB * NB];              // A(n,m)=1/sqrt((n-m)!(n+m)!)
__device__ double c_invA_gs[NB * NB];           // 1/A(n,m)

// m2pWarp slot mapping (neighbour-shuffle scheme): coefficient rows (degrees)
// are packed into 32-lane passes so that a row NEVER straddles a pass boundary
// and coefficients of a row sit on CONSECUTIVE lanes. Then lane k(n,m)+-1 holds
// (n,m+-1), whose column walk produces exactly the I[n+1,m+-1] this lane's
// gradient ladder needs -- two warp shuffles replace two of the three per-lane
// column walks. Packing (MAX_P=12): rows 0..6 -> slots 0..27 (bin 1, 4 idle
// lanes), rows 7..9 -> slots 32..58 (bin 2), row 10 -> slots 64..74 (bin 3),
// row 11 -> slots 75..86, row 12 -> slots 96..108 (bin 4). 0xff = idle slot
// sentinel. Last used slot 108 with 4 passes reads lane index up to 127, so
// SLOT_MAX must be >= 128.
constexpr int SLOT_MAX = 128;
__device__ unsigned char c_sn_gs[SLOT_MAX];     // slot -> degree n (0xff idle)
__device__ unsigned char c_sm_gs[SLOT_MAX];     // slot -> order m
// Total slot span (last used slot + 1) under the never-straddle-a-32-boundary
// packing; NPASS = ceil(slotCount/32). constexpr loop mirrors setup()'s packer
// so it stays correct for any MAX_P (the old p<=9 closed form had a single bin
// jump baked in).
__device__ __host__ inline constexpr int slotCount(int p)
{
  int slot = 0;
  for (int n = 0; n <= p; ++n) {
    if ((slot & 31) + (n + 1) > 32) slot = (slot + 31) & ~31;
    slot += n + 1;
  }
  return slot;
}

// Scene globals smuggled into combine() during the upward pass (combine()'s
// signature is fixed by refit_aggregate). Suffix _gs: cartesian owns _g and
// bary owns _gb; all three policy headers coexist in the same TU.
__device__ const vec3f *d_pos_gs          = nullptr;
// fp64-geometry far field: bucket-ordered fp64 source positions (P2M in combine)
// and target positions (M2P). Same centered frame + bucket order as the fp32
// arrays, so a bucket index selects both. Set by the treecode before upward and
// traverse. Lift the fp32-coordinate accuracy floor without touching the fp32
// node center/scale (a consistent constant shift, promoted to fp64 in the delta).
__device__ const vec3d *d_pos64_gs        = nullptr;
__device__ const vec3d *d_tgt64_gs        = nullptr;
__device__ const vec3d *d_force_gs        = nullptr;
__device__ const int   *d_owner_gs        = nullptr;
__device__ uint32_t    *d_owner_mask_gs   = nullptr;
__device__ int          d_owner_mask_words_gs = 0;
__device__ const int   *d_bucket_begin_gs = nullptr;
__device__ const int   *d_bucket_end_gs   = nullptr;

// rsqrtf seed + ONE fp64 Newton step (rel err ~1e-13); no fp64 div/sqrt chains
// (same pattern as bary_stokes.cuh:293-294 / cartesian_stokes.cuh:195-196).
__device__ __forceinline__ void invR(double r2, double &ir, double &ir2)
{
  ir = (double)rsqrtf((float)r2);
  ir = ir * fma(-0.5 * r2, ir * ir, 1.5);
  ir2 = ir * ir;
}

// ---------------------------------------------------------------------------
// Irregular solid harmonics I_nm (M2P side), per-lane rolling column walk.
//   I[0,0]  = 1/r
//   I[m,m]  = -sqrt((2m-1)/(2m)) * w * ir2 * I[m-1,m-1],   w = x + i*y
//   I[m+1,m]= sqrt(2m+1) * z * ir2 * I[m,m]
//   I[n,m]  = ((2n-1) z I[n-1,m] - c2 I[n-2,m]) * ir2 / sqrt((n-m)(n+m))
// Walk column `col` from its diagonal seed to degree `nt`; returns I[nt,col]
// and (in *prev, when nt > col) I[nt-1,col]. MAXDEG is the compile-time unroll
// bound (ORDER+1); per-lane trip counts differ, so every step is predicated --
// control flow stays warp-convergent, state stays in rolling registers.
// ---------------------------------------------------------------------------
template<int MAXDEG>
__device__ __forceinline__ dcplx
irrWalk(dcplx seed, int col, int nt, double z, double ir2, dcplx *prev)
{
  dcplx a = seed;                               // I[col+s-2, col]
  dcplx b = seed;                               // I[col+s-1, col]
  if (nt > col)
    b = cscale(seed, __ldg(&c_sq2mp1_gs[col]) * z * ir2);
#pragma unroll
  for (int s = 2; s <= MAXDEG; ++s) {
    const int n = col + s;
    if (n <= nt) {
      const double f1 = (double)(2 * n - 1) * z;
      const double f2 = __ldg(&c_c2_gs[n * NB + col]);
      const double fs = ir2 * __ldg(&c_invpair_gs[n * NB + col]);
      const dcplx c = {(f1 * b.re - f2 * a.re) * fs,
                       (f1 * b.im - f2 * a.im) * fs};
      a = b; b = c;
    }
  }
  if (prev) *prev = a;
  return b;
}

// val = I[n,m](D) and its Cartesian D-gradient for one coefficient (n,m), via
// the degree-(n+1) neighbour ladder (demo irregular_basis):
//   dplus  = cpl * I[n+1,m+1]
//   dminus = (m==0 ? +1 : -1) * cmi * I[n+1,m-1]   (m==0: conj(I[n+1,1]))
//   gx = (dplus+dminus)/2, gy = -i(dplus-dminus)/2, gz = -cz * I[n+1,m]
// Three predicated column walks (m-1, m, m+1) off one shared diagonal chain.
template<int ORDER>
__device__ __forceinline__ void
irrValGrad(double Dx, double Dy, double Dz, double ir, double ir2,
           int n, int m, dcplx &val, dcplx &gx, dcplx &gy, dcplx &gz)
{
  const dcplx w = {Dx, Dy};                     // e^{+im phi} convention
  const int mMinus = (m >= 1) ? (m - 1) : 1;    // column -1 == conj(column 1)
  dcplx diag = {ir, 0.0};                       // I[0,0]
  dcplx sMi = diag, sM = diag, sP = diag;       // j==0 captures
#pragma unroll
  for (int j = 1; j <= ORDER + 1; ++j) {
    if (j <= m + 1) {
      diag = cscale(cmul(w, diag), __ldg(&c_diagstep_gs[j]) * ir2);
      if (j == mMinus) sMi = diag;
      if (j == m)      sM  = diag;
      if (j == m + 1)  sP  = diag;
    }
  }
  const int nt = n + 1;
  dcplx Iz = irrWalk<ORDER + 1>(sM, m, nt, Dz, ir2, &val);      // I[n+1,m], val=I[n,m]
  dcplx Ipl = irrWalk<ORDER + 1>(sP, m + 1, nt, Dz, ir2, nullptr);
  dcplx Imi = irrWalk<ORDER + 1>(sMi, mMinus, nt, Dz, ir2, nullptr);
  if (m == 0) Imi = cconj(Imi);

  const dcplx dplus  = cscale(Ipl, __ldg(&c_cpl_gs[n * NB + m]));
  const double sgn   = (m == 0) ? 1.0 : -1.0;
  const dcplx dminus = cscale(Imi, sgn * __ldg(&c_cmi_gs[n * NB + m]));
  gx = {0.5 * (dplus.re + dminus.re), 0.5 * (dplus.im + dminus.im)};
  gy = {0.5 * (dplus.im - dminus.im), -0.5 * (dplus.re - dminus.re)};
  gz = cscale(Iz, -__ldg(&c_cz_gs[n * NB + m]));
}

// ---------------------------------------------------------------------------
// Regular solid harmonics R_nm (P2M / M2M-RD side), streamed column-by-column
// (conj phase, w = x - i*y). The Sink receives (k, R) for every coefficient
// k(n,m), m-major generation order -- one source of truth for the regular
// recurrence, shared by P2M accumulation and the M2M RD table. ORDER is a
// compile-time bound: the nest fully unrolls, so every sink(k, R) index folds
// to a constant and the caller's K-sized accumulators index statically.
//   R[0,0]  = 1
//   R[m,m]  = -sqrt((2m-1)/(2m)) * w * R[m-1,m-1]
//   R[m+1,m]= sqrt(2m+1) * z * R[m,m]
//   R[n,m]  = ((2n-1) z R[n-1,m] - c2 r2 R[n-2,m]) / sqrt((n-m)(n+m))
// ---------------------------------------------------------------------------
template<int ORDER, class Sink>
__device__ inline void
regularWalk(double Sx, double Sy, double Sz, Sink &&sink)
{
  const double r2 = Sx * Sx + Sy * Sy + Sz * Sz;
  const dcplx w = {Sx, -Sy};                    // e^{-im phi} convention
  dcplx diag = {1.0, 0.0};
#pragma unroll
  for (int m = 0; m <= ORDER; ++m) {
    if (m > 0) diag = cscale(cmul(w, diag), c_diagstep_gs[m]);
    sink(kOf(m, m), diag);
    if (m == ORDER) continue;
    dcplx a = diag;
    dcplx b = cscale(diag, c_sq2mp1_gs[m] * Sz);
    sink(kOf(m + 1, m), b);
#pragma unroll
    for (int n = m + 2; n <= ORDER; ++n) {
      const double f1 = (double)(2 * n - 1) * Sz;
      const double f2 = c_c2_gs[n * NB + m] * r2;
      const double fs = c_invpair_gs[n * NB + m];
      const dcplx c = {(f1 * b.re - f2 * a.re) * fs,
                       (f1 * b.im - f2 * a.im) * fs};
      sink(kOf(n, m), c);
      a = b; b = c;
    }
  }
}

// Fused P2M for one source: charges q = (fx, fy, fz, f.S) share one basis pass.
template<int ORDER>
__device__ inline void
p2mPoint(const double S[3], const double f[3],
         double (&Macc)[4][kdimOf(ORDER)][2])
{
  const double q[4] = {f[0], f[1], f[2],
                       f[0] * S[0] + f[1] * S[1] + f[2] * S[2]};
  regularWalk<ORDER>(S[0], S[1], S[2], [&](int k, dcplx R) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      Macc[c][k][0] += q[c] * R.re;
      Macc[c][k][1] += q[c] * R.im;
    }
  });
}

// Full flat regular table (for the M2M translation's RD).
template<int ORDER>
__device__ inline void
regularTable(const double v[3], double (&RD)[kdimOf(ORDER)][2])
{
  regularWalk<ORDER>(v[0], v[1], v[2], [&](int k, dcplx R) {
    RD[k][0] = R.re; RD[k][1] = R.im;
  });
}

// Multiply by i^e (e may be negative): reduce mod 4, apply as swap/sign.
__device__ __forceinline__ dcplx applyPhase(dcplx z, int e)
{
  switch (static_cast<unsigned>(e) & 3u) {
  case 1:  return {-z.im,  z.re};
  case 2:  return {-z.re, -z.im};
  case 3:  return { z.im, -z.re};
  default: return z;
  }
}

// ---------------------------------------------------------------------------
// M2M: translate one child's moments (fp64, normalized child storage) to the
// parent center/scale and ACCUMULATE into Macc. Greengard-Rokhlin translation
// per channel (demo _m2m_channel):
//   T[j,k] = sum_{n<=j, |m|<=n, |k-m|<=j-n} ratio^(j-n) * Mc(j-n, k-m)
//            * i^(|k|-|m|-|k-m|) * A(n,m) A(j-n,k-m) / A(j,k) * RD(n,m)
// with RD = regular_solid(delta_norm), delta_norm = (c_child - c_parent)/a_p,
// ratio = a_child/a_parent; negative orders access by conjugation. Channel 3
// (normalized q3 storage; see header): one extra ratio on the translated
// channel plus the per-child coupling with the TRANSLATED channels 0..2:
//   Macc[3] += ratio * T3 + delta_norm . (T0, T1, T2).
// Factored out of combine() so sph_selftest can validate it in fp64 isolation.
// ORDER is compile-time so the local RD/rpow arrays are K-sized; the O(K^2)
// translation loops keep their loop structure (constant bounds, no forced
// unroll -- full unrolling would blow up codegen x9 orders for no gain).
// ---------------------------------------------------------------------------
template<int ORDER>
__device__ inline void
m2mTranslate(const double (&Mc)[4][kdimOf(ORDER)][2], double childScale,
             const double childCenter[3], const double parentCenter[3],
             double parentScale, double (&Macc)[4][kdimOf(ORDER)][2])
{
  const double invAp = 1.0 / parentScale;
  const double dn[3] = {(childCenter[0] - parentCenter[0]) * invAp,
                        (childCenter[1] - parentCenter[1]) * invAp,
                        (childCenter[2] - parentCenter[2]) * invAp};
  const double ratio = childScale * invAp;

  double RD[kdimOf(ORDER)][2];
  regularTable<ORDER>(dn, RD);
  double rpow[ORDER + 1];
  rpow[0] = 1.0;
  for (int j = 1; j <= ORDER; ++j) rpow[j] = rpow[j - 1] * ratio;

  for (int j = 0; j <= ORDER; ++j)
    for (int kk = 0; kk <= j; ++kk) {
      const double invA_val = c_invA_gs[j * NB + kk];
      dcplx acc[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
      for (int n = 0; n <= j; ++n) {
        const int jn = j - n;
        const double rpow_invA = invA_val * rpow[jn];
        for (int m = -n; m <= n; ++m) {
          const int km = kk - m;
          const int akm = km < 0 ? -km : km;
          if (akm > jn) continue;
          const int am = m < 0 ? -m : m;
          const double coef = c_A_gs[n * NB + am] * c_A_gs[jn * NB + akm]
                            * rpow_invA;
          dcplx R = {RD[kOf(n, am)][0], RD[kOf(n, am)][1]};
          if (m < 0) R = cconj(R);
          R = cscale(applyPhase(R, kk - am - akm), coef);
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            dcplx Mm = {Mc[c][kOf(jn, akm)][0], Mc[c][kOf(jn, akm)][1]};
            if (km < 0) Mm = cconj(Mm);
            const dcplx t = cmul(Mm, R);
            acc[c].re += t.re; acc[c].im += t.im;
          }
        }
      }
      const int k = kOf(j, kk);
      Macc[0][k][0] += acc[0].re;  Macc[0][k][1] += acc[0].im;
      Macc[1][k][0] += acc[1].re;  Macc[1][k][1] += acc[1].im;
      Macc[2][k][0] += acc[2].re;  Macc[2][k][1] += acc[2].im;
      Macc[3][k][0] += ratio * acc[3].re
                     + dn[0] * acc[0].re + dn[1] * acc[1].re + dn[2] * acc[2].re;
      Macc[3][k][1] += ratio * acc[3].im
                     + dn[0] * acc[0].im + dn[1] * acc[1].im + dn[2] * acc[2].im;
    }
}

} // namespace sph

// ===========================================================================
//  The policy.
// ===========================================================================
struct SphericalStokes {
  static constexpr int  MAX_ORDER = sph::MAX_P;    // 12
  static constexpr int  K_MAX     = sph::K_MAX;    // 91 coefficients/channel
  // m2pWarp is warp-cooperative (lane k owns coefficient k, k+32): K is 28..55
  // for the useful orders, so a warp is well fed, and the per-lane column walks
  // keep registers bounded -- the opposite trade to CartesianStokes' 35-slot
  // register-resident serial m2p.
  static constexpr bool LANE_BATCH_M2P = false;
  // fp64-geometry far field: the engine feeds m2pWarp fp64 target positions (via
  // d_tgt64_gs) so the M2P delta D=(T-c)/a isn't floored by fp32 target-coord
  // quantization. Other policies leave this false and keep the fp32 target path.
  // -DSPH_FP32_GEOM reverts to fp32 far-field geometry (A/B anchor; see combine).
#ifdef SPH_FP32_GEOM
  static constexpr bool WANTS_FP64_TARGET = false;
#else
  static constexpr bool WANTS_FP64_TARGET = true;
#endif
  // Per-node far-field contribution type (fp64 here; only BaryStokes has the
  // fp32 fast path, see WIDEBVH_FP32_LEVEL in stokes_kernel.cuh).
  using FarVec = vec3d;

  // SoA node types. Traversal loads only NodeMAC (16 bytes) per node visit.
  struct __align__(16) NodeMAC {
    float cx, cy, cz, halfDiag2;
  };

  struct __align__(16) NodeM2P {
    int ownerMin, ownerMax;
    // Node length scale a (half-diagonal). Moments are stored in NORMALIZED
    // coordinates S = (y-c)/a (fp32 dynamic-range uniformity across levels);
    // m2p evaluates at D = (x-c)/a and multiplies the velocity by 1/a once.
    float scale, pad0;
    // [coeff][channel 0..3][re,im]: lane k's chunk M[k] is one 16B-aligned,
    // 64-byte block of 4 complex channels (four double2 loads); a warp covering
    // k=0..31 reads 2048 contiguous bytes. 16 + 91*64 = 5840 B/node (only the
    // k < kdimOf(order) head is read/written at run time; the tail is cold at
    // orders below MAX).
    double M[K_MAX][4][2];
  };

  static const char *name() { return "spherical(FMM3D-Stokes)"; }
  static void setup(int order);

  // Point the P2M (combine) / M2P at the fp64 bucket-ordered source / target
  // positions. Set by the treecode before the upward pass and the traverse; the
  // pointers are stable across reapply (geometry is fixed).
  static void setSourcePos64Symbol(const vec3d *p)
  { CUDA_CHECK(cudaMemcpyToSymbol(sph::d_pos64_gs, &p, sizeof(p))); }
  static void setTargetPos64Symbol(const vec3d *p)
  { CUDA_CHECK(cudaMemcpyToSymbol(sph::d_tgt64_gs, &p, sizeof(p))); }

  // TVec is the target-position type (vec3f fp32 or vec3d fp64-geometry); the
  // engine deduces it from the argument and Tp widens to fp64 in the delta.
  template<int ORDER, class TVec>
  static __device__ inline vec3d m2p(const NodeM2P &m2p_data,
                                     vec3f center, TVec Tp);

  template<int ORDER, class TVec>
  static __device__ inline vec3d m2pWarp(const NodeM2P &m2p_data,
                                         vec3f center, TVec Tp, int lane,
                                         unsigned int mask);

  static void upwardPass(bvh3f bvh, NodeMAC *d_mac, NodeM2P *d_m2p,
                         const vec3f *d_pos, const vec3d *d_force,
                         const int *d_owner, uint32_t *d_ownerMask,
                         int ownerMaskWords,
                         const int *d_bucketBegin, const int *d_bucketEnd,
                         int order, cudaStream_t stream = 0);

  // Order-specialized refit callback: all upward local arrays are K-sized
  // (K = kdimOf(ORDER)) instead of K_MAX-sized, and the moments are written
  // straight to d_m2p_gs[nodeID] (no local NodeM2P staging buffer). At the
  // default p=6 this shrinks the callback's stack frame ~11.4 KB -> ~4 KB.
  template<int ORDER>
  static __device__ void combine(bvh3f bvh, NodeMAC mac[], int nodeID);
};

__device__ SphericalStokes::NodeM2P *d_m2p_gs = nullptr;

// One combine instantiation per order; upwardPass picks by runtime order.
__device__ void (*SphericalStokes_combine_fps[SphericalStokes::MAX_ORDER])(
    bvh3f, SphericalStokes::NodeMAC[], int) = {
  &SphericalStokes::combine<1>, &SphericalStokes::combine<2>,
  &SphericalStokes::combine<3>, &SphericalStokes::combine<4>,
  &SphericalStokes::combine<5>, &SphericalStokes::combine<6>,
  &SphericalStokes::combine<7>, &SphericalStokes::combine<8>,
  &SphericalStokes::combine<9>, &SphericalStokes::combine<10>,
  &SphericalStokes::combine<11>, &SphericalStokes::combine<12>,
};

// ===========================================================================
//  Definitions.
// ===========================================================================

namespace sph {

// Per-coefficient H-contraction: one 64-byte coalesced moment chunk for k (four
// double2 loads, one per channel); H is formed in registers and consumed
// immediately. Shared by the serial m2p and both m2pWarp variants.
__device__ __forceinline__ void
contractK(const SphericalStokes::NodeM2P &node,
          double Dx, double Dy, double Dz, int k, int m,
          dcplx val, dcplx gx, dcplx gy, dcplx gz,
          double &ux, double &uy, double &uz)
{
  const double2 *mc = reinterpret_cast<const double2 *>(node.M[k]);
  const double2 d0 = __ldg(&mc[0]);             // M0.re M0.im
  const double2 d1 = __ldg(&mc[1]);             // M1.re M1.im
  const double2 d2 = __ldg(&mc[2]);             // M2.re M2.im
  const double2 d3 = __ldg(&mc[3]);             // M3.re M3.im
  const dcplx M0 = {d0.x, d0.y};
  const dcplx M1 = {d1.x, d1.y};
  const dcplx M2 = {d2.x, d2.y};
  const dcplx M3 = {d3.x, d3.y};
  const dcplx H  = {M3.re - Dx * M0.re - Dy * M1.re - Dz * M2.re,
                    M3.im - Dx * M0.im - Dy * M1.im - Dz * M2.im};

  const double w = (m == 0) ? 1.0 : 2.0;        // real-pair weight
  ux += w * (M0.re * val.re - M0.im * val.im + H.re * gx.re - H.im * gx.im);
  uy += w * (M1.re * val.re - M1.im * val.im + H.re * gy.re - H.im * gy.im);
  uz += w * (M2.re * val.re - M2.im * val.im + H.re * gz.re - H.im * gz.im);
}

// Per-coefficient M2P contribution via the self-contained 3-walk basis
// (irrValGrad): the serial reference path, independent of the shuffle scheme.
template<int ORDER>
__device__ __forceinline__ void
m2pLane(const SphericalStokes::NodeM2P &node,
        double Dx, double Dy, double Dz, double ir, double ir2, int k,
        double &ux, double &uy, double &uz)
{
  const int n = __ldg(&c_kn_gs[k]);
  const int m = __ldg(&c_km_gs[k]);
  dcplx val, gx, gy, gz;
  irrValGrad<ORDER>(Dx, Dy, Dz, ir, ir2, n, m, val, gx, gy, gz);
  contractK(node, Dx, Dy, Dz, k, m, val, gx, gy, gz, ux, uy, uz);
}

} // namespace sph

// Serial reference evaluation (the engine never calls this while
// LANE_BATCH_M2P == false; sph_selftest cross-checks m2pWarp against it).
template<int ORDER, class TVec>
__device__ inline vec3d
SphericalStokes::m2p(const NodeM2P &m2p_data, vec3f center, TVec Tp)
{
  constexpr int K = sph::kdimOf(ORDER);
  const double invA = 1.0 / (double)m2p_data.scale;
  const double Dx = ((double)Tp.x - (double)center.x) * invA;
  const double Dy = ((double)Tp.y - (double)center.y) * invA;
  const double Dz = ((double)Tp.z - (double)center.z) * invA;
  double ir, ir2;
  sph::invR(Dx * Dx + Dy * Dy + Dz * Dz, ir, ir2);
  double ux = 0.0, uy = 0.0, uz = 0.0;
  for (int k = 0; k < K; ++k)
    sph::m2pLane<ORDER>(m2p_data, Dx, Dy, Dz, ir, ir2, k, ux, uy, uz);
  return vec3d(ux * invA, uy * invA, uz * invA);
}

// Warp-cooperative M2P. Default: neighbour-shuffle scheme -- lane s%32 owns
// the coefficient of slot s (row-packed mapping, see c_sn_gs); each lane runs
// ONE column walk (column m to degree n+1, yielding val=I[n,m] and
// Iz=I[n+1,m]) and obtains the gradient neighbours I[n+1,m+-1] from the
// adjacent lanes' Iz via two warp shuffles (they own (n,m+-1) by construction).
// Boundary lanes never consume shuffled garbage: the row-end (m==n) takes
// I[n+1,n+1] from its own diagonal chain, the row-start (m==0) conjugates
// I[n+1,1]. This does ~1/3 of the walk arithmetic of the self-contained
// 3-walk basis (kept as m2pLane for the serial reference, and selectable here
// with -DSPH_M2P_NAIVE for A/B).
template<int ORDER, class TVec>
__device__ inline vec3d
SphericalStokes::m2pWarp(const NodeM2P &m2p_data, vec3f center, TVec Tp,
                         int lane, unsigned int mask)
{
  using namespace sph;
  const double invA = 1.0 / (double)m2p_data.scale;
  const double Dx = ((double)Tp.x - (double)center.x) * invA;
  const double Dy = ((double)Tp.y - (double)center.y) * invA;
  const double Dz = ((double)Tp.z - (double)center.z) * invA;
  double ir, ir2;
  invR(Dx * Dx + Dy * Dy + Dz * Dz, ir, ir2);
  double ux = 0.0, uy = 0.0, uz = 0.0;

#ifdef SPH_M2P_NAIVE
  constexpr int K = kdimOf(ORDER);
#pragma unroll
  for (int k = lane; k < K; k += 32)            // 1 pass p<=6, 2 passes p>=7
    m2pLane<ORDER>(m2p_data, Dx, Dy, Dz, ir, ir2, k, ux, uy, uz);
#else
  constexpr int S = slotCount(ORDER);
  constexpr int NPASS = (S + 31) / 32;
  const dcplx wc = {Dx, Dy};                    // e^{+im phi} convention
#pragma unroll
  for (int pass = 0; pass < NPASS; ++pass) {
    const int s = pass * 32 + lane;             // <= 127 < SLOT_MAX always
    const int sn = __ldg(&c_sn_gs[s]);
    const bool active = (s < S) && (sn != 0xff);
    const int n = active ? sn : 0;
    const int m = active ? __ldg(&c_sm_gs[s]) : 0;

    // Diagonal chain to column m+1: sM = I[m,m] (walk seed), sP = I[m+1,m+1]
    // (the row-end lane's own dplus input).
    dcplx diag = {ir, 0.0};
    dcplx sM = diag, sP = diag;
#pragma unroll
    for (int j = 1; j <= ORDER + 1; ++j) {
      if (j <= m + 1) {
        diag = cscale(cmul(wc, diag), __ldg(&c_diagstep_gs[j]) * ir2);
        if (j == m)     sM = diag;
        if (j == m + 1) sP = diag;
      }
    }
    dcplx val;
    const dcplx Iz = irrWalk<ORDER + 1>(sM, m, n + 1, Dz, ir2, &val);

    // ALL lanes execute the shuffles (full-warp mask contract).
    dcplx IzUp, IzDown;
    IzUp.re   = __shfl_down_sync(mask, Iz.re, 1);   // from lane+1: I[n+1,m+1]
    IzUp.im   = __shfl_down_sync(mask, Iz.im, 1);
    IzDown.re = __shfl_up_sync(mask, Iz.re, 1);     // from lane-1: I[n+1,m-1]
    IzDown.im = __shfl_up_sync(mask, Iz.im, 1);

    if (active) {
      const dcplx Ipl = (m == n) ? sP : IzUp;
      const dcplx Imi = (m == 0) ? cconj(Ipl) : IzDown;
      const dcplx dplus  = cscale(Ipl, __ldg(&c_cpl_gs[n * NB + m]));
      const double sgn   = (m == 0) ? 1.0 : -1.0;
      const dcplx dminus = cscale(Imi, sgn * __ldg(&c_cmi_gs[n * NB + m]));
      const dcplx gx = {0.5 * (dplus.re + dminus.re), 0.5 * (dplus.im + dminus.im)};
      const dcplx gy = {0.5 * (dplus.im - dminus.im), -0.5 * (dplus.re - dminus.re)};
      const dcplx gz = cscale(Iz, -__ldg(&c_cz_gs[n * NB + m]));
      contractK(m2p_data, Dx, Dy, Dz, kOf(n, m), m, val, gx, gy, gz, ux, uy, uz);
    }
  }
#endif

  for (int offset = 16; offset > 0; offset >>= 1) {
    ux += __shfl_down_sync(mask, ux, offset);
    uy += __shfl_down_sync(mask, uy, offset);
    uz += __shfl_down_sync(mask, uz, offset);
  }
  return (lane == 0) ? vec3d(ux * invA, uy * invA, uz * invA)
                     : vec3d(0.0, 0.0, 0.0);
}

template<int ORDER>
__device__ void
SphericalStokes::combine(bvh3f bvh, NodeMAC mac[], int nodeID)
{
  using namespace sph;
  constexpr int K = kdimOf(ORDER);
  const auto node = bvh.nodes[nodeID];
  const vec3f c = node.bounds.center();
  const vec3f sz = node.bounds.size();

  NodeMAC nm;
  nm.cx = c.x; nm.cy = c.y; nm.cz = c.z;
  nm.halfDiag2 = 0.25f * cuBQL::sqrLength(sz);

  // Moment scale (see NodeM2P); the floor guards degenerate (point) boxes --
  // their offsets are exactly 0, so any positive scale is consistent.
  const float nodeS = fmaxf(sqrtf(nm.halfDiag2), 1e-12f);
  const double invS = 1.0 / (double)nodeS;

  // No local NodeM2P staging: moments/owners are written straight to the
  // node's global slot. Safe under refit_aggregate's release protocol
  // (__threadfence() before the parent-release atomicAdd), and entries
  // k >= K are never written NOR read (m2p<ORDER> reads k < K only).
  NodeM2P &out = d_m2p_gs[nodeID];
  int ownerMin = 0x7fffffff;
  int ownerMax = -1;
  if (d_owner_mask_gs) {
    uint32_t *mask = d_owner_mask_gs + (size_t)nodeID * d_owner_mask_words_gs;
    for (int w = 0; w < d_owner_mask_words_gs; ++w) mask[w] = 0u;
  }
  double Macc[4][K][2];
  for (int ch = 0; ch < 4; ++ch)
    for (int k = 0; k < K; ++k)
      Macc[ch][k][0] = Macc[ch][k][1] = 0.0;

  if (node.admin.count != 0) {
    const uint32_t off = node.admin.offset;
    for (uint32_t t = 0; t < node.admin.count; ++t) {
      const uint32_t bid = bvh.primIDs[off + t];
      const int begin = d_bucket_begin_gs[bid];
      const int end   = d_bucket_end_gs[bid];
      for (int pid = begin; pid < end; ++pid) {
        // fp64-geometry P2M: accurate source coords, fp32 node center promoted
        // to fp64 in the (exact) subtraction. Same bucket index as the fp32 array.
        // -DSPH_FP32_GEOM reverts to the fp32 bucket coords (A/B anchor).
#ifdef SPH_FP32_GEOM
        const vec3f p = d_pos_gs[pid];
#else
        const vec3d p = d_pos64_gs[pid];
#endif
        const double S[3] = {((double)p.x - (double)c.x) * invS,
                             ((double)p.y - (double)c.y) * invS,
                             ((double)p.z - (double)c.z) * invS};
        const vec3d fv = d_force_gs[pid];
        const double f[3] = {fv.x, fv.y, fv.z};
        if (d_owner_gs) {
          const int owner = d_owner_gs[pid];
          ownerMin = min(ownerMin, owner);
          ownerMax = max(ownerMax, owner);
          if (d_owner_mask_gs) {
            uint32_t *mask = d_owner_mask_gs + (size_t)nodeID * d_owner_mask_words_gs;
            mask[owner >> 5] |= (1u << (owner & 31));
          }
        }
        p2mPoint<ORDER>(S, f, Macc);
      }
    }
  } else {
    const double pc[3] = {(double)c.x, (double)c.y, (double)c.z};
    for (int ch = 0; ch < 2; ++ch) {
      const int cid = node.admin.offset + ch;
      const NodeMAC &cmac = mac[cid];
      const NodeM2P &cm2p = d_m2p_gs[cid];
      ownerMin = min(ownerMin, cm2p.ownerMin);
      ownerMax = max(ownerMax, cm2p.ownerMax);
      if (d_owner_mask_gs) {
        uint32_t *mask = d_owner_mask_gs + (size_t)nodeID * d_owner_mask_words_gs;
        const uint32_t *childMask =
            d_owner_mask_gs + (size_t)cid * d_owner_mask_words_gs;
        for (int w = 0; w < d_owner_mask_words_gs; ++w) mask[w] |= childMask[w];
      }
      // Copy the child's fp64 moments into a local buffer once; the translation
      // re-reads them O(K) times each, so the local copy beats repeated global
      // reads.
      double Mc[4][K][2];
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          Mc[cc][k][0] = (double)cm2p.M[k][cc][0];
          Mc[cc][k][1] = (double)cm2p.M[k][cc][1];
        }
      const double ccen[3] = {(double)cmac.cx, (double)cmac.cy, (double)cmac.cz};
      m2mTranslate<ORDER>(Mc, (double)cm2p.scale, ccen, pc, (double)nodeS, Macc);
    }
  }

  out.ownerMin = ownerMin;
  out.ownerMax = ownerMax;
  out.scale = nodeS;
  out.pad0  = 0.f;
  for (int k = 0; k < K; ++k)
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
      out.M[k][ch][0] = Macc[ch][k][0];
      out.M[k][ch][1] = Macc[ch][k][1];
    }
  mac[nodeID] = nm;
}

inline void
SphericalStokes::setup(int /*order*/)
{
  using namespace sph;
  // Fill to MAX_ORDER bounds regardless of the runtime order (idempotent;
  // called once per Treecode::build()).
  unsigned char kn[K_MAX], km[K_MAX];
  for (int n = 0, k = 0; n <= MAX_P; ++n)
    for (int m = 0; m <= n; ++m, ++k) {
      kn[k] = (unsigned char)n;
      km[k] = (unsigned char)m;
    }
  // m2pWarp slot map: pack rows into 32-lane passes without straddling.
  unsigned char sn[SLOT_MAX], sm[SLOT_MAX];
  for (int s = 0; s < SLOT_MAX; ++s) sn[s] = sm[s] = 0xff;
  for (int n = 0, slot = 0; n <= MAX_P; ++n) {
    if ((slot & 31) + (n + 1) > 32) slot = (slot + 31) & ~31;
    for (int m = 0; m <= n; ++m, ++slot) {
      sn[slot] = (unsigned char)n;
      sm[slot] = (unsigned char)m;
    }
  }
  double diagstep[NB], sq2mp1[NB];
  double c2[NB * NB], invpair[NB * NB], cpl[NB * NB], cmi[NB * NB], cz[NB * NB];
  double A[NB * NB], invA[NB * NB];
  for (int m = 0; m < NB; ++m) {
    diagstep[m] = (m >= 1) ? -std::sqrt((2.0 * m - 1.0) / (2.0 * m)) : 0.0;
    sq2mp1[m]   = std::sqrt(2.0 * m + 1.0);
  }
  for (int n = 0; n < NB; ++n)
    for (int m = 0; m < NB; ++m) {
      const int i = n * NB + m;
      c2[i]      = (n >= m + 2) ? std::sqrt((double)(n + m - 1) * (n - m - 1)) : 0.0;
      invpair[i] = (n >= m + 1) ? 1.0 / std::sqrt((double)(n - m) * (n + m)) : 0.0;
      cpl[i]     = std::sqrt((double)(n + m + 1) * (n + m + 2));
      cmi[i]     = std::sqrt((double)(n - m + 1) * (n - m + 2));
      cz[i]      = std::sqrt((double)(n + m + 1) * (n - m + 1));
      A[i]       = (m <= n) ? std::exp(-0.5 * (std::lgamma((double)(n - m + 1))
                                             + std::lgamma((double)(n + m + 1))))
                            : 0.0;
      invA[i]    = (m <= n) ? 1.0 / A[i] : 0.0;
    }
  CUDA_CHECK(cudaMemcpyToSymbol(c_kn_gs,       kn,       sizeof(kn)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_km_gs,       km,       sizeof(km)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_sn_gs,       sn,       sizeof(sn)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_sm_gs,       sm,       sizeof(sm)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_diagstep_gs, diagstep, sizeof(diagstep)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_sq2mp1_gs,   sq2mp1,   sizeof(sq2mp1)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_c2_gs,       c2,       sizeof(c2)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_invpair_gs,  invpair,  sizeof(invpair)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_cpl_gs,      cpl,      sizeof(cpl)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_cmi_gs,      cmi,      sizeof(cmi)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_cz_gs,       cz,       sizeof(cz)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_A_gs,        A,        sizeof(A)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_invA_gs,     invA,     sizeof(invA)));
}

inline void
SphericalStokes::upwardPass(bvh3f bvh, NodeMAC *d_mac, NodeM2P *d_m2p,
                            const vec3f *d_pos, const vec3d *d_force,
                            const int *d_owner, uint32_t *d_ownerMask,
                            int ownerMaskWords,
                            const int *d_bucketBegin, const int *d_bucketEnd,
                            int order, cudaStream_t stream)
{
  using namespace sph;
  CUDA_CHECK(cudaMemcpyToSymbol(d_pos_gs,          &d_pos,         sizeof(d_pos)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_force_gs,        &d_force,       sizeof(d_force)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_owner_gs,        &d_owner,       sizeof(d_owner)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_owner_mask_gs,   &d_ownerMask,   sizeof(d_ownerMask)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_owner_mask_words_gs, &ownerMaskWords, sizeof(ownerMaskWords)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_bucket_begin_gs, &d_bucketBegin, sizeof(d_bucketBegin)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_bucket_end_gs,   &d_bucketEnd,   sizeof(d_bucketEnd)));
  CUDA_CHECK(cudaMemcpyToSymbol(mp::d_m2p_gs,      &d_m2p,         sizeof(d_m2p)));

  // Select the order-specialized combine instantiation.
  void (*hostFp)(bvh3f, NodeMAC[], int) = nullptr;
  CUDA_CHECK(cudaMemcpyFromSymbol(&hostFp, SphericalStokes_combine_fps,
                                  sizeof(hostFp),
                                  (size_t)(order - 1) * sizeof(hostFp)));
  cuBQL::cuda::refit_aggregate(bvh, d_mac, hostFp, stream);
}

} // namespace mp
