// SPDX-License-Identifier: Apache-2.0
//
// Shared *physical kernel* for the Stokeslet treecode: the exact pairwise Oseen
// (Stokeslet) interaction and the 1/(8*pi*mu) prefactor. These are independent
// of the multipole *representation* -- every multipole module (Cartesian,
// spherical, KIFMM, ...) reuses them unchanged. Only swapping the physical
// kernel itself would touch this file; swapping the expansion does not.
#pragma once

#include <cmath>

#include "cuBQL/math/vec.h"

namespace stokes {

using cuBQL::vec3f;
using cuBQL::vec3d;

// Viscosity, factored into the prefactor below.
static constexpr double MU = 1.0;

// Stokeslet prefactor 1/(8*pi*mu).
inline float prefactor() { return (float)(1.0 / (8.0 * M_PI * MU)); }

// Traction-kernel prefactor -3/(4*pi) (T_ijk = -3/(4*pi) r_i r_j r_k / r^5; no
// viscosity -- mu cancels between the stress and the Stokeslet). fp64: the
// traction path is fp64-eval only, so there is no reason to round the
// prefactor to fp32.
inline double tractionPrefactor() { return -3.0 / (4.0 * M_PI); }

// ---------------------------------------------------------------------------
// Optional Rotne-Prager-Yamakawa regularization of the Stokeslet.
//
// For spheres of radius a at separation r >= 2a the RPY mobility is
//
//   M = 1/(8*pi*mu) [ I/r + RR/r^3 + (2a^2/3) I/r^3 - 2a^2 RR/r^5 ]
//
// i.e. the Stokeslet plus a degenerate-quadrupole (finite-size) correction. The
// overlapping r < 2a branch is deliberately NOT implemented: the only consumer
// (the NeMO far field) excludes everything inside its near-field cutoff, which
// is several particle diameters.
//
// This is a COMPILE-TIME knob, not a Config field, because it sits in the
// innermost loop of every P2P and M2P. WIDEBVH_RPY_A defaults to 0, which makes
// RPY_ON false and every accumulator below reduce -- literally, expression for
// expression -- to the code that was there before. Every pre-existing target
// therefore compiles unchanged PTX. Only the NeMO library sets a > 0.
// ---------------------------------------------------------------------------
#ifndef WIDEBVH_RPY_A
#define WIDEBVH_RPY_A 0.0
#endif
static constexpr double RPY_A  = (double)(WIDEBVH_RPY_A);
static constexpr double RPY_C  = 2.0 * RPY_A * RPY_A / 3.0;   // 2a^2/3
static constexpr bool   RPY_ON = (RPY_A > 0.0);

// ---------------------------------------------------------------------------
// fp32 fast-path level for consumer GPUs (COMPILE-TIME, like WIDEBVH_RPY_A).
//
// The two hot kernels -- BaryStokes::m2pWarp and the near-field p2p -- run their
// inner loops in fp64 on fp32 inputs. On an H200 (fp64 at 1/2 the fp32 rate)
// that is nearly free; on Ada/Ampere GeForce parts fp64 issues at 1/64 the fp32
// rate and the far field becomes ~60x slower than the H200's for the same
// work. This ladder swaps those loops for all-fp32 evaluation, level by level,
// in order of expected fp64 cost:
//
//   0  (default) production: fp64 M2P / P2P / upward pass. Every #if below is
//      off and the compiled code is byte-identical to the pre-fp32 engine.
//   1  fp32 M2P: barycentric far-field evaluation in fp32 (fp32 geometry from
//      the fp32 target/center/half-extents, fp32 Chebyshev offset table, fp32
//      rsqrtf, fp32 lane accumulators, fp32 per-target accumulation in the
//      warp-specialized traversal), converted to fp64 once per target.
//   2  + fp32 P2P on the split-warpspec-atomic path: fp32 bucket coordinates,
//      fp32 forces, fp32 lane accumulators and warp reduce, fp32 global atomics
//      into a per-apply scratch that is folded into the fp64 output at scatter.
//   3  + fp32 upward pass (P2M / M2M shared-memory accumulators and the
//      Lagrange basis).
//
// Accuracy: the truncation error of the expansion (mac, PDEG) dominates by
// orders of magnitude at NeMO's operating point; the fp32 arithmetic adds
// ~1e-6..1e-5 relative (measured in
// pinn-stokes/artifacts/consumer_gpu_far_field_report.md). MAC / acceptance
// arithmetic is untouched at every level -- the interaction lists are the same,
// only the evaluation precision changes.
//
// One .so per level (see add_widebvh_nemo_library in CMakeLists.txt);
// wbnemo_fp32_level() reports it through the C ABI.
// ---------------------------------------------------------------------------
#ifndef WIDEBVH_FP32_LEVEL
#define WIDEBVH_FP32_LEVEL 0
#endif
static constexpr int FP32_LEVEL = WIDEBVH_FP32_LEVEL;

// ---------------------------------------------------------------------------
// Near-field exclusion radius (squared), in device constant memory.
//
// Set to 0 (the default) the treecode computes the whole kernel sum, exactly as
// before. Set to rc^2 > 0 it computes only the part of the sum with r >= rc:
// direct pairs closer than rc are skipped here, and the traversal additionally
// refuses to accept any node whose bounding sphere reaches inside rc (see
// nodeOutsideNearRadius in treecode.cuh), so a near pair can never be absorbed
// into a multipole. Together those two rules make the result exactly the
// complement of a hard r < rc cutoff, which is what a near/far-field split with
// an external near-field operator needs.
//
// Constant memory rather than a kernel argument: the value is read by ~9 call
// sites across 6 kernels and 3 traversal passes, and the count pass and the
// pair-write pass MUST make bit-identical accept decisions or the emitted pair
// list desyncs from its counts. One symbol they all read cannot drift; a
// parameter threaded through 12 launch sites can. It is warp-uniform, so the
// disabled case costs a predicted branch on an L1-resident scalar.
//
// Treecode::syncNearCutoffSymbols() writes it at every apply()/reapply().
__constant__ float  c_nearCut2f = 0.f;   // node/fp32-geometry test
__constant__ double c_nearCut2d = 0.0;   // fp64-geometry P2P test

// Shared velocity accumulator in the FACTORED form used by the three hot
// evaluators (both p2p overloads and BaryStokes::m2pWarp): given 1/r and
// q = (R.f)/r^2, the Stokeslet is u_i += (1/r)(f_i + R_i q).
//
// The RPY form factors the same way -- with c = 2a^2/3,
//   u_i += A f_i + (B q) R_i,   A = 1/r + c/r^3,   B = 1/r - 3c/r^3
// -- so it stays two nested FMAs per component and costs 4 extra fp64 ops
// (ir3, A, B, B*q) shared across all three.
__device__ __forceinline__ void
accumStokesFactored(double ir, double q,
                    double Rx, double Ry, double Rz,
                    double fx, double fy, double fz,
                    double &u0, double &u1, double &u2)
{
  if constexpr (!RPY_ON) {
    u0 = fma(ir, fma(Rx, q, fx), u0);
    u1 = fma(ir, fma(Ry, q, fy), u1);
    u2 = fma(ir, fma(Rz, q, fz), u2);
  } else {
    const double ir3 = ir * (ir * ir);
    const double A   = fma(RPY_C, ir3, ir);              // 1/r + c/r^3
    const double Bq  = fma(-3.0 * RPY_C, ir3, ir) * q;   // (1/r - 3c/r^3) q
    u0 = fma(A, fx, fma(Bq, Rx, u0));
    u1 = fma(A, fy, fma(Bq, Ry, u1));
    u2 = fma(A, fz, fma(Bq, Rz, u2));
  }
}

// fp32 twin of accumStokesFactored, used by the WIDEBVH_FP32_LEVEL >= 1 M2P and
// the level >= 2 P2P. Same algebra, fmaf for fma, the RPY constant rounded to
// fp32 once. Overload resolution is unambiguous: every fp64 call site passes
// all-double arguments.
__device__ __forceinline__ void
accumStokesFactored(float ir, float q,
                    float Rx, float Ry, float Rz,
                    float fx, float fy, float fz,
                    float &u0, float &u1, float &u2)
{
  if constexpr (!RPY_ON) {
    u0 = fmaf(ir, fmaf(Rx, q, fx), u0);
    u1 = fmaf(ir, fmaf(Ry, q, fy), u1);
    u2 = fmaf(ir, fmaf(Rz, q, fz), u2);
  } else {
    const float ir3 = ir * (ir * ir);
    const float A   = fmaf((float)RPY_C, ir3, ir);               // 1/r + c/r^3
    const float Bq  = fmaf(-3.0f * (float)RPY_C, ir3, ir) * q;   // (1/r - 3c/r^3) q
    u0 = fmaf(A, fx, fmaf(Bq, Rx, u0));
    u1 = fmaf(A, fy, fmaf(Bq, Ry, u1));
    u2 = fmaf(A, fz, fmaf(Bq, Rz, u2));
  }
}

// Direct Oseen pair contribution: u += f/r + R (R.f)/r^3, R = T - src. UNSCALED:
// the 1/(8*pi*mu) prefactor is applied ONCE by the caller after the warp reduce
// (same convention as the M2P far field, e.g. bary_stokes.cuh::m2pWarp), so it is
// no longer a per-source fp64 multiply. Skips self / coincident (r==0) and
// accumulates the raw sum into the fp64 accumulator u[3].
//
// 1/r via the fp32 rsqrtf intrinsic (single SFU op) -- no fp32 sqrt/divide chain,
// matching the M2P treatment (bary_stokes.cuh:293-295). Geometry stays fp32; the
// force dot-product and the accumulation are fp64.
//
// The velocity is evaluated in the FACTORED form u_i = (1/r)*(f_i + R_i*q) with
// q = (R.f)/r^2, instead of the expanded f_i/r + R_i*(R.f)/r^3. This is
// algebraically identical (differs only at fp64 round-off) but pulls 1/r out of
// both terms, so each component is two nested fused-multiply-adds and q is shared
// across the three components -- ~6 fewer instructions/interaction. The p2pKernel
// is instruction-issue-bound (not fp64-pipe/memory bound), so cutting instruction
// count is the lever; the contraction stays fp64 (precision-neutral).
template <class ForceVec>
__device__ inline void p2p(vec3f T, vec3f src, ForceVec f, double u[3])
{
  const vec3f R = T - src;
  const float r2 = cuBQL::dot(R, R);
  if (r2 == 0.f) return;                  // skip self / coincident
  if (r2 < c_nearCut2f) return;           // owned by the near-field operator
  const float ir  = rsqrtf(r2);
  const float ir2 = ir * ir;              // 1/r^2
  const double fx = (double)f.x;
  const double fy = (double)f.y;
  const double fz = (double)f.z;
  const double rdf = (double)R.x * fx + (double)R.y * fy + (double)R.z * fz;
  const double q   = rdf * (double)ir2;   // (R.f)/r^2, shared across components
  accumStokesFactored((double)ir, q, (double)R.x, (double)R.y, (double)R.z,
                      fx, fy, fz, u[0], u[1], u[2]);
}

// fp64-GEOMETRY overload: identical physics to the fp32 p2p above, but source and
// target coordinates are the resident fp64 bucket positions (Treecode::points64()
// / targetPoints64()), so the near-field difference R and r^2 carry no fp32
// coordinate-quantization error. 1/r seeds from the single-SFU fp32 rsqrtf
// (~22-bit) and is refined by one fp64 Newton-Raphson step
// (y <- y*(1.5 - 0.5*r2*y^2)), which squares the relative error to ~fp64 without a
// full fp64 sqrt+divide (~4-8% near-field cost vs ~40% for 1.0/sqrt). The unscaled
// 1/(8*pi*mu) convention and fp64 accumulation match the fp32 overload exactly.
template <class ForceVec>
__device__ inline void p2p(vec3d T, vec3d src, ForceVec f, double u[3])
{
  const double Rx = T.x - src.x;
  const double Ry = T.y - src.y;
  const double Rz = T.z - src.z;
  const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
  if (r2 == 0.0) return;                    // skip self / coincident
  if (r2 < c_nearCut2d) return;             // owned by the near-field operator
  double ir = (double)rsqrtf((float)r2);    // ~22-bit fp32 seed
  ir = ir * fma(-0.5 * r2, ir * ir, 1.5);   // one fp64 Newton step -> ~fp64 1/r
  const double ir2 = ir * ir;               // 1/r^2
  const double fx = (double)f.x;
  const double fy = (double)f.y;
  const double fz = (double)f.z;
  const double rdf = Rx * fx + Ry * fy + Rz * fz;
  const double q   = rdf * ir2;             // (R.f)/r^2, shared across components
  accumStokesFactored(ir, q, Rx, Ry, Rz, fx, fy, fz, u[0], u[1], u[2]);
}

// ALL-fp32 pair kernel for the WIDEBVH_FP32_LEVEL >= 2 near field
// (treecode.cuh::p2pAtomicKernel32): fp32 bucket coordinates
// (Treecode::points()/targetPoints()), fp32 forces, fp32 rsqrtf (no Newton
// step: ~2 ulp), fp32 accumulation. Same self / near-cutoff skips as the two
// overloads above, using the fp32 cutoff constant.
//
// On the near/far boundary: r^2 here carries the fp32 quantization of the
// bucket coordinates (~1e-5 relative at r = 6 in a ~1000-wide box), so a pair
// within that band of rc can land on the other side of the split than the fp64
// overload puts it. That is the same band in which the caller's own fp32
// neighbour search (NeMO's Warp hash grid) already disagrees with the fp64
// test, so the level-2 partition is no less consistent with the near-field
// operator than level 0's -- see the report cited at WIDEBVH_FP32_LEVEL.
__device__ __forceinline__ void p2p32(vec3f T, vec3f src, vec3f f, float u[3])
{
  const vec3f R = T - src;
  const float r2 = cuBQL::dot(R, R);
  if (r2 == 0.f) return;                    // skip self / coincident
  if (r2 < c_nearCut2f) return;             // owned by the near-field operator
  const float ir  = rsqrtf(r2);
  const float rdf = fmaf(R.x, f.x, fmaf(R.y, f.y, R.z * f.z));
  const float q   = rdf * (ir * ir);        // (R.f)/r^2, shared across components
  accumStokesFactored(ir, q, R.x, R.y, R.z, f.x, f.y, f.z, u[0], u[1], u[2]);
}

// Target-normal-contracted traction pair (single-layer traction / stresslet
// with the normal at the TARGET): t_i += R_i * (R.q) * (R.n) / r^5, R = T-src,
// n = outward normal at the target. UNSCALED: the caller multiplies by
// tractionPrefactor() = -3/(4*pi) once after the reduce, same convention as
// p2p. fp64-geometry only (the traction path is consumed exclusively by the
// skel block P2P and skel M2I, both fp64-eval): rsqrtf seed + one fp64 Newton
// step for 1/r, factored so s = (R.q)(R.n)/r^5 is shared across components.
template <class ForceVec>
__device__ inline void traction_p2p(vec3d T, vec3d n, vec3d src, ForceVec q,
                                    double t[3])
{
  const double Rx = T.x - src.x;
  const double Ry = T.y - src.y;
  const double Rz = T.z - src.z;
  const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
  if (r2 == 0.0) return;                    // skip self / coincident
  double ir = (double)rsqrtf((float)r2);    // ~22-bit fp32 seed
  ir = ir * fma(-0.5 * r2, ir * ir, 1.5);   // one fp64 Newton step -> ~fp64 1/r
  const double ir2 = ir * ir;
  const double rdq = Rx * (double)q.x + Ry * (double)q.y + Rz * (double)q.z;
  const double rdn = Rx * n.x + Ry * n.y + Rz * n.z;
  const double s = (rdq * rdn) * ((ir2 * ir2) * ir);   // (R.q)(R.n)/r^5
  t[0] = fma(Rx, s, t[0]);
  t[1] = fma(Ry, s, t[1]);
  t[2] = fma(Rz, s, t[2]);
}

} // namespace stokes
