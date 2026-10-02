// SPDX-License-Identifier: Apache-2.0
//
// Reusable GPU Barnes-Hut *treecode* engine for the 3D Stokeslet (Oseen
// tensor), built on a cuBQL BinaryBVH over grid-constrained particle buckets.
// This header is the GENERIC engine: the BVH traversal kernels, the M2P (far
// field) + P2P (near field) split pipeline, and a small `Treecode` class that
// owns the spatial structure and node moments.
//
// Pipeline (treecode, NOT FMM -> no M2L / L2L / L2P):
//   apply()   : build geometry -> upward pass -> M2P traversal -> P2P
//   reapply() : reuse geometry + cached P2P list, recompute moments from forces
//
// Precision policy: particle positions / BVH geometry stay fp32, source
// strengths stay fp64 in the internal bucket order, and per-target velocity
// accumulation runs in fp64. Moment precision is set by the multipole policy.
//
// The multipole expansion is a swappable policy `MP` (the Treecode template
// parameter, default mp::CartesianStokes). To swap it (spherical, KIFMM, ...)
// instantiate `Treecode<mp::OtherModule>`; the module header must expose the
// same surface (Moments, MAX_ORDER, name(), setup(), upwardPass(), m2p<ORDER>(),
// combine()). The shared physical kernel (p2p, prefactor) is in stokes_kernel.cuh.
//
// Self-interaction is the default when the target pointer is null. Distinct
// source/target geometry is supported in the same centered fp32 frame.
//
// ====================================================================
//  ODR / LINK CONSTRAINT -- READ BEFORE ADDING A NEW DRIVER
// --------------------------------------------------------------------
//  The multipole module headers (cartesian_stokes.cuh, bary_stokes.cuh) define
//  non-inline, namespace-scope `__device__` globals and a `__device__` function
//  pointer. They must be defined exactly once per executable. Therefore this
//  header (which includes them) must be included by EXACTLY ONE translation
//  unit per executable -- in practice, the single .cu that owns main(). Adding
//  a second .cu to a driver target that also includes treecode.cuh yields
//  duplicate-symbol link errors. The companion .cu files compiled into each
//  driver (grid_buckets.cu, common.cu) must NOT include this header (they don't).
//
//  Every driver target also requires CUDA_SEPARABLE_COMPILATION (+ resolve
//  device symbols), because the upward pass takes combine()'s device address
//  across the TU boundary via cudaMemcpyFromSymbol (see add_treecode_driver in
//  CMakeLists.txt).
// ====================================================================
#pragma once

#include "common.cuh"
#include "grid_buckets.cuh"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <type_traits>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>

#include <cub/cub.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"

// Defer to the includer's CUDA_CHECK if it already has one; otherwise provide
// the richer file:line version. Defined BEFORE the multipole module headers so
// their #ifndef fallback resolves to this one.
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess)                                                \
      throw std::runtime_error(std::string("CUDA error: ") +             \
                               cudaGetErrorString(_e) + " @ " +           \
                               __FILE__ ":" + std::to_string(__LINE__));  \
  } while (0)
#endif

using cuBQL::vec3f;
using cuBQL::vec3d;
using cuBQL::bvh3f;

// Shared physical kernel (p2p, prefactor) + the swappable multipole modules.
#include "stokes_kernel.cuh"
#include "struct_recon.cuh"
#include "cartesian_stokes.cuh"
#include "bary_stokes.cuh"
#include "spherical_stokes.cuh"

// Per-target far-field accumulator of the warp-specialized traversal
// (traverseSplitWarpSpecBody's s_results): fp32 on the WIDEBVH_FP32_LEVEL >= 1
// path so that no per-node fp64 operation survives between m2pWarp and the
// once-per-target widening to the fp64 output; double otherwise (unchanged).
using FarAcc = std::conditional_t<(stokes::FP32_LEVEL >= 1), float, double>;

// Target-skeleton ID build for the TC_PATH=skel path (host-side, one-time per
// apply; needs HostClock/elapsed_ms, so it is included after them below).

using stokes::p2p;               // near-field, used unqualified by the kernels
using stokes::traction_p2p;      // traction near-field (KernelKind::Traction)

// Per-target traversal statistics (# of BVH nodes popped/examined per target)
// are collected ONLY in debug builds: CMake defines NDEBUG for Release/
// RelWithDebInfo but not for Debug. Defined here so the kernel signature and the
// launch macro in evaluate() see one consistent definition.
#ifndef NDEBUG
#  define TREECODE_STATS 1
#else
#  define TREECODE_STATS 0
#endif

// Host timing helpers (shared with the driver, which also includes this header).
using HostClock = std::chrono::steady_clock;
inline double elapsed_ms(HostClock::time_point t0, HostClock::time_point t1)
{
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

#include "target_skeleton.cuh"

inline bool treecodeEnvFlagValue(const char *v, bool defaultValue)
{
  if (!v || !*v) return defaultValue;
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
      std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 ||
      std::strcmp(v, "OFF") == 0 || std::strcmp(v, "no") == 0 ||
      std::strcmp(v, "NO") == 0)
    return false;
  return true;
}

inline bool treecodeEnvFlag(const char *name, bool defaultValue)
{
  return treecodeEnvFlagValue(std::getenv(name), defaultValue);
}

// Per-apply chatter switch (TC_QUIET=1). The treecode normally prints its
// selected path, bucketizer and BVH builder on every apply()/reapply(), which is
// fine for a one-shot driver but drowns a caller that applies the operator once
// per timestep -- or once per column of an assembled matrix. Read once, cached.
inline bool treecodeQuiet()
{
  static const bool q = treecodeEnvFlag("TC_QUIET", false);
  return q;
}

inline bool treecodeEnvFlagIfPresent(const char *name, bool &out)
{
  const char *v = std::getenv(name);
  if (!v || !*v) return false;
  out = treecodeEnvFlagValue(v, false);
  return true;
}

static constexpr int TC_INTERACTION_M2P = 0;
static constexpr int TC_INTERACTION_P2P = 1;

__host__ __device__ inline int encodeInteractionNode(uint32_t nodeID, int kind)
{
  return ((int)nodeID << 1) | kind;
}

__host__ __device__ inline uint32_t decodeInteractionNode(int encoded)
{
  return (uint32_t)encoded >> 1;
}

__host__ __device__ inline int decodeInteractionKind(int encoded)
{
  return encoded & 1;
}


// ======================================================================
//  Kernel thread-block sizes (compile-time; tune here).
//
//  Every kernel launch in this file takes its block size from one of the
//  constants below. For these kernels the block size is a PURE launch parameter
//  -- it affects occupancy / consumer-warp count but never the numerical result
//  -- so the sizes live at file scope as compile-time constants, editable in one
//  place, instead of being chosen at runtime. All must be multiples of 32.
// ======================================================================

// Warp-per-target split traversal: traverseM2PKernel_Warp /
// traverseM2POnlyKernel_Warp run block/32 warps, one target per warp.
static constexpr int TRAVERSE_WARP_BLOCK   = 128;

// Split warp-specialized producer/consumer M2P ring
// (traverseSplitWarpSpecKernel): warp 0 produces, the remaining block/32 - 1
// warps consume M2P items. No shared array depends on it, so it is a free launch
// parameter (>= 64); larger = more consumer warps.
// Overridable at compile time via -DWIDEBVH_SPLIT_WARPSPEC_BLOCK=<N> (used by the
// block-size perf sweep to build variant binaries); default unchanged.
#ifndef WIDEBVH_SPLIT_WARPSPEC_BLOCK
#define WIDEBVH_SPLIT_WARPSPEC_BLOCK 512
#endif
static constexpr int SPLIT_WARPSPEC_BLOCK  = WIDEBVH_SPLIT_WARPSPEC_BLOCK;

// Block size for the __launch_bounds__ shell used by LANE_BATCH_M2P policies
// (CartesianStokes). Lane-batched consumers finish M2P so fast that the single
// producer warp is the bottleneck; smaller blocks put MORE producer warps per
// SM (launch_bounds(B, 512/B) pins the 128-reg cap, so 512/B blocks co-reside:
// 512 -> 1 producer/SM, 256 -> 2, 128 -> 4) at a similar total consumer count.
static constexpr int SPLIT_WARPSPEC_LB_BLOCK = 128;

// Direct-near merged warp-specialized ring (traverseWarpSpecKernel). NOT free:
// the kernel sizes its per-consumer-warp temp storage from this same constant,
// so the launch and the kernel read one value and cannot drift.
// Overridable at compile time via -DWIDEBVH_DIRECT_WARPSPEC_BLOCK=<N>; default unchanged.
#ifndef WIDEBVH_DIRECT_WARPSPEC_BLOCK
#define WIDEBVH_DIRECT_WARPSPEC_BLOCK 512
#endif
static constexpr int DIRECT_WARPSPEC_BLOCK = WIDEBVH_DIRECT_WARPSPEC_BLOCK;

// Near-field P2P, one warp per near pair (p2pKernel).
static constexpr int P2P_BLOCK             = 256;
static constexpr int P2P_WARPS             = P2P_BLOCK / 32;

// One-thread-per-element helper kernels: pairCountsKernel, scatterAddKernel,
// scatterCompToInputOrderKernel.
static constexpr int ELEMENTWISE_BLOCK     = 128;

// double-traverse pass 2 (traversePairWriteKernel): one thread per target, each
// doing a full DFS. Pure launch parameter (occupancy only).
static constexpr int TRAVERSE_PAIRWRITE_BLOCK = 128;

// TC_PATH=skel: group traversal (one warp per particle, count then write) and
// skeleton M2P eval (one warp per (particle, skeleton target)). Pure launch
// parameters.
static constexpr int SKEL_TRAVERSE_BLOCK = 128;
static constexpr int SKEL_EVAL_BLOCK     = 256;

// TC_SKEL_P2P=block: one THREAD per target, block per (group, target-chunk),
// GPU-Gems-style shared-memory source tiles. Block size = chunk size; TILE =
// staged sources per round (shared bytes = TILE * (48 + 4)).
static constexpr int SKEL_P2P_BLOCK = 256;
static constexpr int SKEL_P2P_TILE  = 256;


// Per-node traversal decision, shared by the warp-specialized producer
// (traverseSplitWarpSpecKernel) and the double-traverse pair-write kernel
// (traversePairWriteKernel) so their rejected-leaf sets are GUARANTEED identical:
// the per-target count produced in pass 1 must exactly equal the number of pairs
// written in pass 2. Keep the MAC arithmetic verbatim (the fmaf form and
// `mac2 * r2`); any reassociation can flip a borderline node and desync the two
// passes. `adminCount == 0` marks an internal node (descend to both children).
//
// P2P/multipole crossover (TC_XOVER, `xoverThresh`): when enabled (> 0) a node
// that PASSES the MAC but summarizes fewer than `xoverThresh` real sources is
// cheaper (and exact) to evaluate by direct P2P than by the fixed (PDEG+1)^3-point
// M2P. Small accepted LEAVES emit a near pair; small accepted INTERNAL nodes
// descend (there is no contiguous source range for an internal subtree, so we let
// its small leaves each emit through the existing machinery -- keeping every near
// pair a single-bucket leaf). `nodeCount[nid]` is read only on accepted nodes and
// only when the crossover is on, so the disabled path is bit-identical and
// `nodeCount` may be null.
enum TcNodeAction { TC_GO_CHILDREN = 0, TC_DO_M2P = 1, TC_DO_P2P = 2 };

// Second acceptance condition, active only when a near-field cutoff rc > 0 is
// configured (stokes::c_nearCut2f; see the note there). True iff the node's
// bounding sphere -- center at distance r, radius hd -- lies entirely outside
// radius rc of the target, i.e. r - hd >= rc. If it does not, the node may
// contain a source the near-field operator already owns, so it must NOT be
// summarized into a multipole; traversal descends instead and the per-pair
// r < rc test in stokes::p2p does the rest.
//
// Kept sqrt-free (r, hd and rc all appear only squared) so it stays in the same
// fp32 arithmetic as the MAC itself:
//   r - hd >= rc  <=>  r^2 - rc^2 - hd^2 >= 2*rc*hd, with the LHS non-negative
//                 <=>  L >= 0 and L^2 >= 4*rc^2*hd^2.
__device__ __forceinline__ bool
nodeOutsideNearRadius(float hd2, float r2)
{
  const float rc2 = stokes::c_nearCut2f;
  if (rc2 <= 0.f) return true;             // cutoff disabled: legacy behaviour
  const float L = r2 - rc2 - hd2;
  return (L >= 0.f) && (L * L >= 4.f * rc2 * hd2);
}

template<class NodeMAC>
__device__ __forceinline__ TcNodeAction
classifyTraversalNode(const NodeMAC &nm, uint64_t adminCount,
                      vec3f T, uint32_t nid,
                      const uint32_t *ownerMask, int ownerMaskWords,
                      int tgtOwn, int skipSameGroup, float mac2,
                      const int *nodeCount, int xoverThresh,
                      int emitInternalNodes)
{
  const float dx = T.x - nm.cx;
  const float dy = T.y - nm.cy;
  const float dz = T.z - nm.cz;
  const float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
  bool mayContainSelf = false;
  if (skipSameGroup && ownerMask && tgtOwn >= 0) {
    const uint32_t word =
        ownerMask[(size_t)nid * ownerMaskWords + (tgtOwn >> 5)];
    mayContainSelf = (word & (1u << (tgtOwn & 31))) != 0u;
  }
  if (r2 > 0.f && !mayContainSelf && nm.halfDiag2 < mac2 * r2 &&
      nodeOutsideNearRadius(nm.halfDiag2, r2)) {
    // MAC accepted. Multipole worth it iff enough sources (crossover off => yes).
    if (xoverThresh <= 0 || __ldg(&nodeCount[nid]) >= xoverThresh)
      return TC_DO_M2P;
    if (adminCount != 0)
      return TC_DO_P2P;         // small accepted leaf -> exact near-field P2P
    // small accepted internal: node-range mode emits ONE (target, node) pair;
    // otherwise descend to its leaves (each emits its own leaf pair).
    if (emitInternalNodes)
      return TC_DO_P2P;
    return TC_GO_CHILDREN;
  }
  if (adminCount != 0)
    return TC_DO_P2P;           // rejected leaf -> near-field P2P
  return TC_GO_CHILDREN;        // internal node -> descend
}


template<class MP, int ORDER>
__global__ void traverseM2PKernel_Warp(bvh3f bvh,
                                       const typename MP::NodeMAC *macArr,
                                       const typename MP::NodeM2P *m2pArr,
                                       const vec3f *pos,
                                       const int *targetOwner,
                                       const uint32_t *ownerMask,
                                       int ownerMaskWords,
                                       const int *nodeCount, int xoverThresh,
                                       int targetStart, int count, int N,
                                       float mac, float pref,
                                       int skipSameGroup,
                                       double *potential,
                                       int *pairTarget, int *pairLeaf,
                                       unsigned long long *pairCount,
                                       long long pairCapacity
#if TREECODE_STATS
                                       , uint32_t *visited,
                                       uint32_t *m2pAccepted
#endif
                                       )
{
  // [targetStart, targetStart+count) is this launch's target sub-range; N is the
  // full target count = component-major stride of `potential`. Full-range callers
  // pass targetStart=0, count=N (identical to the original single-pass behavior);
  // the P2P tiling path passes a sub-range so the near-pair buffer stays bounded.
  const int lane = threadIdx.x & 31;
  const int warpInBlock = threadIdx.x >> 5;
  const int warpsPerBlock = blockDim.x >> 5;
  const int local = blockIdx.x * warpsPerBlock + warpInBlock;
  if (local >= count) return;
  const int tid = targetStart + local;

  const unsigned int mask = 0xffffffffu;
  const vec3f T = pos[tid];
  const int tgtOwner =
      (skipSameGroup && targetOwner) ? targetOwner[tid] : -1;
  const float mac2 = mac * mac;
  double far0 = 0.0, far1 = 0.0, far2 = 0.0;
#if TREECODE_STATS
  uint32_t nVisited = 0;
  uint32_t nM2P = 0;
#endif

  uint32_t stack[64];
  int sp = 0;
  stack[sp++] = 0;

  while (sp > 0) {
    const uint32_t nid = stack[--sp];
    const auto admin = bvh.nodes[nid].admin;
    const auto &nm = macArr[nid];
#if TREECODE_STATS
    ++nVisited;
#endif
    const float dx = T.x - nm.cx;
    const float dy = T.y - nm.cy;
    const float dz = T.z - nm.cz;
    const float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
    bool mayContainSelf = false;
    if (skipSameGroup && ownerMask && tgtOwner >= 0) {
      const uint32_t word =
          ownerMask[(size_t)nid * ownerMaskWords + (tgtOwner >> 5)];
      mayContainSelf = (word & (1u << (tgtOwner & 31))) != 0u;
    }

    // Crossover: a small accepted node is exact-and-cheaper via P2P. accept &&
    // big => M2P; otherwise a leaf emits a near pair (rejected OR small-accepted)
    // and an internal node descends (rejected OR small-accepted).
    const bool accept =
        (r2 > 0.f && !mayContainSelf && nm.halfDiag2 < mac2 * r2 &&
         nodeOutsideNearRadius(nm.halfDiag2, r2));
    const bool doM2P = accept &&
        (xoverThresh <= 0 || __ldg(&nodeCount[nid]) >= xoverThresh);
    if (doM2P) {
      const vec3f c(nm.cx, nm.cy, nm.cz);
      typename MP::FarVec du;
      if constexpr (MP::WANTS_FP64_TARGET)
        du = MP::template m2pWarp<ORDER>(
            m2pArr[nid], c, mp::sph::d_tgt64_gs[tid], lane, mask);
      else
        du = MP::template m2pWarp<ORDER>(m2pArr[nid], c, T, lane, mask);
      if (lane == 0) {
        far0 += (double)du.x;
        far1 += (double)du.y;
        far2 += (double)du.z;
      }
#if TREECODE_STATS
      ++nM2P;
#endif
    } else if (admin.count != 0) {
      if (lane == 0) {
        // 64-bit counter so the whole-range probe counts correctly past 2^31
        // (the cap < INT_MAX still bounds the buffer write below).
        const long long slot = (long long)atomicAdd(pairCount, 1ull);
        if (slot < pairCapacity) {
          pairTarget[slot] = tid;
          pairLeaf[slot] = (int)nid;
        }
      }
    } else {
      stack[sp++] = admin.offset + 0;
      stack[sp++] = admin.offset + 1;
    }
  }

  if (lane == 0) {
    const double scale = (double)pref;
    potential[(size_t)0 * (size_t)N + (size_t)tid] = scale * far0;
    potential[(size_t)1 * (size_t)N + (size_t)tid] = scale * far1;
    potential[(size_t)2 * (size_t)N + (size_t)tid] = scale * far2;
#if TREECODE_STATS
    visited[tid] = nVisited;
    m2pAccepted[tid] = nM2P;
#endif
  }
}

template<class MP, int ORDER>
__global__ void traverseM2POnlyKernel_Warp(
    bvh3f bvh,
    const typename MP::NodeMAC *macArr,
    const typename MP::NodeM2P *m2pArr,
    const vec3f *pos, const int *targetOwner,
    const uint32_t *ownerMask, int ownerMaskWords,
    const int *nodeCount, int xoverThresh,
    int N, float mac, float pref, int skipSameGroup,
    double *potential
#if TREECODE_STATS
    , uint32_t *visited,
    uint32_t *m2pAccepted
#endif
    )
{
  const int lane = threadIdx.x & 31;
  const int warpInBlock = threadIdx.x >> 5;
  const int warpsPerBlock = blockDim.x >> 5;
  const int tid = blockIdx.x * warpsPerBlock + warpInBlock;
  if (tid >= N) return;

  const unsigned int mask = 0xffffffffu;
  const vec3f T = pos[tid];
  const int tgtOwner =
      (skipSameGroup && targetOwner) ? targetOwner[tid] : -1;
  const float mac2 = mac * mac;
  double far0 = 0.0, far1 = 0.0, far2 = 0.0;
#if TREECODE_STATS
  uint32_t nVisited = 0;
  uint32_t nM2P = 0;
#endif

  uint32_t stack[64];
  int sp = 0;
  stack[sp++] = 0;

  while (sp > 0) {
    const uint32_t nid = stack[--sp];
    const auto admin = bvh.nodes[nid].admin;
    const auto &nm = macArr[nid];
#if TREECODE_STATS
    ++nVisited;
#endif
    const float dx = T.x - nm.cx;
    const float dy = T.y - nm.cy;
    const float dz = T.z - nm.cz;
    const float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
    bool mayContainSelf = false;
    if (skipSameGroup && ownerMask && tgtOwner >= 0) {
      const uint32_t word =
          ownerMask[(size_t)nid * ownerMaskWords + (tgtOwner >> 5)];
      mayContainSelf = (word & (1u << (tgtOwner & 31))) != 0u;
    }

    // Crossover reapply: M2P only the big accepted nodes (the small accepted ones
    // are replayed from the cached near-pair list). accept && big => M2P; a small
    // accepted internal node still descends; a small accepted / rejected leaf is
    // handled by the cached P2P path, so it falls through to nothing here.
    const bool accept =
        (r2 > 0.f && !mayContainSelf && nm.halfDiag2 < mac2 * r2 &&
         nodeOutsideNearRadius(nm.halfDiag2, r2));
    const bool doM2P = accept &&
        (xoverThresh <= 0 || __ldg(&nodeCount[nid]) >= xoverThresh);
    if (doM2P) {
      const vec3f c(nm.cx, nm.cy, nm.cz);
      typename MP::FarVec du;
      if constexpr (MP::WANTS_FP64_TARGET)
        du = MP::template m2pWarp<ORDER>(
            m2pArr[nid], c, mp::sph::d_tgt64_gs[tid], lane, mask);
      else
        du = MP::template m2pWarp<ORDER>(m2pArr[nid], c, T, lane, mask);
      if (lane == 0) {
        far0 += (double)du.x;
        far1 += (double)du.y;
        far2 += (double)du.z;
      }
#if TREECODE_STATS
      ++nM2P;
#endif
    } else if (admin.count == 0) {
      stack[sp++] = admin.offset + 0;
      stack[sp++] = admin.offset + 1;
    }
  }

  if (lane == 0) {
    const double scale = (double)pref;
    potential[(size_t)0 * (size_t)N + (size_t)tid] = scale * far0;
    potential[(size_t)1 * (size_t)N + (size_t)tid] = scale * far1;
    potential[(size_t)2 * (size_t)N + (size_t)tid] = scale * far2;
#if TREECODE_STATS
    visited[tid] = nVisited;
    m2pAccepted[tid] = nM2P;
#endif
  }
}


// ======================================================================
//  Diagnostic: traversal-ONLY interaction counter (TC_PATH=traverse-count).
//
//  One THREAD per target walks the source BVH using the exact same MAC test
//  (classifyTraversalNode) as the real paths, and only counts how many nodes
//  it would accept for M2P and how many leaves it would reject to P2P. There
//  is NO m2pWarp evaluation, NO near-pair emit, NO P2P -- just the walk + the
//  MAC branch + two integer increments. Timing this kernel and contrasting it
//  with the merged apply/reapply traverse time (stats_.travMs, a_m2p / r_m2p)
//  isolates the cost of the traversal itself vs the M2P multipole compute.
//  Reads only bvh + NodeMAC (centers + halfDiag2); moments are not touched.
// ======================================================================
template<class MP>
__global__ void traverseCountOnlyKernel(
    bvh3f bvh,
    const typename MP::NodeMAC *macArr,
    const vec3f *pos, const int *targetOwner,
    const uint32_t *ownerMask, int ownerMaskWords,
    const int *nodeCount, int xoverThresh,
    int N, float mac, int skipSameGroup,
    int *m2pCount, int *p2pCount)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  const vec3f T    = pos[tid];
  const int tgtOwn = targetOwner ? targetOwner[tid] : -1;
  const float mac2 = mac * mac;

  uint32_t stack[64];
  int sp = 0;
  stack[sp++] = 0;
  int nM2P = 0, nP2P = 0;
  while (sp > 0) {
    const uint32_t nid = stack[--sp];
    const auto admin   = bvh.nodes[nid].admin;
    const auto &nm     = macArr[nid];
    const TcNodeAction act = classifyTraversalNode(
        nm, admin.count, T, nid, ownerMask, ownerMaskWords,
        tgtOwn, skipSameGroup, mac2, nodeCount, xoverThresh,
        /*emitInternalNodes=*/0);
    if (act == TC_DO_M2P) {
      ++nM2P;
    } else if (act == TC_DO_P2P) {
      ++nP2P;
    } else {  // TC_GO_CHILDREN
      stack[sp++] = admin.offset + 0;
      stack[sp++] = admin.offset + 1;
    }
  }
  m2pCount[tid] = nM2P;   // accepted nodes  (= # m2pWarp calls in the real path)
  p2pCount[tid] = nP2P;   // rejected leaves (= near pairs for this target)
}


// ======================================================================
//  Warp-specialized producer/consumer merged traversal (LOCK-FREE RING).
//
//  Block layout: 512 threads = 16 warps (launched with wsBlock=512).
//    Warp 0 (producer):  32 threads each traverse the BVH for a different
//                         target, enqueuing accepted-M2P and rejected-P2P
//                         items to a shared-memory circular queue.
//    Warps 1-15 (consumers): pop items from the queue and evaluate them --
//                           M2P via m2pWarp, P2P via lane-strided source loop.
//                           Results accumulate per-target in shared memory
//                           via fp64 atomicAdd.
//  Consumers spin-poll the ring head/tail; this fine-grained item-level overlap
//  is fast in practice and outperforms the staged-tile barrier variant below.
//  This is the default `direct-warpspec` path.
// ======================================================================
template<class MP, int ORDER>
__global__ void traverseWarpSpecKernel(
    bvh3f bvh,
    const typename MP::NodeMAC *macArr,
    const typename MP::NodeM2P *m2pArr,
    // Near-field P2P uses fp64 GEOMETRY (srcPos64/tgtPos64), matching the split
    // paths' p2pKernel/p2pAtomicKernel: fp32 coordinates have O(1) relative
    // error in R = T - S for near-duplicate particle pairs (separation ~ fp32
    // ulp), which poisons the large mutual Stokeslet term. Traversal MAC and
    // M2P keep the fp32 target (same contract as the split producers).
    const vec3d *srcPos64, const vec3d *srcForce,
    const vec3f *tgtPos, const vec3d *tgtPos64,
    const int *bucketBegin, const int *bucketEnd,
    const int *srcOwner, const int *targetOwner,
    const uint32_t *ownerMask, int ownerMaskWords,
    const int *nodeCount, int xoverThresh,
    int N, float mac, float pref, int skipSameGroup,
    double *potential, unsigned long long *nearLeafCount,
    unsigned long long *p2pCount
#if TREECODE_STATS
    , uint32_t *visited,
    uint32_t *m2pAccepted
#endif
    )
{
  // Block size is fixed at compile time by DIRECT_WARPSPEC_BLOCK (file scope);
  // the launcher uses the same constant, so the per-consumer-warp temp storage
  // below is always sized to match the launch.
  static constexpr int WARPSPEC_BLOCK    = DIRECT_WARPSPEC_BLOCK;
  static constexpr int NUM_WARPS         = WARPSPEC_BLOCK / 32;
  static constexpr int CONSUMER_WARPS    = NUM_WARPS - 1;
  static constexpr int TARGETS_PER_BLOCK = 32;
  static constexpr int QUEUE_CAP         = 512;
  static constexpr int QUEUE_MASK        = QUEUE_CAP - 1;

  typedef cub::WarpReduce<double> WarpReduceD;

  __shared__ vec3f  s_targetPos[TARGETS_PER_BLOCK];
  __shared__ vec3d  s_targetPos64[TARGETS_PER_BLOCK];
  __shared__ int    s_targetOwner[TARGETS_PER_BLOCK];
  __shared__ double s_results[TARGETS_PER_BLOCK][3];
  __shared__ int    s_qTarget[QUEUE_CAP];
  __shared__ int    s_qNode[QUEUE_CAP];
  __shared__ int    s_qTail;
  __shared__ int    s_qHead;
  __shared__ int    s_producerDone;
  __shared__ typename WarpReduceD::TempStorage s_warpTemp[CONSUMER_WARPS][3];

  const int warpId = threadIdx.x >> 5;
  const int lane   = threadIdx.x & 31;
  const float mac2 = mac * mac;

  // ---- Block-level init ----
  if (threadIdx.x < TARGETS_PER_BLOCK) {
    const int gid = blockIdx.x * TARGETS_PER_BLOCK + threadIdx.x;
    if (gid < N) {
      s_targetPos[threadIdx.x]   = tgtPos[gid];
      s_targetPos64[threadIdx.x] = tgtPos64[gid];
      s_targetOwner[threadIdx.x] =
          (skipSameGroup && targetOwner) ? targetOwner[gid] : -1;
    } else {
      s_targetPos[threadIdx.x]   = vec3f(0.f, 0.f, 0.f);
      s_targetPos64[threadIdx.x] = vec3d(0.0, 0.0, 0.0);
      s_targetOwner[threadIdx.x] = -1;
    }
    s_results[threadIdx.x][0] = 0.0;
    s_results[threadIdx.x][1] = 0.0;
    s_results[threadIdx.x][2] = 0.0;
  }
  if (threadIdx.x == 0) {
    s_qTail        = 0;
    s_qHead        = 0;
    s_producerDone = 0;
  }
  __syncthreads();

  if (warpId == 0) {
    // ==================== PRODUCER WARP ====================
    const int myTarget = blockIdx.x * TARGETS_PER_BLOCK + lane;
    const bool valid   = myTarget < N;
    const vec3f T      = s_targetPos[lane];
    const int tgtOwn   = s_targetOwner[lane];

#if TREECODE_STATS
    uint32_t nVisited = 0;
    uint32_t nM2P_    = 0;
#endif
    unsigned long long localNearLeaves = 0;

    uint32_t stack[64];
    int sp = 0;
    if (valid) stack[sp++] = 0;

    bool threadDone = !valid;
    while (__ballot_sync(0xffffffffu, !threadDone)) {
      bool hasWork   = false;
      int  workNode  = -1;

      if (!threadDone) {
        if (sp > 0) {
          const uint32_t nid  = stack[--sp];
          const auto node     = bvh.nodes[nid];
          const auto admin    = node.admin;
          const auto &nm      = macArr[nid];
#if TREECODE_STATS
          ++nVisited;
#endif
          const float dx = T.x - nm.cx;
          const float dy = T.y - nm.cy;
          const float dz = T.z - nm.cz;
          const float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
          bool mayContainSelf = false;
          if (skipSameGroup && ownerMask && tgtOwn >= 0) {
            const uint32_t word =
                ownerMask[(size_t)nid * ownerMaskWords + (tgtOwn >> 5)];
            mayContainSelf = (word & (1u << (tgtOwn & 31))) != 0u;
          }

          // Crossover: small accepted nodes go P2P (leaf) / descend (internal).
          const bool accept =
              (r2 > 0.f && !mayContainSelf && nm.halfDiag2 < mac2 * r2 &&
         nodeOutsideNearRadius(nm.halfDiag2, r2));
          const bool doM2P = accept &&
              (xoverThresh <= 0 || __ldg(&nodeCount[nid]) >= xoverThresh);
          if (doM2P) {
            workNode = encodeInteractionNode(nid, TC_INTERACTION_M2P);
            hasWork  = true;
#if TREECODE_STATS
            ++nM2P_;
#endif
          } else if (admin.count != 0) {
            workNode = encodeInteractionNode(nid, TC_INTERACTION_P2P);
            hasWork  = true;
            ++localNearLeaves;
          } else {
            stack[sp++] = admin.offset + 0;
            stack[sp++] = admin.offset + 1;
          }
        }
        if (sp == 0) threadDone = true;
      }

      // Warp-aggregated enqueue
      const unsigned workMask = __ballot_sync(0xffffffffu, hasWork);
      if (workMask) {
        const int leader = __ffs(workMask) - 1;
        const int count  = __popc(workMask);
        int base;
        if (lane == leader) {
          // Spin until queue has enough space
          int localTail = s_qTail;
          while (localTail - atomicAdd(&s_qHead, 0) + count > QUEUE_CAP)
            ;
          base = localTail;
        }
        base = __shfl_sync(0xffffffffu, base, leader);
        if (hasWork) {
          const int slot =
              base + __popc(workMask & ((1u << lane) - 1u));
          s_qTarget[slot & QUEUE_MASK] = lane;
          s_qNode[slot & QUEUE_MASK]   = workNode;
        }
        __syncwarp();
        __threadfence_block();
        if (lane == leader)
          atomicExch(&s_qTail, base + count);
      }
    }

    // Producer finished
    __syncwarp();
    if (lane == 0) {
      atomicExch(&s_producerDone, 1);
      __threadfence_block();
    }

    // Warp-reduce near-leaf counts via shuffle (no shared memory needed)
    for (int off = 16; off > 0; off >>= 1)
      localNearLeaves += __shfl_down_sync(0xffffffffu, localNearLeaves, off);
    if (lane == 0 && nearLeafCount)
      atomicAdd(nearLeafCount, localNearLeaves);
#if TREECODE_STATS
    if (valid) {
      visited[myTarget]     = nVisited;
      m2pAccepted[myTarget] = nM2P_;
    }
#endif

  } else {
    // ==================== CONSUMER WARPS ====================
    const int consumerIdx = warpId - 1;
    unsigned long long localP2P = 0;

    for (;;) {
      int tgtLocal = -1, encoded = -1;
      if (lane == 0) {
        for (;;) {
          const int h = atomicAdd(&s_qHead, 0);
          const int t = atomicAdd(&s_qTail, 0);
          if (h < t) {
            __threadfence_block();
            const int preTarget = s_qTarget[h & QUEUE_MASK];
            const int preNode   = s_qNode[h & QUEUE_MASK];
            if (atomicCAS(&s_qHead, h, h + 1) == h) {
              tgtLocal = preTarget;
              encoded  = preNode;
              break;
            }
          } else if (atomicAdd(&s_producerDone, 0)) {
            const int h2 = atomicAdd(&s_qHead, 0);
            const int t2 = atomicAdd(&s_qTail, 0);
            if (h2 >= t2) break;
          }
        }
      }
      tgtLocal = __shfl_sync(0xffffffffu, tgtLocal, 0);
      encoded  = __shfl_sync(0xffffffffu, encoded, 0);
      if (tgtLocal < 0) break;

      const vec3f T        = s_targetPos[tgtLocal];
      const uint32_t nid   = decodeInteractionNode(encoded);
      const int kind       = decodeInteractionKind(encoded);

      double u[3] = {0.0, 0.0, 0.0};

      if (kind == TC_INTERACTION_M2P) {
        const auto &nm = macArr[nid];
        const vec3f c(nm.cx, nm.cy, nm.cz);
        typename MP::FarVec du;
        if constexpr (MP::WANTS_FP64_TARGET)
          du = MP::template m2pWarp<ORDER>(m2pArr[nid], c,
              mp::sph::d_tgt64_gs[blockIdx.x * TARGETS_PER_BLOCK + tgtLocal],
              lane, 0xffffffffu);
        else
          du = MP::template m2pWarp<ORDER>(m2pArr[nid], c, T, lane, 0xffffffffu);
        if (lane == 0) {
          u[0] = (double)pref * (double)du.x;
          u[1] = (double)pref * (double)du.y;
          u[2] = (double)pref * (double)du.z;
        }
      } else {
        const int tgtOwn = s_targetOwner[tgtLocal];
        const vec3d T64  = s_targetPos64[tgtLocal];
        const auto admin = bvh.nodes[nid].admin;
        for (uint32_t p = 0; p < admin.count; ++p) {
          const uint32_t bid = bvh.primIDs[admin.offset + p];
          const int begin = bucketBegin[bid];
          const int end   = bucketEnd[bid];
          for (int src = begin + lane; src < end; src += 32) {
            if (skipSameGroup && srcOwner && tgtOwn >= 0 &&
                srcOwner[src] == tgtOwn)
              continue;
            p2p(T64, srcPos64[src], srcForce[src], u);
            ++localP2P;
          }
        }
        u[0] = WarpReduceD(s_warpTemp[consumerIdx][0]).Sum(u[0]);
        u[1] = WarpReduceD(s_warpTemp[consumerIdx][1]).Sum(u[1]);
        u[2] = WarpReduceD(s_warpTemp[consumerIdx][2]).Sum(u[2]);
        // Prefactor applied once here (post-reduce), not per source inside p2p,
        // matching the M2P branch above.
        u[0] *= (double)pref;
        u[1] *= (double)pref;
        u[2] *= (double)pref;
      }

      if (lane == 0) {
        atomicAdd(&s_results[tgtLocal][0], u[0]);
        atomicAdd(&s_results[tgtLocal][1], u[1]);
        atomicAdd(&s_results[tgtLocal][2], u[2]);
      }
    }

    // Warp-reduce P2P counts via shuffle
    for (int off = 16; off > 0; off >>= 1)
      localP2P += __shfl_down_sync(0xffffffffu, localP2P, off);
    if (lane == 0 && p2pCount)
      atomicAdd(p2pCount, localP2P);
  }

  __syncthreads();

  // ---- Write results to global memory ----
  if (threadIdx.x < TARGETS_PER_BLOCK) {
    const int gid = blockIdx.x * TARGETS_PER_BLOCK + threadIdx.x;
    if (gid < N) {
      potential[(size_t)0 * (size_t)N + (size_t)gid] =
          s_results[threadIdx.x][0];
      potential[(size_t)1 * (size_t)N + (size_t)gid] =
          s_results[threadIdx.x][1];
      potential[(size_t)2 * (size_t)N + (size_t)gid] =
          s_results[threadIdx.x][2];
    }
  }
}


// ======================================================================
//  Split warp-specialized traversal (cached-P2P variant of the warpspec ring).
//
//  Block layout: blockDim.x threads (multiple of 32, >= 64). Warp 0 is the
//  producer; warps 1.. are consumers. Block size is a pure launch parameter --
//  no shared array depends on it -- and selects the consumer-warp count. It is
//  the compile-time SPLIT_WARPSPEC_BLOCK (file scope), not a runtime token.
//    Warp 0 (producer): 32 lanes each traverse the source BVH for one target.
//                       Accepted nodes (M2P) are enqueued into a lock-free
//                       shared ring; rejected leaves (P2P) are appended to the
//                       GLOBAL near-pair list (when EMIT_PAIRS), exactly like
//                       traverseM2PKernel_Warp, for the cached p2pKernel pass.
//    Consumers:         pop M2P items from the ring (one warp per item) and
//                       evaluate the far field via m2pWarp; accumulate per
//                       target in shared memory via fp64 atomicAdd.
//  Only M2P flows through the ring, so NO CUB WarpReduce temp is needed
//  (m2pWarp self-reduces via __shfl_down_sync). That keeps every shared array
//  block-size-independent. EMIT_PAIRS=false is the reapply far-field pass: the
//  cached P2P list is replayed separately, so the P2P-append code is compiled
//  out and the pair pointers may be null.
// ======================================================================

// PAIR_MODE: EMIT_PAIRS=true appends rejected leaves to the global near-pair list
// (split-warpspec apply). COUNT_ONLY=true instead counts rejected leaves per
// target into perTargetCount[] with no global writes (double-traverse apply pass
// 1); the actual pairs are written later by traversePairWriteKernel at prefix-sum
// offsets. Both false => M2P-only far field (reapply). EMIT_PAIRS and COUNT_ONLY
// are mutually exclusive. The M2P producer/consumer path is identical in all
// modes, so the far field is bit-identical.
// The kernel body lives in this __device__ function; two thin __global__
// shells below differ only in __launch_bounds__. Policies whose lane-batched
// m2p is register-hungry (CartesianStokes fp64 order-3 wants 132 regs) get the
// LB shell, which caps ptxas at 128 regs so the 512-thread launch always fits.
// Policies that fit comfortably (BaryStokes: 62-72 regs) get the unbounded
// shell: giving ptxas the 128-reg budget inflates their register usage and
// halves occupancy (2 blocks/SM -> 1; measured -35% on Bary M2P).
template<class MP, int ORDER, bool EMIT_PAIRS, bool COUNT_ONLY>
__device__ __forceinline__ void
traverseSplitWarpSpecBody(
    bvh3f bvh,
    const typename MP::NodeMAC *macArr,
    const typename MP::NodeM2P *m2pArr,
    const vec3f *tgtPos, const int *targetOwner,
    const uint32_t *ownerMask, int ownerMaskWords,
    const int *nodeCount, int xoverThresh, int emitInternalNodes,
    int N, float mac, float pref, int skipSameGroup,
    double *potential,
    int *pairTarget, int *pairLeaf,
    unsigned long long *pairCount, long long pairCapacity,
    int *perTargetCount
#if TREECODE_STATS
    , uint32_t *visited,
    uint32_t *m2pAccepted
#endif
    )
{
  // TARGETS_PER_BLOCK is the producer warp width (fixed 32, independent of the
  // launch block size). QUEUE_CAP is fixed too: a single warp-aggregated
  // enqueue reserves at most 32 <= QUEUE_CAP slots regardless of block size.
  static constexpr int TARGETS_PER_BLOCK = 32;
  static constexpr int QUEUE_CAP         = 512;
  static constexpr int QUEUE_MASK        = QUEUE_CAP - 1;

  __shared__ vec3f  s_targetPos[TARGETS_PER_BLOCK];
  __shared__ int    s_targetOwner[TARGETS_PER_BLOCK];
  // FarAcc: fp64, or fp32 on the WIDEBVH_FP32_LEVEL >= 1 path (native shared
  // fp32 atomics instead of the CAS+DADD loop an fp64 shared atomicAdd is).
  __shared__ FarAcc s_results[TARGETS_PER_BLOCK][3];
  __shared__ int    s_qTarget[QUEUE_CAP];
  __shared__ int    s_qNode[QUEUE_CAP];
  __shared__ int    s_qTail;
  __shared__ int    s_qHead;
  __shared__ int    s_producerDone;

  const int warpId = threadIdx.x >> 5;
  const int lane   = threadIdx.x & 31;
  const float mac2 = mac * mac;

  // ---- Block-level init ----
  if (threadIdx.x < TARGETS_PER_BLOCK) {
    const int gid = blockIdx.x * TARGETS_PER_BLOCK + threadIdx.x;
    if (gid < N) {
      s_targetPos[threadIdx.x]   = tgtPos[gid];
      s_targetOwner[threadIdx.x] =
          (skipSameGroup && targetOwner) ? targetOwner[gid] : -1;
    } else {
      s_targetPos[threadIdx.x]   = vec3f(0.f, 0.f, 0.f);
      s_targetOwner[threadIdx.x] = -1;
    }
    s_results[threadIdx.x][0] = FarAcc(0);
    s_results[threadIdx.x][1] = FarAcc(0);
    s_results[threadIdx.x][2] = FarAcc(0);
  }
  if (threadIdx.x == 0) {
    s_qTail        = 0;
    s_qHead        = 0;
    s_producerDone = 0;
  }
  __syncthreads();

  if (warpId == 0) {
    // ==================== PRODUCER WARP ====================
    const int myTarget = blockIdx.x * TARGETS_PER_BLOCK + lane;
    const bool valid   = myTarget < N;
    const vec3f T      = s_targetPos[lane];
    const int tgtOwn   = s_targetOwner[lane];

#if TREECODE_STATS
    uint32_t nVisited = 0;
    uint32_t nM2P_    = 0;
#endif
    // double-traverse Count mode: per-lane (= per-target) rejected-leaf counter.
    int myLeafCount = 0;

    uint32_t stack[64];
    int sp = 0;
    if (valid) stack[sp++] = 0;

    bool threadDone = !valid;
    while (__ballot_sync(0xffffffffu, !threadDone)) {
      bool     m2pWork = false;
      bool     p2pWork = false;
      uint32_t workNid = 0;

      if (!threadDone) {
        if (sp > 0) {
          const uint32_t nid  = stack[--sp];
          const auto admin    = bvh.nodes[nid].admin;
          const auto &nm      = macArr[nid];
#if TREECODE_STATS
          ++nVisited;
#endif
          const TcNodeAction act = classifyTraversalNode(
              nm, admin.count, T, nid, ownerMask, ownerMaskWords,
              tgtOwn, skipSameGroup, mac2, nodeCount, xoverThresh,
              emitInternalNodes);
          if (act == TC_DO_M2P) {
            workNid = nid;
            m2pWork = true;
#if TREECODE_STATS
            ++nM2P_;
#endif
          } else if (act == TC_DO_P2P) {
            workNid = nid;
            p2pWork = true;       // rejected leaf -> near-field P2P
            if constexpr (COUNT_ONLY) ++myLeafCount;
          } else {
            stack[sp++] = admin.offset + 0;
            stack[sp++] = admin.offset + 1;
          }
        }
        if (sp == 0) threadDone = true;
      }

      // ---- M2P -> shared ring (warp-aggregated enqueue) ----
      const unsigned m2pMask = __ballot_sync(0xffffffffu, m2pWork);
      if (m2pMask) {
        const int leader = __ffs(m2pMask) - 1;
        const int count  = __popc(m2pMask);
        int base;
        if (lane == leader) {
          // Spin until the ring has room (consumers advance s_qHead).
          int localTail = s_qTail;
          while (localTail - atomicAdd(&s_qHead, 0) + count > QUEUE_CAP)
            ;
          base = localTail;
        }
        base = __shfl_sync(0xffffffffu, base, leader);
        if (m2pWork) {
          const int slot = base + __popc(m2pMask & ((1u << lane) - 1u));
          s_qTarget[slot & QUEUE_MASK] = lane;
          s_qNode[slot & QUEUE_MASK]   = (int)workNid;
        }
        __syncwarp();
        __threadfence_block();
        if (lane == leader)
          atomicExch(&s_qTail, base + count);
      }

      // ---- P2P -> global near-pair list (apply only; warp-aggregated) ----
      if constexpr (EMIT_PAIRS) {
        const unsigned p2pMask = __ballot_sync(0xffffffffu, p2pWork);
        if (p2pMask) {
          const int leader = __ffs(p2pMask) - 1;   // independent of m2pMask
          const int cnt    = __popc(p2pMask);
          unsigned long long base;                  // 64-bit global slot alloc
          if (lane == leader)
            base = atomicAdd(pairCount, (unsigned long long)cnt);
          base = __shfl_sync(0xffffffffu, base, leader);
          if (p2pWork) {
            const long long slot =
                (long long)base + __popc(p2pMask & ((1u << lane) - 1u));
            if (slot < pairCapacity) {              // guard enables overflow detect
              pairTarget[slot] = myTarget;
              pairLeaf[slot]   = (int)workNid;
            }
          }
        }
      }
    }

    // Producer finished -> signal consumers.
    __syncwarp();
    if (lane == 0) {
      atomicExch(&s_producerDone, 1);
      __threadfence_block();
    }

    // double-traverse Count mode: publish this target's rejected-leaf count.
    if constexpr (COUNT_ONLY) {
      if (valid) perTargetCount[myTarget] = myLeafCount;
    }

#if TREECODE_STATS
    if (valid) {
      visited[myTarget]     = nVisited;
      m2pAccepted[myTarget] = nM2P_;
    }
#endif

  } else if constexpr (MP::LANE_BATCH_M2P) {
    // ============ CONSUMER WARPS (M2P only, lane-per-item batch) ============
    // The policy's m2p is cheap and register-resident, so instead of 32 lanes
    // cooperating on one item, the warp claims up to 32 items per head-CAS and
    // each lane runs the FULL serial m2p on its own (target, node) item. The
    // scatter uses the same shared fp64 atomicAdd as the per-item path below
    // (same-target concurrency across warps already exists there). Entries
    // must be read BEFORE the head CAS: the producer's ring-full spin treats
    // slots >= head as unconsumed, so a slot may be overwritten as soon as the
    // head moves past it (read-then-CAS, like the per-item pop).
    for (;;) {
      int h = 0, avail = 0, done = 0;
      if (lane == 0) {
        h     = atomicAdd(&s_qHead, 0);
        avail = atomicAdd(&s_qTail, 0) - h;
        if (avail <= 0) done = atomicAdd(&s_producerDone, 0);
      }
      h     = __shfl_sync(0xffffffffu, h, 0);
      avail = __shfl_sync(0xffffffffu, avail, 0);
      done  = __shfl_sync(0xffffffffu, done, 0);
      if (avail <= 0) {
        if (!done) continue;
        // Producer done: recheck emptiness (items may have landed between the
        // tail read and the done read) before the warp-uniform exit.
        int empty = 0;
        if (lane == 0)
          empty = (atomicAdd(&s_qHead, 0) >= atomicAdd(&s_qTail, 0));
        if (__shfl_sync(0xffffffffu, empty, 0)) break;
        continue;
      }

      const int want = min(32, avail);
      __threadfence_block();
      int myT = -1, myN = -1;
      if (lane < want) {
        const int slot = (h + lane) & QUEUE_MASK;
        myT = s_qTarget[slot];
        myN = s_qNode[slot];
      }
      __syncwarp();
      int claimed = 0;
      if (lane == 0)
        claimed = (atomicCAS(&s_qHead, h, h + want) == h);
      if (!__shfl_sync(0xffffffffu, claimed, 0)) continue;   // lost the race

      if (lane < want) {
        const uint32_t nid = (uint32_t)myN;
        const auto &nm     = macArr[nid];
        const vec3f c(nm.cx, nm.cy, nm.cz);
        const vec3d du = MP::template m2p<ORDER>(m2pArr[nid], c,
                                                 s_targetPos[myT]);
        atomicAdd(&s_results[myT][0], (FarAcc)((double)pref * du.x));
        atomicAdd(&s_results[myT][1], (FarAcc)((double)pref * du.y));
        atomicAdd(&s_results[myT][2], (FarAcc)((double)pref * du.z));
      }
    }
  } else {
    // ==================== CONSUMER WARPS (M2P only) ====================
    for (;;) {
      int tgtLocal = -1, nidEnc = -1;
      if (lane == 0) {
        for (;;) {
          const int h = atomicAdd(&s_qHead, 0);
          const int t = atomicAdd(&s_qTail, 0);
          if (h < t) {
            __threadfence_block();
            const int preTarget = s_qTarget[h & QUEUE_MASK];
            const int preNode   = s_qNode[h & QUEUE_MASK];
            if (atomicCAS(&s_qHead, h, h + 1) == h) {
              tgtLocal = preTarget;
              nidEnc   = preNode;
              break;
            }
          } else if (atomicAdd(&s_producerDone, 0)) {
            const int h2 = atomicAdd(&s_qHead, 0);
            const int t2 = atomicAdd(&s_qTail, 0);
            if (h2 >= t2) break;
          }
        }
      }
      tgtLocal = __shfl_sync(0xffffffffu, tgtLocal, 0);
      nidEnc   = __shfl_sync(0xffffffffu, nidEnc, 0);
      if (tgtLocal < 0) break;                      // warp-uniform exit

      const uint32_t nid = (uint32_t)nidEnc;
      const auto &nm     = macArr[nid];
      const vec3f c(nm.cx, nm.cy, nm.cz);
      typename MP::FarVec du;
      if constexpr (MP::WANTS_FP64_TARGET)
        du = MP::template m2pWarp<ORDER>(m2pArr[nid], c,
            mp::sph::d_tgt64_gs[blockIdx.x * TARGETS_PER_BLOCK + tgtLocal],
            lane, 0xffffffffu);
      else
        du = MP::template m2pWarp<ORDER>(m2pArr[nid], c, s_targetPos[tgtLocal],
            lane, 0xffffffffu);
      if (lane == 0) {
        // Level 0: (double)pref * (double)du -- unchanged. Level >= 1: FarVec
        // and FarAcc are both float, so this is one FMUL and a native shared
        // fp32 atomic per component.
        atomicAdd(&s_results[tgtLocal][0], (FarAcc)pref * (FarAcc)du.x);
        atomicAdd(&s_results[tgtLocal][1], (FarAcc)pref * (FarAcc)du.y);
        atomicAdd(&s_results[tgtLocal][2], (FarAcc)pref * (FarAcc)du.z);
      }
    }
  }

  __syncthreads();

  // ---- Write M2P far field to global memory (near field added later) ----
  if (threadIdx.x < TARGETS_PER_BLOCK) {
    const int gid = blockIdx.x * TARGETS_PER_BLOCK + threadIdx.x;
    if (gid < N) {
      // The one fp32 -> fp64 widening per target on the fp32 path.
      potential[(size_t)0 * (size_t)N + (size_t)gid] = (double)s_results[threadIdx.x][0];
      potential[(size_t)1 * (size_t)N + (size_t)gid] = (double)s_results[threadIdx.x][1];
      potential[(size_t)2 * (size_t)N + (size_t)gid] = (double)s_results[threadIdx.x][2];
    }
  }
}

// Shell arguments are forwarded verbatim to traverseSplitWarpSpecBody; see the
// comment above it for why there are two shells.
#if TREECODE_STATS
#define TC_SWS_STATS_PARAMS , uint32_t *visited, uint32_t *m2pAccepted
#define TC_SWS_STATS_ARGS   , visited, m2pAccepted
#else
#define TC_SWS_STATS_PARAMS
#define TC_SWS_STATS_ARGS
#endif

#define TC_SWS_PARAMS                                                     \
    bvh3f bvh,                                                            \
    const typename MP::NodeMAC *macArr,                                   \
    const typename MP::NodeM2P *m2pArr,                                   \
    const vec3f *tgtPos, const int *targetOwner,                          \
    const uint32_t *ownerMask, int ownerMaskWords,                        \
    const int *nodeCount, int xoverThresh, int emitInternalNodes,         \
    int N, float mac, float pref, int skipSameGroup,                      \
    double *potential,                                                    \
    int *pairTarget, int *pairLeaf,                                       \
    unsigned long long *pairCount, long long pairCapacity,                \
    int *perTargetCount TC_SWS_STATS_PARAMS

#define TC_SWS_ARGS                                                       \
    bvh, macArr, m2pArr, tgtPos, targetOwner, ownerMask, ownerMaskWords,  \
    nodeCount, xoverThresh, emitInternalNodes,                            \
    N, mac, pref, skipSameGroup, potential, pairTarget, pairLeaf,         \
    pairCount, pairCapacity, perTargetCount TC_SWS_STATS_ARGS

template<class MP, int ORDER, bool EMIT_PAIRS, bool COUNT_ONLY = false>
__global__ void
traverseSplitWarpSpecKernel(TC_SWS_PARAMS)
{
  traverseSplitWarpSpecBody<MP, ORDER, EMIT_PAIRS, COUNT_ONLY>(TC_SWS_ARGS);
}

template<class MP, int ORDER, bool EMIT_PAIRS, bool COUNT_ONLY = false>
__global__ void __launch_bounds__(SPLIT_WARPSPEC_LB_BLOCK,
                                  SPLIT_WARPSPEC_BLOCK / SPLIT_WARPSPEC_LB_BLOCK)
traverseSplitWarpSpecKernelLB(TC_SWS_PARAMS)
{
  traverseSplitWarpSpecBody<MP, ORDER, EMIT_PAIRS, COUNT_ONLY>(TC_SWS_ARGS);
}

#undef TC_SWS_PARAMS
#undef TC_SWS_ARGS
#undef TC_SWS_STATS_PARAMS
#undef TC_SWS_STATS_ARGS

// Compile-time shell selection: only the returned shell is instantiated for a
// given (policy, order, mode), so each policy pays only its own codegen.
template<class MP, int ORDER, bool EMIT_PAIRS, bool COUNT_ONLY = false>
constexpr auto splitWarpSpecKernel()
{
  if constexpr (MP::LANE_BATCH_M2P)
    return &traverseSplitWarpSpecKernelLB<MP, ORDER, EMIT_PAIRS, COUNT_ONLY>;
  else
    return &traverseSplitWarpSpecKernel<MP, ORDER, EMIT_PAIRS, COUNT_ONLY>;
}


// Per near-pair source range. BuildConfig(1) => each leaf is exactly one
// bucket, so a leaf maps to one contiguous particle range [begin,end).
__global__ void pairCountsKernel(bvh3f bvh,
                                 const int *bucketBegin, const int *bucketEnd,
                                 const int *sortedLeaf, int nPairs,
                                 int *pairBegin, int *pairCnt)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nPairs) return;
  const auto node = bvh.nodes[(uint32_t)sortedLeaf[i]];
  assert(node.admin.count == 1);                // single-bucket leaf invariant
  const uint32_t bid = bvh.primIDs[node.admin.offset];
  const int begin = bucketBegin[bid];
  pairBegin[i] = begin;
  pairCnt[i]   = bucketEnd[bid] - begin;
}

// double-traverse pass 2: one THREAD per target re-traverses the BVH and writes
// its rejected leaves as (target, leaf) pairs into the contiguous block
// [pairOffset[t], pairOffset[t+1]) -- so the output is already grouped/sorted by
// target and the radix sort is skipped. The per-node decision MUST match the
// Count-mode producer exactly (same classifyTraversalNode, same child push
// order), so off advances exactly perTargetCount[t] times and lands on
// pairOffset[t+1]. No atomics.
template<class MP>
__global__ void traversePairWriteKernel(
    bvh3f bvh, const typename MP::NodeMAC *macArr,
    const vec3f *tgtPos, const int *targetOwner,
    const uint32_t *ownerMask, int ownerMaskWords,
    const int *nodeCount, int xoverThresh,
    int N, float mac, int skipSameGroup,
    const unsigned long long *pairOffset,
    int *sortedTarget, int *sortedLeaf)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;

  const float mac2   = mac * mac;
  const vec3f T      = tgtPos[t];
  const int   tgtOwn = (skipSameGroup && targetOwner) ? targetOwner[t] : -1;
  unsigned long long off = pairOffset[t];

  uint32_t stack[64];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const uint32_t nid = stack[--sp];
    const auto admin   = bvh.nodes[nid].admin;
    const auto &nm     = macArr[nid];
    const TcNodeAction act = classifyTraversalNode(
        nm, admin.count, T, nid, ownerMask, ownerMaskWords,
        tgtOwn, skipSameGroup, mac2, nodeCount, xoverThresh,
        /*emitInternalNodes=*/0);
    if (act == TC_DO_M2P) {
      // far field handled by pass 1; nothing to write here
    } else if (act == TC_DO_P2P) {
      sortedTarget[off] = t;
      sortedLeaf[off]   = (int)nid;
      ++off;
    } else {
      stack[sp++] = admin.offset + 0;
      stack[sp++] = admin.offset + 1;
    }
  }
}

// ======================================================================
//  TC_PATH=skel: per-particle target grouping + skeleton M2P ("target-side
//  template lift").
//
//  All targets of one particle share ONE far/near split, decided by a
//  cluster-cluster MAC against the particle's target-group box:
//      (halfDiag_node + r_group) < mac * dist(C_group, node_center).
//  This is REQUIRED for correctness of the lift: the skeleton targets must
//  sample the same far-field source set as every other target of the group,
//  otherwise interpolating their values to the group is meaningless. It is
//  slightly stricter than the per-target MAC (more near-field work, slightly
//  lower far-field error); the near/far trade is retuned via mac.
//
//  skelGroupTraverseKernel: one WARP per group, two passes (COUNT then WRITE,
//  identical control flow, same pattern as double-traverse). Outputs a
//  group->accepted-node CSR (the first cached M2P list in the engine) and the
//  rejected leaves EXPANDED to per-target (target, leaf) pairs -- format-
//  identical to the unsorted list p2pAtomicKernel consumes, so the whole
//  near-field path is reused unchanged. With object source leaves and
//  same-group skipping, a group's OWN source leaf is dropped at emit time.
//
//  skelM2PEvalKernel: one warp per (group, skeleton target); evaluates the
//  group's CSR node list with the policy's m2pWarp into the compact
//  u_skel buffer (col-major nSkel x 3G, column j = comp*G + group). The lift
//  to all B targets is one shared cuBLAS DGEMM: velBucket(B x 3G) = T * u_skel.
// ======================================================================
template<class MP, bool WRITE>
__global__ void skelGroupTraverseKernel(
    bvh3f bvh, const typename MP::NodeMAC *macArr,
    const cuBQL::box3f *groupBoxes, int nGroups, int B, float mac,
    const uint32_t *ownerMask, int ownerMaskWords,
    int skipSameGroup, int srcObjectLeaves,
    const int *nodeCount, int xoverThresh,
    int *groupNodeCount, int *groupLeafCount,
    const int *nodeOffset, int *nodeList,
    const unsigned long long *pairOffset, int *pairTarget, int *pairLeaf,
    const int *leafOffset, int *leafList, int emitExpanded)
{
  const int lane = threadIdx.x & 31;
  const int g =
      (int)((((long long)blockIdx.x * blockDim.x) + threadIdx.x) >> 5);
  if (g >= nGroups) return;

  // All 32 lanes traverse identically (cheap, divergence-free); lanes matter
  // only for the strided expanded-pair writes.
  const cuBQL::box3f gb = groupBoxes[g];
  const float cgx = 0.5f * (gb.lower.x + gb.upper.x);
  const float cgy = 0.5f * (gb.lower.y + gb.upper.y);
  const float cgz = 0.5f * (gb.lower.z + gb.upper.z);
  const float ex = gb.upper.x - gb.lower.x;
  const float ey = gb.upper.y - gb.lower.y;
  const float ez = gb.upper.z - gb.lower.z;
  const float rg = 0.5f * sqrtf(ex * ex + ey * ey + ez * ez);
  const float mac2 = mac * mac;

  int nNodes = 0;
  int nLeaves = 0;
  const int nodeOff = WRITE ? nodeOffset[g] : 0;
  const unsigned long long pairBase =
      (WRITE && emitExpanded) ? pairOffset[g] : 0ull;
  const int leafOff = (WRITE && !emitExpanded) ? leafOffset[g] : 0;

  uint32_t stack[64];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const uint32_t nid = stack[--sp];
    const auto admin = bvh.nodes[nid].admin;
    const auto &nm = macArr[nid];
    const float dx = cgx - nm.cx;
    const float dy = cgy - nm.cy;
    const float dz = cgz - nm.cz;
    const float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
    bool mayContainSelf = false;
    if (skipSameGroup && ownerMask) {
      const uint32_t word = ownerMask[(size_t)nid * ownerMaskWords + (g >> 5)];
      mayContainSelf = (word & (1u << (g & 31))) != 0u;
    }
    const float sr = sqrtf(nm.halfDiag2) + rg;
    TcNodeAction act;
    if (r2 > 0.f && !mayContainSelf && sr * sr < mac2 * r2) {
      // Group MAC accepted; crossover routes tiny accepted leaves to exact P2P
      // (small accepted internals descend, exactly like classifyTraversalNode
      // with emitInternalNodes=0).
      if (xoverThresh <= 0 || __ldg(&nodeCount[nid]) >= xoverThresh)
        act = TC_DO_M2P;
      else
        act = (admin.count != 0) ? TC_DO_P2P : TC_GO_CHILDREN;
    } else {
      act = (admin.count != 0) ? TC_DO_P2P : TC_GO_CHILDREN;
    }
    if (act == TC_DO_M2P) {
      if (WRITE && lane == 0) nodeList[nodeOff + nNodes] = (int)nid;
      ++nNodes;
    } else if (act == TC_DO_P2P) {
      // Rejected (or tiny accepted) leaf -> near field for ALL B targets of the
      // group. With object source leaves + same-group skipping, the group's own
      // source leaf is dropped here instead of per-source in the P2P loop.
      bool skipOwn = false;
      if (skipSameGroup && srcObjectLeaves) {
        const uint32_t bid = bvh.primIDs[admin.offset];
        skipOwn = ((int)bid == g);
      }
      if (!skipOwn) {
        if (WRITE) {
          if (emitExpanded) {
            const unsigned long long base =
                pairBase + (unsigned long long)nLeaves * (unsigned long long)B;
            const int first = g * B;
            for (int j = lane; j < B; j += 32) {
              pairTarget[base + (unsigned long long)j] = first + j;
              pairLeaf[base + (unsigned long long)j] = (int)nid;
            }
          } else if (lane == 0) {
            leafList[leafOff + nLeaves] = (int)nid;   // group-level leaf CSR
          }
        }
        ++nLeaves;
      }
    } else {
      stack[sp++] = admin.offset + 0;
      stack[sp++] = admin.offset + 1;
    }
  }
  if (!WRITE && lane == 0) {
    groupNodeCount[g] = nNodes;
    groupLeafCount[g] = nLeaves;
  }
}

// One warp per (group, skeleton target): evaluate the group's cached CSR node
// list with the policy's warp-cooperative m2pWarp, write the prefactored fp64
// far field into u_skel (col-major nSkel x 3G, column j = comp*nGroups + g).
template<class MP, int ORDER>
__global__ void skelM2PEvalKernel(
    const typename MP::NodeMAC *macArr, const typename MP::NodeM2P *m2pArr,
    const int *nodeBegin, const int *nodeList,
    const vec3f *tgtPos, const int *skelIdx, int nSkel, int B, int nGroups,
    float pref, double *uSkel)
{
  const int lane = threadIdx.x & 31;
  const long long w =
      (((long long)blockIdx.x * blockDim.x) + threadIdx.x) >> 5;
  if (w >= (long long)nGroups * nSkel) return;
  const int g = (int)(w / nSkel);
  const int s = (int)(w - (long long)g * nSkel);
  const vec3f T = tgtPos[(size_t)g * B + skelIdx[s]];
  const unsigned int mask = 0xffffffffu;

  double u0 = 0.0, u1 = 0.0, u2 = 0.0;
  const int kEnd = nodeBegin[g + 1];
  for (int k = nodeBegin[g]; k < kEnd; ++k) {
    const int nid = nodeList[k];
    const auto &nm = macArr[nid];
    const vec3f c(nm.cx, nm.cy, nm.cz);
    const auto du =
        MP::template m2pWarp<ORDER>(m2pArr[nid], c, T, lane, mask);
    if (lane == 0) { u0 += du.x; u1 += du.y; u2 += du.z; }
  }
  if (lane == 0) {
    const double pd = (double)pref;
    uSkel[((size_t)0 * nGroups + g) * (size_t)nSkel + s] = pd * u0;
    uSkel[((size_t)1 * nGroups + g) * (size_t)nSkel + s] = pd * u1;
    uSkel[((size_t)2 * nGroups + g) * (size_t)nSkel + s] = pd * u2;
  }
}

// TC_SKEL_CHECK diagnostic: evaluate the SAME cached CSR far field at ALL B
// targets of a few sampled groups (one warp per (sample, target)), so the lift
// error can be isolated from the MAC-change error. Output layout
// uFull[(comp*nSample + sampleIdx)*B + j].
template<class MP, int ORDER>
__global__ void skelFullEvalKernel(
    const typename MP::NodeMAC *macArr, const typename MP::NodeM2P *m2pArr,
    const int *nodeBegin, const int *nodeList,
    const vec3f *tgtPos, const int *sampleGroups, int nSample, int B,
    float pref, double *uFull)
{
  const int lane = threadIdx.x & 31;
  const long long w =
      (((long long)blockIdx.x * blockDim.x) + threadIdx.x) >> 5;
  if (w >= (long long)nSample * B) return;
  const int gi = (int)(w / B);
  const int j = (int)(w - (long long)gi * B);
  const int g = sampleGroups[gi];
  const vec3f T = tgtPos[(size_t)g * B + j];
  const unsigned int mask = 0xffffffffu;

  double u0 = 0.0, u1 = 0.0, u2 = 0.0;
  const int kEnd = nodeBegin[g + 1];
  for (int k = nodeBegin[g]; k < kEnd; ++k) {
    const int nid = nodeList[k];
    const auto &nm = macArr[nid];
    const vec3f c(nm.cx, nm.cy, nm.cz);
    const auto du =
        MP::template m2pWarp<ORDER>(m2pArr[nid], c, T, lane, mask);
    if (lane == 0) { u0 += du.x; u1 += du.y; u2 += du.z; }
  }
  if (lane == 0) {
    const double pd = (double)pref;
    uFull[((size_t)0 * nSample + gi) * (size_t)B + j] = pd * u0;
    uFull[((size_t)1 * nSample + gi) * (size_t)B + j] = pd * u1;
    uFull[((size_t)2 * nSample + gi) * (size_t)B + j] = pd * u2;
  }
}

// TC_SKEL_P2P=block near field: since every target of a group shares the SAME
// rejected-leaf set (group-consistent MAC), the near field is consumed at
// (group, leaf) granularity with a GPU-Gems-style tile kernel. One THREAD owns
// one target (block per (group, target-chunk)); the block cooperatively stages
// each leaf's sources (pos+force, optional owner) into shared memory in tiles,
// every thread then streams the staged tile from shared (broadcast reads) and
// accumulates its target's fp64 velocity in registers across ALL of the
// group's leaves. One plain global += per component at the end -- each target
// belongs to exactly one thread, so NO atomics, and the accumulation order is
// fixed, so the result is deterministic. `srcOwner` is non-null only when the
// per-source same-group test is still needed (skipSameGroup with non-object
// source buckets; with object sources the group's own leaf was already dropped
// at emit time and no other leaf contains its sources).
__global__ void skelP2PBlockKernel(
    bvh3f bvh, const int *bucketBegin, const int *bucketEnd,
    const vec3d *srcPos, const vec3d *srcForce, const vec3d *tgtPos,
    const int *srcOwner,
    const int *leafBegin, const int *leafList,
    int B, int N, float pref, double *potential)
{
  __shared__ vec3d sPos[SKEL_P2P_TILE];
  __shared__ vec3d sFrc[SKEL_P2P_TILE];
  __shared__ int   sOwn[SKEL_P2P_TILE];

  const int g = (int)blockIdx.x;
  const int j = (int)blockIdx.y * blockDim.x + (int)threadIdx.x;
  const bool active = j < B;
  const int tid = g * B + j;
  const vec3d T = active ? tgtPos[tid] : vec3d(0.0, 0.0, 0.0);

  double u[3] = {0.0, 0.0, 0.0};
  const int kEnd = leafBegin[g + 1];
  for (int k = leafBegin[g]; k < kEnd; ++k) {
    const int nid = leafList[k];
    const auto nd = bvh.nodes[(uint32_t)nid];
    assert(nd.admin.count == 1);              // single-bucket leaf invariant
    const uint32_t bid = bvh.primIDs[nd.admin.offset];
    const int begin = bucketBegin[bid];
    const int end   = bucketEnd[bid];
    for (int tile = begin; tile < end; tile += SKEL_P2P_TILE) {
      const int n = min(SKEL_P2P_TILE, end - tile);
      __syncthreads();                        // previous tile fully consumed
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sPos[i] = srcPos[tile + i];
        sFrc[i] = srcForce[tile + i];
        if (srcOwner) sOwn[i] = srcOwner[tile + i];
      }
      __syncthreads();
      if (active) {
        for (int i = 0; i < n; ++i) {
          if (srcOwner && sOwn[i] == g) continue;
          p2p(T, sPos[i], sFrc[i], u);
        }
      }
    }
  }
  if (active) {
    const double pd = (double)pref;
    potential[(size_t)0 * (size_t)N + (size_t)tid] += pd * u[0];
    potential[(size_t)1 * (size_t)N + (size_t)tid] += pd * u[1];
    potential[(size_t)2 * (size_t)N + (size_t)tid] += pd * u[2];
  }
}

// ======================================================================
//  KernelKind::Traction siblings of the three skel evaluation kernels
//  (KAFMM-style kernel aggregation): the SAME Stokeslet proxy moments and the
//  SAME cached traversal CSRs are consumed, only the target-side pair kernel
//  changes to the target-normal-contracted traction t = -3/(4pi) R(R.q)(R.n)/r^5.
//  Every kernel takes the bucket-ordered target normals and a DOUBLE prefactor
//  (stokes::tractionPrefactor()).
// ======================================================================

// Traction M2I: one warp per (group, traction-skeleton target).
template<class MP, int ORDER>
__global__ void skelM2PTractionEvalKernel(
    const typename MP::NodeMAC *macArr, const typename MP::NodeM2P *m2pArr,
    const int *nodeBegin, const int *nodeList,
    const vec3f *tgtPos, const vec3d *tgtNrm,
    const int *skelIdx, int nSkel, int B, int nGroups,
    double pref, double *uSkel)
{
  const int lane = threadIdx.x & 31;
  const long long w =
      (((long long)blockIdx.x * blockDim.x) + threadIdx.x) >> 5;
  if (w >= (long long)nGroups * nSkel) return;
  const int g = (int)(w / nSkel);
  const int s = (int)(w - (long long)g * nSkel);
  const size_t tid = (size_t)g * B + skelIdx[s];
  const vec3f T = tgtPos[tid];
  const vec3d nA = tgtNrm[tid];
  const unsigned int mask = 0xffffffffu;

  // Per-lane accumulation across ALL accepted nodes; ONE warp reduction at the
  // end instead of one per node (differs from the per-node reduce at round-off
  // only).
  double t0 = 0.0, t1 = 0.0, t2 = 0.0;
  const int kEnd = nodeBegin[g + 1];
  for (int k = nodeBegin[g]; k < kEnd; ++k) {
    const int nid = nodeList[k];
    const auto &nm = macArr[nid];
    const vec3f c(nm.cx, nm.cy, nm.cz);
    MP::template tractionWarpAccum<ORDER>(m2pArr[nid], c, T, nA, lane,
                                          t0, t1, t2);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    t0 += __shfl_down_sync(mask, t0, offset);
    t1 += __shfl_down_sync(mask, t1, offset);
    t2 += __shfl_down_sync(mask, t2, offset);
  }
  if (lane == 0) {
    uSkel[((size_t)0 * nGroups + g) * (size_t)nSkel + s] = pref * t0;
    uSkel[((size_t)1 * nGroups + g) * (size_t)nSkel + s] = pref * t1;
    uSkel[((size_t)2 * nGroups + g) * (size_t)nSkel + s] = pref * t2;
  }
}

// TC_SKEL_CHECK diagnostic, traction variant of skelFullEvalKernel.
template<class MP, int ORDER>
__global__ void skelFullEvalTractionKernel(
    const typename MP::NodeMAC *macArr, const typename MP::NodeM2P *m2pArr,
    const int *nodeBegin, const int *nodeList,
    const vec3f *tgtPos, const vec3d *tgtNrm,
    const int *sampleGroups, int nSample, int B,
    double pref, double *uFull)
{
  const int lane = threadIdx.x & 31;
  const long long w =
      (((long long)blockIdx.x * blockDim.x) + threadIdx.x) >> 5;
  if (w >= (long long)nSample * B) return;
  const int gi = (int)(w / B);
  const int j = (int)(w - (long long)gi * B);
  const int g = sampleGroups[gi];
  const size_t tid = (size_t)g * B + j;
  const vec3f T = tgtPos[tid];
  const vec3d nA = tgtNrm[tid];
  const unsigned int mask = 0xffffffffu;

  double t0 = 0.0, t1 = 0.0, t2 = 0.0;
  const int kEnd = nodeBegin[g + 1];
  for (int k = nodeBegin[g]; k < kEnd; ++k) {
    const int nid = nodeList[k];
    const auto &nm = macArr[nid];
    const vec3f c(nm.cx, nm.cy, nm.cz);
    const vec3d dt =
        MP::template tractionWarp<ORDER>(m2pArr[nid], c, T, nA, lane, mask);
    if (lane == 0) { t0 += dt.x; t1 += dt.y; t2 += dt.z; }
  }
  if (lane == 0) {
    uFull[((size_t)0 * nSample + gi) * (size_t)B + j] = pref * t0;
    uFull[((size_t)1 * nSample + gi) * (size_t)B + j] = pref * t1;
    uFull[((size_t)2 * nSample + gi) * (size_t)B + j] = pref * t2;
  }
}

// Traction near field, TC_SKEL_P2P=block only: identical tiling/determinism to
// skelP2PBlockKernel, with stokes::traction_p2p in the tile loop and one extra
// per-thread register for the target normal.
__global__ void skelP2PBlockTractionKernel(
    bvh3f bvh, const int *bucketBegin, const int *bucketEnd,
    const vec3d *srcPos, const vec3d *srcForce, const vec3d *tgtPos,
    const vec3d *tgtNrm, const int *srcOwner,
    const int *leafBegin, const int *leafList,
    int B, int N, double pref, double *potential)
{
  __shared__ vec3d sPos[SKEL_P2P_TILE];
  __shared__ vec3d sFrc[SKEL_P2P_TILE];
  __shared__ int   sOwn[SKEL_P2P_TILE];

  const int g = (int)blockIdx.x;
  const int j = (int)blockIdx.y * blockDim.x + (int)threadIdx.x;
  const bool active = j < B;
  const int tid = g * B + j;
  const vec3d T = active ? tgtPos[tid] : vec3d(0.0, 0.0, 0.0);
  const vec3d nA = active ? tgtNrm[tid] : vec3d(0.0, 0.0, 0.0);

  double t[3] = {0.0, 0.0, 0.0};
  const int kEnd = leafBegin[g + 1];
  for (int k = leafBegin[g]; k < kEnd; ++k) {
    const int nid = leafList[k];
    const auto nd = bvh.nodes[(uint32_t)nid];
    assert(nd.admin.count == 1);              // single-bucket leaf invariant
    const uint32_t bid = bvh.primIDs[nd.admin.offset];
    const int begin = bucketBegin[bid];
    const int end   = bucketEnd[bid];
    for (int tile = begin; tile < end; tile += SKEL_P2P_TILE) {
      const int n = min(SKEL_P2P_TILE, end - tile);
      __syncthreads();                        // previous tile fully consumed
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sPos[i] = srcPos[tile + i];
        sFrc[i] = srcForce[tile + i];
        if (srcOwner) sOwn[i] = srcOwner[tile + i];
      }
      __syncthreads();
      if (active) {
        for (int i = 0; i < n; ++i) {
          if (srcOwner && sOwn[i] == g) continue;
          traction_p2p(T, nA, sPos[i], sFrc[i], t);
        }
      }
    }
  }
  if (active) {
    potential[(size_t)0 * (size_t)N + (size_t)tid] += pref * t[0];
    potential[(size_t)1 * (size_t)N + (size_t)tid] += pref * t[1];
    potential[(size_t)2 * (size_t)N + (size_t)tid] += pref * t[2];
  }
}


// Split evaluation, pass 2: one WARP per near-pair (Oseen direct sum).
// Warp w owns pair w. Its 32 lanes stride the source leaf
// [pairBegin[w], pairBegin[w]+pairCnt[w]) with coalesced pos/force loads,
// accumulate the Stokeslet in registers, then a CUB WarpReduce sums each
// component across the warp. Lane 0 writes ONE partial double3 per pair into
// `pairPartial` (component-major [3*nPairs]); NO atomics, NO write into the
// final potential. The per-target near field is formed afterward by a
// reduce-by-key over the (target-sorted) pairs + a scatter-add.
// Block size: P2P_BLOCK / P2P_WARPS (file-scope, top of file).

// `srcPos`/`srcForce` index source-bucket particles; `tgtPos` indexes targets.
// For self-interaction (Treecode::evaluate) the caller passes the same array for
// both source and target positions; the MFS treecode passes distinct sets.
__global__ void p2pKernel(const vec3d *srcPos, const vec3d *srcForce,
                          const vec3d *tgtPos,
                          const int *srcOwner, const int *targetOwner,
                          const int *sortedTarget, const int *pairBegin,
                          const int *pairCnt, int nPairs, float pref,
                          int skipSameGroup,
                          double *pairPartial)
{
  typedef cub::WarpReduce<double> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp[P2P_WARPS][3];

  const int lane      = threadIdx.x & 31;
  const int warpInBlk = threadIdx.x >> 5;
  const long long w   =
      (((long long)blockIdx.x * (long long)blockDim.x) + threadIdx.x) >> 5;
  if (w >= nPairs) return;

  const int target = sortedTarget[(size_t)w];
  const int begin  = pairBegin[(size_t)w];
  const int cnt    = pairCnt[(size_t)w];
  const vec3d T    = tgtPos[target];
  const int tgtOwner = (skipSameGroup && targetOwner) ? targetOwner[target] : -1;

  double u[3] = {0.0, 0.0, 0.0};
  for (int j = lane; j < cnt; j += 32) {          // coalesced across the warp
    const int src = begin + j;
    if (skipSameGroup && srcOwner && tgtOwner >= 0 && srcOwner[src] == tgtOwner)
      continue;
    p2p(T, srcPos[src], srcForce[src], u);
  }

  // Prefactor applied once here (post-reduce), not per source inside p2p.
  const double pd = (double)pref;
  const double s0 = WarpReduce(temp[warpInBlk][0]).Sum(u[0]);
  const double s1 = WarpReduce(temp[warpInBlk][1]).Sum(u[1]);
  const double s2 = WarpReduce(temp[warpInBlk][2]).Sum(u[2]);
  if (lane == 0) {
    pairPartial[(size_t)0 * (size_t)nPairs + (size_t)w] = pd * s0;
    pairPartial[(size_t)1 * (size_t)nPairs + (size_t)w] = pd * s1;
    pairPartial[(size_t)2 * (size_t)nPairs + (size_t)w] = pd * s2;
  }
}


// split-warpspec-atomic near-field kernel: one warp owns one UNSORTED
// (target, leaf) near pair. It folds in pairCountsKernel (inline leaf -> bucket
// source range lookup) and replaces the p2pKernel + reduce_by_key + scatterAdd
// chain: each warp accumulates the Stokeslet in registers, warp-reduces each
// component, then lane 0 atomicAdds the three components directly onto the far
// field already in `potential` (component-major [3*N], fp64). Because the pair
// list is unsorted, many warps can target the same row -> the atomics serialize
// hot targets, but no radix sort / reduce-by-key is needed and the cache is just
// two int arrays. Atomic ordering is nondeterministic, so the result differs
// from the sorted path at round-off only.
__global__ void p2pAtomicKernel(bvh3f bvh,
                                const int *bucketBegin, const int *bucketEnd,
                                const vec3d *srcPos, const vec3d *srcForce,
                                const vec3d *tgtPos,
                                const int *srcOwner, const int *targetOwner,
                                const int *pairTarget, const int *pairLeaf,
                                const int *nodeDfsFirst, const int *nodeBucketCount,
                                const int *dfsBuckets,
                                int nPairs, float pref, int skipSameGroup,
                                int N, double *potential)
{
  typedef cub::WarpReduce<double> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp[P2P_WARPS][3];

  const int lane      = threadIdx.x & 31;
  const int warpInBlk = threadIdx.x >> 5;
  const long long w   =
      (((long long)blockIdx.x * (long long)blockDim.x) + threadIdx.x) >> 5;
  if (w >= nPairs) return;

  const int target = pairTarget[(size_t)w];
  const int node   = pairLeaf[(size_t)w];
  const vec3d T   = tgtPos[target];
  const int tgtOwner = (skipSameGroup && targetOwner) ? targetOwner[target] : -1;

  double u[3] = {0.0, 0.0, 0.0};
  if (dfsBuckets) {
    // Node-range: the node may be internal. Loop its contiguous DFS bucket run;
    // each bucket's particles are strided across the warp. Handles leaves (nb=1)
    // and internals (nb>1) uniformly, so no BuildConfig(1) single-bucket assert.
    const int first = nodeDfsFirst[node];
    const int nb    = nodeBucketCount[node];
    for (int b = 0; b < nb; ++b) {
      const int bid   = dfsBuckets[first + b];
      const int begin = bucketBegin[bid];
      const int end   = bucketEnd[bid];
      for (int src = begin + lane; src < end; src += 32) {
        if (skipSameGroup && srcOwner && tgtOwner >= 0 &&
            srcOwner[src] == tgtOwner)
          continue;
        p2p(T, srcPos[src], srcForce[src], u);
      }
    }
  } else {
    // Single-bucket leaf (BuildConfig(1) => one bucket per leaf).
    const auto nd = bvh.nodes[(uint32_t)node];
    assert(nd.admin.count == 1);
    const uint32_t bid = bvh.primIDs[nd.admin.offset];
    const int begin = bucketBegin[bid];
    const int cnt   = bucketEnd[bid] - begin;
    for (int j = lane; j < cnt; j += 32) {          // coalesced across the warp
      const int src = begin + j;
      if (skipSameGroup && srcOwner && tgtOwner >= 0 && srcOwner[src] == tgtOwner)
        continue;
      p2p(T, srcPos[src], srcForce[src], u);
    }
  }

  // Prefactor applied once here (post-reduce), not per source inside p2p.
  const double pd = (double)pref;
  const double s0 = WarpReduce(temp[warpInBlk][0]).Sum(u[0]);
  const double s1 = WarpReduce(temp[warpInBlk][1]).Sum(u[1]);
  const double s2 = WarpReduce(temp[warpInBlk][2]).Sum(u[2]);
  if (lane == 0) {
    atomicAdd(&potential[(size_t)0 * (size_t)N + (size_t)target], pd * s0);
    atomicAdd(&potential[(size_t)1 * (size_t)N + (size_t)target], pd * s1);
    atomicAdd(&potential[(size_t)2 * (size_t)N + (size_t)target], pd * s2);
  }
}

#if WIDEBVH_FP32_LEVEL >= 2
// ALL-fp32 twin of p2pAtomicKernel (WIDEBVH_FP32_LEVEL >= 2, see
// stokes_kernel.cuh): fp32 bucket coordinates for sources and targets, an fp32
// copy of the gathered forces, stokes::p2p32 per interaction, fp32 lane
// accumulators, cub::WarpReduce<float>, and native fp32 global atomics into a
// per-apply zeroed scratch `near32` (component-major [3*N]) that
// scatterCompToInputOrderFoldKernel adds onto the fp64 far field. Same pair
// list, same leaf -> bucket lookup, same skips as the fp64 kernel above.
__global__ void p2pAtomicKernel32(bvh3f bvh,
                                  const int *bucketBegin, const int *bucketEnd,
                                  const vec3f *srcPos, const vec3f *srcForce,
                                  const vec3f *tgtPos,
                                  const int *srcOwner, const int *targetOwner,
                                  const int *pairTarget, const int *pairLeaf,
                                  const int *nodeDfsFirst, const int *nodeBucketCount,
                                  const int *dfsBuckets,
                                  int nPairs, float pref, int skipSameGroup,
                                  int N, float *near32)
{
  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp[P2P_WARPS][3];

  const int lane      = threadIdx.x & 31;
  const int warpInBlk = threadIdx.x >> 5;
  const long long w   =
      (((long long)blockIdx.x * (long long)blockDim.x) + threadIdx.x) >> 5;
  if (w >= nPairs) return;

  const int target = pairTarget[(size_t)w];
  const int node   = pairLeaf[(size_t)w];
  const vec3f T   = tgtPos[target];
  const int tgtOwner = (skipSameGroup && targetOwner) ? targetOwner[target] : -1;

  float u[3] = {0.f, 0.f, 0.f};
  if (dfsBuckets) {
    const int first = nodeDfsFirst[node];
    const int nb    = nodeBucketCount[node];
    for (int b = 0; b < nb; ++b) {
      const int bid   = dfsBuckets[first + b];
      const int begin = bucketBegin[bid];
      const int end   = bucketEnd[bid];
      for (int src = begin + lane; src < end; src += 32) {
        if (skipSameGroup && srcOwner && tgtOwner >= 0 &&
            srcOwner[src] == tgtOwner)
          continue;
        stokes::p2p32(T, srcPos[src], srcForce[src], u);
      }
    }
  } else {
    const auto nd = bvh.nodes[(uint32_t)node];
    assert(nd.admin.count == 1);
    const uint32_t bid = bvh.primIDs[nd.admin.offset];
    const int begin = bucketBegin[bid];
    const int cnt   = bucketEnd[bid] - begin;
    for (int j = lane; j < cnt; j += 32) {
      const int src = begin + j;
      if (skipSameGroup && srcOwner && tgtOwner >= 0 && srcOwner[src] == tgtOwner)
        continue;
      stokes::p2p32(T, srcPos[src], srcForce[src], u);
    }
  }

  const float s0 = WarpReduce(temp[warpInBlk][0]).Sum(u[0]);
  const float s1 = WarpReduce(temp[warpInBlk][1]).Sum(u[1]);
  const float s2 = WarpReduce(temp[warpInBlk][2]).Sum(u[2]);
  if (lane == 0) {
    atomicAdd(&near32[(size_t)0 * (size_t)N + (size_t)target], pref * s0);
    atomicAdd(&near32[(size_t)1 * (size_t)N + (size_t)target], pref * s1);
    atomicAdd(&near32[(size_t)2 * (size_t)N + (size_t)target], pref * s2);
  }
}
#endif  // WIDEBVH_FP32_LEVEL >= 2


// ---- structured leaf-centric near field (TC_SRC_TEMPLATE, atomic path) --------
// Group the emitted (target, leaf) pairs by object id via a counting sort. In
// object mode a BVH leaf is one bucket == one object, so the object id is
// primIDs[node.admin.offset]. Pass 1 histograms per object; pass 2 scatters each
// pair's target into its object's CSR slot.
__global__ void structLeafBidHistoKernel(bvh3f bvh, const int *pairLeaf,
                                         int nPairs, int *count)
{
  const long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nPairs) return;
  const auto nd = bvh.nodes[(uint32_t)pairLeaf[(size_t)w]];
  const uint32_t bid = bvh.primIDs[nd.admin.offset];   // object id
  atomicAdd(&count[bid], 1);
}

__global__ void structLeafScatterKernel(bvh3f bvh, const int *pairTarget,
                                        const int *pairLeaf, int nPairs,
                                        int *cursor, int *groupTargets)
{
  const long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nPairs) return;
  const auto nd = bvh.nodes[(uint32_t)pairLeaf[(size_t)w]];
  const uint32_t bid = bvh.primIDs[nd.admin.offset];
  const int slot = atomicAdd(&cursor[bid], 1);
  groupTargets[slot] = pairTarget[(size_t)w];
}

// One BLOCK per object (grid = nObj). It reconstructs the object's nPts source
// positions ONCE into shared memory (shared template rotated+translated by the
// object's transform), then the block's warps stream that object's near targets
// (its CSR segment) against the shared cloud and atomicAdd into the velocity
// array. Reconstruction is amortized over all the object's near pairs. Numerics
// match p2pAtomicKernel (fp64 p2p + fp64 atomic scatter); only the source
// positions come from reconstruction (== input - shift to fp64 round-off).
__global__ void p2pLeafGroupedStructuredKernel(
    const double *R, const vec3d *center, const vec3d *tmpl, int nPts,
    const int *bucketBegin, const vec3d *srcForce, const vec3d *tgtPos,
    const int *groupTargets, const int *groupOffset,
    const int *targetOwner, int skipSameGroup, int N,
    float pref, double *potential)
{
  typedef cub::WarpReduce<double> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp[P2P_WARPS][3];
  extern __shared__ vec3d shPos[];        // nPts reconstructed sources
  __shared__ double sR[9];
  __shared__ vec3d  sc;

  const int bid    = blockIdx.x;
  const int segBeg = groupOffset[bid];
  const int segEnd = groupOffset[bid + 1];
  if (segBeg == segEnd) return;           // object has no near targets

  const int tid = threadIdx.x;
  if (tid < 9) sR[tid] = R[(size_t)9 * (size_t)bid + tid];
  if (tid == 0) sc = center[bid];
  __syncthreads();

  for (int i = tid; i < nPts; i += blockDim.x)
    shPos[i] = reconstructSrc64(sR, tmpl[i], sc);
  __syncthreads();

  const int fbase     = bucketBegin[bid];   // this object's force base (= bid*nPts)
  const int lane      = tid & 31;
  const int warpInBlk = tid >> 5;
  const int nWarps    = blockDim.x >> 5;
  const double pd     = (double)pref;

  for (int s = segBeg + warpInBlk; s < segEnd; s += nWarps) {
    const int target = groupTargets[s];
    // skipSameGroup: every source here belongs to object bid, so a same-group
    // target skips the whole pair (no-op unless skipSameGroup is on).
    if (skipSameGroup && targetOwner && targetOwner[target] == bid) continue;
    const vec3d T = tgtPos[target];
    double u[3] = {0.0, 0.0, 0.0};
    for (int j = lane; j < nPts; j += 32)
      p2p(T, shPos[j], srcForce[fbase + j], u);
    const double s0 = WarpReduce(temp[warpInBlk][0]).Sum(u[0]);
    const double s1 = WarpReduce(temp[warpInBlk][1]).Sum(u[1]);
    const double s2 = WarpReduce(temp[warpInBlk][2]).Sum(u[2]);
    if (lane == 0) {
      atomicAdd(&potential[(size_t)0 * (size_t)N + (size_t)target], pd * s0);
      atomicAdd(&potential[(size_t)1 * (size_t)N + (size_t)target], pd * s1);
      atomicAdd(&potential[(size_t)2 * (size_t)N + (size_t)target], pd * s2);
    }
  }
}


// Add each distinct target's summed near field (from the reduce-by-key) onto the
// far field already stored in `potential` (component-major [3*N]). Unique targets
// are distinct, so the additions never race -- no atomics needed.
__global__ void scatterAddKernel(const int *uniqueTarget,
                                 const double *sum0, const double *sum1,
                                 const double *sum2, int nUnique, int N,
                                 double *potential)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nUnique) return;
  const int t = uniqueTarget[k];
  potential[(size_t)0 * (size_t)N + (size_t)t] += sum0[k];
  potential[(size_t)1 * (size_t)N + (size_t)t] += sum1[k];
  potential[(size_t)2 * (size_t)N + (size_t)t] += sum2[k];
}

// Source forces are provided in caller/original order. Buckets own a permutation
// bucket slot -> original index, so this gathers into source-bucket order.
struct SourceForceGatherF {
  const vec3f *force;
  __host__ __device__ vec3d operator()(uint32_t originalIndex) const
  {
    const vec3f f = force[originalIndex];
    return vec3d((double)f.x, (double)f.y, (double)f.z);
  }
};

// fp64 -> fp32 force narrowing for the WIDEBVH_FP32_LEVEL >= 2 near field when
// the caller supplied fp64 forces (the drivers; NeMO passes fp32).
struct NarrowForceD2F {
  __host__ __device__ vec3f operator()(const vec3d &f) const
  {
    return vec3f((float)f.x, (float)f.y, (float)f.z);
  }
};

struct SourceForceGatherD {
  const vec3d *force;
  __host__ __device__ vec3d operator()(uint32_t originalIndex) const
  {
    return force[originalIndex];
  }
};

struct OwnerFromOriginalIndex {
  int groupSize;
  __host__ __device__ int operator()(uint32_t originalIndex) const
  {
    return (groupSize > 0) ? (int)(originalIndex / (uint32_t)groupSize) : -1;
  }
};

// TC_PATH=skel helpers: per-group bounding-sphere radius from the target-group
// AABB, and the leaf-count -> expanded-pair-count map for the pair-offset scan.
struct BoxHalfDiagF {
  __host__ __device__ float operator()(const cuBQL::box3f &b) const
  {
    const float ex = b.upper.x - b.lower.x;
    const float ey = b.upper.y - b.lower.y;
    const float ez = b.upper.z - b.lower.z;
    return 0.5f * sqrtf(ex * ex + ey * ey + ez * ez);
  }
};

struct LeafCntToPairsULL {
  int B;
  __host__ __device__ unsigned long long operator()(int c) const
  {
    return (unsigned long long)c * (unsigned long long)B;
  }
};

// Structured source: shift a per-object input-frame center into the treecode's
// centered frame (c - outputShift), so reconstruction lands in the same frame as
// points64() = input - bounds.center().
struct SubShiftD {
  cuBQL::vec3d shift;
  __host__ __device__ vec3d operator()(const vec3d &c) const
  {
    return c - shift;
  }
};

// Component-major velocity in target-bucket order -> component-major velocity
// in caller/original target order.
__global__ void scatterCompToInputOrderKernel(const double *bucketVel,
                                              const uint32_t *targetPerm,
                                              int nTargets,
                                              double *outVel)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nTargets) return;
  const size_t original = (size_t)targetPerm[i];
  outVel[(size_t)0 * (size_t)nTargets + original] =
      bucketVel[(size_t)0 * (size_t)nTargets + (size_t)i];
  outVel[(size_t)1 * (size_t)nTargets + original] =
      bucketVel[(size_t)1 * (size_t)nTargets + (size_t)i];
  outVel[(size_t)2 * (size_t)nTargets + original] =
      bucketVel[(size_t)2 * (size_t)nTargets + (size_t)i];
}

#if WIDEBVH_FP32_LEVEL >= 2
// Same permutation, plus the fp32 near-field scratch of p2pAtomicKernel32
// folded onto the fp64 far field on the way out (one widening per component
// per target).
__global__ void scatterCompToInputOrderFoldKernel(const double *bucketVel,
                                                  const float *near32,
                                                  const uint32_t *targetPerm,
                                                  int nTargets,
                                                  double *outVel)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nTargets) return;
  const size_t original = (size_t)targetPerm[i];
  const size_t nT = (size_t)nTargets;
  outVel[0 * nT + original] = bucketVel[0 * nT + (size_t)i] + (double)near32[0 * nT + (size_t)i];
  outVel[1 * nT + original] = bucketVel[1 * nT + (size_t)i] + (double)near32[1 * nT + (size_t)i];
  outVel[2 * nT + original] = bucketVel[2 * nT + (size_t)i] + (double)near32[2 * nT + (size_t)i];
}
#endif  // WIDEBVH_FP32_LEVEL >= 2

// ======================================================================
//  Node-range crossover: build-time DFS bucket ordering (engine-owned).
//
//  The SAH/median/ELH builders lay primIDs out per-LEAF (BFS-numbered nodeIDs +
//  sort-by-leaf-nodeID), so an internal node's subtree is NOT a contiguous primIDs
//  range. To emit ONE (target, internal_node) near pair we build our own DFS
//  bucket ordering after the build: every node then maps to a contiguous run
//  [dfsFirst[node], dfsFirst[node]+bucketCount[node]) of dfsBuckets[]. Geometry
//  only -- built once, reused across reapply.
// ======================================================================

// Pass 1: bottom-up subtree bucket count via cuBQL refit_aggregate
// (child-before-parent arrival protocol). Leaf = admin.count buckets;
// internal = child0 + child1. Consumed by the top-down offset pass.
__device__ void nodeBucketCountCombine(bvh3f bvh, int cnt[], int nodeID)
{
  const auto admin = bvh.nodes[nodeID].admin;
  if (admin.count != 0) cnt[nodeID] = (int)admin.count;                 // leaf
  else cnt[nodeID] = cnt[admin.offset] + cnt[admin.offset + 1];         // internal
}
__device__ void (*nodeBucketCountCombine_fp)(bvh3f, int[], int)
    = &nodeBucketCountCombine;

// Init for the top-down pass: dfsFirst = -1 everywhere except the root (node 0).
__global__ void dfsInitKernel(int *dfsFirst, int numNodes)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numNodes) return;
  dfsFirst[i] = (i == 0) ? 0 : -1;
}

// Pass 2 (one launch per tree level): each internal node whose offset F is ready
// (>= 0) assigns its children's DFS offsets -- left = F, right = F +
// bucketCount[left] -- and sets *changed. Idempotent: a node with an
// already-assigned left child (>=0) skips. Parent id < child id (BFS allocation),
// so this converges top-down in ~tree-depth rounds.
__global__ void dfsOffsetRoundKernel(bvh3f bvh, const int *bucketCount,
                                     int *dfsFirst, int *changed)
{
  const int nid = blockIdx.x * blockDim.x + threadIdx.x;
  if (nid >= (int)bvh.numNodes || nid == 1) return;   // node 1 is unused in cuBQL
  const auto admin = bvh.nodes[nid].admin;
  if (admin.count != 0) return;                        // leaf: nothing to propagate
  const int F = dfsFirst[nid];
  if (F < 0) return;                                   // my offset not ready yet
  const int c0 = (int)admin.offset;
  if (dfsFirst[c0] >= 0) return;                       // already propagated
  dfsFirst[c0]     = F;
  dfsFirst[c0 + 1] = F + bucketCount[c0];
  *changed = 1;
}

// Pass 3: scatter each leaf's buckets into DFS order. dfsBuckets[dfsFirst[leaf]+k]
// = primIDs[admin.offset+k]. Skips node 1 and any unreachable node (dfsFirst < 0).
__global__ void dfsScatterKernel(bvh3f bvh, const int *dfsFirst, int *dfsBuckets)
{
  const int nid = blockIdx.x * blockDim.x + threadIdx.x;
  if (nid >= (int)bvh.numNodes) return;
  const auto admin = bvh.nodes[nid].admin;
  if (admin.count == 0) return;                        // internal: skip
  const int base = dfsFirst[nid];
  if (base < 0) return;                                // unused / unreachable
  for (uint32_t k = 0; k < admin.count; ++k)
    dfsBuckets[base + (int)k] = (int)bvh.primIDs[admin.offset + k];
}

template <class T>
inline void releaseDeviceVector(thrust::device_vector<T> &v)
{
  thrust::device_vector<T>().swap(v);
}

#if TREECODE_STATS
inline void printTraversalDebugStats(const char *label,
                                     const uint32_t *d_visited,
                                     const uint32_t *d_m2pAccepted,
                                     size_t nTarget)
{
  if (nTarget == 0) return;
  std::vector<uint32_t> vis(nTarget);
  std::vector<uint32_t> m2p(nTarget);
  CUDA_CHECK(cudaMemcpy(vis.data(), d_visited, nTarget * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(m2p.data(), d_m2pAccepted, nTarget * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  std::sort(vis.begin(), vis.end());
  double visMean = 0.0, m2pMean = 0.0;
  for (uint32_t v : vis) visMean += v;
  for (uint32_t c : m2p) m2pMean += c;
  visMean /= (double)nTarget;
  m2pMean /= (double)nTarget;
  std::sort(m2p.begin(), m2p.end());
  const size_t iP99 = std::min((size_t)(0.99 * (nTarget - 1) + 0.5), nTarget - 1);
  printf("# [debug] nodes traversed per target (%s, over all %zu targets):\n",
         label, nTarget);
  printf("#   min=%u  median=%u  mean=%.1f  p99=%u  max=%u\n",
         vis.front(), vis[nTarget / 2], visMean, vis[iP99], vis.back());
  printf("# [debug] accepted m2p interactions per target (%s, over all %zu targets):\n",
         label, nTarget);
  printf("#   min=%u  median=%u  mean=%.1f  p99=%u  max=%u\n",
         m2p.front(), m2p[nTarget / 2], m2pMean, m2p[iP99], m2p.back());
}
#endif


// ======================================================================
//  Treecode: owns the spatial structure (grid buckets + BVH), the per-node
//  multipole moments, and the cached P2P list for apply-many runs.
// ======================================================================
template<class MP = mp::CartesianStokes>
class Treecode {
public:
  static constexpr int MAX_ORDER = MP::MAX_ORDER;
  static const char *multipoleName() { return MP::name(); }

  // ====================================================================
  // Execution paths (the single flat list).
  //
  // Exactly one path runs per Treecode, selected only by the TC_PATH
  // environment variable (see parsePathKind). There are no nested knobs and
  // no Config fields: the path is chosen entirely from this one flat list.
  // apply()/reapply() print the selected token + description.
  //
  //   TC_PATH            traversal: apply / reapply                   near-field
  //   ----------------   ------------------------------------------   --------------------------
  //   split-warp  (def)  traverseM2PKernel_Warp / _Warp               cached p2pKernel pair list
  //   split-warpspec     traverseSplitWarpSpecKernel / same           cached p2pKernel pair list
  //                      (warp 0 produces; consumer warps do M2P;     (block size is the
  //                      P2P appended to the global near-pair list)    compile-time
  //                                                                    SPLIT_WARPSPEC_BLOCK)
  //   double-traverse    traverseSplitWarpSpecKernel<COUNT_ONLY> /     cached p2pKernel pair list
  //                      traverseSplitWarpSpecKernel<EMIT=false>       (built by a 2nd traversal
  //                      (apply: warpspec M2P counts leaves; a 2nd      at prefix-sum offsets, so
  //                      one-thread-per-target pass writes pairs at     NO radix sort)
  //                      prefix-sum offsets. reapply == split-warpspec)
  //   direct-warpspec    traverseWarpSpecKernel (lock-free ring)       merged inline, no cache
  // ====================================================================
  enum class PathKind {
    SplitWarp,
    SplitWarpSpec,
    SplitWarpSpecAtomic,
    DoubleTraverse,
    DirectWarpSpec,
    Skel,           // target-skeleton lift: group traversal + skeleton M2P +
                    // shared per-template DGEMM lift + atomic-scatter P2P
    TraverseCount   // diagnostic: traversal-only interaction counting (no compute)
  };

  // Canonical TC_PATH token for a path.
  static const char *pathToken(PathKind p)
  {
    switch (p) {
      case PathKind::SplitWarp:           return "split-warp";
      case PathKind::SplitWarpSpec:       return "split-warpspec";
      case PathKind::SplitWarpSpecAtomic: return "split-warpspec-atomic";
      case PathKind::DoubleTraverse:      return "double-traverse";
      case PathKind::DirectWarpSpec:      return "direct-warpspec";
      case PathKind::Skel:                return "skel";
      case PathKind::TraverseCount:       return "traverse-count";
    }
    return "unknown";
  }

  // Human-readable summary (kernels + caching) printed by apply()/reapply().
  static const char *pathDescription(PathKind p)
  {
    switch (p) {
      case PathKind::SplitWarp:
        return "split: warp M2P emit + cached P2P";
      case PathKind::SplitWarpSpec:
        return "split: warp-specialized producer/consumer M2P (ring) + cached P2P";
      case PathKind::SplitWarpSpecAtomic:
        return "split: warpspec M2P (ring) + unsorted near pairs + atomic-scatter "
               "P2P (inline range lookup, no sort/reduce)";
      case PathKind::DoubleTraverse:
        return "split: warpspec M2P (count) + double-traverse pair write (no sort) + cached P2P";
      case PathKind::DirectWarpSpec:
        return "direct-near merged, warp-specialized producer/consumer (lock-free ring)";
      case PathKind::Skel:
        return "target-skeleton lift: per-group traversal + skeleton-only M2P + "
               "shared template DGEMM lift + atomic-scatter P2P";
      case PathKind::TraverseCount:
        return "diagnostic: traversal-only thread-per-target (count M2P/P2P interactions, no compute)";
    }
    return "unknown";
  }

  // Map a TC_PATH value to a PathKind. Null/empty selects dflt; an unrecognized
  // value throws with the accepted tokens listed.
  static PathKind parsePathKind(const char *env, PathKind dflt)
  {
    if (!env || !*env) return dflt;
    if (std::strcmp(env, "split-warp") == 0)      return PathKind::SplitWarp;
    if (std::strcmp(env, "split-warpspec") == 0)  return PathKind::SplitWarpSpec;
    if (std::strcmp(env, "split-warpspec-atomic") == 0)
      return PathKind::SplitWarpSpecAtomic;
    if (std::strcmp(env, "double-traverse") == 0) return PathKind::DoubleTraverse;
    if (std::strcmp(env, "direct-warpspec") == 0) return PathKind::DirectWarpSpec;
    if (std::strcmp(env, "skel") == 0)            return PathKind::Skel;
    if (std::strcmp(env, "traverse-count") == 0)  return PathKind::TraverseCount;
    throw std::runtime_error(
        std::string("bad TC_PATH=") + env +
        "; expected one of: split-warp, split-warpspec, split-warpspec-atomic, "
        "double-traverse, direct-warpspec, skel, traverse-count");
  }

  // Source-BVH builder selectable via TC_BVH_BUILDER. Default/Sah/Elh map to
  // cuBQL::BuildConfig::BuildMethod and go through gpuBuilder(); Radix is cuBQL's
  // standalone fast Morton/LBVH builder (cuBQL::cuda::radixBuilder), dispatched
  // separately in buildBvh(). makeLeafThreshold=1 (one bucket per BVH leaf) holds
  // for all of them.
  enum class BvhBuilder { Default, Sah, Elh, Radix };

  // Canonical TC_BVH_BUILDER token for a builder.
  static const char *bvhBuilderToken(BvhBuilder b)
  {
    switch (b) {
      case BvhBuilder::Default: return "default";
      case BvhBuilder::Sah:     return "sah";
      case BvhBuilder::Elh:     return "elh";
      case BvhBuilder::Radix:   return "radix";
    }
    return "unknown";
  }

  // Map a TC_BVH_BUILDER value to a builder. Null/empty selects `dflt` (the
  // project default is Sah); an unrecognized value throws.
  // Accepted: default|median|spatial-median (= adaptive spatial median), sah,
  // elh, radix|morton|lbvh (= fast Morton/LBVH radix builder).
  static BvhBuilder parseBvhBuilder(const char *env, BvhBuilder dflt)
  {
    if (!env || !*env) return dflt;
    if (std::strcmp(env, "default") == 0 ||
        std::strcmp(env, "median") == 0 ||
        std::strcmp(env, "spatial-median") == 0)
      return BvhBuilder::Default;
    if (std::strcmp(env, "sah") == 0) return BvhBuilder::Sah;
    if (std::strcmp(env, "elh") == 0) return BvhBuilder::Elh;
    if (std::strcmp(env, "radix") == 0 ||
        std::strcmp(env, "morton") == 0 ||
        std::strcmp(env, "lbvh") == 0)
      return BvhBuilder::Radix;
    throw std::runtime_error(
        std::string("bad TC_BVH_BUILDER=") + env +
        "; expected one of: default (=median/spatial-median), sah, elh, "
        "radix (=morton/lbvh)");
  }

  // Bucketizer (TC_BUCKETIZER).
  // GridHilbert (default): uniform grid cells auto-sized from the R_max
  // formula (cell half-diagonal = R_domain / cbrt(max(1024, N/maxLeaf))),
  // particles Hilbert-ordered WITHIN each cell, then chunked to <= maxLeaf
  // like grid. Cells cap bucket extent/occupancy (what global hilbert lacked)
  // while the within-cell Hilbert curve keeps dense-cell chunks compact
  // instead of Morton-stacked. cellEdge is ignored (auto); TC_HILBERT_Q
  // overrides q. 81-config gate: -0.97% geomean vs per-config-tuned grid,
  // 0 REGRESS (octree_results.md 2026-07-07).
  // Grid: the original uniform-cell + within-cell-Morton bucketizer
  // (cellEdge honored; bit-anchored legacy baseline).
  // Hilbert: one global Hilbert-SFC sort (cubized box), fixed-size chunks of
  // maxLeaf consecutive particles -- no spatial cap; still the plummer
  // optimum (~1.2% ahead of grid-hilbert at large h) but loses elsewhere.
  // Object: one bucket per physical object (contiguous run of sourceGroupSize /
  // targetGroupSize input points, e.g. one MFS particle's stokeslet cloud); no
  // sort, no maxLeaf split, cellEdge/maxLeaf ignored. Requires the matching
  // group size > 0 and evenly dividing the point count (opt-in for grouped
  // inputs like the ellipsoid MFS bench).
  enum class Bucketizer { Grid, Hilbert, GridHilbert, Object };

  static const char *bucketizerToken(Bucketizer b)
  {
    switch (b) {
      case Bucketizer::Grid:        return "grid";
      case Bucketizer::Hilbert:     return "hilbert";
      case Bucketizer::GridHilbert: return "grid-hilbert";
      case Bucketizer::Object:      return "object";
    }
    return "unknown";
  }

  static Bucketizer parseBucketizer(const char *env, Bucketizer dflt)
  {
    if (!env || !*env) return dflt;
    if (std::strcmp(env, "grid") == 0)         return Bucketizer::Grid;
    if (std::strcmp(env, "hilbert") == 0)      return Bucketizer::Hilbert;
    if (std::strcmp(env, "grid-hilbert") == 0) return Bucketizer::GridHilbert;
    if (std::strcmp(env, "object") == 0)       return Bucketizer::Object;
    throw std::runtime_error(
        std::string("bad TC_BUCKETIZER=") + env +
        "; expected one of: grid, hilbert, grid-hilbert, object");
  }

  // SFC box for cornerstone key computation (the global Hilbert bucketizer):
  // cubized around the domain center unless TC_OCT_SFC_BOX=domain
  // (cornerstone normalizes each axis independently; see octSfcCube_).
  cuBQL::box3d sfcBoxFor(const cuBQL::box3d &bounds) const
  {
    if (!octSfcCube_) return bounds;
    const cuBQL::vec3d c = bounds.center();
    const cuBQL::vec3d sz = bounds.size();
    const double half = 0.5 * std::max(sz.x, std::max(sz.y, sz.z));
    cuBQL::box3d b;
    b.lower = c - cuBQL::vec3d(half);
    b.upper = c + cuBQL::vec3d(half);
    return b;
  }

  // grid-hilbert auto cell edge for a side with n particles: h = 2*R_max/√3
  // so the CELL half-diagonal equals R_max = R_domain / q,
  // q = TC_HILBERT_Q override or cbrt(B_ref), B_ref = max(1024, n/maxLeaf).
  // On a cubic domain this yields >= B_ref cells (h = domain_edge/q exactly).
  // R_domain is the TIGHT point-bounds half-diagonal, like grid mode (per-cell
  // local curves make the global box aspect irrelevant, so no SFC cube box).
  double gridHilbertCellEdge(size_t n, const cuBQL::box3d &bounds) const
  {
    const double q =
      (gridHilbertQ_ > 0.0)
        ? gridHilbertQ_
        : std::cbrt(std::max(1024.0,
                             (double)n / (double)std::max(1, cfg_.maxLeaf)));
    const cuBQL::vec3d sz = bounds.size();
    const double rDomain =
      0.5 * std::sqrt(sz.x * sz.x + sz.y * sz.y + sz.z * sz.z);
    return 2.0 * (rDomain / q) / std::sqrt(3.0);
  }

  // TC_BUCKETIZER / TC_TGT_BUCKETIZER dispatch. The source side passes
  // bucketizer_, the target side targetBucketizer_ (which defaults to
  // bucketizer_ unless TC_TGT_BUCKETIZER overrides it), so sources can be
  // object-grouped while targets keep the spatial default -- see configure().
  // In object mode the caller passes the matching group size (sourceGroupSize
  // for the source side, targetGroupSize for targets); it is ignored otherwise.
  util::GridBuckets runBucketizer(const vec3d *d_pts, size_t n,
                                  const cuBQL::box3d &bounds,
                                  int objectGroupSize,
                                  Bucketizer which,
                                  bool structuredSource = false) const
  {
    if (which == Bucketizer::Object) {
      if (objectGroupSize < 1)
        throw std::runtime_error(
            "TC_BUCKETIZER=object requires a group size > 0 "
            "(set Config::sourceGroupSize / targetGroupSize)");
      // Structured (TC_SRC_TEMPLATE) source side: build the same object buckets
      // but do NOT materialize the per-point positions (reconstructed on the fly
      // from the shared template). Boxes still come straight off d_pts, so the
      // BVH is bit-identical.
      if (structuredSource)
        return util::buildStructuredObjectBuckets(d_pts, n, bounds.center(),
                                                  objectGroupSize);
      return util::buildObjectBuckets(d_pts, n, bounds.center(),
                                      objectGroupSize);
    }
    if (which == Bucketizer::Hilbert)
      return util::buildHilbertBuckets(d_pts, n, sfcBoxFor(bounds),
                                       bounds.center(), cfg_.maxLeaf);
    if (which == Bucketizer::GridHilbert)
      return util::buildGridBuckets(d_pts, n, bounds, bounds.center(),
                                    gridHilbertCellEdge(n, bounds),
                                    cfg_.maxLeaf, util::FineCurve::Hilbert);
    return util::buildGridBuckets(d_pts, n, bounds, bounds.center(),
                                  cfg_.cellEdge, cfg_.maxLeaf);
  }

  // split-warpspec's block size (consumer-warp count) is the compile-time
  // SPLIT_WARPSPEC_BLOCK at file scope, no longer a runtime TC_PATH token; edit
  // that constant to retune. Block == 32 producer lanes + (block/32 - 1)
  // consumer warps.

  struct Config {
    int    order    = 6;     // multipole expansion order (1..MAX_ORDER)
    float  mac      = 0.5f;  // multipole-acceptance criterion
    double cellEdge = 10.0;  // uniform grid cell edge length (fp64 units)
    int    maxLeaf  = 256;    // max particles per bucket
    int    sourceGroupSize = 0; // optional original-order source points/group
    int    targetGroupSize = 0; // optional original-order target points/group
    bool   skipSameGroup   = false; // exclude source group == target group
    // Near-field exclusion radius rc (0 = off, the classic whole-sum treecode).
    // With rc > 0 the evaluated sum runs over r >= rc only: pairs closer than rc
    // are skipped in P2P and no node reaching inside rc is ever accepted, so the
    // result is the exact complement of a hard r < rc cutoff. Intended for
    // near/far splits where an external operator owns the near field.
    double nearCutoff      = 0.0;
    // The execution path (split-warp / split-warpspec / direct-warpspec) is NOT
    // a Config field: it is chosen solely by the TC_PATH environment variable.
  };

  // apply/reapply counts + phase timings. The driver reads this after apply().
  struct Stats {
    // source geometry
    uint32_t numBuckets       = 0;  // kept for old driver wording: source buckets
    uint32_t numSourceBuckets = 0;
    uint32_t numTargetBuckets = 0;
    uint32_t numNodes         = 0;
    uint32_t traversalNodes   = 0;
    uint32_t traversalInner   = 0;
    uint32_t traversalLeaves  = 0;
    int      minTraversalLeafDepth = 0;
    int      maxTraversalLeafDepth = 0;
    double   meanTraversalLeafDepth = 0.0;
    uint32_t nx = 0, ny = 0, nz = 0;
    uint64_t totalCells    = 0;
    int      occupiedCells = 0;
    int      coarseBits    = 0;
    int      fineBits      = 0;
    // evaluation
    long long nPairs   = 0;  // near-field P2P pairs
    long long totalP2P = 0;
    // phase timings (ms)
    double bucketMs     = 0.0;
    double targetBucketMs = 0.0;
    double prepForcesMs = 0.0;
    double buildBvhMs   = 0.0;
    double upwardMs     = 0.0;
    float  travMs       = 0.f;   // pass-1 traverse + M2P
    float  p2pMs        = 0.f;   // sort + pass-2 P2P + reduce/scatter
    float  groupMs      = 0.f;   // structured leaf-centric: one-time CSR grouping
  };

  Treecode() { configure(cfg_); }
  explicit Treecode(const Config &cfg) { configure(cfg); }
  ~Treecode()
  {
    freeBuild();
    if (cublas_) {
      cublasDestroy(cublas_);
      cublas_ = nullptr;
    }
  }
  Treecode(const Treecode &) = delete;            // owns CUDA handles
  Treecode &operator=(const Treecode &) = delete;

  void setConfig(const Config &cfg)
  {
    assert(!built_);
    configure(cfg);
  }

  void setSkipSameGroup(bool skip)
  {
    if (cfg_.skipSameGroup == skip) return;
    cfg_.skipSameGroup = skip;
    if (built_) clearPairCache();
  }

  // Which target-side pair kernel apply()/reapply() evaluates. Stokeslet
  // (default) is the classic velocity treecode; Traction is the KAFMM-style
  // target-normal-contracted traction of the single layer, -3/(4pi)
  // R(R.q)(R.n)/r^5, from the SAME Stokeslet moments. Cheap runtime toggle:
  // the tree, buckets, owner masks, moments/upward pass, and the skel
  // traversal CSRs are all kernel-independent and stay cached across a
  // switch; only the (skeleton, lift) pair and the eval/P2P kernels differ.
  // Traction is supported ONLY on TC_PATH=skel with TC_SKEL_P2P=block and
  // requires setTargetNormals() (enforced fail-loud at evaluate time).
  enum class KernelKind { Stokeslet, Traction };
  void setKernel(KernelKind k) { kernel_ = k; }
  KernelKind kernel() const { return kernel_; }

  // Per-target OUTWARD unit normals in TARGET INPUT ORDER (vec3d AoS, nTarget
  // entries). Copied into an owned buffer that survives freeBuild() (input
  // state, like the structured-source template); build() gathers it to
  // target-bucket order. Consumed only by the Traction kernels.
  void setTargetNormals(const vec3d *d_normals, size_t nTarget)
  {
    if (d_normals == nullptr || nTarget == 0) {
      releaseDeviceVector(targetNormalsInput_);
      releaseDeviceVector(targetNormals64_);
      return;
    }
    targetNormalsInput_.resize(nTarget);
    CUDA_CHECK(cudaMemcpy(util::devicePtr(targetNormalsInput_), d_normals,
                          nTarget * sizeof(vec3d), cudaMemcpyDeviceToDevice));
    if (built_) {
      if (nTarget != nTarget_)
        throw std::runtime_error("setTargetNormals: size != nTarget of the "
                                 "built tree");
      gatherTargetNormals();
    }
  }

  // Full source-target apply. `d_force` is in source input order. The vec3d
  // overload preserves source strengths in fp64; the vec3f overload is kept for
  // existing drivers and widens once into the bucket-contiguous internal buffer.
  // If `d_target == nullptr`, targets are the sources. `d_velOut` is
  // caller-owned component-major double[3*nTarget] in target input order.
  void apply(const vec3d *d_source, size_t nSource,
             const vec3d *d_target, size_t nTarget,
             const vec3d *d_force, double *d_velOut,
             bool will_reuse_tree = false);
  void apply(const vec3d *d_source, size_t nSource,
             const vec3d *d_target, size_t nTarget,
             const vec3f *d_force, double *d_velOut,
             bool will_reuse_tree = false);

  // Self-target convenience overload.
  void apply(const vec3d *d_source, size_t nSource,
             const vec3d *d_force, double *d_velOut,
             bool will_reuse_tree = false)
  {
    apply(d_source, nSource, nullptr, 0, d_force, d_velOut, will_reuse_tree);
  }
  void apply(const vec3d *d_source, size_t nSource,
             const vec3f *d_force, double *d_velOut,
             bool will_reuse_tree = false)
  {
    apply(d_source, nSource, nullptr, 0, d_force, d_velOut, will_reuse_tree);
  }

  // Reuse the geometry + cached P2P list from apply(..., true), with new source
  // forces in source input order. M2P is retraversed; no M2P list is cached.
  void reapply(const vec3d *d_force, double *d_velOut);
  void reapply(const vec3f *d_force, double *d_velOut);

  // Opt-in monodisperse "structured source" mode (TC_SRC_TEMPLATE). Instead of
  // materializing the per-point source positions, the treecode reconstructs each
  // source on the fly from a single shared reference template + one rigid
  // transform per object: pos[k][i] = R_k * templatePts[i] + centers[k]
  // (R_k row-major 3x3, mirrors mfs_broms.cuh::tileRotateTranslate). Call after
  // the constructor / setConfig and BEFORE apply. Only supported on the
  // split-warpspec-atomic path with TC_BUCKETIZER=object and the BaryStokes
  // policy (build() enforces this). The buffers are copied into device members
  // and persist across reapply; targets stay materialized. Requires
  // sourceGroupSize == nPtsPerObj and nSource == nObj*nPtsPerObj.
  void setStructuredSource(const vec3d *d_templatePts, int nPtsPerObj,
                           const double *d_R, const vec3d *d_centers, int nObj);
  void clearStructuredSource();
  bool hasStructuredSource() const { return hasStructuredSource_; }
  // Bytes the treecode holds for SOURCE geometry (for VRAM A/B): the two
  // materialized position copies (default), or the shared template + per-object
  // transforms (structured). Valid while the tree is alive (apply(reuse=true)).
  size_t sourceGeomBytes() const
  {
    if (hasStructuredSource_)
      return structTemplate64_.capacity() * sizeof(vec3d) +
             structR_.capacity() * sizeof(double) +
             structCenterInput_.capacity() * sizeof(vec3d) +
             structCenter64_.capacity() * sizeof(vec3d);
    return buckets_.points.capacity() * sizeof(vec3f) +
           buckets_.points64.capacity() * sizeof(vec3d);
  }

  size_t        numParticles() const { return nSource_; }
  size_t        numTargets()   const { return nTarget_; }
  const Stats  &stats()        const { return stats_; }
  // Device pointers into the source-bucket-contiguous set, for validation.
  const vec3f  *points()       const { return util::devicePtr(buckets_.points); }
  const vec3d  *points64()     const { return util::devicePtr(buckets_.points64); }
  const vec3d  *forces()       const { return util::devicePtr(force_); }
  const uint32_t *sourcePermutation() const
  { return util::devicePtr(buckets_.perm); }
  const uint32_t *targetPermutation() const
  { return util::devicePtr(targetPermVector()); }

private:
  enum class EvaluationMode {
    Split,
    DirectNear
  };
  enum class PairCacheKind {
    None,
    NearLeaf,
    NearLeafUnsorted,  // split-warpspec-atomic: unsorted (target, leaf) list;
                       // no pairBegin_/pairCnt_, range looked up inline in P2P.
    NearLeafGrouped,   // structured leaf-centric: near targets grouped by object
                       // into a CSR (leafGroupOffset_/leafGroupTargets_); the
                       // leaf-centric P2P reconstructs each object's sources once.
    SkelGroupCSR       // TC_SKEL_P2P=block: near field kept as the per-group
                       // rejected-leaf CSR (skelLeafBegin_/skelLeafList_); no
                       // expanded per-target pair list is ever materialized.
  };

  void configure(const Config &cfg);
  void syncNearCutoffSymbols() const;
  // Print the TC_PATH-selected path. Called by apply()/reapply() outside the
  // CUDA timing-event zones so it never perturbs the measured intervals.
  void logSelectedPath(const char *op) const;
  void freeBuild();
  void clearPairCache();
  void build(const vec3d *d_source, size_t nSource,
             const vec3d *d_target, size_t nTarget);
  // Node-range crossover: compute the DFS bucket ordering + per-node
  // (dfsFirst, bucketCount) after the BVH build. Geometry only.
  void buildDfsBucketOrder();
  void updateTraversalTreeMetrics();
  void upwardPass(const vec3d *d_force);
  void upwardPass(const vec3f *d_force);
  void evaluateCurrent(bool rebuildP2P, double *d_velOut);
  // Diagnostic (TC_PATH=traverse-count): time the traversal-only counting kernel
  // and print per-target M2P/P2P interaction totals; zeros d_velOut.
  void runTraverseCountOnly(double *d_velOut);
  void traverseEmitPairs(double *d_velBucket,
                         thrust::device_vector<int> &pairTarget,
                         thrust::device_vector<int> &pairLeaf,
                         int &nPairs);
  // Split-path P2P tiling: when a whole-range near-pair emit would exceed the
  // pair-buffer budget (pairCap_), process targets in contiguous tiles so the
  // pair list / partials stay bounded (~TC_PAIR_BUDGET_GB). Each tile re-emits
  // (M2P + near pairs) over a target sub-range and runs the cached P2P pipeline,
  // accumulating into d_velBucket. Used only for the oversize case; the common
  // single-tile path is unchanged.
  void evaluateSplitTiled(double *d_velBucket);
  void traverseM2POnly(double *d_velBucket);
  void traverseDirectNear(double *d_velBucket);
  void preparePairList(thrust::device_vector<int> &pairTarget,
                       thrust::device_vector<int> &pairLeaf,
                       int nPairs);
  // split-warpspec-atomic: cache the emitted (target, leaf) pairs UNSORTED (just
  // swap the buffers in; no radix sort, no pairCountsKernel). The bucket range is
  // looked up inline by p2pAtomicKernel and the per-target near field is formed
  // by atomicAdd, so no pairBegin_/pairCnt_ are built.
  void preparePairListUnsorted(thrust::device_vector<int> &pairTarget,
                               thrust::device_vector<int> &pairLeaf,
                               int nPairs);
  // double-traverse apply: pass 1 (Count-mode warpspec) writes the M2P far field
  // into d_velBucket and per-target rejected-leaf counts into perTargetCount.
  void traverseCountM2P(double *d_velBucket,
                        thrust::device_vector<int> &perTargetCount);
  // double-traverse apply: prefix-sum the counts, pass 2 writes the (already
  // target-sorted) near-pair list, then build the executable cache. Returns true
  // if the pair list was built (proceed to runP2P); false if the whole-range
  // count exceeded pairCap_ and evaluateSplitTiled already did the full P2P.
  bool buildPairListDouble(thrust::device_vector<int> &perTargetCount,
                           double *d_velBucket);
  // Map a (target-sorted) sortedLeaf_ list to executable bucket ranges
  // (pairBegin_/pairCnt_) and finalize the near-leaf cache. Shared by
  // preparePairList (split paths) and buildPairListDouble (double-traverse).
  void finalizeNearLeafCache(int nPairs);
  void runP2P(double *d_velBucket);
  // split-warpspec-atomic near field: one warp per cached unsorted (target, leaf)
  // pair, atomicAdd straight into d_velBucket (no partials / reduce / scatter).
  void runP2PAtomic(double *d_velBucket);
  // ---- TC_PATH=skel (target-skeleton lift) ----
  int skelNumGroups() const
  {
    return (cfg_.targetGroupSize > 0)
               ? (int)(nTarget_ / (size_t)cfg_.targetGroupSize) : 0;
  }
  // One-time (per build) host ID build: skeleton indices + shared lift matrix.
  // Built lazily per KernelKind (velocity and traction keep separate skeletons:
  // differentiation amplifies the ID truncation, so the velocity skeleton is
  // never reused for traction).
  void skelBuildLift(KernelKind kind);
  // Fail-loud support check for KernelKind::Traction (skel path + block P2P +
  // normals present); called at evaluate time.
  void skelRequireTractionSupport() const;
  // Gather the input-order target normals into bucket order (targetNormals64_).
  void gatherTargetNormals()
  {
    if (targetNormalsInput_.empty()) return;
    if (targetNormalsInput_.size() != nTarget_)
      throw std::runtime_error("setTargetNormals: size != nTarget");
    targetNormals64_.resize(nTarget_);
    const thrust::device_vector<uint32_t> &perm = targetPermVector();
    thrust::gather(perm.begin(), perm.end(), targetNormalsInput_.begin(),
                   targetNormals64_.begin());
  }
  // Apply-only group traversal (count + write): fills the group->node CSR and
  // the expanded per-target (target, leaf) near-pair list.
  void skelTraverse(thrust::device_vector<int> &pairTarget,
                    thrust::device_vector<int> &pairLeaf, int &nPairs);
  // Far field for apply AND reapply: skeleton M2P into u_skel + DGEMM lift
  // (overwrites d_velBucket; P2P atomic-adds on top afterward).
  void skelEvalFar(double *d_velBucket);
  // TC_SKEL_CHECK=1: full per-target far field on sampled groups vs the lift.
  void skelCheckLift(const double *d_velBucket);
  // TC_SKEL_P2P=block near field: block per (group, target-chunk), shared-mem
  // source tiles, register accumulation, plain (atomic-free) global add.
  void runSkelP2PBlock(double *d_velBucket);
  // Structured leaf-centric near field: group the emitted (target, leaf) pairs by
  // object into a CSR (leafGroupOffset_/leafGroupTargets_) via a counting sort;
  // cached and replayed on reapply. Records stats_.groupMs.
  void preparePairListGrouped(thrust::device_vector<int> &pairTarget,
                              thrust::device_vector<int> &pairLeaf,
                              int nPairs);
  // Structured leaf-centric P2P: one block per object reconstructs its sources
  // ONCE into shared (template + transform), then all its near targets stream
  // that shared cloud and atomicAdd into d_velBucket. Replaces runP2PAtomic when
  // structured. Requires PairCacheKind::NearLeafGrouped.
  void runP2PLeafGrouped(double *d_velBucket);
  void scatterToInputOrder(double *d_outInputOrder);
  const vec3f *targetPoints() const
  {
    return selfTargets_ ? util::devicePtr(buckets_.points)
                        : util::devicePtr(targetBuckets_.points);
  }
  // fp64 bucket-ordered targets (same order as targetPoints()), for the
  // fp64-geometry M2P path (SphericalStokes).
  const vec3d *targetPoints64() const
  {
    return selfTargets_ ? util::devicePtr(buckets_.points64)
                        : util::devicePtr(targetBuckets_.points64);
  }
  const thrust::device_vector<uint32_t> &targetPermVector() const
  {
    return selfTargets_ ? buckets_.perm : targetBuckets_.perm;
  }
  const int *sourceOwnerPtr() const
  {
    return sourceOwner_.empty() ? nullptr : util::devicePtr(sourceOwner_);
  }
  const int *targetOwnerPtr() const
  {
    return targetOwner_.empty() ? nullptr : util::devicePtr(targetOwner_);
  }
  const uint32_t *ownerMaskPtr() const { return d_ownerMask_; }
  int skipSameGroupFlag() const
  {
    return (cfg_.skipSameGroup && !sourceOwner_.empty() && !targetOwner_.empty())
               ? 1 : 0;
  }
  // Node-range crossover accessors: null / 0 unless the mode is on.
  int emitInternalNodesFlag() const { return emitInternalNodes_ ? 1 : 0; }
  const int *nodeDfsFirstPtr() const
  { return emitInternalNodes_ ? d_nodeDfsFirst_ : nullptr; }
  const int *nodeBucketCountPtr() const
  { return emitInternalNodes_ ? d_nodeBucketCount_ : nullptr; }
  const int *dfsBucketsPtr() const
  { return emitInternalNodes_ ? d_dfsBuckets_ : nullptr; }

  template<int ORDER>
  void launchTraverseEmit(int gridN, int block, double *d_velBucket,
                          int *d_pairTarget, int *d_pairLeaf,
                          unsigned long long *d_pairCount, long long pairCapacity
#if TREECODE_STATS
                          , uint32_t *d_visited, uint32_t *d_m2pAccepted
#endif
                          );
  template<int ORDER>
  void launchTraverseOnly(int gridN, int block, double *d_velBucket
#if TREECODE_STATS
                          , uint32_t *d_visited, uint32_t *d_m2pAccepted
#endif
                          );
  // double-traverse pass 1: warpspec M2P producer/consumer in Count mode
  // (EMIT_PAIRS=false, COUNT_ONLY=true) -> far field + per-target leaf counts.
  template<int ORDER>
  void launchTraverseCount(double *d_velBucket, int *d_perTargetCount);
  // Range-restricted split-warp emit (M2P + near pairs) for [targetStart,
  // targetStart+count); used by evaluateSplitTiled. Always the warp kernel,
  // independent of pathKind_ (correctness-identical to split-warp).
  template<int ORDER>
  void launchTraverseEmitWarpRange(int targetStart, int count,
                                   double *d_velBucket,
                                   int *d_pairTarget, int *d_pairLeaf,
                                   unsigned long long *d_pairCount,
                                   int pairCapacity);
  template<int ORDER>
  void launchTraverseDirect(int gridN, int block, double *d_velBucket,
                            unsigned long long *d_nearLeafCount,
                            unsigned long long *d_p2pCount
#if TREECODE_STATS
                            , uint32_t *d_visited, uint32_t *d_m2pAccepted
#endif
                            );

  util::GridBuckets            buckets_;
  util::GridBuckets            targetBuckets_;
  thrust::device_vector<vec3d> force_;
  // WIDEBVH_FP32_LEVEL >= 2 only (empty otherwise): fp32 copy of the gathered
  // forces for p2pAtomicKernel32, and its per-apply fp32 near-field scratch
  // (component-major [3*nTarget], zeroed in evaluateCurrent, folded into the
  // fp64 output by scatterToInputOrder).
  thrust::device_vector<vec3f> force32_;
  thrust::device_vector<float> velNear32_;
  thrust::device_vector<int>   sourceOwner_;
  thrust::device_vector<int>   targetOwner_;
  thrust::device_vector<double> velBucket_;
  bvh3f                        bvh_{};
  // Use regular cudaMalloc/cudaFree for cuBQL-owned BVH allocations. Nsight
  // Systems' CUDA memory tracker has been fragile around large cudaMallocAsync
  // pool teardown, while these BVH allocations are not a hot inner-loop cost.
  cuBQL::DeviceMemoryResource   bvhMem_;
  typename MP::NodeMAC        *d_mac_ = nullptr;
  typename MP::NodeM2P        *d_m2p_ = nullptr;
  uint32_t                    *d_ownerMask_ = nullptr;
  int                          ownerMaskWords_ = 0;
  // P2P/multipole crossover (TC_XOVER). d_nodeCount_ holds per-BVH-node total
  // source counts (filled by the Bary refit); xoverThresh_ > 0 routes accepted
  // nodes summarizing fewer than xoverThresh_ sources to exact P2P. Only
  // allocated/filled for the BaryStokes policy when the crossover is on.
  int                         *d_nodeCount_ = nullptr;
  int                          xoverThresh_ = 0;

  // Node-range crossover (TC_XOVER_NODERANGE, split-warpspec-atomic only). A small
  // accepted INTERNAL node emits ONE (target, node) near pair instead of descending
  // to K leaf pairs. Needs a subtree-contiguous bucket ordering (the SAH tree is
  // NOT contiguous), so after the build we compute our own DFS bucket permutation:
  //   d_dfsBuckets_[numBuckets]    - DFS-ordered permutation of bucket ids
  //   d_nodeDfsFirst_[numNodes]    - each node's start into d_dfsBuckets_
  //   d_nodeBucketCount_[numNodes] - # buckets (BVH leaves) under the node
  // All geometry-fixed (built once, reused across reapply); only when the mode is on.
  int                         *d_dfsBuckets_ = nullptr;
  int                         *d_nodeDfsFirst_ = nullptr;
  int                         *d_nodeBucketCount_ = nullptr;
  bool                         emitInternalNodes_ = false;

  // Cached near-field list. sortedTarget_/sortedLeaf_ are the requested
  // (target, leaf_id) list; pairBegin_/pairCnt_ are its executable ranges.
  // For PairCacheKind::NearLeaf they are target-sorted and pairBegin_/pairCnt_
  // hold the per-pair bucket ranges. For PairCacheKind::NearLeafUnsorted
  // (split-warpspec-atomic) sortedTarget_/sortedLeaf_ are UNSORTED, sortedLeaf_
  // is retained, and pairBegin_/pairCnt_ are empty (the range is looked up
  // inline in p2pAtomicKernel).
  thrust::device_vector<int> sortedTarget_;
  thrust::device_vector<int> sortedLeaf_;
  thrust::device_vector<int> pairBegin_;
  thrust::device_vector<int> pairCnt_;

  // ---- structured (monodisperse) source, opt-in via setStructuredSource() ----
  // INPUT buffers survive freeBuild() (set once before apply, like cfg_); only the
  // DERIVED buffers below are rebuilt per build/pair-cache.
  bool                          hasStructuredSource_ = false;
  int                           structNPtsPerObj_ = 0;   // e.g. 656
  int                           structNObj_       = 0;   // e.g. 10000
  thrust::device_vector<vec3d>  structTemplate64_;       // template, reference frame (nPts)
  thrust::device_vector<double> structR_;                // nObj*9 row-major fp64
  thrust::device_vector<vec3d>  structCenterInput_;      // nObj input-frame centers
  thrust::device_vector<vec3d>  structCenter64_;         // nObj SHIFTED (c - outputShift)
  cuBQL::vec3d                  structShift_{0.0, 0.0, 0.0};
  // Grouped near-field cache (CSR by object) for the leaf-centric structured P2P.
  thrust::device_vector<int>    leafGroupOffset_;        // nObj+1 CSR row offsets
  thrust::device_vector<int>    leafGroupTargets_;       // nPairs targets grouped by object

  // ---- TC_PATH=skel (target-skeleton lift) state ----
  // skelIdx_/skelT_ are the one-time ID build (template indices + B x nSkel
  // col-major fp64 lift matrix, shared by every particle); skelNodeBegin_/
  // skelNodeList_ are the cached per-group M2P CSR (rebuilt each apply, reused
  // by reapply); skelU_ is the compact skeleton far-field buffer.
  int  skelN_ = 128;              // requested nSkel (TC_TGT_SKEL)
  int  skelNTr_ = 128;            // traction nSkel (TC_TGT_SKEL_TRACTION)
  int  skelProxyPerShell_ = 250;  // proxy sphere sampling (TC_SKEL_PROXY_M)
  bool skelCheck_ = false;        // TC_SKEL_CHECK
  bool skelTiming_ = false;       // TC_SKEL_TIMING
  bool skelP2PBlock_ = false;     // TC_SKEL_P2P=block (default atomic)
  bool skelBuilt_ = false;
  bool skelBuiltTr_ = false;      // traction (skeleton, lift) built
  thrust::device_vector<int>    skelIdx_;
  thrust::device_vector<double> skelT_;
  thrust::device_vector<int>    skelIdxT_;       // traction skeleton indices
  thrust::device_vector<double> skelTT_;         // traction lift (B x nSkelT)
  thrust::device_vector<int>    skelNodeBegin_;
  thrust::device_vector<int>    skelNodeList_;
  thrust::device_vector<int>    skelLeafBegin_;  // block P2P: per-group leaf CSR
  thrust::device_vector<int>    skelLeafList_;
  thrust::device_vector<double> skelU_;
  cublasHandle_t                cublas_ = nullptr;

  // ---- KernelKind::Traction state ----
  // kernel_ and targetNormalsInput_ are INPUT state (survive freeBuild, like
  // the structured-source buffers); targetNormals64_ is derived per build.
  KernelKind                    kernel_ = KernelKind::Stokeslet;
  thrust::device_vector<vec3d>  targetNormalsInput_;  // target input order
  thrust::device_vector<vec3d>  targetNormals64_;     // target-bucket order

  Config                       cfg_{};
  PathKind                     pathKind_ = PathKind::SplitWarpSpec;
  // TC_OCT_SFC_BOX=cube|domain (default cube): SFC box for the global Hilbert
  // bucketizer's cornerstone keys. Cornerstone normalizes each axis
  // independently, so with the raw (domain) box every Hilbert cell inherits
  // the domain's aspect ratio (4:1 bricks on two_ball_t100's 460x460x1863
  // domain); cubizing keeps cells isotropic and is a no-op on near-cubic
  // domains.
  bool                         octSfcCube_ = true;
  // Bucketizer (TC_BUCKETIZER, default grid-hilbert).
  Bucketizer                   bucketizer_ = Bucketizer::GridHilbert;
  // Target-side bucketizer (TC_TGT_BUCKETIZER). Defaults to bucketizer_, so the
  // source and target sides match unless explicitly decoupled. Lets sources be
  // object-grouped (tight one-object leaves) while targets keep a spatial
  // ordering ("fully independent" targets), e.g. TC_BUCKETIZER=object
  // TC_TGT_BUCKETIZER=grid-hilbert in the MFS mobility solve.
  Bucketizer                   targetBucketizer_ = Bucketizer::GridHilbert;
  // grid-hilbert auto-cell-edge q override (TC_HILBERT_Q): <= 0 = auto
  // (cbrt(B_ref)), > 0 = explicit q. Only read in GridHilbert mode.
  double                       gridHilbertQ_ = -1.0;
  // cuBQL source-BVH builder algorithm, selected by TC_BVH_BUILDER (default =
  // SAH). Affects tree quality, so build time AND traversal.
  BvhBuilder bvhBuildMethod_ = BvhBuilder::Sah;
  EvaluationMode               evalMode_ = EvaluationMode::Split;
  PairCacheKind                pairCacheKind_ = PairCacheKind::None;
  Stats                        stats_{};
  size_t                       nSource_ = 0;
  size_t                       nTarget_ = 0;
  int                          nPairs_ = 0;
  // Split-path P2P near-pair budget. Per-tile near-pair count is capped at
  // pairCap_ so the pair list + fp64 partials fit ~TC_PAIR_BUDGET_GB (default
  // 40 GB) of VRAM; clamped < INT_MAX so the per-tile CUB sort / int kernels
  // stay valid. emitOverflowed_ is set by the whole-range probe when the count
  // exceeds pairCap_; oversizeTiled_ remembers it so reapply also tiles.
  size_t                       pairCap_ = 0;
  bool                         emitOverflowed_ = false;
  bool                         oversizeTiled_ = false;
  bool                         built_ = false;
  bool                         hasBvh_ = false;
  bool                         reusable_ = false;
  bool                         selfTargets_ = true;
  bool                         pairsBuilt_ = false;
};


// Push the configured near-field cutoff into the device constant symbols that
// every traversal / P2P kernel reads. Called from each public entry point rather
// than once at configure() time so that several Treecode instances with
// different cutoffs can coexist in one process: kernels of a given apply() are
// all issued on the same stream after this copy, so the value they see is always
// the one belonging to the instance being applied.
template<class MP>
void Treecode<MP>::syncNearCutoffSymbols() const
{
  const double rc2d = cfg_.nearCutoff * cfg_.nearCutoff;
  const float  rc2f = (float)rc2d;
  CUDA_CHECK(cudaMemcpyToSymbol(stokes::c_nearCut2d, &rc2d, sizeof(double)));
  CUDA_CHECK(cudaMemcpyToSymbol(stokes::c_nearCut2f, &rc2f, sizeof(float)));
}


template<class MP>
void Treecode<MP>::configure(const Config &cfg)
{
  cfg_ = cfg;

  if (cfg_.nearCutoff < 0.0)
    throw std::runtime_error("Treecode: nearCutoff must be >= 0");

  // The execution path is chosen solely by TC_PATH (default: split-warpspec).
  // The split-warpspec block size is the compile-time SPLIT_WARPSPEC_BLOCK.
  pathKind_ = parsePathKind(std::getenv("TC_PATH"), PathKind::SplitWarpSpec);

  // P2P/multipole crossover threshold (TC_XOVER). 0/unset disables it and the
  // traversal is bit-identical to the pre-crossover path. When enabled (> 0), an
  // accepted node summarizing fewer than TC_XOVER sources is evaluated by exact
  // P2P instead of the fixed (PDEG+1)^3-point M2P (small leaves emit a near pair,
  // small internal nodes descend). Only the BaryStokes policy fills a per-node
  // source count, so the crossover is a no-op for other policies.
  xoverThresh_ = 0;
  if constexpr (std::is_same_v<MP, mp::BaryStokes>) {
    if (const char *e = std::getenv("TC_XOVER")) {
      const int v = std::atoi(e);
      if (v > 0) xoverThresh_ = v;
    }
  }

  // Node-range crossover (TC_XOVER_NODERANGE=1): a small accepted INTERNAL node
  // emits ONE (target, node) near pair instead of descending to its leaves. Needs
  // a subtree-contiguous bucket ordering we build ourselves (see build()), and the
  // node-range atomic P2P kernel to consume it -- so it is honored ONLY on the
  // split-warpspec-atomic path with the crossover on. Ignored otherwise.
  emitInternalNodes_ = false;
  if (xoverThresh_ > 0 && pathKind_ == PathKind::SplitWarpSpecAtomic &&
      treecodeEnvFlag("TC_XOVER_NODERANGE", false))
    emitInternalNodes_ = true;

  // Target-skeleton (TC_PATH=skel) knobs. TC_TGT_SKEL = number of skeleton
  // targets kept per particle (the ID rank); TC_SKEL_PROXY_M = proxy points per
  // sampling shell for the one-time ID build; TC_SKEL_CHECK compares the lifted
  // far field against a full per-target evaluation on sampled groups;
  // TC_SKEL_TIMING prints per-phase (traverse/eval/lift) timings.
  skelN_ = 128;
  if (const char *e = std::getenv("TC_TGT_SKEL")) {
    const int v = std::atoi(e);
    if (v < 1)
      throw std::runtime_error(std::string("bad TC_TGT_SKEL=") + e +
                               "; expected an integer >= 1");
    skelN_ = v;
  }
  // Traction skeleton rank (TC_TGT_SKEL_TRACTION), default = TC_TGT_SKEL.
  // Kept separate because the traction fields are derivative-rougher than the
  // velocity fields, so the traction ID usually needs a higher rank;
  // TC_TGT_SKEL_TRACTION=<groupSize> degenerates to an exact identity lift.
  skelNTr_ = skelN_;
  if (const char *e = std::getenv("TC_TGT_SKEL_TRACTION")) {
    const int v = std::atoi(e);
    if (v < 1)
      throw std::runtime_error(std::string("bad TC_TGT_SKEL_TRACTION=") + e +
                               "; expected an integer >= 1");
    skelNTr_ = v;
  }
  skelProxyPerShell_ = 250;
  if (const char *e = std::getenv("TC_SKEL_PROXY_M")) {
    const int v = std::atoi(e);
    if (v < 8)
      throw std::runtime_error(std::string("bad TC_SKEL_PROXY_M=") + e +
                               "; expected an integer >= 8");
    skelProxyPerShell_ = v;
  }
  skelCheck_  = treecodeEnvFlag("TC_SKEL_CHECK", false);
  skelTiming_ = treecodeEnvFlag("TC_SKEL_TIMING", false);
  // Near-field consumer for the skel path. `atomic` (default) expands the
  // rejected leaves to per-target pairs for p2pAtomicKernel; `block` keeps the
  // per-group leaf CSR and runs the group-blocked shared-memory tile kernel
  // (no expanded list, no atomics, no pair budget).
  skelP2PBlock_ = false;
  if (const char *e = std::getenv("TC_SKEL_P2P")) {
    if (std::strcmp(e, "atomic") == 0)      skelP2PBlock_ = false;
    else if (std::strcmp(e, "block") == 0)  skelP2PBlock_ = true;
    else
      throw std::runtime_error(std::string("bad TC_SKEL_P2P=") + e +
                               "; expected one of: atomic, block");
  }

  // Source-BVH builder algorithm, selected by TC_BVH_BUILDER (default = SAH,
  // surface-area heuristic). Applied at the gpuBuilder call in buildBvh().
  bvhBuildMethod_ =
      parseBvhBuilder(std::getenv("TC_BVH_BUILDER"), BvhBuilder::Sah);

  bucketizer_ =
      parseBucketizer(std::getenv("TC_BUCKETIZER"), Bucketizer::GridHilbert);
  // Target-side bucketizer defaults to the source bucketizer (so a bare
  // TC_BUCKETIZER=object keeps the old both-sides behavior); TC_TGT_BUCKETIZER
  // decouples it to leave targets on a spatial ordering while sources group by
  // object.
  targetBucketizer_ =
      parseBucketizer(std::getenv("TC_TGT_BUCKETIZER"), bucketizer_);
  // grid-hilbert auto-cell-edge q override (see gridHilbertCellEdge).
  gridHilbertQ_ = -1.0;
  if (const char *e = std::getenv("TC_HILBERT_Q")) {
    gridHilbertQ_ = std::atof(e);
    if (!(gridHilbertQ_ > 0.0))
      throw std::runtime_error(std::string("bad TC_HILBERT_Q=") + e +
                               "; expected a value > 0");
  }
  if (const char *e = std::getenv("TC_OCT_SFC_BOX")) {
    if (std::strcmp(e, "cube") == 0)        octSfcCube_ = true;
    else if (std::strcmp(e, "domain") == 0) octSfcCube_ = false;
    else
      throw std::runtime_error(std::string("unknown TC_OCT_SFC_BOX token '") +
                               e + "'; expected one of: cube, domain");
  }

  // Split-path near-pair budget. ~kBytesPerPair (sorted target + pair ranges +
  // 3 fp64 partials, minus the freed emit/sortedLeaf buffers) of VRAM per pair;
  // TC_PAIR_BUDGET_GB overrides the default 40 GB. Clamp the resulting cap below
  // INT_MAX so each tile's CUB sort / p2pKernel (int counts) stay valid.
  {
    constexpr double kBytesPerPair = 40.0;
    double budgetGB = 40.0;
    if (const char *e = std::getenv("TC_PAIR_BUDGET_GB")) {
      const double v = std::atof(e);
      if (v > 0.0) budgetGB = v;
    }
    const double capPairs = budgetGB * (double)(1ull << 30) / kBytesPerPair;
    const size_t kMaxTilePairs = 2000000000ull;  // < INT_MAX, headroom for CUB
    pairCap_ = (size_t)std::min(capPairs, (double)kMaxTilePairs);
    if (pairCap_ < (1u << 20)) pairCap_ = (1u << 20);  // sane floor (1M)
  }

  // Derive the internal eval mode from pathKind_.
  switch (pathKind_) {
    case PathKind::SplitWarp:
    case PathKind::SplitWarpSpec:
    case PathKind::SplitWarpSpecAtomic:
    case PathKind::DoubleTraverse:
    case PathKind::Skel:
      evalMode_ = EvaluationMode::Split;
      break;
    case PathKind::DirectWarpSpec:
      evalMode_ = EvaluationMode::DirectNear;
      break;
    case PathKind::TraverseCount:
      // Diagnostic path: short-circuited in evaluateCurrent; mode unused.
      evalMode_ = EvaluationMode::Split;
      break;
  }
}


template<class MP>
void Treecode<MP>::logSelectedPath(const char *op) const
{
  if (treecodeQuiet()) return;
  std::printf("[treecode] %s path=%s (%s)\n", op, pathToken(pathKind_),
              pathDescription(pathKind_));
  if (pathKind_ == PathKind::Skel)
    std::printf("[treecode] %s skel kernel=%s nSkel=%d%s proxies=%d/shell "
                "p2p=%s (TC_TGT_SKEL / TC_TGT_SKEL_TRACTION / "
                "TC_SKEL_PROXY_M / TC_SKEL_P2P)\n", op,
                kernel_ == KernelKind::Traction ? "traction" : "stokeslet",
                kernel_ == KernelKind::Traction ? skelNTr_ : skelN_,
                kernel_ == KernelKind::Traction ? " (traction)" : "",
                skelProxyPerShell_, skelP2PBlock_ ? "block" : "atomic");
  if (xoverThresh_ > 0)
    std::printf("[treecode] %s xover=%d (accepted nodes with <%d sources -> "
                "exact P2P)%s\n", op, xoverThresh_, xoverThresh_,
                emitInternalNodes_
                    ? " [node-range: small internal nodes emit ONE pair]"
                    : "");
  std::fflush(stdout);
}


template<class MP>
void Treecode<MP>::freeBuild()
{
  if (hasBvh_) {
    cuBQL::cuda::free(bvh_, 0, bvhMem_);
    bvh_ = bvh3f{};
  }
  if (d_mac_) {
    cudaFree(d_mac_);
    d_mac_ = nullptr;
  }
  if (d_m2p_) {
    cudaFree(d_m2p_);
    d_m2p_ = nullptr;
  }
  if (d_ownerMask_) {
    cudaFree(d_ownerMask_);
    d_ownerMask_ = nullptr;
  }
  if (d_nodeCount_) {
    cudaFree(d_nodeCount_);
    d_nodeCount_ = nullptr;
  }
  if (d_dfsBuckets_) {
    cudaFree(d_dfsBuckets_);
    d_dfsBuckets_ = nullptr;
  }
  if (d_nodeDfsFirst_) {
    cudaFree(d_nodeDfsFirst_);
    d_nodeDfsFirst_ = nullptr;
  }
  if (d_nodeBucketCount_) {
    cudaFree(d_nodeBucketCount_);
    d_nodeBucketCount_ = nullptr;
  }
  buckets_ = util::GridBuckets{};
  targetBuckets_ = util::GridBuckets{};
  releaseDeviceVector(force_);
  releaseDeviceVector(force32_);
  releaseDeviceVector(velNear32_);
  releaseDeviceVector(sourceOwner_);
  releaseDeviceVector(targetOwner_);
  releaseDeviceVector(velBucket_);
  releaseDeviceVector(skelIdx_);
  releaseDeviceVector(skelT_);
  releaseDeviceVector(skelIdxT_);
  releaseDeviceVector(skelTT_);
  releaseDeviceVector(skelNodeBegin_);
  releaseDeviceVector(skelNodeList_);
  releaseDeviceVector(skelLeafBegin_);
  releaseDeviceVector(skelLeafList_);
  releaseDeviceVector(skelU_);
  releaseDeviceVector(targetNormals64_);  // derived; the INPUT normals persist
  skelBuilt_ = false;
  skelBuiltTr_ = false;
  clearPairCache();
  nSource_ = 0;
  nTarget_ = 0;
  nPairs_ = 0;
  ownerMaskWords_ = 0;
  built_ = false;
  hasBvh_ = false;
  reusable_ = false;
  selfTargets_ = true;
}


template<class MP>
void Treecode<MP>::clearPairCache()
{
  releaseDeviceVector(sortedTarget_);
  releaseDeviceVector(sortedLeaf_);
  releaseDeviceVector(pairBegin_);
  releaseDeviceVector(pairCnt_);
  releaseDeviceVector(leafGroupOffset_);
  releaseDeviceVector(leafGroupTargets_);
  nPairs_ = 0;
  pairsBuilt_ = false;
  pairCacheKind_ = PairCacheKind::None;
}


template<class MP>
void Treecode<MP>::setStructuredSource(const vec3d *d_templatePts, int nPtsPerObj,
                                       const double *d_R, const vec3d *d_centers,
                                       int nObj)
{
  assert(!built_);
  if (nPtsPerObj < 1 || nObj < 1)
    throw std::runtime_error(
        "setStructuredSource: nPtsPerObj and nObj must be >= 1");
  structNPtsPerObj_ = nPtsPerObj;
  structNObj_ = nObj;
  // Copy the (tiny ~1 MB) template + per-object transforms into device members so
  // the treecode owns them and they survive across reapply.
  structTemplate64_.resize(nPtsPerObj);
  thrust::copy(thrust::device_pointer_cast(d_templatePts),
               thrust::device_pointer_cast(d_templatePts + nPtsPerObj),
               structTemplate64_.begin());
  structR_.resize((size_t)nObj * 9);
  thrust::copy(thrust::device_pointer_cast(d_R),
               thrust::device_pointer_cast(d_R + (size_t)nObj * 9),
               structR_.begin());
  structCenterInput_.resize(nObj);
  thrust::copy(thrust::device_pointer_cast(d_centers),
               thrust::device_pointer_cast(d_centers + nObj),
               structCenterInput_.begin());
  hasStructuredSource_ = true;
}


template<class MP>
void Treecode<MP>::clearStructuredSource()
{
  hasStructuredSource_ = false;
  structNPtsPerObj_ = 0;
  structNObj_ = 0;
  releaseDeviceVector(structTemplate64_);
  releaseDeviceVector(structR_);
  releaseDeviceVector(structCenterInput_);
  releaseDeviceVector(structCenter64_);
}


template<class MP>
void Treecode<MP>::updateTraversalTreeMetrics()
{
#if TREECODE_STATS
  const util::TreeMetrics m = util::computeTreeMetrics(bvh_);
  stats_.traversalNodes = m.numNodes;
  stats_.traversalInner = m.numInner;
  stats_.traversalLeaves = m.numLeaves;
  stats_.minTraversalLeafDepth = m.minLeafDepth;
  stats_.maxTraversalLeafDepth = m.maxLeafDepth;
  stats_.meanTraversalLeafDepth = m.meanLeafDepth;
#else
  stats_.traversalNodes = bvh_.numNodes;
#endif
}


template<class MP>
void Treecode<MP>::build(const vec3d *d_source, size_t nSource,
                         const vec3d *d_target, size_t nTarget)
{
  freeBuild();
  assert(d_source != nullptr);
  assert(nSource > 0);
  assert(cfg_.order > 0 && cfg_.order <= MAX_ORDER);
  assert(cfg_.mac > 0.f);
  assert(cfg_.cellEdge > 0.0);
  assert(cfg_.maxLeaf > 0);
  assert(nSource <= (size_t)std::numeric_limits<int>::max());
  if (d_target == nullptr) {
    nTarget = nSource;
    selfTargets_ = true;
  } else {
    assert(nTarget > 0);
    assert(nTarget <= (size_t)std::numeric_limits<int>::max());
    selfTargets_ = false;
  }

  if (hasStructuredSource_) {
    // The structured (template-reconstruction) path is only wired for the
    // split-warpspec-atomic near field + object bucketizer + BaryStokes policy,
    // distinct targets, and no crossover. Fail loud rather than silently produce
    // wrong results; every other path stays byte-identical (it never sees this).
    if (pathKind_ != PathKind::SplitWarpSpecAtomic)
      throw std::runtime_error(
          "structured source (TC_SRC_TEMPLATE) requires TC_PATH=split-warpspec-atomic");
    if (bucketizer_ != Bucketizer::Object)
      throw std::runtime_error(
          "structured source requires TC_BUCKETIZER=object");
    if (!(std::is_same_v<MP, mp::BaryStokes>))
      throw std::runtime_error(
          "structured source is only supported for the BaryStokes policy");
    if (selfTargets_)
      throw std::runtime_error(
          "structured source requires distinct targets (self-target unsupported)");
    if (emitInternalNodes_)
      throw std::runtime_error(
          "structured source is incompatible with node-range crossover "
          "(TC_XOVER_NODERANGE)");
    if (cfg_.sourceGroupSize != structNPtsPerObj_)
      throw std::runtime_error(
          "structured source: Config::sourceGroupSize must equal nPtsPerObj");
    if (nSource != (size_t)structNObj_ * (size_t)structNPtsPerObj_)
      throw std::runtime_error(
          "structured source: nSource must equal nObj * nPtsPerObj");
  }

  if (pathKind_ == PathKind::Skel) {
    // The lift is defined per target GROUP (= particle): it needs object target
    // buckets (contiguous per-particle runs, identity perm) and a well-defined
    // group MAC. Fail loud on unsupported combinations.
    if (selfTargets_)
      throw std::runtime_error("TC_PATH=skel requires distinct targets");
    if (cfg_.targetGroupSize < 1)
      throw std::runtime_error(
          "TC_PATH=skel requires Config::targetGroupSize > 0");
    if (targetBucketizer_ != Bucketizer::Object)
      throw std::runtime_error(
          "TC_PATH=skel requires object target buckets "
          "(TC_TGT_BUCKETIZER=object or TC_BUCKETIZER=object)");
    if (skelN_ > cfg_.targetGroupSize)
      throw std::runtime_error(
          "TC_TGT_SKEL must be <= Config::targetGroupSize");
    if constexpr (MP::WANTS_FP64_TARGET)
      throw std::runtime_error(
          "TC_PATH=skel does not support fp64-target policies");
    if (emitInternalNodes_)
      throw std::runtime_error(
          "TC_PATH=skel is incompatible with node-range crossover");
  }

  if (cfg_.nearCutoff > 0.0) {
    // The skel path replaces the per-target MAC with a group MAC over a target
    // skeleton, and evaluates its near field through a separate blocked kernel
    // and an M2I lift -- neither of which consults nodeOutsideNearRadius or
    // stokes::p2p. It would silently keep the near pairs.
    if (pathKind_ == PathKind::Skel)
      throw std::runtime_error(
          "Config::nearCutoff is not supported on TC_PATH=skel");
    // The traction kernel has its own accumulator (traction_p2p) with no cutoff.
    if (kernel_ != KernelKind::Stokeslet)
      throw std::runtime_error(
          "Config::nearCutoff is only supported for the Stokeslet kernel");
  }

  stats_ = Stats{};
  nSource_ = nSource;
  nTarget_ = nTarget;
  MP::setup(cfg_.order);

  auto t0 = HostClock::now();
  const cuBQL::box3d sourceBounds = util::computeBounds(d_source, nSource);
  cuBQL::box3d bounds = sourceBounds;
  if (!selfTargets_) {
    const cuBQL::box3d targetBounds = util::computeBounds(d_target, nTarget);
    bounds = bounds.including(targetBounds);
  }
  if (treecodeQuiet())
    { /* per-build banners suppressed; see treecodeQuiet() */ }
  else if (bucketizer_ == Bucketizer::Hilbert)
    std::printf("[treecode] bucketizer=hilbert (global Hilbert sort, "
                "chunk=%d, sfcbox=%s; cellEdge ignored)\n",
                cfg_.maxLeaf, octSfcCube_ ? "cube" : "domain");
  else if (bucketizer_ == Bucketizer::GridHilbert)
    std::printf("[treecode] bucketizer=grid-hilbert (auto cellEdge=%g "
                "(q=%g%s), within-cell Hilbert, maxLeaf=%d; cellEdge cfg "
                "ignored)\n",
                gridHilbertCellEdge(nSource, bounds),
                gridHilbertQ_ > 0.0
                  ? gridHilbertQ_
                  : std::cbrt(std::max(1024.0,
                                       (double)nSource /
                                         (double)std::max(1, cfg_.maxLeaf))),
                gridHilbertQ_ > 0.0 ? "" : " auto", cfg_.maxLeaf);
  else if (bucketizer_ == Bucketizer::Object)
    std::printf("[treecode] bucketizer=object (one bucket per object: "
                "sourceGroupSize=%d, targetGroupSize=%d; cellEdge/maxLeaf "
                "ignored)\n",
                cfg_.sourceGroupSize, cfg_.targetGroupSize);
  else
    std::printf("[treecode] bucketizer=grid (cellEdge=%g, maxLeaf=%d)\n",
                cfg_.cellEdge, cfg_.maxLeaf);
  std::fflush(stdout);

  if (hasStructuredSource_) {
    // Pre-shift the per-object centers into the centered frame once the shift is
    // known, so reconstruction (R*template + shiftedCenter) lands where points64
    // would (input - shift). Fixed geometry: recomputed each build, reused across
    // reapply.
    structShift_ = bounds.center();
    structCenter64_.resize(structNObj_);
    thrust::transform(structCenterInput_.begin(), structCenterInput_.end(),
                      structCenter64_.begin(), SubShiftD{structShift_});
  }

  buckets_ = runBucketizer(d_source, nSource, bounds, cfg_.sourceGroupSize,
                           bucketizer_, hasStructuredSource_);
  CUDA_CHECK(cudaDeviceSynchronize());
  stats_.bucketMs = elapsed_ms(t0, HostClock::now());

  if (!selfTargets_) {
    if (targetBucketizer_ != bucketizer_ && !treecodeQuiet())
      std::printf("[treecode] target bucketizer=%s (decoupled from source)\n",
                  bucketizerToken(targetBucketizer_));
    t0 = HostClock::now();
    targetBuckets_ = runBucketizer(d_target, nTarget, bounds,
                                   cfg_.targetGroupSize, targetBucketizer_);
    CUDA_CHECK(cudaDeviceSynchronize());
    stats_.targetBucketMs = elapsed_ms(t0, HostClock::now());
  }

  t0 = HostClock::now();
  // BuildConfig(1): makeLeafThreshold=1 => one bucket per BVH leaf (the whole
  // pipeline relies on this). The builder is the TC_BVH_BUILDER selection.
  cuBQL::BuildConfig buildCfg(1);
  if (!treecodeQuiet()) {
    std::printf("[treecode] bvh_builder=%s\n", bvhBuilderToken(bvhBuildMethod_));
    std::fflush(stdout);
  }
  if (bvhBuildMethod_ == BvhBuilder::Radix) {
    // cuBQL's standalone fast Morton/LBVH radix builder. It is not part of the
    // BuildMethod enum that gpuBuilder() dispatches on, so call it directly; it
    // still honors makeLeafThreshold=1 (one bucket per leaf) and refits the node
    // boxes internally, matching gpuBuilder()'s output contract.
    cuBQL::cuda::radixBuilder(bvh_, util::devicePtr(buckets_.boxes),
                              buckets_.numBuckets(), buildCfg, 0,
                              bvhMem_);
  } else {
    buildCfg.buildMethod =
        (bvhBuildMethod_ == BvhBuilder::Sah) ? cuBQL::BuildConfig::SAH :
        (bvhBuildMethod_ == BvhBuilder::Elh) ? cuBQL::BuildConfig::ELH :
                                               cuBQL::BuildConfig::SPATIAL_MEDIAN;
    cuBQL::gpuBuilder(bvh_, util::devicePtr(buckets_.boxes),
                      buckets_.numBuckets(), buildCfg, 0,
                      bvhMem_);
  }
  hasBvh_ = true;
  if (bvh_.numNodes > (uint32_t)(std::numeric_limits<int>::max() >> 1))
    throw std::runtime_error("treecode BVH has too many nodes for tagged "
                             "interaction-list encoding");
  CUDA_CHECK(cudaDeviceSynchronize());
  stats_.buildBvhMs = elapsed_ms(t0, HostClock::now());

  CUDA_CHECK(cudaMalloc(&d_mac_,
                        bvh_.numNodes * sizeof(typename MP::NodeMAC)));
  CUDA_CHECK(cudaMalloc(&d_m2p_,
                        bvh_.numNodes * sizeof(typename MP::NodeM2P)));
  // Per-node source counts for the P2P/multipole crossover (TC_XOVER); one int
  // per BVH node, filled by the refit. Only when the crossover is enabled.
  if (xoverThresh_ > 0)
    CUDA_CHECK(cudaMalloc(&d_nodeCount_, bvh_.numNodes * sizeof(int)));
  // Node-range crossover: DFS bucket ordering + per-node (dfsFirst, bucketCount).
  // Geometry only, built once here and reused across reapply.
  if (emitInternalNodes_) {
    CUDA_CHECK(cudaMalloc(&d_nodeBucketCount_, bvh_.numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodeDfsFirst_, bvh_.numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dfsBuckets_,
                          (size_t)buckets_.numBuckets() * sizeof(int)));
    buildDfsBucketOrder();
  }
  force_.resize(nSource_);
  if (cfg_.sourceGroupSize > 0) {
    sourceOwner_.resize(nSource_);
    thrust::transform(buckets_.perm.begin(), buckets_.perm.end(),
                      sourceOwner_.begin(),
                      OwnerFromOriginalIndex{cfg_.sourceGroupSize});
    const size_t nGroups =
        (nSource_ + (size_t)cfg_.sourceGroupSize - 1) /
        (size_t)cfg_.sourceGroupSize;
    ownerMaskWords_ = (int)((nGroups + 31) / 32);
    CUDA_CHECK(cudaMalloc(&d_ownerMask_,
                          (size_t)bvh_.numNodes * (size_t)ownerMaskWords_
                              * sizeof(uint32_t)));
  }
  if (cfg_.targetGroupSize > 0) {
    targetOwner_.resize(nTarget_);
    const thrust::device_vector<uint32_t> &targetPerm = targetPermVector();
    thrust::transform(targetPerm.begin(), targetPerm.end(),
                      targetOwner_.begin(),
                      OwnerFromOriginalIndex{cfg_.targetGroupSize});
  }
  gatherTargetNormals();   // no-op unless setTargetNormals() was called
  velBucket_.resize((size_t)3 * nTarget_);

  stats_.numBuckets       = buckets_.numBuckets();
  stats_.numSourceBuckets = buckets_.numBuckets();
  stats_.numTargetBuckets = selfTargets_ ? buckets_.numBuckets()
                                         : targetBuckets_.numBuckets();
  stats_.numNodes      = bvh_.numNodes;
  stats_.traversalNodes = bvh_.numNodes;
  stats_.nx            = buckets_.nx;
  stats_.ny            = buckets_.ny;
  stats_.nz            = buckets_.nz;
  stats_.totalCells    = buckets_.totalCells;
  stats_.occupiedCells = buckets_.occupiedCells;
  stats_.coarseBits    = buckets_.coarseBits;
  stats_.fineBits      = buckets_.fineBits;
  updateTraversalTreeMetrics();
  built_ = true;
}


// Node-range crossover: build the DFS bucket ordering + per-node (dfsFirst,
// bucketCount). Three one-time geometry passes (see the kernels above). Assumes
// d_nodeBucketCount_/d_nodeDfsFirst_/d_dfsBuckets_ are already allocated.
template<class MP>
void Treecode<MP>::buildDfsBucketOrder()
{
  const int numNodes = (int)bvh_.numNodes;
  const int block = 256;
  const int gridNodes = (numNodes + block - 1) / block;

  // Pass 1: bottom-up subtree bucket count.
  void (*hostFp)(bvh3f, int[], int) = nullptr;
  CUDA_CHECK(cudaMemcpyFromSymbol(&hostFp, nodeBucketCountCombine_fp,
                                  sizeof(hostFp)));
  cuBQL::cuda::refit_aggregate(bvh_, d_nodeBucketCount_, hostFp, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Pass 2: top-down DFS offsets, one launch per tree level until it stops moving.
  dfsInitKernel<<<gridNodes, block>>>(d_nodeDfsFirst_, numNodes);
  int *d_changed = nullptr;
  CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
  for (int round = 0; round < numNodes; ++round) {   // numNodes is a safe cap
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
    dfsOffsetRoundKernel<<<gridNodes, block>>>(
        bvh_, d_nodeBucketCount_, d_nodeDfsFirst_, d_changed);
    int changed = 0;
    CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (!changed) break;
  }
  cudaFree(d_changed);

  // Pass 3: scatter buckets into DFS order.
  dfsScatterKernel<<<gridNodes, block>>>(bvh_, d_nodeDfsFirst_, d_dfsBuckets_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}


template<class MP>
void Treecode<MP>::upwardPass(const vec3f *d_force)
{
  assert(built_);
  assert(d_force != nullptr);

  auto t0 = HostClock::now();
  thrust::transform(buckets_.perm.begin(), buckets_.perm.end(),
                    force_.begin(), SourceForceGatherF{d_force});
#if WIDEBVH_FP32_LEVEL >= 2
  force32_.resize(nSource_);
  thrust::gather(buckets_.perm.begin(), buckets_.perm.end(),
                 thrust::device_pointer_cast(d_force), force32_.begin());
#endif
  // Stream-scoped sync (NOT cudaDeviceSynchronize): this wait exists only to
  // host-clock the phase, and all engine work runs on the legacy default
  // stream, so syncing stream 0 times exactly the same work. A device-wide
  // sync here would also drag in work a CALLER may have in flight on a side
  // stream (e.g. the sphere-BIM VSH near correction overlapping reapply).
  CUDA_CHECK(cudaStreamSynchronize(cudaStream_t(0)));
  stats_.prepForcesMs = elapsed_ms(t0, HostClock::now());

  // Point the Bary refit at the per-node source-count array when the crossover is
  // on (null otherwise, so the count write is skipped). No-op for other policies.
  if constexpr (std::is_same_v<MP, mp::BaryStokes>) {
    mp::BaryStokes::setNodeCountSymbol(xoverThresh_ > 0 ? d_nodeCount_ : nullptr);
    // Structured source: point the P2M leaf loops at the template + transforms
    // (null when off => byte-identical legacy moments).
    mp::BaryStokes::setStructuredSourceSymbols(
        hasStructuredSource_ ? util::devicePtr(structTemplate64_) : nullptr,
        hasStructuredSource_ ? util::devicePtr(structR_)          : nullptr,
        hasStructuredSource_ ? util::devicePtr(structCenter64_)   : nullptr);
  }

  // fp64-geometry far field: point P2M/M2P at the fp64 bucket-ordered positions
  // (stable across reapply). No-op for other policies.
  if constexpr (std::is_same_v<MP, mp::SphericalStokes>) {
    mp::SphericalStokes::setSourcePos64Symbol(util::devicePtr(buckets_.points64));
    mp::SphericalStokes::setTargetPos64Symbol(targetPoints64());
  }

  t0 = HostClock::now();
  MP::upwardPass(bvh_, d_mac_, d_m2p_,
                 util::devicePtr(buckets_.points), util::devicePtr(force_),
                 sourceOwnerPtr(), d_ownerMask_, ownerMaskWords_,
                 util::devicePtr(buckets_.begin), util::devicePtr(buckets_.end),
                 cfg_.order);
  CUDA_CHECK(cudaStreamSynchronize(cudaStream_t(0)));   // see prepForcesMs note
  stats_.upwardMs = elapsed_ms(t0, HostClock::now());
}


template<class MP>
void Treecode<MP>::upwardPass(const vec3d *d_force)
{
  assert(built_);
  assert(d_force != nullptr);

  auto t0 = HostClock::now();
  thrust::transform(buckets_.perm.begin(), buckets_.perm.end(),
                    force_.begin(), SourceForceGatherD{d_force});
#if WIDEBVH_FP32_LEVEL >= 2
  force32_.resize(nSource_);
  thrust::transform(force_.begin(), force_.end(), force32_.begin(),
                    NarrowForceD2F{});
#endif
  // Stream-scoped sync (NOT cudaDeviceSynchronize): this wait exists only to
  // host-clock the phase, and all engine work runs on the legacy default
  // stream, so syncing stream 0 times exactly the same work. A device-wide
  // sync here would also drag in work a CALLER may have in flight on a side
  // stream (e.g. the sphere-BIM VSH near correction overlapping reapply).
  CUDA_CHECK(cudaStreamSynchronize(cudaStream_t(0)));
  stats_.prepForcesMs = elapsed_ms(t0, HostClock::now());

  // Point the Bary refit at the per-node source-count array when the crossover is
  // on (null otherwise, so the count write is skipped). No-op for other policies.
  if constexpr (std::is_same_v<MP, mp::BaryStokes>) {
    mp::BaryStokes::setNodeCountSymbol(xoverThresh_ > 0 ? d_nodeCount_ : nullptr);
    // Structured source: point the P2M leaf loops at the template + transforms
    // (null when off => byte-identical legacy moments).
    mp::BaryStokes::setStructuredSourceSymbols(
        hasStructuredSource_ ? util::devicePtr(structTemplate64_) : nullptr,
        hasStructuredSource_ ? util::devicePtr(structR_)          : nullptr,
        hasStructuredSource_ ? util::devicePtr(structCenter64_)   : nullptr);
  }

  // fp64-geometry far field: point P2M/M2P at the fp64 bucket-ordered positions
  // (stable across reapply). No-op for other policies.
  if constexpr (std::is_same_v<MP, mp::SphericalStokes>) {
    mp::SphericalStokes::setSourcePos64Symbol(util::devicePtr(buckets_.points64));
    mp::SphericalStokes::setTargetPos64Symbol(targetPoints64());
  }

  t0 = HostClock::now();
  MP::upwardPass(bvh_, d_mac_, d_m2p_,
                 util::devicePtr(buckets_.points), util::devicePtr(force_),
                 sourceOwnerPtr(), d_ownerMask_, ownerMaskWords_,
                 util::devicePtr(buckets_.begin), util::devicePtr(buckets_.end),
                 cfg_.order);
  CUDA_CHECK(cudaStreamSynchronize(cudaStream_t(0)));   // see prepForcesMs note
  stats_.upwardMs = elapsed_ms(t0, HostClock::now());
}


template<class MP>
template<int ORDER>
void Treecode<MP>::launchTraverseEmit(int gridN, int block,
                                      double *d_velBucket,
                                      int *d_pairTarget, int *d_pairLeaf,
                                      unsigned long long *d_pairCount,
                                      long long pairCapacity
#if TREECODE_STATS
                                      , uint32_t *d_visited, uint32_t *d_m2pAccepted
#endif
                                      )
{
  if (pathKind_ == PathKind::SplitWarpSpec ||
      pathKind_ == PathKind::SplitWarpSpecAtomic) {
    // split-warpspec[-atomic]: warp 0 produces; consumer warps evaluate M2P via a
    // ring; rejected leaves are appended to the global near-pair list (EMIT_PAIRS).
    // Identical emit for both paths -- they differ only in how the emitted pair
    // list is consumed afterward (sorted+reduce vs unsorted+atomic). Block size
    // (consumer-warp count) is the compile-time SPLIT_WARPSPEC_BLOCK; the passed
    // gridN/block are for the split-warp kernel and ignored.
    (void)gridN;
    (void)block;
    const int gridWS = (int)((nTarget_ + 31) / 32);
    constexpr int wsBlock =
        MP::LANE_BATCH_M2P ? SPLIT_WARPSPEC_LB_BLOCK : SPLIT_WARPSPEC_BLOCK;
    auto kfn = splitWarpSpecKernel<MP, ORDER, /*EMIT_PAIRS=*/true>();
    kfn<<<gridWS, wsBlock>>>(
        bvh_, d_mac_, d_m2p_, targetPoints(), targetOwnerPtr(),
        ownerMaskPtr(), ownerMaskWords_, d_nodeCount_, xoverThresh_,
        emitInternalNodesFlag(),
        (int)nTarget_, cfg_.mac, stokes::prefactor(), skipSameGroupFlag(),
        d_velBucket, d_pairTarget, d_pairLeaf, d_pairCount, pairCapacity,
        /*perTargetCount=*/nullptr
#if TREECODE_STATS
        , d_visited, d_m2pAccepted
#endif
        );
    return;
  }
  // split-warp: one warp per target (full range: targetStart=0, count=N=nTarget_).
  (void)gridN;
  const int warpsPerBlock = block / 32;
  const int gridW = (int)((nTarget_ + (size_t)warpsPerBlock - 1)
                          / (size_t)warpsPerBlock);
  traverseM2PKernel_Warp<MP, ORDER><<<gridW, block>>>(
      bvh_, d_mac_, d_m2p_, targetPoints(), targetOwnerPtr(),
      ownerMaskPtr(), ownerMaskWords_, d_nodeCount_, xoverThresh_,
      /*targetStart=*/0, /*count=*/(int)nTarget_, (int)nTarget_,
      cfg_.mac, stokes::prefactor(), skipSameGroupFlag(),
      d_velBucket, d_pairTarget, d_pairLeaf,
      d_pairCount, pairCapacity
#if TREECODE_STATS
      , d_visited, d_m2pAccepted
#endif
      );
}


template<class MP>
template<int ORDER>
void Treecode<MP>::launchTraverseEmitWarpRange(
    int targetStart, int count, double *d_velBucket,
    int *d_pairTarget, int *d_pairLeaf,
    unsigned long long *d_pairCount, int pairCapacity)
{
  if (count <= 0) return;
  const int block = TRAVERSE_WARP_BLOCK;
  const int warpsPerBlock = block / 32;
  const int gridW =
      (int)(((size_t)count + (size_t)warpsPerBlock - 1) / (size_t)warpsPerBlock);
  // Always the warp emit kernel over [targetStart, targetStart+count); the full
  // target count nTarget_ is the component-major stride of d_velBucket.
  traverseM2PKernel_Warp<MP, ORDER><<<gridW, block>>>(
      bvh_, d_mac_, d_m2p_, targetPoints(), targetOwnerPtr(),
      ownerMaskPtr(), ownerMaskWords_, d_nodeCount_, xoverThresh_,
      targetStart, count, (int)nTarget_,
      cfg_.mac, stokes::prefactor(), skipSameGroupFlag(),
      d_velBucket, d_pairTarget, d_pairLeaf,
      d_pairCount, (long long)pairCapacity
#if TREECODE_STATS
      // tiling is for oversize runs; per-tile traversal stats are not collected.
      , nullptr, nullptr
#endif
      );
}


template<class MP>
template<int ORDER>
void Treecode<MP>::launchTraverseOnly(int gridN, int block,
                                      double *d_velBucket
#if TREECODE_STATS
                                      , uint32_t *d_visited, uint32_t *d_m2pAccepted
#endif
                                      )
{
  if (pathKind_ == PathKind::SplitWarpSpec ||
      pathKind_ == PathKind::SplitWarpSpecAtomic ||
      pathKind_ == PathKind::DoubleTraverse) {
    // split-warpspec[-atomic] / double-traverse reapply far field: M2P only
    // (cached P2P replayed later, so EMIT_PAIRS=false and the pair buffers are
    // null). Block size is the compile-time SPLIT_WARPSPEC_BLOCK.
    (void)gridN;
    (void)block;
    const int gridWS = (int)((nTarget_ + 31) / 32);
    constexpr int wsBlock =
        MP::LANE_BATCH_M2P ? SPLIT_WARPSPEC_LB_BLOCK : SPLIT_WARPSPEC_BLOCK;
    auto kfn = splitWarpSpecKernel<MP, ORDER, /*EMIT_PAIRS=*/false>();
    kfn<<<gridWS, wsBlock>>>(
        bvh_, d_mac_, d_m2p_, targetPoints(), targetOwnerPtr(),
        ownerMaskPtr(), ownerMaskWords_, d_nodeCount_, xoverThresh_,
        emitInternalNodesFlag(),
        (int)nTarget_, cfg_.mac, stokes::prefactor(), skipSameGroupFlag(),
        d_velBucket, nullptr, nullptr, nullptr, 0, /*perTargetCount=*/nullptr
#if TREECODE_STATS
        , d_visited, d_m2pAccepted
#endif
        );
    return;
  }
  // split-warp reapply far field: one warp per target.
  (void)gridN;
  const int warpsPerBlock = block / 32;
  const int gridW = (int)((nTarget_ + (size_t)warpsPerBlock - 1)
                          / (size_t)warpsPerBlock);
  traverseM2POnlyKernel_Warp<MP, ORDER><<<gridW, block>>>(
      bvh_, d_mac_, d_m2p_, targetPoints(), targetOwnerPtr(),
      ownerMaskPtr(), ownerMaskWords_, d_nodeCount_, xoverThresh_,
      (int)nTarget_, cfg_.mac, stokes::prefactor(), skipSameGroupFlag(),
      d_velBucket
#if TREECODE_STATS
      , d_visited, d_m2pAccepted
#endif
      );
}


template<class MP>
template<int ORDER>
void Treecode<MP>::launchTraverseCount(double *d_velBucket,
                                       int *d_perTargetCount)
{
  // double-traverse pass 1: same warpspec M2P producer/consumer as split-warpspec
  // (far field bit-identical), but the producer counts rejected leaves per target
  // (COUNT_ONLY) instead of emitting pairs. No near-pair list, no atomics.
  const int gridWS = (int)((nTarget_ + 31) / 32);
  constexpr int wsBlock =
      MP::LANE_BATCH_M2P ? SPLIT_WARPSPEC_LB_BLOCK : SPLIT_WARPSPEC_BLOCK;
  auto kfn = splitWarpSpecKernel<MP, ORDER, /*EMIT_PAIRS=*/false, /*COUNT_ONLY=*/true>();
  kfn<<<gridWS, wsBlock>>>(
      bvh_, d_mac_, d_m2p_, targetPoints(), targetOwnerPtr(),
      ownerMaskPtr(), ownerMaskWords_, d_nodeCount_, xoverThresh_,
      emitInternalNodesFlag(),
      (int)nTarget_, cfg_.mac, stokes::prefactor(), skipSameGroupFlag(),
      d_velBucket, nullptr, nullptr, nullptr, 0, d_perTargetCount
#if TREECODE_STATS
      , nullptr, nullptr
#endif
      );
}


template<class MP>
template<int ORDER>
void Treecode<MP>::launchTraverseDirect(int gridN, int block,
                                        double *d_velBucket,
                                        unsigned long long *d_nearLeafCount,
                                        unsigned long long *d_p2pCount
#if TREECODE_STATS
                                        , uint32_t *d_visited, uint32_t *d_m2pAccepted
#endif
                                        )
{
  // direct-warpspec: warp-specialized producer/consumer merged traversal over a
  // lock-free shared ring. The block size is the compile-time
  // DIRECT_WARPSPEC_BLOCK (the kernel's WARPSPEC_BLOCK reads the same constant);
  // gridN/block are unused here.
  (void)gridN;
  (void)block;
  const int wsBlock = DIRECT_WARPSPEC_BLOCK;
  const int gridWS = (int)((nTarget_ + 31) / 32);
  traverseWarpSpecKernel<MP, ORDER><<<gridWS, wsBlock>>>(
      bvh_, d_mac_, d_m2p_, points64(), util::devicePtr(force_),
      targetPoints(), targetPoints64(), util::devicePtr(buckets_.begin),
      util::devicePtr(buckets_.end), sourceOwnerPtr(), targetOwnerPtr(),
      ownerMaskPtr(), ownerMaskWords_, d_nodeCount_, xoverThresh_,
      (int)nTarget_, cfg_.mac,
      stokes::prefactor(), skipSameGroupFlag(), d_velBucket,
      d_nearLeafCount, d_p2pCount
#if TREECODE_STATS
      , d_visited, d_m2pAccepted
#endif
      );
}


// double-traverse apply, pass 1: M2P far field + per-target rejected-leaf counts.
// No host sync here -- the count kernel is enqueued and the trav-region event is
// recorded right after it; buildPairListDouble's prefix-sum readback provides the
// ordering point.
template<class MP>
void Treecode<MP>::traverseCountM2P(double *d_velBucket,
                                    thrust::device_vector<int> &perTargetCount)
{
  assert(built_);
  perTargetCount.resize(nTarget_);
  int *d_cnt = util::devicePtr(perTargetCount);
#define TC_LAUNCH_COUNT(ORD)                                              \
  case ORD:                                                               \
    if constexpr (MP::MAX_ORDER >= ORD) {                                 \
      launchTraverseCount<ORD>(d_velBucket, d_cnt);                       \
    } else {                                                              \
      assert(false);                                                     \
    }                                                                     \
    break
  switch (cfg_.order) {
    TC_LAUNCH_COUNT(1);
    TC_LAUNCH_COUNT(2);
    TC_LAUNCH_COUNT(3);
    TC_LAUNCH_COUNT(4);
    TC_LAUNCH_COUNT(5);
    TC_LAUNCH_COUNT(6);
    TC_LAUNCH_COUNT(7);
    TC_LAUNCH_COUNT(8);
    TC_LAUNCH_COUNT(9);
    TC_LAUNCH_COUNT(10);
    TC_LAUNCH_COUNT(11);
    TC_LAUNCH_COUNT(12);
    default: assert(false); break;
  }
#undef TC_LAUNCH_COUNT
  CUDA_CHECK(cudaGetLastError());
}


template<class MP>
void Treecode<MP>::traverseEmitPairs(double *d_velBucket,
                                     thrust::device_vector<int> &pairTarget,
                                     thrust::device_vector<int> &pairLeaf,
                                     int &nPairs)
{
  assert(built_);
  const int block = TRAVERSE_WARP_BLOCK;
  const int gridN = (int)((nTarget_ + block - 1) / block);

#if TREECODE_STATS
  uint32_t *d_visited = nullptr;
  uint32_t *d_m2pAccepted = nullptr;
  CUDA_CHECK(cudaMalloc(&d_visited, nTarget_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_m2pAccepted, nTarget_ * sizeof(uint32_t)));
#endif

  // Whole-range probe. Cap the pair buffer at pairCap_ (the ~40 GB budget). The
  // 64-bit counter records the TRUE total even when it exceeds the buffer, so we
  // can detect the oversize case safely (no int overflow / OOB write) and switch
  // to the tiled path. emitOverflowed_ => evaluateSplitTiled handles this solve.
  static constexpr int AVG_NEAR_LEAVES = 128;
  emitOverflowed_ = false;
  size_t pairCapacity = std::min(nTarget_ * (size_t)AVG_NEAR_LEAVES, pairCap_);
  if (pairCapacity < 1) pairCapacity = 1;
  pairTarget.resize(pairCapacity);
  pairLeaf.resize(pairCapacity);
  thrust::device_vector<unsigned long long> pairCount(1);
  unsigned long long hPairs = 0;

  for (;;) {
    CUDA_CHECK(cudaMemset(util::devicePtr(pairCount), 0,
                          sizeof(unsigned long long)));
#if TREECODE_STATS
#  define TC_VISITED_ARG , d_visited, d_m2pAccepted
#else
#  define TC_VISITED_ARG
#endif
#define TC_LAUNCH_EMIT(ORD)                                                \
    case ORD:                                                              \
      if constexpr (MP::MAX_ORDER >= ORD) {                                \
        launchTraverseEmit<ORD>(gridN, block, d_velBucket,                 \
                                util::devicePtr(pairTarget),               \
                                util::devicePtr(pairLeaf),                 \
                                util::devicePtr(pairCount),                \
                                (long long)pairCapacity TC_VISITED_ARG);   \
      } else {                                                             \
        assert(false);                                                     \
      }                                                                    \
      break
    switch (cfg_.order) {
      TC_LAUNCH_EMIT(1);
      TC_LAUNCH_EMIT(2);
      TC_LAUNCH_EMIT(3);
      TC_LAUNCH_EMIT(4);
      TC_LAUNCH_EMIT(5);
      TC_LAUNCH_EMIT(6);
      TC_LAUNCH_EMIT(7);
      TC_LAUNCH_EMIT(8);
      TC_LAUNCH_EMIT(9);
      TC_LAUNCH_EMIT(10);
      TC_LAUNCH_EMIT(11);
      TC_LAUNCH_EMIT(12);
      default: assert(false); break;
    }
#undef TC_LAUNCH_EMIT
#undef TC_VISITED_ARG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    hPairs = pairCount[0];
    if (hPairs <= pairCapacity) break;          // fits the current buffer
    if (pairCapacity >= pairCap_) {             // hit the budget -> tile instead
      emitOverflowed_ = true;
      break;
    }
    // grow toward the cap and re-run the whole-range emit
    pairCapacity = std::min((size_t)hPairs, pairCap_);
    pairTarget.resize(pairCapacity);
    pairLeaf.resize(pairCapacity);
  }
  // When not overflowing, hPairs <= pairCap_ < INT_MAX, so the int return is safe.
  nPairs = emitOverflowed_ ? 0 : (int)hPairs;

#if TREECODE_STATS
  printTraversalDebugStats("split emit", d_visited, d_m2pAccepted, nTarget_);
  cudaFree(d_m2pAccepted);
  cudaFree(d_visited);
#endif
}


// Oversize split path: when the whole-range near-pair count exceeds pairCap_,
// process targets in contiguous tiles. Each tile re-emits (M2P=, near pairs) over
// its sub-range with the warp emit kernel, then runs the cached P2P pipeline,
// accumulating near field into d_velBucket. M2P output is per-target (disjoint
// tiles => no cross-tile interference); each target is traversed once per tile it
// belongs to. Per-tile pair count is held under pairCap_ (< INT_MAX), so the
// existing int sort/p2p machinery is reused unchanged.
template<class MP>
void Treecode<MP>::evaluateSplitTiled(double *d_velBucket)
{
  assert(built_);
  const int nT = (int)nTarget_;
  static constexpr int AVG_NEAR_LEAVES = 128;

  unsigned long long totalPairs = 0;
  long long          totalP2P   = 0;
  int                tilesUsed  = 0;

  thrust::device_vector<int> pTgt, pLeaf;
  thrust::device_vector<unsigned long long> pCount(1);

  // LIFO stack of [t0, t0+cnt) target ranges to process.
  std::vector<std::pair<int, int>> work;
  work.emplace_back(0, nT);

  while (!work.empty()) {
    const int t0  = work.back().first;
    const int cnt = work.back().second;
    work.pop_back();
    if (cnt <= 0) continue;

    // Emit pairs for this range, growing the buffer toward pairCap_.
    size_t cap = std::min((size_t)cnt * (size_t)AVG_NEAR_LEAVES, pairCap_);
    if (cap < 1) cap = 1;
    pTgt.resize(cap);
    pLeaf.resize(cap);
    unsigned long long np = 0;
    bool over = false;
    for (;;) {
      CUDA_CHECK(cudaMemset(util::devicePtr(pCount), 0,
                            sizeof(unsigned long long)));
#define TC_TILE_EMIT(ORD)                                                     \
      case ORD:                                                               \
        if constexpr (MP::MAX_ORDER >= ORD) {                                 \
          launchTraverseEmitWarpRange<ORD>(                                   \
              t0, cnt, d_velBucket, util::devicePtr(pTgt),                    \
              util::devicePtr(pLeaf), util::devicePtr(pCount), (int)cap);     \
        } else {                                                              \
          assert(false);                                                      \
        }                                                                     \
        break
      switch (cfg_.order) {
        TC_TILE_EMIT(1);
        TC_TILE_EMIT(2);
        TC_TILE_EMIT(3);
        TC_TILE_EMIT(4);
        TC_TILE_EMIT(5);
        TC_TILE_EMIT(6);
        TC_TILE_EMIT(7);
        TC_TILE_EMIT(8);
        TC_TILE_EMIT(9);
        TC_TILE_EMIT(10);
        TC_TILE_EMIT(11);
        TC_TILE_EMIT(12);
        default: assert(false); break;
      }
#undef TC_TILE_EMIT
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      np = pCount[0];
      if (np <= cap) break;                 // fits this buffer
      if (cap >= pairCap_) { over = true; break; }  // hit budget -> subdivide
      cap = std::min((size_t)np, pairCap_);
      pTgt.resize(cap);
      pLeaf.resize(cap);
    }

    if (over) {
      if (cnt <= 1)
        throw std::runtime_error(
            "treecode: a single target's near-pair count exceeds the P2P "
            "budget; raise TC_PAIR_BUDGET_GB");
      // Subdivide into k contiguous sub-ranges sized from the measured count.
      int k = (int)((np + pairCap_ - 1) / pairCap_);
      if (k < 2) k = 2;
      if (k > cnt) k = cnt;
      const int base = cnt / k, extra = cnt % k;
      int s = t0;
      for (int i = 0; i < k; ++i) {
        const int w = base + (i < extra ? 1 : 0);
        if (w > 0) work.emplace_back(s, w);
        s += w;
      }
      continue;
    }

    // Process this tile, then accumulate into d_velBucket. For the atomic path,
    // cache the tile's pairs unsorted and atomicAdd (atomics make any tiling
    // correct); otherwise sort + ranges + cached P2P.
    if (pathKind_ == PathKind::SplitWarpSpecAtomic) {
      preparePairListUnsorted(pTgt, pLeaf, (int)np);  // np <= pairCap_ < INT_MAX
      releaseDeviceVector(pTgt);
      releaseDeviceVector(pLeaf);
      runP2PAtomic(d_velBucket);
    } else {
      preparePairList(pTgt, pLeaf, (int)np);  // np <= pairCap_ < INT_MAX
      releaseDeviceVector(pTgt);              // free emit buffer before partials
      releaseDeviceVector(pLeaf);
      runP2P(d_velBucket);
    }
    totalPairs += np;
    totalP2P   += stats_.totalP2P;
    ++tilesUsed;
  }

  // Cache members now reflect only the last tile; the oversize path always
  // recomputes on reapply (oversizeTiled_), so the partial cache is never reused.
  stats_.nPairs = (long long)std::min(
      totalPairs, (unsigned long long)std::numeric_limits<long long>::max());
  stats_.totalP2P = totalP2P;
  fprintf(stderr,
          "[treecode] split P2P tiled over %d target tile(s) "
          "(near pairs=%llu, budget/tile=%zu pairs)\n",
          tilesUsed, totalPairs, pairCap_);
}


template<class MP>
void Treecode<MP>::traverseDirectNear(double *d_velBucket)
{
  assert(built_);
  const int block = TRAVERSE_WARP_BLOCK;
  const int gridN = (int)((nTarget_ + block - 1) / block);

  unsigned long long *d_counts = nullptr;
  CUDA_CHECK(cudaMalloc(&d_counts, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_counts, 0, 2 * sizeof(unsigned long long)));

#if TREECODE_STATS
  uint32_t *d_visited = nullptr;
  uint32_t *d_m2pAccepted = nullptr;
  CUDA_CHECK(cudaMalloc(&d_visited, nTarget_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_m2pAccepted, nTarget_ * sizeof(uint32_t)));
#endif

#if TREECODE_STATS
#  define TC_VISITED_ARG , d_visited, d_m2pAccepted
#else
#  define TC_VISITED_ARG
#endif
#define TC_LAUNCH_DIRECT(ORD)                                             \
  case ORD:                                                               \
    if constexpr (MP::MAX_ORDER >= ORD) {                                 \
      launchTraverseDirect<ORD>(gridN, block, d_velBucket,                \
                                d_counts, d_counts + 1 TC_VISITED_ARG);   \
    } else {                                                              \
      assert(false);                                                      \
    }                                                                     \
    break
  switch (cfg_.order) {
    TC_LAUNCH_DIRECT(1);
    TC_LAUNCH_DIRECT(2);
    TC_LAUNCH_DIRECT(3);
    TC_LAUNCH_DIRECT(4);
    TC_LAUNCH_DIRECT(5);
    TC_LAUNCH_DIRECT(6);
    TC_LAUNCH_DIRECT(7);
    TC_LAUNCH_DIRECT(8);
    TC_LAUNCH_DIRECT(9);
    TC_LAUNCH_DIRECT(10);
    TC_LAUNCH_DIRECT(11);
    TC_LAUNCH_DIRECT(12);
    default: assert(false); break;
  }
#undef TC_LAUNCH_DIRECT
#undef TC_VISITED_ARG
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long h_counts[2] = {0, 0};
  CUDA_CHECK(cudaMemcpy(h_counts, d_counts, 2 * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  stats_.nPairs = (h_counts[0] >
                   (unsigned long long)std::numeric_limits<long long>::max())
                      ? std::numeric_limits<long long>::max()
                      : (long long)h_counts[0];
  stats_.totalP2P = (h_counts[1] >
                     (unsigned long long)std::numeric_limits<long long>::max())
                        ? std::numeric_limits<long long>::max()
                        : (long long)h_counts[1];

#if TREECODE_STATS
  printTraversalDebugStats("direct-near", d_visited, d_m2pAccepted, nTarget_);
  cudaFree(d_m2pAccepted);
  cudaFree(d_visited);
#endif
  cudaFree(d_counts);
}


template<class MP>
void Treecode<MP>::traverseM2POnly(double *d_velBucket)
{
  assert(built_);
  const int block = TRAVERSE_WARP_BLOCK;
  const int gridN = (int)((nTarget_ + block - 1) / block);

#if TREECODE_STATS
  uint32_t *d_visited = nullptr;
  uint32_t *d_m2pAccepted = nullptr;
  CUDA_CHECK(cudaMalloc(&d_visited, nTarget_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_m2pAccepted, nTarget_ * sizeof(uint32_t)));
#endif

#if TREECODE_STATS
#  define TC_VISITED_ARG , d_visited, d_m2pAccepted
#else
#  define TC_VISITED_ARG
#endif
#define TC_LAUNCH_ONLY(ORD)                                                \
  case ORD:                                                               \
    if constexpr (MP::MAX_ORDER >= ORD) {                                 \
      launchTraverseOnly<ORD>(gridN, block, d_velBucket TC_VISITED_ARG);  \
    } else {                                                              \
      assert(false);                                                      \
    }                                                                     \
    break
  switch (cfg_.order) {
    TC_LAUNCH_ONLY(1);
    TC_LAUNCH_ONLY(2);
    TC_LAUNCH_ONLY(3);
    TC_LAUNCH_ONLY(4);
    TC_LAUNCH_ONLY(5);
    TC_LAUNCH_ONLY(6);
    TC_LAUNCH_ONLY(7);
    TC_LAUNCH_ONLY(8);
    TC_LAUNCH_ONLY(9);
    TC_LAUNCH_ONLY(10);
    TC_LAUNCH_ONLY(11);
    TC_LAUNCH_ONLY(12);
    default: assert(false); break;
  }
#undef TC_LAUNCH_ONLY
#undef TC_VISITED_ARG
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

#if TREECODE_STATS
  printTraversalDebugStats("m2p-only", d_visited, d_m2pAccepted, nTarget_);
  cudaFree(d_m2pAccepted);
  cudaFree(d_visited);
#endif
}


template<class MP>
void Treecode<MP>::preparePairList(thrust::device_vector<int> &pairTarget,
                                   thrust::device_vector<int> &pairLeaf,
                                   int nPairs)
{
  clearPairCache();
  nPairs_ = nPairs;
  stats_.nPairs = nPairs;
  stats_.totalP2P = 0;
  if (nPairs <= 0) {
    pairsBuilt_ = true;
    pairCacheKind_ = PairCacheKind::NearLeaf;
    return;
  }

  sortedTarget_.resize(nPairs);
  sortedLeaf_.resize(nPairs);
  {
    // Keys are target indices in [0, nTarget_); only the low ceil(log2(nTarget_))
    // bits are significant. Restrict the radix sort to [0, endBit) to drop the
    // upper 8-bit passes. Conservative: endBit is the smallest value with
    // 2^endBit >= nTarget_, so every key (< nTarget_) is fully representable and
    // the ordering is exact -- never narrower than correctness allows.
    int endBit = 1;
    while (((size_t)1 << endBit) < (size_t)nTarget_) ++endBit;
    if (endBit > 32) endBit = 32;
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes,
        util::devicePtr(pairTarget), util::devicePtr(sortedTarget_),
        util::devicePtr(pairLeaf),   util::devicePtr(sortedLeaf_),
        nPairs, 0, endBit));
    thrust::device_vector<char> tmp(bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        util::devicePtr(tmp), bytes,
        util::devicePtr(pairTarget), util::devicePtr(sortedTarget_),
        util::devicePtr(pairLeaf),   util::devicePtr(sortedLeaf_),
        nPairs, 0, endBit));
  }

  finalizeNearLeafCache(nPairs);
}


// split-warpspec-atomic: cache the emitted (target, leaf) pairs as-is (unsorted).
// No radix sort and no pairCountsKernel: p2pAtomicKernel reads the unsorted list,
// looks up each leaf's bucket range inline, and atomicAdds into the velocity
// array. sortedTarget_/sortedLeaf_ are populated by an O(1) buffer swap;
// pairBegin_/pairCnt_ stay empty.
template<class MP>
void Treecode<MP>::preparePairListUnsorted(
    thrust::device_vector<int> &pairTarget,
    thrust::device_vector<int> &pairLeaf,
    int nPairs)
{
  clearPairCache();
  nPairs_ = nPairs;
  stats_.nPairs = nPairs;
  // totalP2P (sum of per-pair bucket counts) is not computed here: it would add a
  // reduction inside the timed P2P zone for a stats-only number. Reported as 0.
  stats_.totalP2P = 0;
  if (nPairs <= 0) {
    pairsBuilt_ = true;
    pairCacheKind_ = PairCacheKind::NearLeafUnsorted;
    return;
  }
  // Take ownership of the emit buffers (no copy). The caller's vectors are left
  // holding the old cache buffers and are released right after.
  sortedTarget_.swap(pairTarget);
  sortedLeaf_.swap(pairLeaf);
  pairsBuilt_ = true;
  pairCacheKind_ = PairCacheKind::NearLeafUnsorted;
}


// Map the (target-sorted) sortedLeaf_ list to executable bucket ranges and mark
// the near-leaf cache ready. Assumes sortedTarget_/sortedLeaf_ already hold the
// nPairs sorted pairs. Shared by preparePairList (radix-sorted) and
// buildPairListDouble (double-traverse, sorted by construction).
template<class MP>
void Treecode<MP>::finalizeNearLeafCache(int nPairs)
{
  pairBegin_.resize(nPairs);
  pairCnt_.resize(nPairs);
  const int block = ELEMENTWISE_BLOCK;
  const int gridP = (nPairs + block - 1) / block;
  pairCountsKernel<<<gridP, block>>>(
      bvh_, util::devicePtr(buckets_.begin), util::devicePtr(buckets_.end),
      util::devicePtr(sortedLeaf_), nPairs,
      util::devicePtr(pairBegin_), util::devicePtr(pairCnt_));
  CUDA_CHECK(cudaGetLastError());
  // sortedLeaf_ is only consumed by pairCountsKernel; the cached P2P replay
  // (runP2P) needs only sortedTarget_/pairBegin_/pairCnt_. Free it now to keep
  // the near-field footprint small (important for the tiled budget).
  releaseDeviceVector(sortedLeaf_);

  stats_.totalP2P = thrust::reduce(pairCnt_.begin(), pairCnt_.end(),
                                   (long long)0, cuda::std::plus<long long>{});
  pairsBuilt_ = true;
  pairCacheKind_ = PairCacheKind::NearLeaf;
}


// double-traverse apply: turn the per-target counts from pass 1 into the cached
// near-pair list without a radix sort. Prefix-sum the counts to get per-target
// begin offsets (+ exact total), then one thread per target re-traverses and
// writes its pairs into [offset[t], offset[t+1]) -- already grouped by target.
template<class MP>
bool Treecode<MP>::buildPairListDouble(
    thrust::device_vector<int> &perTargetCount, double *d_velBucket)
{
  clearPairCache();

  // Exclusive begin-offsets in pairOffset[0..nTarget_-1]; pairOffset[nTarget_] is
  // the total. inclusive_scan into [1..] with [0]=0 gives exactly that. Scan in
  // 64-bit so the running sum can't overflow even if the total exceeds INT_MAX
  // (we never cast to int until after the pairCap_ check below).
  thrust::device_vector<unsigned long long> pairOffset(nTarget_ + 1);
  pairOffset[0] = 0ull;
  thrust::inclusive_scan(perTargetCount.begin(), perTargetCount.end(),
                         pairOffset.begin() + 1);
  const unsigned long long total = pairOffset[nTarget_];   // D2H read (syncs)

  if (total > (unsigned long long)pairCap_) {
    // Over the VRAM budget: drop the double-traverse state and recompute this
    // solve in bounded target tiles (rare; reuses the proven tiled split path,
    // which rewrites the M2P far field with `=`, overwriting pass 1's cleanly).
    releaseDeviceVector(perTargetCount);
    releaseDeviceVector(pairOffset);
    evaluateSplitTiled(d_velBucket);
    oversizeTiled_ = true;
    return false;
  }
  oversizeTiled_ = false;

  const int nPairs = (int)total;       // safe: total <= pairCap_ < INT_MAX
  nPairs_ = nPairs;
  stats_.nPairs = nPairs;
  stats_.totalP2P = 0;
  if (nPairs <= 0) {
    releaseDeviceVector(perTargetCount);
    releaseDeviceVector(pairOffset);
    pairsBuilt_ = true;
    pairCacheKind_ = PairCacheKind::NearLeaf;
    return true;
  }

  sortedTarget_.resize(nPairs);
  sortedLeaf_.resize(nPairs);
  {
    const int block = TRAVERSE_PAIRWRITE_BLOCK;
    const int grid  = (int)((nTarget_ + (size_t)block - 1) / (size_t)block);
    traversePairWriteKernel<MP><<<grid, block>>>(
        bvh_, d_mac_, targetPoints(), targetOwnerPtr(),
        ownerMaskPtr(), ownerMaskWords_, d_nodeCount_, xoverThresh_,
        (int)nTarget_, cfg_.mac, skipSameGroupFlag(),
        util::devicePtr(pairOffset),
        util::devicePtr(sortedTarget_), util::devicePtr(sortedLeaf_));
    CUDA_CHECK(cudaGetLastError());
  }
  releaseDeviceVector(perTargetCount);
  releaseDeviceVector(pairOffset);

  finalizeNearLeafCache(nPairs);
  return true;
}


template<class MP>
void Treecode<MP>::runP2P(double *d_velBucket)
{
  assert(pairsBuilt_);
  assert(pairCacheKind_ == PairCacheKind::NearLeaf);
  if (nPairs_ <= 0) return;
  const int nT = (int)nTarget_;

  thrust::device_vector<double> pairPartial((size_t)3 * (size_t)nPairs_);
  const long long grid2 = ((long long)nPairs_ + P2P_WARPS - 1) / P2P_WARPS;
  p2pKernel<<<(unsigned)grid2, P2P_BLOCK>>>(
      points64(), util::devicePtr(force_),
      targetPoints64(), sourceOwnerPtr(), targetOwnerPtr(),
      util::devicePtr(sortedTarget_),
      util::devicePtr(pairBegin_), util::devicePtr(pairCnt_),
      nPairs_, stokes::prefactor(), skipSameGroupFlag(),
      util::devicePtr(pairPartial));
  CUDA_CHECK(cudaGetLastError());

  // reduce_by_key emits one entry per DISTINCT target; sortedTarget_ holds target
  // indices in [0, nTarget_), so at most nTarget_ outputs (<< nPairs_ when the
  // near field is dense). Sizing to nTarget_ instead of nPairs_ removes ~28 B/pair
  // of VRAM, which is what lets the ~40 GB budget hold ~1B pairs in one tile.
  thrust::device_vector<int> uniqueTarget(nTarget_);
  thrust::device_vector<double> sum0(nTarget_), sum1(nTarget_), sum2(nTarget_);
  auto kend = thrust::reduce_by_key(
      sortedTarget_.begin(), sortedTarget_.end(),
      pairPartial.begin() + (ptrdiff_t)0 * nPairs_,
      uniqueTarget.begin(), sum0.begin());
  const int nUnique = (int)(kend.first - uniqueTarget.begin());
  thrust::reduce_by_key(
      sortedTarget_.begin(), sortedTarget_.end(),
      pairPartial.begin() + (ptrdiff_t)1 * nPairs_,
      thrust::make_discard_iterator(), sum1.begin());
  thrust::reduce_by_key(
      sortedTarget_.begin(), sortedTarget_.end(),
      pairPartial.begin() + (ptrdiff_t)2 * nPairs_,
      thrust::make_discard_iterator(), sum2.begin());

  if (nUnique > 0) {
    const int block = ELEMENTWISE_BLOCK;
    const int gridU = (nUnique + block - 1) / block;
    scatterAddKernel<<<gridU, block>>>(
        util::devicePtr(uniqueTarget),
        util::devicePtr(sum0), util::devicePtr(sum1),
        util::devicePtr(sum2), nUnique, nT, d_velBucket);
    CUDA_CHECK(cudaGetLastError());
  }
}


// split-warpspec-atomic near field: one warp per cached UNSORTED (target, leaf)
// pair. p2pAtomicKernel inlines the leaf -> bucket range lookup and atomicAdds the
// warp-reduced contribution straight into d_velBucket -- no partials, no sort, no
// reduce_by_key, no scatterAdd. Replaces runP2P() for the atomic path.
template<class MP>
void Treecode<MP>::runP2PAtomic(double *d_velBucket)
{
  assert(pairsBuilt_);
  assert(pairCacheKind_ == PairCacheKind::NearLeafUnsorted);
  if (nPairs_ <= 0) return;
  const int nT = (int)nTarget_;

  const long long grid2 = ((long long)nPairs_ + P2P_WARPS - 1) / P2P_WARPS;
#if WIDEBVH_FP32_LEVEL >= 2
  // fp32 near field into the per-apply scratch (zeroed in evaluateCurrent);
  // d_velBucket keeps only the far field until scatterToInputOrder folds them.
  (void)d_velBucket;
  assert(velNear32_.size() == (size_t)3 * nTarget_);
  assert(force32_.size() == nSource_);
  p2pAtomicKernel32<<<(unsigned)grid2, P2P_BLOCK>>>(
      bvh_, util::devicePtr(buckets_.begin), util::devicePtr(buckets_.end),
      points(), util::devicePtr(force32_),
      targetPoints(), sourceOwnerPtr(), targetOwnerPtr(),
      util::devicePtr(sortedTarget_), util::devicePtr(sortedLeaf_),
      nodeDfsFirstPtr(), nodeBucketCountPtr(), dfsBucketsPtr(),
      nPairs_, stokes::prefactor(), skipSameGroupFlag(), nT,
      util::devicePtr(velNear32_));
#else
  p2pAtomicKernel<<<(unsigned)grid2, P2P_BLOCK>>>(
      bvh_, util::devicePtr(buckets_.begin), util::devicePtr(buckets_.end),
      points64(), util::devicePtr(force_),
      targetPoints64(), sourceOwnerPtr(), targetOwnerPtr(),
      util::devicePtr(sortedTarget_), util::devicePtr(sortedLeaf_),
      nodeDfsFirstPtr(), nodeBucketCountPtr(), dfsBucketsPtr(),
      nPairs_, stokes::prefactor(), skipSameGroupFlag(), nT, d_velBucket);
#endif
  CUDA_CHECK(cudaGetLastError());
}


// TC_PATH=skel one-time ID build (host + one cuBLAS DGEMM). The proxy sphere
// radius is the WORST-CASE nearest MAC-admissible source distance over all
// groups: the group MAC guarantees every accepted source is >= r_g / mac from
// its group-box center, so sampling at safety * min_g(r_g) / mac makes the
// skeleton conservative for every particle (fields from farther sources are a
// subset of the sampled space). Rotation needs no per-particle handling -- see
// target_skeleton.cuh.
template<class MP>
void Treecode<MP>::skelBuildLift(KernelKind kind)
{
  assert(built_);
  const bool traction = (kind == KernelKind::Traction);
  const int B = cfg_.targetGroupSize;
  const int G = skelNumGroups();
  if ((size_t)G * (size_t)B != nTarget_)
    throw std::runtime_error("skel: targetGroupSize must divide nTarget");
  if (!cublas_) CUBLAS_CHECK(cublasCreate(&cublas_));

  const float minRg = thrust::reduce(
      thrust::make_transform_iterator(targetBuckets_.boxes.begin(),
                                      BoxHalfDiagF{}),
      thrust::make_transform_iterator(targetBuckets_.boxes.end(),
                                      BoxHalfDiagF{}),
      std::numeric_limits<float>::max(), thrust::minimum<float>());
  if (!(minRg > 0.f))
    throw std::runtime_error("skel: degenerate target-group box");

  cuBQL::box3f box0;
  CUDA_CHECK(cudaMemcpy(&box0, util::devicePtr(targetBuckets_.boxes),
                        sizeof(box0), cudaMemcpyDeviceToHost));
  const cuBQL::vec3d c0(0.5 * ((double)box0.lower.x + (double)box0.upper.x),
                        0.5 * ((double)box0.lower.y + (double)box0.upper.y),
                        0.5 * ((double)box0.lower.z + (double)box0.upper.z));

  const int nReq = traction ? skelNTr_ : skelN_;
  if (nReq > B)
    throw std::runtime_error(traction
        ? "TC_TGT_SKEL_TRACTION must be <= Config::targetGroupSize"
        : "TC_TGT_SKEL must be <= Config::targetGroupSize");
  const double rProxy = 0.95 * (double)minRg / (double)cfg_.mac;
  // Traction ID: sample the traction fields of the proxy Stokeslets (template
  // = group 0 = the first B bucket slots; object target buckets are contiguous
  // per-particle runs, so its normals are the first B of targetNormals64_).
  tskel::TargetSkeleton sk = tskel::buildTargetSkeleton(
      cublas_, targetPoints64(), B, c0, rProxy, nReq, skelProxyPerShell_,
      traction ? tskel::SampleKind::Traction : tskel::SampleKind::Stokeslet,
      traction ? util::devicePtr(targetNormals64_) : nullptr);
  if (sk.nSkel < nReq)
    std::printf("[treecode] skel: numerical-rank clamp nSkel %d -> %d\n",
                nReq, sk.nSkel);
  if (traction) {
    skelIdxT_.assign(sk.idx.begin(), sk.idx.end());
    skelTT_.assign(sk.T.begin(), sk.T.end());
    skelBuiltTr_ = true;
  } else {
    skelIdx_.assign(sk.idx.begin(), sk.idx.end());
    skelT_.assign(sk.T.begin(), sk.T.end());
    skelBuilt_ = true;
  }
  std::printf("[treecode] skel build(%s): B=%d nSkel=%d proxies=%d/shell "
              "r_proxy=%.4g min_rg=%.4g pred_rel_resid=%.3e build_ms=%.1f\n",
              traction ? "traction" : "stokeslet", B, sk.nSkel,
              skelProxyPerShell_, rProxy, (double)minRg,
              sk.relResid, sk.buildMs);
  std::fflush(stdout);
}


// KernelKind::Traction support guard: only the skel path evaluates traction,
// and only with the block near field (no traction variant of the atomic
// expanded-pair kernel exists) and bucket-ordered target normals.
template<class MP>
void Treecode<MP>::skelRequireTractionSupport() const
{
  if constexpr (!std::is_same_v<MP, mp::BaryStokes>)
    throw std::runtime_error(
        "KernelKind::Traction is only supported for the BaryStokes policy");
  if (pathKind_ != PathKind::Skel)
    throw std::runtime_error(
        "KernelKind::Traction requires TC_PATH=skel");
  if (!skelP2PBlock_)
    throw std::runtime_error(
        "KernelKind::Traction requires TC_SKEL_P2P=block");
  if (targetNormals64_.size() != nTarget_)
    throw std::runtime_error(
        "KernelKind::Traction requires setTargetNormals() before apply()");
}


// TC_PATH=skel apply traversal: warp-per-group count pass, host scans, then the
// write pass fills the group->node CSR and the expanded per-target near list.
template<class MP>
void Treecode<MP>::skelTraverse(thrust::device_vector<int> &pairTarget,
                                thrust::device_vector<int> &pairLeaf,
                                int &nPairs)
{
  assert(built_ && (skelBuilt_ || skelBuiltTr_));
  const int B = cfg_.targetGroupSize;
  const int G = skelNumGroups();
  const int srcObjectLeaves =
      (bucketizer_ == Bucketizer::Object && cfg_.sourceGroupSize > 0) ? 1 : 0;

  const auto t0 = HostClock::now();
  thrust::device_vector<int> nodeCnt(G);
  thrust::device_vector<int> leafCnt(G);
  const int warpsPerBlock = SKEL_TRAVERSE_BLOCK / 32;
  const int grid = (G + warpsPerBlock - 1) / warpsPerBlock;
  skelGroupTraverseKernel<MP, /*WRITE=*/false><<<grid, SKEL_TRAVERSE_BLOCK>>>(
      bvh_, d_mac_, util::devicePtr(targetBuckets_.boxes), G, B, cfg_.mac,
      ownerMaskPtr(), ownerMaskWords_, skipSameGroupFlag(), srcObjectLeaves,
      d_nodeCount_, xoverThresh_,
      util::devicePtr(nodeCnt), util::devicePtr(leafCnt),
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
  CUDA_CHECK(cudaGetLastError());

  skelNodeBegin_.resize((size_t)G + 1);
  thrust::exclusive_scan(nodeCnt.begin(), nodeCnt.end(),
                         skelNodeBegin_.begin());
  const long long totalNodes =
      (long long)skelNodeBegin_[G - 1] + (long long)nodeCnt[G - 1];
  if (totalNodes > (long long)std::numeric_limits<int>::max())
    throw std::runtime_error("skel: M2P node list exceeds INT_MAX");
  skelNodeBegin_[G] = (int)totalNodes;
  skelNodeList_.resize((size_t)totalNodes);

  const long long totalLeaves =
      thrust::reduce(leafCnt.begin(), leafCnt.end(), (long long)0,
                     cuda::std::plus<long long>{});
  const unsigned long long totalPairs =
      (unsigned long long)totalLeaves * (unsigned long long)B;

  if (skelP2PBlock_) {
    // Group-blocked near field: keep the (group, leaf) CSR only; the expanded
    // per-target list is never materialized (no pair budget applies).
    skelLeafBegin_.resize((size_t)G + 1);
    thrust::exclusive_scan(leafCnt.begin(), leafCnt.end(),
                           skelLeafBegin_.begin());
    skelLeafBegin_[G] = (int)totalLeaves;
    skelLeafList_.resize((size_t)totalLeaves);
    skelGroupTraverseKernel<MP, /*WRITE=*/true><<<grid, SKEL_TRAVERSE_BLOCK>>>(
        bvh_, d_mac_, util::devicePtr(targetBuckets_.boxes), G, B, cfg_.mac,
        ownerMaskPtr(), ownerMaskWords_, skipSameGroupFlag(), srcObjectLeaves,
        d_nodeCount_, xoverThresh_,
        nullptr, nullptr,
        util::devicePtr(skelNodeBegin_), util::devicePtr(skelNodeList_),
        nullptr, nullptr, nullptr,
        util::devicePtr(skelLeafBegin_), util::devicePtr(skelLeafList_), 0);
    CUDA_CHECK(cudaGetLastError());
    nPairs = 0;
    clearPairCache();
    pairsBuilt_ = true;
    pairCacheKind_ = PairCacheKind::SkelGroupCSR;
    stats_.nPairs = (long long)totalPairs;   // expanded-equivalent, comparable
    stats_.totalP2P = 0;
  } else {
    thrust::device_vector<unsigned long long> pairOff((size_t)G + 1);
    thrust::exclusive_scan(
        thrust::make_transform_iterator(leafCnt.begin(), LeafCntToPairsULL{B}),
        thrust::make_transform_iterator(leafCnt.end(), LeafCntToPairsULL{B}),
        pairOff.begin());
    pairOff[G] = totalPairs;
    if (totalPairs > (unsigned long long)pairCap_)
      throw std::runtime_error(
          "skel: expanded near-pair list exceeds the pair budget; raise "
          "TC_PAIR_BUDGET_GB or loosen mac (or use TC_SKEL_P2P=block)");
    pairTarget.resize((size_t)totalPairs);
    pairLeaf.resize((size_t)totalPairs);
    skelGroupTraverseKernel<MP, /*WRITE=*/true><<<grid, SKEL_TRAVERSE_BLOCK>>>(
        bvh_, d_mac_, util::devicePtr(targetBuckets_.boxes), G, B, cfg_.mac,
        ownerMaskPtr(), ownerMaskWords_, skipSameGroupFlag(), srcObjectLeaves,
        d_nodeCount_, xoverThresh_,
        nullptr, nullptr,
        util::devicePtr(skelNodeBegin_), util::devicePtr(skelNodeList_),
        util::devicePtr(pairOff), util::devicePtr(pairTarget),
        util::devicePtr(pairLeaf), nullptr, nullptr, 1);
    CUDA_CHECK(cudaGetLastError());
    nPairs = (int)totalPairs;
  }

  if (skelTiming_) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("[treecode] skel timing: traverse(count+write)=%.3f ms\n",
                elapsed_ms(t0, HostClock::now()));
  }
  std::printf("[treecode] skel traverse: groups=%d m2p_nodes=%lld (%.1f/group) "
              "near_leaves=%lld expanded_pairs=%llu%s\n",
              G, totalNodes, (double)totalNodes / (double)G,
              totalLeaves, totalPairs,
              skelP2PBlock_ ? " (CSR only, not materialized)" : "");
  std::fflush(stdout);
}


// TC_SKEL_P2P=block near field: block per (group, target-chunk) over the
// cached per-group leaf CSR; see skelP2PBlockKernel.
template<class MP>
void Treecode<MP>::runSkelP2PBlock(double *d_velBucket)
{
  assert(pairsBuilt_);
  assert(pairCacheKind_ == PairCacheKind::SkelGroupCSR);
  if (skelLeafList_.empty()) return;
  const int B = cfg_.targetGroupSize;
  const int G = skelNumGroups();
  const int nChunk = (B + SKEL_P2P_BLOCK - 1) / SKEL_P2P_BLOCK;
  // Per-source owner test only when object source leaves cannot guarantee the
  // group's own sources were excluded at emit time.
  const bool srcObjectLeaves =
      (bucketizer_ == Bucketizer::Object && cfg_.sourceGroupSize > 0);
  const int *ownPtr =
      (skipSameGroupFlag() && !srcObjectLeaves) ? sourceOwnerPtr() : nullptr;
  dim3 grid((unsigned)G, (unsigned)nChunk);
  if (kernel_ == KernelKind::Traction)
    skelP2PBlockTractionKernel<<<grid, SKEL_P2P_BLOCK>>>(
        bvh_, util::devicePtr(buckets_.begin), util::devicePtr(buckets_.end),
        points64(), util::devicePtr(force_), targetPoints64(),
        util::devicePtr(targetNormals64_), ownPtr,
        util::devicePtr(skelLeafBegin_), util::devicePtr(skelLeafList_),
        B, (int)nTarget_, stokes::tractionPrefactor(), d_velBucket);
  else
    skelP2PBlockKernel<<<grid, SKEL_P2P_BLOCK>>>(
        bvh_, util::devicePtr(buckets_.begin), util::devicePtr(buckets_.end),
        points64(), util::devicePtr(force_), targetPoints64(), ownPtr,
        util::devicePtr(skelLeafBegin_), util::devicePtr(skelLeafList_),
        B, (int)nTarget_, stokes::prefactor(), d_velBucket);
  CUDA_CHECK(cudaGetLastError());
}


// TC_PATH=skel far field (apply AND reapply): skeleton-only M2P into the
// compact u_skel buffer, then ONE shared cuBLAS DGEMM lifts to all B targets of
// every group, OVERWRITING d_velBucket (beta = 0). Skeleton rows of the lift
// matrix are exact identity, so skeleton targets are reproduced to round-off;
// P2P atomic-adds the near field on top afterward.
template<class MP>
void Treecode<MP>::skelEvalFar(double *d_velBucket)
{
  const bool traction = (kernel_ == KernelKind::Traction);
  assert(built_ && (traction ? skelBuiltTr_ : skelBuilt_));
  const int B = cfg_.targetGroupSize;
  const int G = skelNumGroups();
  const int nS = (int)(traction ? skelIdxT_.size() : skelIdx_.size());
  skelU_.resize((size_t)nS * (size_t)3 * (size_t)G);

  cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
  if (skelTiming_) {
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventRecord(e0));
  }

  const long long warps = (long long)G * nS;
  const long long grid = (warps * 32 + SKEL_EVAL_BLOCK - 1) / SKEL_EVAL_BLOCK;
#define TC_LAUNCH_SKEL(ORD)                                                   \
  case ORD:                                                                   \
    if constexpr (MP::MAX_ORDER >= ORD) {                                     \
      if (traction) {                                                         \
        /* Traction is BaryStokes-only (skelRequireTractionSupport throws     \
           first); the constexpr guard keeps other policies compiling. */     \
        if constexpr (std::is_same_v<MP, mp::BaryStokes>)                     \
          skelM2PTractionEvalKernel<MP, ORD><<<(unsigned)grid,                \
                                               SKEL_EVAL_BLOCK>>>(            \
              d_mac_, d_m2p_, util::devicePtr(skelNodeBegin_),                \
              util::devicePtr(skelNodeList_), targetPoints(),                 \
              util::devicePtr(targetNormals64_),                              \
              util::devicePtr(skelIdxT_), nS, B, G,                           \
              stokes::tractionPrefactor(), util::devicePtr(skelU_));          \
        else                                                                  \
          assert(false);                                                      \
      } else                                                                  \
        skelM2PEvalKernel<MP, ORD><<<(unsigned)grid, SKEL_EVAL_BLOCK>>>(      \
            d_mac_, d_m2p_, util::devicePtr(skelNodeBegin_),                  \
            util::devicePtr(skelNodeList_), targetPoints(),                   \
            util::devicePtr(skelIdx_), nS, B, G, stokes::prefactor(),         \
            util::devicePtr(skelU_));                                         \
    } else {                                                                  \
      assert(false);                                                          \
    }                                                                         \
    break
  switch (cfg_.order) {
    TC_LAUNCH_SKEL(1);
    TC_LAUNCH_SKEL(2);
    TC_LAUNCH_SKEL(3);
    TC_LAUNCH_SKEL(4);
    TC_LAUNCH_SKEL(5);
    TC_LAUNCH_SKEL(6);
    TC_LAUNCH_SKEL(7);
    TC_LAUNCH_SKEL(8);
    TC_LAUNCH_SKEL(9);
    TC_LAUNCH_SKEL(10);
    TC_LAUNCH_SKEL(11);
    TC_LAUNCH_SKEL(12);
    default: assert(false); break;
  }
#undef TC_LAUNCH_SKEL
  CUDA_CHECK(cudaGetLastError());

  if (skelTiming_) CUDA_CHECK(cudaEventRecord(e1));

  const double one = 1.0, zero = 0.0;
  CUBLAS_CHECK(cublasDgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, B, 3 * G, nS,
                           &one,
                           traction ? util::devicePtr(skelTT_)
                                    : util::devicePtr(skelT_), B,
                           util::devicePtr(skelU_), nS,
                           &zero, d_velBucket, B));

  if (skelTiming_) {
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaEventSynchronize(e2));
    float evalMs = 0.f, liftMs = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&evalMs, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&liftMs, e1, e2));
    std::printf("[treecode] skel timing: m2p_eval=%.3f ms lift_dgemm=%.3f ms "
                "(nSkel=%d)\n", evalMs, liftMs, nS);
    std::fflush(stdout);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
  }
}


// TC_SKEL_CHECK=1: on a few sampled groups, evaluate the far field at ALL B
// targets with the SAME cached node list and compare against the lifted rows in
// d_velBucket (which at this point holds exactly the lifted far field). This
// isolates the ID truncation error from the (separate) group-MAC change.
template<class MP>
void Treecode<MP>::skelCheckLift(const double *d_velBucket)
{
  const int B = cfg_.targetGroupSize;
  const int G = skelNumGroups();
  const int nSample = std::min(G, 16);
  const int stride = std::max(1, G / nSample);
  std::vector<int> h_s(nSample);
  for (int k = 0; k < nSample; ++k) h_s[k] = std::min(G - 1, k * stride);
  thrust::device_vector<int> d_s(h_s.begin(), h_s.end());
  thrust::device_vector<double> d_full((size_t)3 * nSample * B);

  const bool traction = (kernel_ == KernelKind::Traction);
  const long long warps = (long long)nSample * B;
  const long long grid = (warps * 32 + SKEL_EVAL_BLOCK - 1) / SKEL_EVAL_BLOCK;
#define TC_LAUNCH_SKELCHK(ORD)                                                \
  case ORD:                                                                   \
    if constexpr (MP::MAX_ORDER >= ORD) {                                     \
      if (traction) {                                                         \
        if constexpr (std::is_same_v<MP, mp::BaryStokes>)                     \
          skelFullEvalTractionKernel<MP, ORD><<<(unsigned)grid,               \
                                                SKEL_EVAL_BLOCK>>>(           \
              d_mac_, d_m2p_, util::devicePtr(skelNodeBegin_),                \
              util::devicePtr(skelNodeList_), targetPoints(),                 \
              util::devicePtr(targetNormals64_),                              \
              util::devicePtr(d_s), nSample, B,                               \
              stokes::tractionPrefactor(), util::devicePtr(d_full));          \
        else                                                                  \
          assert(false);                                                      \
      } else                                                                  \
        skelFullEvalKernel<MP, ORD><<<(unsigned)grid, SKEL_EVAL_BLOCK>>>(     \
            d_mac_, d_m2p_, util::devicePtr(skelNodeBegin_),                  \
            util::devicePtr(skelNodeList_), targetPoints(),                   \
            util::devicePtr(d_s), nSample, B, stokes::prefactor(),            \
            util::devicePtr(d_full));                                         \
    } else {                                                                  \
      assert(false);                                                          \
    }                                                                         \
    break
  switch (cfg_.order) {
    TC_LAUNCH_SKELCHK(1);
    TC_LAUNCH_SKELCHK(2);
    TC_LAUNCH_SKELCHK(3);
    TC_LAUNCH_SKELCHK(4);
    TC_LAUNCH_SKELCHK(5);
    TC_LAUNCH_SKELCHK(6);
    TC_LAUNCH_SKELCHK(7);
    TC_LAUNCH_SKELCHK(8);
    TC_LAUNCH_SKELCHK(9);
    TC_LAUNCH_SKELCHK(10);
    TC_LAUNCH_SKELCHK(11);
    TC_LAUNCH_SKELCHK(12);
    default: assert(false); break;
  }
#undef TC_LAUNCH_SKELCHK
  CUDA_CHECK(cudaGetLastError());

  std::vector<double> full((size_t)3 * nSample * B);
  CUDA_CHECK(cudaMemcpy(full.data(), util::devicePtr(d_full),
                        full.size() * sizeof(double), cudaMemcpyDeviceToHost));
  std::vector<double> lift((size_t)3 * nSample * B);
  for (int c = 0; c < 3; ++c)
    for (int k = 0; k < nSample; ++k)
      CUDA_CHECK(cudaMemcpy(
          &lift[((size_t)c * nSample + k) * B],
          d_velBucket + (size_t)c * nTarget_ + (size_t)h_s[k] * B,
          (size_t)B * sizeof(double), cudaMemcpyDeviceToHost));

  double maxAbs = 0.0, maxRef = 0.0, num2 = 0.0, den2 = 0.0;
  for (size_t i = 0; i < full.size(); ++i) {
    const double e = lift[i] - full[i];
    num2 += e * e;
    den2 += full[i] * full[i];
    maxAbs = std::max(maxAbs, std::abs(e));
    maxRef = std::max(maxRef, std::abs(full[i]));
  }
  std::printf("[treecode] skel check (far field, %d sampled groups): "
              "max_abs=%.3e rel_max=%.3e rel_l2=%.3e\n",
              nSample, maxAbs, (maxRef > 0.0) ? maxAbs / maxRef : maxAbs,
              (den2 > 0.0) ? std::sqrt(num2 / den2) : std::sqrt(num2));
  std::fflush(stdout);
}


// Structured leaf-centric grouping: counting-sort the emitted (target, leaf)
// pairs by object id into a CSR (leafGroupOffset_/leafGroupTargets_). This is the
// one-time cost the plain atomic path avoids (it swaps the buffers in unsorted),
// but the CSR is cached and reused across reapply. Records stats_.groupMs.
template<class MP>
void Treecode<MP>::preparePairListGrouped(
    thrust::device_vector<int> &pairTarget,
    thrust::device_vector<int> &pairLeaf,
    int nPairs)
{
  clearPairCache();
  nPairs_ = nPairs;
  stats_.nPairs = nPairs;
  stats_.totalP2P = 0;
  const int nObj = structNObj_;
  leafGroupOffset_.assign((size_t)nObj + 1, 0);
  if (nPairs <= 0) {
    pairsBuilt_ = true;
    pairCacheKind_ = PairCacheKind::NearLeafGrouped;
    return;
  }

  cudaEvent_t g0, g1;
  CUDA_CHECK(cudaEventCreate(&g0));
  CUDA_CHECK(cudaEventCreate(&g1));
  CUDA_CHECK(cudaEventRecord(g0));

  thrust::device_vector<int> count(nObj, 0);
  const int block = ELEMENTWISE_BLOCK;
  const long long grid = ((long long)nPairs + block - 1) / block;
  structLeafBidHistoKernel<<<(unsigned)grid, block>>>(
      bvh_, util::devicePtr(pairLeaf), nPairs, util::devicePtr(count));
  CUDA_CHECK(cudaGetLastError());

  // Exclusive scan -> per-object begin offsets [0..nObj-1]; [nObj] = total nPairs.
  thrust::exclusive_scan(count.begin(), count.end(), leafGroupOffset_.begin());
  leafGroupOffset_[nObj] = nPairs;   // one D2H proxy write (syncs; one-time)

  leafGroupTargets_.resize(nPairs);
  thrust::device_vector<int> cursor(leafGroupOffset_.begin(),
                                    leafGroupOffset_.begin() + nObj);
  structLeafScatterKernel<<<(unsigned)grid, block>>>(
      bvh_, util::devicePtr(pairTarget), util::devicePtr(pairLeaf), nPairs,
      util::devicePtr(cursor), util::devicePtr(leafGroupTargets_));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(g1));
  CUDA_CHECK(cudaEventSynchronize(g1));
  CUDA_CHECK(cudaEventElapsedTime(&stats_.groupMs, g0, g1));
  cudaEventDestroy(g0);
  cudaEventDestroy(g1);

  pairsBuilt_ = true;
  pairCacheKind_ = PairCacheKind::NearLeafGrouped;
}


// Structured leaf-centric near field: one block per object reconstructs its
// sources once into shared and streams its near targets (cached CSR), atomicAdd
// into d_velBucket. Replaces runP2PAtomic when structured.
template<class MP>
void Treecode<MP>::runP2PLeafGrouped(double *d_velBucket)
{
  assert(pairsBuilt_);
  assert(pairCacheKind_ == PairCacheKind::NearLeafGrouped);
  assert(hasStructuredSource_);
  if (nPairs_ <= 0) return;
  const size_t shBytes = (size_t)structNPtsPerObj_ * sizeof(vec3d);
  p2pLeafGroupedStructuredKernel<<<(unsigned)structNObj_, P2P_BLOCK, shBytes>>>(
      util::devicePtr(structR_), util::devicePtr(structCenter64_),
      util::devicePtr(structTemplate64_), structNPtsPerObj_,
      util::devicePtr(buckets_.begin), util::devicePtr(force_),
      targetPoints64(), util::devicePtr(leafGroupTargets_),
      util::devicePtr(leafGroupOffset_), targetOwnerPtr(),
      skipSameGroupFlag(), (int)nTarget_, stokes::prefactor(), d_velBucket);
  CUDA_CHECK(cudaGetLastError());
}


template<class MP>
void Treecode<MP>::scatterToInputOrder(double *d_outInputOrder)
{
  const int block = ELEMENTWISE_BLOCK;
  const int nT = (int)nTarget_;
  const int grid = (nT + block - 1) / block;
#if WIDEBVH_FP32_LEVEL >= 2
  // The fp32 near-field scratch exists (zeroed) on every evaluateCurrent, and
  // is non-zero only after p2pAtomicKernel32 ran; paths that still evaluate
  // their P2P in fp64 straight into velBucket_ (split-warp, split-warpspec,
  // double-traverse, direct-warpspec, structured, skel) fold zeros.
  assert(velNear32_.size() == (size_t)3 * nTarget_);
  scatterCompToInputOrderFoldKernel<<<grid, block>>>(
      util::devicePtr(velBucket_), util::devicePtr(velNear32_),
      util::devicePtr(targetPermVector()), nT, d_outInputOrder);
#else
  scatterCompToInputOrderKernel<<<grid, block>>>(
      util::devicePtr(velBucket_), util::devicePtr(targetPermVector()),
      nT, d_outInputOrder);
#endif
  CUDA_CHECK(cudaGetLastError());
}


template<class MP>
void Treecode<MP>::evaluateCurrent(bool rebuildP2P, double *d_velOut)
{
  assert(built_);
  assert(d_velOut != nullptr);

  // Traction is only wired for the skel path (+ block P2P + normals); every
  // other path would silently evaluate Stokeslets, so fail loud up front.
  if (kernel_ == KernelKind::Traction) skelRequireTractionSupport();

  if (pathKind_ == PathKind::TraverseCount) {
    // Diagnostic: time the traversal-only walk and report interaction counts.
    // No M2P/P2P compute, no pair cache; d_velOut is zeroed.
    runTraverseCountOnly(d_velOut);
    return;
  }

  if (!rebuildP2P && !pairsBuilt_) rebuildP2P = true;

  velBucket_.resize((size_t)3 * nTarget_);
  double *d_velBucket = util::devicePtr(velBucket_);
#if WIDEBVH_FP32_LEVEL >= 2
  // fp32 near-field scratch for p2pAtomicKernel32: zeroed ONCE per evaluation
  // (apply and reapply alike), before any traversal or tile, so the tiled
  // fallback's per-tile runP2PAtomic calls accumulate into it correctly.
  velNear32_.resize((size_t)3 * nTarget_);
  CUDA_CHECK(cudaMemsetAsync(util::devicePtr(velNear32_), 0,
                             sizeof(float) * (size_t)3 * nTarget_, 0));
#endif

  thrust::device_vector<int> pairTarget;
  thrust::device_vector<int> pairLeaf;
  int emittedPairs = 0;
  bool tiledHandled = false;   // set when evaluateSplitTiled did the P2P work

  cudaEvent_t e0, eTrav, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&eTrav));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));

  const bool useDirectNear  = evalMode_ == EvaluationMode::DirectNear;
  const bool doubleTraverse = pathKind_ == PathKind::DoubleTraverse;
  const bool atomicPath     = pathKind_ == PathKind::SplitWarpSpecAtomic;
  const bool skelPath       = pathKind_ == PathKind::Skel;
  thrust::device_vector<int> perTargetCount;  // double-traverse pass-1 counts

  if (useDirectNear) {
    // direct-warpspec: merged traversal writes far + near in one kernel.
    traverseDirectNear(d_velBucket);
  } else if (skelPath) {
    // TC_PATH=skel: apply builds the lift + group CSR + expanded near list;
    // reapply reuses all three (no traversal at all). The far field is written
    // by skelEvalFar (skeleton M2P + DGEMM lift, overwrites d_velBucket); the
    // near field atomic-adds on top in the P2P section below.
    // The lift build is lazy PER KERNEL and deliberately OUTSIDE the
    // rebuildP2P condition: a Stokeslet reapply on a tree whose apply ran in
    // Traction mode (mobility solve: ~15 traction GMRES reapplies, then one
    // velocity evaluation) builds the missing (skeleton, lift) pair on first
    // use. The traversal CSRs are kernel-independent and shared.
    if (kernel_ == KernelKind::Stokeslet && !skelBuilt_)
      skelBuildLift(KernelKind::Stokeslet);
    if (kernel_ == KernelKind::Traction && !skelBuiltTr_)
      skelBuildLift(KernelKind::Traction);
    if (rebuildP2P)
      skelTraverse(pairTarget, pairLeaf, emittedPairs);
    skelEvalFar(d_velBucket);
    if (skelCheck_) skelCheckLift(d_velBucket);
  } else if (rebuildP2P && doubleTraverse) {
    // double-traverse apply pass 1: M2P far field + per-target near-leaf counts
    // (no near-pair emit, no atomics). The pair list is built in the p2p region.
    traverseCountM2P(d_velBucket, perTargetCount);
  } else if (rebuildP2P) {
    // split apply (split-warp / split-warpspec): emit M2P far field + near pairs.
    traverseEmitPairs(d_velBucket, pairTarget, pairLeaf, emittedPairs);
    if (emitOverflowed_) {
      // Whole-range near-pair count exceeds the budget. Drop the truncated probe
      // buffer and recompute this solve in bounded target tiles.
      releaseDeviceVector(pairTarget);
      releaseDeviceVector(pairLeaf);
      evaluateSplitTiled(d_velBucket);
      oversizeTiled_ = true;
      tiledHandled = true;
    } else {
      oversizeTiled_ = false;
    }
  } else if (oversizeTiled_) {
    // Reapply for an oversize solve: re-emit + recompute in tiles (no per-tile
    // cache; oversize reuse is rare and not perf-critical for the experiments).
    evaluateSplitTiled(d_velBucket);
    tiledHandled = true;
  } else {
    // split reapply: recompute the M2P far field; the cached P2P list is replayed.
    traverseM2POnly(d_velBucket);
  }
  CUDA_CHECK(cudaEventRecord(eTrav));

  bool doRunP2P = !useDirectNear && !tiledHandled;
  if (doRunP2P) {
    if (rebuildP2P && doubleTraverse) {
      // double-traverse pass 2 + cache build (no sort). On budget overflow it
      // falls back to evaluateSplitTiled (which does the full P2P) and we skip
      // the cached runP2P below.
      if (!buildPairListDouble(perTargetCount, d_velBucket)) {
        tiledHandled = true;
        doRunP2P = false;
      }
    } else if (rebuildP2P && (atomicPath || (skelPath && !skelP2PBlock_))) {
      // split-warpspec-atomic (and skel/atomic, whose expanded per-target list
      // is format-identical): cache the emitted pairs UNSORTED (O(1) swap; no
      // radix sort, no pairCountsKernel). p2pAtomicKernel looks up each leaf's
      // bucket range inline and atomicAdds into d_velBucket. In structured mode
      // the leaf-centric P2P needs the pairs grouped by object (CSR) instead.
      if (hasStructuredSource_)
        preparePairListGrouped(pairTarget, pairLeaf, emittedPairs);
      else
        preparePairListUnsorted(pairTarget, pairLeaf, emittedPairs);
      releaseDeviceVector(pairTarget);
      releaseDeviceVector(pairLeaf);
    } else if (rebuildP2P && !skelPath) {
      // (skel with TC_SKEL_P2P=block set its SkelGroupCSR cache inside
      // skelTraverse; nothing to prepare here.)
      preparePairList(pairTarget, pairLeaf, emittedPairs);
      // pairs are sorted into the cache members; free the emit buffer before
      // runP2P allocates the fp64 partials (keeps the single-tile peak ~40 B/pair).
      releaseDeviceVector(pairTarget);
      releaseDeviceVector(pairLeaf);
    }
    if (doRunP2P) {
      if (skelPath && skelP2PBlock_) {
        runSkelP2PBlock(d_velBucket);
      } else if (atomicPath || skelPath) {
        if (hasStructuredSource_) runP2PLeafGrouped(d_velBucket);
        else                      runP2PAtomic(d_velBucket);
      } else {
        runP2P(d_velBucket);
      }
    }
  }
  scatterToInputOrder(d_velOut);

  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  CUDA_CHECK(cudaEventElapsedTime(&stats_.travMs, e0, eTrav));
  CUDA_CHECK(cudaEventElapsedTime(&stats_.p2pMs, eTrav, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(eTrav);
  cudaEventDestroy(e1);
}


// Diagnostic path (TC_PATH=traverse-count): launch the traversal-only counting
// kernel, time it (warmup + repeat, min/mean), reduce the per-target M2P/P2P
// interaction counts to totals, and print one summary line. d_velOut is zeroed
// (this path computes no velocities). stats_.travMs holds the mean kernel time
// so the driver's existing "traverse=" print reflects the traversal-only cost.
template<class MP>
void Treecode<MP>::runTraverseCountOnly(double *d_velOut)
{
  assert(built_);
  const int N = (int)nTarget_;

  thrust::device_vector<int> m2pCount(nTarget_);
  thrust::device_vector<int> p2pCount(nTarget_);
  int *d_m2p = util::devicePtr(m2pCount);
  int *d_p2p = util::devicePtr(p2pCount);

  const int block = 128;
  const int grid  = (int)((nTarget_ + (size_t)block - 1) / (size_t)block);

  auto launch = [&]() {
    traverseCountOnlyKernel<MP><<<grid, block>>>(
        bvh_, d_mac_, targetPoints(), targetOwnerPtr(),
        ownerMaskPtr(), ownerMaskWords_, d_nodeCount_, xoverThresh_,
        N, cfg_.mac, skipSameGroupFlag(), d_m2p, d_p2p);
  };

  // Warmup.
  for (int i = 0; i < 3; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed repeats: track min and mean over R iterations.
  constexpr int R = 10;
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  float minMs = 1e30f, sumMs = 0.f;
  for (int i = 0; i < R; ++i) {
    CUDA_CHECK(cudaEventRecord(e0));
    launch();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    minMs = std::min(minMs, ms);
    sumMs += ms;
  }
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  const float meanMs = sumMs / (float)R;

  // Counts are force-independent and identical across iterations; reduce once.
  const long long totalM2P =
      thrust::reduce(m2pCount.begin(), m2pCount.end(),
                     (long long)0, cuda::std::plus<long long>{});
  const long long totalP2P =
      thrust::reduce(p2pCount.begin(), p2pCount.end(),
                     (long long)0, cuda::std::plus<long long>{});

  // Report into stats so the driver's "traverse=" line shows the mean.
  stats_.travMs = meanMs;
  stats_.p2pMs  = 0.f;

  std::printf(
      "[treecode] traverse-count: N=%d mac=%.3f time_ms(min/mean)=%.4f/%.4f "
      "total_m2p=%lld total_p2p_leaves=%lld mean_m2p=%.2f mean_p2p_leaves=%.2f\n",
      N, (double)cfg_.mac, (double)minMs, (double)meanMs,
      totalM2P, totalP2P,
      (double)totalM2P / (double)N, (double)totalP2P / (double)N);
  std::fflush(stdout);

  // No velocities computed; zero the caller-visible output (bucket order).
  CUDA_CHECK(cudaMemset(d_velOut, 0, sizeof(double) * (size_t)3 * nTarget_));
}


template<class MP>
void
Treecode<MP>::apply(const vec3d *d_source, size_t nSource,
                    const vec3d *d_target, size_t nTarget,
                    const vec3d *d_force, double *d_velOut,
                    bool will_reuse_tree)
{
  assert(d_force != nullptr);
  assert(d_velOut != nullptr);
  logSelectedPath("apply");
  syncNearCutoffSymbols();
  build(d_source, nSource, d_target, nTarget);
  upwardPass(d_force);
  reusable_ = will_reuse_tree;
  evaluateCurrent(/*rebuildP2P=*/true, d_velOut);
  if (!will_reuse_tree)
    freeBuild();
}


template<class MP>
void
Treecode<MP>::apply(const vec3d *d_source, size_t nSource,
                    const vec3d *d_target, size_t nTarget,
                    const vec3f *d_force, double *d_velOut,
                    bool will_reuse_tree)
{
  assert(d_force != nullptr);
  assert(d_velOut != nullptr);
  logSelectedPath("apply");
  syncNearCutoffSymbols();
  build(d_source, nSource, d_target, nTarget);
  upwardPass(d_force);
  reusable_ = will_reuse_tree;
  evaluateCurrent(/*rebuildP2P=*/true, d_velOut);
  if (!will_reuse_tree)
    freeBuild();
}


template<class MP>
void Treecode<MP>::reapply(const vec3d *d_force, double *d_velOut)
{
  assert(built_);
  assert(reusable_);
  assert(evalMode_ == EvaluationMode::DirectNear || pairsBuilt_ ||
         pathKind_ == PathKind::TraverseCount);
  assert(d_force != nullptr);
  assert(d_velOut != nullptr);
  logSelectedPath("reapply");
  syncNearCutoffSymbols();
  upwardPass(d_force);
  evaluateCurrent(/*rebuildP2P=*/false, d_velOut);
}


template<class MP>
void Treecode<MP>::reapply(const vec3f *d_force, double *d_velOut)
{
  assert(built_);
  assert(reusable_);
  assert(evalMode_ == EvaluationMode::DirectNear || pairsBuilt_ ||
         pathKind_ == PathKind::TraverseCount);
  assert(d_force != nullptr);
  assert(d_velOut != nullptr);
  logSelectedPath("reapply");
  syncNearCutoffSymbols();
  upwardPass(d_force);
  evaluateCurrent(/*rebuildP2P=*/false, d_velOut);
}


