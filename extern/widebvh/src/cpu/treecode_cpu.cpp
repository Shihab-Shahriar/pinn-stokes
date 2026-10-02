// SPDX-License-Identifier: Apache-2.0
//
// See treecode_cpu.h. Ports:
//   build()    <- Treecode::build (treecode.cuh:3341-3438), minus targets/
//                 owners/crossover; BVH via cuBQL::cpu::spatialMedian.
//   upward()   <- BaryStokes upward pass: scalar combine() split into the
//                 leaf P2M (parallel over leaves) and inner M2M (serial
//                 descending-index sweep; the CPU builder guarantees child
//                 index > parent index, so children are final when the sweep
//                 reaches their parent).
//   evaluate() <- traverseCountOnlyKernel's per-target stack walk
//                 (treecode.cuh:524-563) with classifyTraversalNode's MAC
//                 arithmetic verbatim (:269-279) and the merged inline
//                 M2P/P2P eval of traverseWarpSpecKernel (:805-835), then
//                 scatterCompToInputOrderKernel (:2056-2070).
#include "treecode_cpu.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include <omp.h>

namespace tccpu {

namespace {
constexpr uint64_t kOffMask = (1ull << 48) - 1ull;
constexpr int kStackCap = 128;   // GPU uses 64; spatial-median can chain deeper
}

TreecodeCpu::~TreecodeCpu()
{
  if (bvh_.nodes || bvh_.primIDs) cuBQL::cpu::freeBVH(bvh_);
}

void TreecodeCpu::build(const vec3d *pts, size_t n)
{
  if (n == 0) throw std::runtime_error("no particles");
  n_ = n;

  // -- tight fp64 input bounds (port of util::computeBounds) ---------------
  auto t0 = HostClock::now();
  box3d bounds;
#pragma omp parallel
  {
    box3d local;
#pragma omp for schedule(static) nowait
    for (long long i = 0; i < (long long)n; ++i)
      local = local.including(box3d(pts[i]));
#pragma omp critical
    bounds = bounds.including(local);
  }

  // -- source buckets, shifted by the bounds center: object mode (one bucket
  //    per contiguous group) or grid-hilbert ------------------------------
  if (cfg_.objectGroupSize > 0) {
    buckets_ = buildObjectBucketsCpu(pts, n, bounds, bounds.center(),
                                     cfg_.objectGroupSize);
  } else {
    const double edge =
        (cfg_.cellEdge > 0.0)
            ? cfg_.cellEdge
            : gridHilbertCellEdgeCpu(n, bounds, cfg_.maxLeaf, cfg_.gridHilbertQ);
    buckets_ = buildGridHilbertBucketsCpu(pts, n, bounds, bounds.center(),
                                          edge, cfg_.maxLeaf);
  }
  stats_.bucketMs = elapsed_ms(t0, HostClock::now());
  stats_.numBuckets = (uint32_t)buckets_.numBuckets();

  // -- host BVH over the bucket AABBs. makeLeafThreshold=1 => exactly one
  //    bucket per BVH leaf (the same invariant the GPU pipeline relies on;
  //    the P2P range lookup below depends on it). ---------------------------
  t0 = HostClock::now();
  if (bvh_.nodes || bvh_.primIDs) cuBQL::cpu::freeBVH(bvh_);
  cuBQL::BuildConfig buildCfg(1);
  cuBQL::cpu::spatialMedian(bvh_, buckets_.boxes.data(),
                            (uint32_t)buckets_.numBuckets(), buildCfg);
  stats_.buildBvhMs = elapsed_ms(t0, HostClock::now());
  stats_.numNodes = bvh_.numNodes;

  // -- dense hot arrays + leaf list + invariant checks ---------------------
  adminBits_.resize(bvh_.numNodes);
  leafIds_.clear();
  leafIds_.reserve(buckets_.numBuckets());
  for (uint32_t nid = 0; nid < bvh_.numNodes; ++nid) {
    const auto &ad = bvh_.nodes[nid].admin;
    adminBits_[nid] = (uint64_t)ad.offset | ((uint64_t)ad.count << 48);
    if (ad.count != 0) {
      if (ad.count != 1)
        throw std::runtime_error("BVH leaf holds more than one bucket "
                                 "(makeLeafThreshold=1 invariant violated)");
      leafIds_.push_back(nid);
    }
  }
  stats_.numLeaves = (uint32_t)leafIds_.size();
  if (leafIds_.size() != (size_t)buckets_.numBuckets())
    throw std::runtime_error("BVH leaf count != bucket count");

  // -- prim-rank intervals + inverse prim map (same-particle exclusion).
  //    cuBQL's cpu::spatialMedian partitions each node's primID range in
  //    place ([begin,mid) -> child0, [mid,end) -> child1), so every subtree
  //    owns a CONTIGUOUS rank interval; the sweep below computes it exactly
  //    and asserts the contiguity invariant once per build. Children have
  //    higher indices than their parent, so a descending sweep sees children
  //    first. ---------------------------------------------------------------
  {
    const uint32_t numBuckets = (uint32_t)buckets_.numBuckets();
    invPrim_.assign(numBuckets, 0u);
    for (uint32_t r = 0; r < numBuckets; ++r)
      invPrim_[bvh_.primIDs[r]] = r;
    rankLo_.resize(bvh_.numNodes);
    rankHi_.resize(bvh_.numNodes);
    for (long long nid = (long long)bvh_.numNodes - 1; nid >= 0; --nid) {
      const uint64_t admin = adminBits_[(size_t)nid];
      const uint32_t count = (uint32_t)(admin >> 48);
      const uint32_t off = (uint32_t)(admin & kOffMask);
      if (count != 0) {
        rankLo_[(size_t)nid] = off;
        rankHi_[(size_t)nid] = off + count;
      } else {
        if (rankHi_[off + 0] != rankLo_[off + 1])
          throw std::runtime_error("BVH subtree prim ranges not contiguous");
        rankLo_[(size_t)nid] = rankLo_[off + 0];
        rankHi_[(size_t)nid] = rankHi_[off + 1];
      }
    }
    if (rankLo_[0] != 0 || rankHi_[0] != numBuckets)
      throw std::runtime_error("BVH root prim range != [0, numBuckets)");
  }

  // -- M2M level schedule: depth by one ascending pass (child index > parent
  //    index), then internal nodes CSR-binned deepest level first ----------
  {
    std::vector<uint32_t> depth(bvh_.numNodes, 0u);
    uint32_t maxDepth = 0;
    for (uint32_t nid = 0; nid < bvh_.numNodes; ++nid) {
      const uint64_t admin = adminBits_[nid];
      if ((admin >> 48) != 0) continue;   // leaf: no children
      const uint32_t off = (uint32_t)(admin & kOffMask);
      const uint32_t d = depth[nid] + 1;
      depth[off + 0] = d;
      depth[off + 1] = d;
      if (d > maxDepth) maxDepth = d;
    }
    const uint32_t numLevels = maxDepth + 1;
    std::vector<uint32_t> cnt(numLevels, 0u);
    for (uint32_t nid = 0; nid < bvh_.numNodes; ++nid)
      if ((adminBits_[nid] >> 48) == 0) ++cnt[depth[nid]];
    m2mLevelOffsets_.assign(numLevels + 1, 0u);
    for (uint32_t b = 0; b < numLevels; ++b)   // bin b <=> depth maxDepth - b
      m2mLevelOffsets_[b + 1] = m2mLevelOffsets_[b] + cnt[maxDepth - b];
    m2mLevelNodes_.resize(m2mLevelOffsets_.back());
    std::vector<uint32_t> cursor(m2mLevelOffsets_.begin(),
                                 m2mLevelOffsets_.end() - 1);
    for (uint32_t nid = 0; nid < bvh_.numNodes; ++nid)
      if ((adminBits_[nid] >> 48) == 0)
        m2mLevelNodes_[cursor[maxDepth - depth[nid]]++] = nid;
  }

  mac_.assign(bvh_.numNodes, bary_cpu::NodeMAC{});
  // Uninitialized on purpose: the parallel P2M pass does the NUMA first touch.
  m2p_.reset(new bary_cpu::NodeM2P[bvh_.numNodes]);
  stats_.nodeM2PMB =
      (double)(sizeof(bary_cpu::NodeM2P) * (size_t)bvh_.numNodes) / (1024.0 * 1024.0);

  force64_.resize(n);
  velBucket_.clear();               // self-target only; sized in evaluate()

  // SoA position streams for the SIMD P2P loop (static schedule: first touch).
  srcX_.resize(n); srcY_.resize(n); srcZ_.resize(n);
  fX_.resize(n); fY_.resize(n); fZ_.resize(n);
  const vec3d *pos64 = buckets_.points64.data();
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)n; ++i) {
    srcX_[i] = pos64[i].x;
    srcY_[i] = pos64[i].y;
    srcZ_[i] = pos64[i].z;
  }
}

void TreecodeCpu::upward(const vec3f *forceInput)
{
  if (!bvh_.nodes) throw std::runtime_error("upward() before build()");
  const long long n = (long long)n_;

  // -- gather caller-order fp32 forces into bucket-ordered fp64 (port of the
  //    SourceForceGatherF step) --------------------------------------------
  auto t0 = HostClock::now();
  const uint32_t *perm = buckets_.perm.data();
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < n; ++i) {
    const vec3f f = forceInput[perm[i]];
    force64_[i] = vec3d((double)f.x, (double)f.y, (double)f.z);
    fX_[i] = (double)f.x;
    fY_[i] = (double)f.y;
    fZ_[i] = (double)f.z;
  }
  stats_.prepForcesMs = elapsed_ms(t0, HostClock::now());
  upwardMoments();
}

void TreecodeCpu::upward(const vec3d *forceInput)
{
  if (!bvh_.nodes) throw std::runtime_error("upward() before build()");
  const long long n = (long long)n_;

  // fp64 gather (MFS strengths): no rounding through float.
  auto t0 = HostClock::now();
  const uint32_t *perm = buckets_.perm.data();
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < n; ++i) {
    const vec3d f = forceInput[perm[i]];
    force64_[i] = f;
    fX_[i] = f.x;
    fY_[i] = f.y;
    fZ_[i] = f.z;
  }
  stats_.prepForcesMs = elapsed_ms(t0, HostClock::now());
  upwardMoments();
}

void TreecodeCpu::upwardMoments()
{
  // -- P2M over leaves (the per-source-heavy part; leaves are independent).
  //    dynamic,1: bucket populations vary 1..maxLeaf. -----------------------
  auto t0 = HostClock::now();
  const int numLeaves = (int)leafIds_.size();
#pragma omp parallel for schedule(dynamic, 1)
  for (int l = 0; l < numLeaves; ++l) {
    const uint32_t nid = leafIds_[l];
    const uint64_t admin = adminBits_[nid];
    bary_cpu::p2mLeaf(bvh_.nodes[nid].bounds,
                      bvh_.primIDs, admin & kOffMask, (uint32_t)(admin >> 48),
                      buckets_.begin.data(), buckets_.end.data(),
                      buckets_.points.data(), force64_.data(),
                      mac_[nid], m2p_[nid]);
  }

  // -- M2M: level-parallel descending sweep over the depth bins built in
  //    build() (deepest level first). Per-node math is identical to the old
  //    serial descending-index sweep -- every input of a level-L node was
  //    finalized in a strictly deeper bin -- so moments are bit-identical;
  //    only the schedule changed (upward runs once per MFS matvec, so the
  //    serial O(numNodes) sweep was a real cost at 100k+ internal nodes). ---
  const int numLevels = (int)m2mLevelOffsets_.size() - 1;
  for (int L = 0; L < numLevels; ++L) {
    const long long b = (long long)m2mLevelOffsets_[L];
    const long long e = (long long)m2mLevelOffsets_[L + 1];
#pragma omp parallel for schedule(dynamic, 8)
    for (long long k = b; k < e; ++k) {
      const uint32_t nid = m2mLevelNodes_[(size_t)k];
      const uint64_t admin = adminBits_[nid];
      bary_cpu::m2mInner(bvh_.nodes[nid].bounds, admin & kOffMask,
                         mac_.data(), m2p_.get(),
                         mac_[nid], m2p_[nid]);
    }
  }
  stats_.upwardMs = elapsed_ms(t0, HostClock::now());
}

void TreecodeCpu::setTargets(const vec3d *pts, size_t nT, size_t targetsPerOwner)
{
  if (!bvh_.nodes) throw std::runtime_error("setTargets() before build()");
  if (nT == 0) throw std::runtime_error("setTargets(): no targets");
  if (targetsPerOwner > 0) {
    if (nT % targetsPerOwner != 0)
      throw std::runtime_error("setTargets(): nTargets not divisible by "
                               "targetsPerOwner");
    if (cfg_.objectGroupSize > 0 &&
        nT / targetsPerOwner != (size_t)buckets_.numBuckets())
      throw std::runtime_error("setTargets(): target owner count != source "
                               "bucket count (object mode maps owner k to "
                               "source bucket k)");
  }
  auto t0 = HostClock::now();
  nTargets_ = nT;
  targetsPerOwner_ = targetsPerOwner;
  tgt64_.resize(nT);
  tgt32_.resize(nT);
  const vec3d shift = buckets_.outputShift;   // SAME shift as the sources
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)nT; ++i) {
    const vec3d p = pts[i] - shift;
    tgt64_[i] = p;
    tgt32_[i] = vec3f((float)p.x, (float)p.y, (float)p.z);
  }
  stats_.setTargetsMs = elapsed_ms(t0, HostClock::now());
}

void TreecodeCpu::evaluate(double *velOut)
{
  if (!bvh_.nodes) throw std::runtime_error("evaluate() before build()/upward()");
  const long long N = (long long)n_;
  const float mac2 = cfg_.mac * cfg_.mac;
  const double pref = prefactor64();
  if (velBucket_.size() != (size_t)3 * n_)
    velBucket_.assign((size_t)3 * n_, 0.0);

  const uint64_t *adminBits = adminBits_.data();
  const bary_cpu::NodeMAC *macArr = mac_.data();
  const bary_cpu::NodeM2P *m2pArr = m2p_.get();
  const uint32_t *primIDs = bvh_.primIDs;
  const vec3f *pos = buckets_.points.data();
  const vec3d *pos64 = buckets_.points64.data();
  const double *srcX = srcX_.data();
  const double *srcY = srcY_.data();
  const double *srcZ = srcZ_.data();
  const double *fX = fX_.data();
  const double *fY = fY_.data();
  const double *fZ = fZ_.data();
  const int *bucketBegin = buckets_.begin.data();
  const int *bucketEnd = buckets_.end.data();
  double *velBucket = velBucket_.data();

  long long m2pTot = 0, p2pLeafTot = 0, p2pTot = 0;
  int maxSp = 0;

  // One thread owns one target: local fp64 accumulators, exactly three stores
  // at the end (no shared writes -> no atomics, no false sharing beyond chunk
  // boundaries; results are bit-identical for every thread count).
  // dynamic,32 over Hilbert-ordered targets: work stealing levels the
  // clustered-distro tails while neighboring targets keep sharing accepted
  // nodes in cache.
  auto t0 = HostClock::now();
#pragma omp parallel for schedule(dynamic, 32) \
    reduction(+ : m2pTot, p2pLeafTot, p2pTot) reduction(max : maxSp)
  for (long long t = 0; t < N; ++t) {
    const vec3f T = pos[t];        // fp32: MAC + M2P (GPU contract)
    const vec3d T64 = pos64[t];    // fp64: P2P (GPU contract)
    double u0 = 0.0, u1 = 0.0, u2 = 0.0;

    uint32_t stack[kStackCap];
    int sp = 0;
    stack[sp++] = 0;               // root
    while (sp > 0) {
      const uint32_t nid = stack[--sp];
      const uint64_t admin = adminBits[nid];
      const uint32_t count = (uint32_t)(admin >> 48);
      const bary_cpu::NodeMAC &nm = macArr[nid];
      // MAC test: keep classifyTraversalNode's fmaf form verbatim
      // (treecode.cuh:269-279) -- fp32, accept iff halfDiag2 < mac^2 * r2.
      const float dx = T.x - nm.cx;
      const float dy = T.y - nm.cy;
      const float dz = T.z - nm.cz;
      const float r2 = std::fmaf(dx, dx, std::fmaf(dy, dy, dz * dz));
      if (r2 > 0.f && nm.halfDiag2 < mac2 * r2) {
        // accepted node -> inline far field
        bary_cpu::m2pAccum(m2pArr[nid], nm, T, u0, u1, u2);
        ++m2pTot;
      } else if (count != 0) {
        // rejected leaf -> exact near field over its bucket span(s).
        // count == 1 under BuildConfig(1); keep the prim loop for shape
        // parity with the GPU kernel. SIMD form of p2p64: unit-stride SoA
        // source/force streams and a branchless masked ir (r2==0 self /
        // coincident lanes contribute exact 0), so the span vectorizes like
        // m2pAccum. Lane-summation order differs from the scalar version at
        // round-off only, and stays thread-count independent.
        for (uint32_t p = 0; p < count; ++p) {
          const uint32_t bid = primIDs[(admin & kOffMask) + p];
          const int b = bucketBegin[bid];
          const int e = bucketEnd[bid];
#pragma omp simd reduction(+ : u0, u1, u2)
          for (int s = b; s < e; ++s) {
            const double Rx = T64.x - srcX[s];
            const double Ry = T64.y - srcY[s];
            const double Rz = T64.z - srcZ[s];
            const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
            const double ir = (r2 == 0.0) ? 0.0 : 1.0 / std::sqrt(r2);
            const double q  = (Rx * fX[s] + Ry * fY[s] + Rz * fZ[s]) * (ir * ir);
            u0 += ir * (fX[s] + Rx * q);
            u1 += ir * (fY[s] + Ry * q);
            u2 += ir * (fZ[s] + Rz * q);
          }
          p2pTot += e - b;
        }
        ++p2pLeafTot;
      } else {
        // internal node -> descend
        const uint32_t off = (uint32_t)(admin & kOffMask);
        assert(sp + 2 <= kStackCap && "traversal stack overflow");
        stack[sp++] = off + 0;
        stack[sp++] = off + 1;
        if (sp > maxSp) maxSp = sp;
      }
    }

    velBucket[(size_t)0 * N + t] = pref * u0;
    velBucket[(size_t)1 * N + t] = pref * u1;
    velBucket[(size_t)2 * N + t] = pref * u2;
  }
  stats_.travMs = elapsed_ms(t0, HostClock::now());
  stats_.m2pNodes = m2pTot;
  stats_.p2pLeaves = p2pLeafTot;
  stats_.p2pInteractions = p2pTot;
  stats_.maxStackDepth = maxSp;

  // -- scatter component-major bucket order -> caller order (port of
  //    scatterCompToInputOrderKernel) --------------------------------------
  t0 = HostClock::now();
  const uint32_t *perm = buckets_.perm.data();
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < N; ++i) {
    const size_t o = perm[i];
    velOut[(size_t)0 * N + o] = velBucket[(size_t)0 * N + i];
    velOut[(size_t)1 * N + o] = velBucket[(size_t)1 * N + i];
    velOut[(size_t)2 * N + o] = velBucket[(size_t)2 * N + i];
  }
  stats_.scatterMs = elapsed_ms(t0, HostClock::now());
}

void TreecodeCpu::evaluateAtTargets(double *velOut, bool skipOwnParticle)
{
  if (!bvh_.nodes) throw std::runtime_error("evaluateAtTargets() before build()");
  if (nTargets_ == 0)
    throw std::runtime_error("evaluateAtTargets() before setTargets()");
  if (skipOwnParticle) {
    if (targetsPerOwner_ == 0)
      throw std::runtime_error("skipOwnParticle needs targetsPerOwner > 0");
    if (cfg_.objectGroupSize <= 0)
      throw std::runtime_error("skipOwnParticle needs object source buckets "
                               "(Config::objectGroupSize > 0)");
  }
  const long long NT = (long long)nTargets_;
  const float mac2 = cfg_.mac * cfg_.mac;
  const double pref = prefactor64();

  const uint64_t *adminBits = adminBits_.data();
  const bary_cpu::NodeMAC *macArr = mac_.data();
  const bary_cpu::NodeM2P *m2pArr = m2p_.get();
  const uint32_t *primIDs = bvh_.primIDs;
  const vec3f *tgt32 = tgt32_.data();
  const vec3d *tgt64 = tgt64_.data();
  const double *srcX = srcX_.data();
  const double *srcY = srcY_.data();
  const double *srcZ = srcZ_.data();
  const double *fX = fX_.data();
  const double *fY = fY_.data();
  const double *fZ = fZ_.data();
  const int *bucketBegin = buckets_.begin.data();
  const int *bucketEnd = buckets_.end.data();
  const uint32_t *rankLo = rankLo_.data();
  const uint32_t *rankHi = rankHi_.data();
  const uint32_t *invPrim = invPrim_.data();
  const size_t tgtPerOwner = targetsPerOwner_;

  long long m2pTot = 0, p2pLeafTot = 0, p2pTot = 0;
  int maxSp = 0;

  // Same merged per-thread walk as evaluate(), over the separate target
  // cloud in CALLER order (MFS targets are particle-contiguous, so
  // neighboring targets share accepted nodes in cache), writing interleaved
  // AoS output directly -- no target perm, no scatter.
  //
  // The body is a generic lambda taking the component flags as
  // true_type/false_type, so `if constexpr` drops the unselected evaluation
  // body at compile time: the kAll instantiation is exactly the pre-switch
  // loop, and kTravOnly/kM2P/kP2P are profiling variants (classification and
  // interaction counting always run).
  auto run = [&](auto doM2P, auto doP2P) {
    long long m2pT = 0, p2pLeafT = 0, p2pT = 0;
    int maxSpT = 0;
#pragma omp parallel for schedule(dynamic, 32) \
    reduction(+ : m2pT, p2pLeafT, p2pT) reduction(max : maxSpT)
    for (long long t = 0; t < NT; ++t) {
      const vec3f T = tgt32[t];      // fp32: MAC + M2P (GPU contract)
      const vec3d T64 = tgt64[t];    // fp64: P2P (GPU contract)
      // Rank of the target's own source bucket in the BVH primID order; the
      // exclusion below mirrors the GPU mayContainSelf term: a node whose
      // subtree contains that rank is never MAC-accepted (forced descend),
      // and the own LEAF is dropped entirely -- it is exactly the dense self
      // block the MFS operator handles separately.
      const uint32_t tRank =
          skipOwnParticle ? invPrim[(uint32_t)((size_t)t / tgtPerOwner)] : 0u;
      double u0 = 0.0, u1 = 0.0, u2 = 0.0;

      uint32_t stack[kStackCap];
      int sp = 0;
      stack[sp++] = 0;               // root
      while (sp > 0) {
        const uint32_t nid = stack[--sp];
        const uint64_t admin = adminBits[nid];
        const uint32_t count = (uint32_t)(admin >> 48);
        const bary_cpu::NodeMAC &nm = macArr[nid];
        const bool containsOwn =
            skipOwnParticle && rankLo[nid] <= tRank && tRank < rankHi[nid];
        const float dx = T.x - nm.cx;
        const float dy = T.y - nm.cy;
        const float dz = T.z - nm.cz;
        const float r2 = std::fmaf(dx, dx, std::fmaf(dy, dy, dz * dz));
        if (r2 > 0.f && nm.halfDiag2 < mac2 * r2 && !containsOwn) {
          if constexpr (decltype(doM2P)::value)
            bary_cpu::m2pAccum(m2pArr[nid], nm, T, u0, u1, u2);
          ++m2pT;
        } else if (count != 0) {
          if (!containsOwn) {        // own leaf: skipped in full
            for (uint32_t p = 0; p < count; ++p) {
              const uint32_t bid = primIDs[(admin & kOffMask) + p];
              const int b = bucketBegin[bid];
              const int e = bucketEnd[bid];
              if constexpr (decltype(doP2P)::value) {
#pragma omp simd reduction(+ : u0, u1, u2)
                for (int s = b; s < e; ++s) {
                  const double Rx = T64.x - srcX[s];
                  const double Ry = T64.y - srcY[s];
                  const double Rz = T64.z - srcZ[s];
                  const double r2p = Rx * Rx + Ry * Ry + Rz * Rz;
                  const double ir = (r2p == 0.0) ? 0.0 : 1.0 / std::sqrt(r2p);
                  const double q =
                      (Rx * fX[s] + Ry * fY[s] + Rz * fZ[s]) * (ir * ir);
                  u0 += ir * (fX[s] + Rx * q);
                  u1 += ir * (fY[s] + Ry * q);
                  u2 += ir * (fZ[s] + Rz * q);
                }
              }
              p2pT += e - b;
            }
            ++p2pLeafT;
          }
        } else {
          const uint32_t off = (uint32_t)(admin & kOffMask);
          assert(sp + 2 <= kStackCap && "traversal stack overflow");
          stack[sp++] = off + 0;
          stack[sp++] = off + 1;
          if (sp > maxSpT) maxSpT = sp;
        }
      }

      velOut[3 * t + 0] = pref * u0;   // interleaved AoS, caller order
      velOut[3 * t + 1] = pref * u1;
      velOut[3 * t + 2] = pref * u2;
    }
    m2pTot = m2pT; p2pLeafTot = p2pLeafT; p2pTot = p2pT; maxSp = maxSpT;
  };

  auto t0 = HostClock::now();
  switch (components_) {
  case kAll:      run(std::true_type{},  std::true_type{});  break;
  case kM2P:      run(std::true_type{},  std::false_type{}); break;
  case kP2P:      run(std::false_type{}, std::true_type{});  break;
  default:        run(std::false_type{}, std::false_type{}); break;
  }
  stats_.travMs = elapsed_ms(t0, HostClock::now());
  stats_.scatterMs = 0.0;
  stats_.numTargets = (uint32_t)NT;
  stats_.m2pNodes = m2pTot;
  stats_.p2pLeaves = p2pLeafTot;
  stats_.p2pInteractions = p2pTot;
  stats_.maxStackDepth = maxSp;
}

} // namespace tccpu
