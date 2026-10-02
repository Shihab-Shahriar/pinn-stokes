// SPDX-License-Identifier: Apache-2.0
//
// CPU-only Stokeslet treecode engine (BaryStokes policy), the merged
// single-path port of the GPU engine (src/treecode.cuh): one OpenMP thread
// owns one target, walks the source BVH with a local stack, and evaluates
// M2P and P2P inline as it classifies nodes -- the "direct-warpspec" execution
// model with no producer/consumer ring and no near-pair list.
//
// Pipeline (same stages as Treecode::apply):
//   build()    - tight fp64 bounds, grid-hilbert source buckets, cuBQL host
//                BVH over the bucket AABBs (BuildConfig(1): one bucket per
//                leaf), node-array allocation.
//   upward()   - gather caller-order forces into bucket order, P2M over
//                leaves (parallel), M2M descending-index sweep (serial).
//   evaluate() - merged traversal+eval over all targets (= the sources, in
//                bucket/Hilbert order), then scatter to caller order.
// Splitting build/upward/evaluate keeps a future reapply trivial (rebuild
// moments + re-evaluate on fixed geometry).
//
// The MFS path adds: a separate target cloud (setTargets + evaluateAtTargets,
// caller-order targets, AoS output), fp64 strengths (upward(vec3d*)), the
// object source bucketizer (Config::objectGroupSize), and exact same-particle
// exclusion via per-node prim-rank intervals (skipOwnParticle).
// Dropped vs the GPU engine: target bucketization, crossover (TC_XOVER),
// skeletonization, structured sources, and all TC_PATH variants.
#pragma once

#include "bary_stokes_cpu.h"
#include "grid_buckets_cpu.h"

#include <memory>

namespace tccpu {

class TreecodeCpu {
public:
  struct Config {
    float  mac = 0.5f;
    int    maxLeaf = 256;
    double cellEdge = -1.0;      // <= 0 => auto grid-hilbert edge
    double gridHilbertQ = 0.0;   // TC_HILBERT_Q-style override; 0 => formula
    // > 0 selects the object bucketizer: one bucket per contiguous group of
    // objectGroupSize sources (identity perm; maxLeaf/cellEdge unused). This
    // is the MFS source mode (group = one particle's proxy cloud) and the
    // precondition for skipOwnParticle in evaluateAtTargets.
    int    objectGroupSize = 0;
  };

  struct Stats {
    double bucketMs = 0.0, buildBvhMs = 0.0;
    double prepForcesMs = 0.0, upwardMs = 0.0;
    double travMs = 0.0, scatterMs = 0.0;
    double setTargetsMs = 0.0;
    uint32_t numBuckets = 0, numNodes = 0, numLeaves = 0;
    uint32_t numTargets = 0;        // set by evaluateAtTargets (0 = self mode)
    long long m2pNodes = 0;         // MAC-accepted nodes, summed over targets
    long long p2pLeaves = 0;        // rejected leaves, summed over targets
    long long p2pInteractions = 0;  // near-field pair interactions
    int maxStackDepth = 0;
    double nodeM2PMB = 0.0;         // moment-array footprint
  };

  explicit TreecodeCpu(const Config &cfg) : cfg_(cfg) {}
  ~TreecodeCpu();
  TreecodeCpu(const TreecodeCpu &) = delete;
  TreecodeCpu &operator=(const TreecodeCpu &) = delete;

  // pts: caller-order fp64 positions. Builds buckets + BVH + node arrays.
  void build(const vec3d *pts, size_t n);
  // Separate target cloud (MFS: collocation points; sources stay the proxy
  // cloud passed to build). Stored shifted by the SAME source outputShift,
  // caller order (no target bucketization). targetsPerOwner > 0 assigns
  // target t to owner t / targetsPerOwner for skipOwnParticle; in object
  // mode the owner count must equal the source bucket count (owner k ==
  // source bucket k). Requires build().
  void setTargets(const vec3d *pts, size_t nTargets, size_t targetsPerOwner = 0);
  // forceInput: caller-order fp32 forces (widened to fp64 in bucket order,
  // like the GPU force gather). Requires build().
  void upward(const vec3f *forceInput);
  // fp64 overload (MFS: GMRES densities); no rounding through float.
  void upward(const vec3d *forceInput);
  // velOut: caller-owned double[3*n], component-major caller order
  // [ux[0:n], uy[0:n], uz[0:n]]. Requires upward(). Self-target mode
  // (targets are the sources), untouched by the MFS path.
  void evaluate(double *velOut);
  // velOutAoS: caller-owned double[3*nTargets], interleaved AoS
  // [u0x u0y u0z u1x ...] in target caller order. Requires setTargets() +
  // upward(). skipOwnParticle drops every source of the target's own group
  // exactly (object mode only): a node whose subtree contains the target's
  // own bucket is never MAC-accepted (forced descend, mirroring the GPU
  // mayContainSelf term), and the own leaf itself is dropped.
  void evaluateAtTargets(double *velOutAoS, bool skipOwnParticle = false);

  // Profiling-only component mask for evaluateAtTargets: traversal + MAC
  // classification always run; kM2P/kP2P switch the two evaluation bodies on
  // and off at COMPILE time (four instantiations, one runtime dispatch outside
  // the parallel region), so `trav` isolates the pure walk cost and
  // m2p/p2p split the merged loop. kAll is the production path.
  enum Components { kTravOnly = 0, kM2P = 1, kP2P = 2, kAll = 3 };
  void setComponents(int mask) { components_ = mask & kAll; }

  size_t n() const { return n_; }
  size_t numTargets() const { return nTargets_; }
  const Config &config() const { return cfg_; }
  const Stats &stats() const { return stats_; }
  const GridBucketsCpu &buckets() const { return buckets_; }
  // bucket-ordered fp64 shifted positions / forces + bucket->caller perm,
  // for direct-sum validation (mirrors Treecode::points64()/forces()/
  // sourcePermutation()).
  const std::vector<vec3d> &points64() const { return buckets_.points64; }
  const std::vector<vec3d> &forces64() const { return force64_; }
  const std::vector<uint32_t> &sourcePermutation() const { return buckets_.perm; }

private:
  Config cfg_;
  Stats stats_;
  size_t n_ = 0;
  int components_ = kAll;   // profiling only; kAll = production

  GridBucketsCpu buckets_;
  bvh3f bvh_{};   // heap arrays owned via cuBQL::cpu::freeBVH in ~TreecodeCpu

  // Dense hot-loop copies/arrays indexed by BVH node id:
  //  adminBits_ : offset | (count << 48), explicitly packed (24 B/visit hot
  //               set together with mac_, vs 64 B touching the BVH node)
  //  mac_       : 16 B/node, read every node visit
  //  m2p_       : ~6.2 KB/node moments, read only on MAC accept. Allocated
  //               UNINITIALIZED (new[]) so the parallel P2M/M2M writes do the
  //               NUMA first touch.
  std::vector<uint64_t> adminBits_;
  std::vector<bary_cpu::NodeMAC> mac_;
  std::unique_ptr<bary_cpu::NodeM2P[]> m2p_;
  std::vector<uint32_t> leafIds_;

  std::vector<vec3d> force64_;      // bucket-ordered fp64 forces
  std::vector<double> velBucket_;   // component-major, bucket order
                                    // (self-target mode only; lazily sized)

  // -- separate target cloud (empty => self-target mode) -------------------
  std::vector<vec3d> tgt64_;        // shifted by buckets_.outputShift
  std::vector<vec3f> tgt32_;        // fp32 twins: MAC + M2P (GPU contract)
  size_t nTargets_ = 0;
  size_t targetsPerOwner_ = 0;

  // -- same-particle exclusion (object mode: prim == bucket == particle) ----
  // Each cuBQL cpu::spatialMedian subtree owns a CONTIGUOUS primID range
  // (buildRec partitions [begin,mid)/[mid,end) in place), so a per-node rank
  // interval [rankLo_, rankHi_) makes "subtree contains the target's own
  // bucket" an exact two-compare test; invPrim_ maps bucket id -> its rank
  // in the primID array.
  std::vector<uint32_t> invPrim_;
  std::vector<uint32_t> rankLo_, rankHi_;

  // -- level-binned M2M schedule: internal node ids CSR-grouped by depth,
  //    deepest level first (children of level L live in levels > L slots,
  //    i.e. earlier bins), so levels run serially, nodes within a level in
  //    parallel -- per-node math identical to the serial sweep. ------------
  std::vector<uint32_t> m2mLevelNodes_;
  std::vector<uint32_t> m2mLevelOffsets_;

  void upwardMoments();             // P2M + M2M shared by both upward()s

  // SoA twins of points64/force64 for the SIMD P2P loop: unit-stride fp64
  // streams (the AoS vec3d loads were what blocked auto-vectorization).
  // Positions filled in build(), forces in upward(); AoS copies are kept for
  // P2M, the fp32 gather, and the driver's independent direct-sum reference.
  std::vector<double> srcX_, srcY_, srcZ_;
  std::vector<double> fX_, fY_, fZ_;
};

} // namespace tccpu
