// SPDX-License-Identifier: Apache-2.0
//
// Shared utilities for the two_ball treecode driver. Everything that is
// not the top-level orchestration lives here so main.cu stays readable.
//
// Design notes (see ../CLAUDE.md):
//  - input particle data is fp64; bounding-box + centering are done in
//    fp64 so we don't lose precision before we drop to fp32.
//  - the BVH itself is built over fp32 positions.
//  - all bulk (per-particle) work goes through thrust, never a
//    hand-written kernel.
#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "cuBQL/bvh.h"
#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"

namespace util {

  // ---- I/O -------------------------------------------------------------

  /*! Load a raw, headerless binary file of little-endian fp64 coordinates
      stored as three structure-of-arrays blocks: x[0:N], y[0:N], z[0:N].
      The number of points is inferred from the file size. Returns host-side
      points. */
  std::vector<cuBQL::vec3d> loadPointsFP64(const std::string &path);

  // ---- bulk geometry ops (device, thrust) ------------------------------

  /*! Axis-aligned bounding box of n device-resident fp64 points. */
  cuBQL::box3d computeBounds(const cuBQL::vec3d *d_points, size_t n);

  /*! In place: subtract 'shift' from every device-resident fp64 point. */
  void translatePoints(cuBQL::vec3d *d_points, size_t n, cuBQL::vec3d shift);

  /*! Convert n device fp64 points to fp32 (separate output buffer). */
  void convertToFloat(const cuBQL::vec3d *d_in, cuBQL::vec3f *d_out, size_t n);

  /*! Build one (degenerate) box per fp32 point, as cuBQL's builder wants. */
  void pointsToBoxes(const cuBQL::vec3f *d_points, cuBQL::box3f *d_boxes, size_t n);

  // ---- tree analysis ---------------------------------------------------

  /*! Metrics that matter to a treecode developer: how the builder
      distributed particles across leaves, and how deep traversal gets. */
  struct TreeMetrics {
    uint32_t numNodes  = 0;   //!< total nodes in the binary BVH
    uint32_t numInner  = 0;
    uint32_t numLeaves = 0;
    uint64_t numPrims  = 0;   //!< total particles referenced by leaves

    int    minLeaf    = 0;    //!< particles in the smallest leaf
    int    maxLeaf    = 0;    //!< particles in the largest leaf  (metric a)
    double medianLeaf = 0.0;  //!< median particles per leaf       (metric a)
    double meanLeaf   = 0.0;

    int    minLeafDepth  = 0;
    int    maxLeafDepth  = 0;
    double meanLeafDepth = 0.0;

    // leaf spatial size = volume of the leaf's bounding box (fp64 accum)
    double rootVolume       = 0.0;  //!< volume of the whole-domain (root) box
    double totalLeafVolume  = 0.0;  //!< sum of all leaf box volumes
    double minLeafVolume    = 0.0;
    double maxLeafVolume    = 0.0;
    double medianLeafVolume = 0.0;
    double meanLeafVolume   = 0.0;
    double p99LeafVolume    = 0.0;
    uint32_t numZeroVolLeaves = 0;  //!< leaves whose box has zero volume

    std::vector<int>    leafSizes;    //!< per-leaf particle counts, sorted asc.
    std::vector<double> leafVolumes;  //!< per-leaf box volumes, sorted asc.
  };

  /*! Walk the (host-copied) BVH from the root, gathering per-leaf sizes
      and leaf depths. The BVH lives in device memory; this copies the
      node array to the host (it is tiny: ~2x #leaves nodes). */
  TreeMetrics computeTreeMetrics(const cuBQL::bvh3f &bvh);

  /*! Pretty-print the metrics, including the leaf-size histogram. */
  void printTreeMetrics(const TreeMetrics &m, int leafThreshold);

} // namespace util
