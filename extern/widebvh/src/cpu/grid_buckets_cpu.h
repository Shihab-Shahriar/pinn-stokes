// SPDX-License-Identifier: Apache-2.0
//
// CPU port of the grid-hilbert source bucketizer (grid_buckets.cu:261-424,
// FineCurve::Hilbert): assign fp64 points to uniform grid cells, order each
// cell's particles along a local Hilbert curve, chunk each occupied cell's run
// into buckets of <= maxLeaf particles, and compute one tight fp32 AABB per
// bucket. Only the grid-hilbert flavor is ported (the engine default); the
// Morton fine curve and the hilbert/object/structured builders are not.
#pragma once

#include "cpu_common.h"

namespace tccpu {

struct GridBucketsCpu {
  // bucket-contiguous positions, SHIFTED by outputShift (= domain bounds
  // center) so fp32 coordinates stay small. points64 is the fp64 twin in the
  // identical order (the honest geometry for P2P and validation).
  std::vector<vec3f>    points;
  std::vector<vec3d>    points64;
  std::vector<box3f>    boxes;         // one tight fp32 AABB per bucket
  std::vector<int>      begin, end;    // bucket -> particle range
  std::vector<uint32_t> perm;          // bucket slot -> original input index

  box3d  bounds;                       // tight fp64 input bounds (unshifted)
  vec3d  outputShift{0.0, 0.0, 0.0};
  double cellEdge = 0.0;
  int    maxParticlesPerBucket = 0;

  uint32_t nx = 0, ny = 0, nz = 0;
  uint64_t totalCells = 0;
  int occupiedCells = 0;
  int coarseBitsPerAxis = 0, coarseBits = 0;
  int fineBitsPerAxis = 0, fineBits = 0;

  int numBuckets() const { return (int)begin.size(); }
};

// Auto cell edge for grid-hilbert, verbatim from
// Treecode::gridHilbertCellEdge (treecode.cuh:2385-2396):
// q = gridHilbertQ (>0 override, e.g. TC_HILBERT_Q) or cbrt(max(1024,
// n/maxLeaf)); the cell half-diagonal equals rDomain/q where rDomain is the
// tight point-bounds half-diagonal.
double gridHilbertCellEdgeCpu(size_t n, const box3d &bounds, int maxLeaf,
                              double gridHilbertQ = 0.0);

GridBucketsCpu buildGridHilbertBucketsCpu(const vec3d *pts, size_t n,
                                          const box3d &bounds,
                                          vec3d outputShift,
                                          double cellEdge, int maxLeaf);

// CPU port of buildObjectBuckets (grid_buckets.cu:500-551): bucket k is the
// contiguous input run [k*groupSize, (k+1)*groupSize) -- one bucket per
// object -- with identity perm (no sort; the source BVH does the spatial
// ordering over the per-object boxes). Requires n % groupSize == 0. The grid
// stat fields stay 0 and cellEdge = 0 (no grid layer), matching the GPU
// contract.
GridBucketsCpu buildObjectBucketsCpu(const vec3d *pts, size_t n,
                                     const box3d &bounds,
                                     vec3d outputShift, int groupSize);

} // namespace tccpu
