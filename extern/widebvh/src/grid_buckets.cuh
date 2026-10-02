// SPDX-License-Identifier: Apache-2.0
//
// GPU construction of leaf-constrained particle buckets:
//   1. assign particles to uniform grid cells of edge h,
//   2. order particles within each occupied cell by a fine Morton key,
//   3. split each cell run into chunks of <= maxParticlesPerBucket,
//   4. compute one tight AABB per chunk.
//
// The output particle array is sorted into bucket-contiguous order. The caller
// chooses the fp64 shift used before casting positions to fp32, so treecode can
// keep centered coordinates while the grid assignment still uses the original
// fp64 bounds.
#pragma once

#include <cstddef>
#include <cstdint>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "cuBQL/bvh.h"

namespace util {

template <class T>
inline T *devicePtr(thrust::device_vector<T> &v)
{
  return thrust::raw_pointer_cast(v.data());
}

template <class T>
inline const T *devicePtr(const thrust::device_vector<T> &v)
{
  return thrust::raw_pointer_cast(v.data());
}

struct GridBuckets {
  thrust::device_vector<cuBQL::vec3f> points;    // particle positions, bucket-contiguous
  thrust::device_vector<cuBQL::vec3d> points64;  // same, fp64 (fp64-geometry P2M/M2P)
  thrust::device_vector<cuBQL::box3f> boxes;     // one primitive AABB per bucket
  thrust::device_vector<int> begin;              // bucket -> sorted particle begin
  thrust::device_vector<int> end;                // bucket -> sorted particle end
  thrust::device_vector<int> count;              // bucket particle count
  thrust::device_vector<cuBQL::vec3f> centroid;  // placeholder P2M/debug summary
  thrust::device_vector<uint32_t> perm;          // bucket slot -> original input index

  cuBQL::box3d bounds;
  cuBQL::vec3d outputShift;
  double cellEdge = 0.0;
  int maxParticlesPerBucket = 0;

  uint32_t nx = 0, ny = 0, nz = 0;
  uint64_t totalCells = 0;
  int occupiedCells = 0;
  int coarseBits = 0;
  int coarseBitsPerAxis = 0;
  int fineBits = 0;
  int fineBitsPerAxis = 0;

  size_t numParticles() const { return points.size(); }
  uint32_t numBuckets() const { return (uint32_t)boxes.size(); }
};

/*! Within-cell fine ordering curve for buildGridBuckets. Morton is the
    original scheme. Hilbert orders each cell's particles along a per-cell
    Hilbert curve instead, so when a dense cell is chopped into <=maxLeaf
    chunks the chunks are contiguous curve segments (compact, near-disjoint)
    rather than Morton-interleaved slabs that stack on top of each other. */
enum class FineCurve { Morton, Hilbert };

GridBuckets buildGridBuckets(const cuBQL::vec3d *d_points,
                             size_t n,
                             cuBQL::box3d bounds,
                             cuBQL::vec3d outputShift,
                             double cellEdge,
                             int maxParticlesPerBucket,
                             FineCurve fineCurve = FineCurve::Morton);

/*! Grid-free bucketizer: one global Hilbert-SFC sort (local iHilbert21 keys
    over sfcBox -- pass a CUBIZED box, each axis is normalized independently),
    then fixed-size chunks of maxParticlesPerBucket consecutive particles per
    bucket. Buckets are contiguous curve segments, so their AABBs are compact
    and (unlike Morton-chunked grid cells) nearly non-overlapping. Output
    contract identical to buildGridBuckets; the grid-stat fields are 0, cellEdge
    is 0, and centroid is left empty. */
GridBuckets buildHilbertBuckets(const cuBQL::vec3d *d_points,
                                size_t n,
                                cuBQL::box3d sfcBox,
                                cuBQL::vec3d outputShift,
                                int maxParticlesPerBucket);

/*! Object-per-bucket builder: each bucket is one physical object's contiguous
    run of `groupSize` consecutive input points (e.g. one MFS particle's
    stokeslet/proxy cloud). No spatial sort and no maxLeaf splitting -- points
    keep their input order (perm = identity; the BVH reorders the per-object
    boxes anyway) and the run [k*groupSize, (k+1)*groupSize) becomes bucket k.
    Requires n % groupSize == 0 (contiguous, evenly divisible groups). Output
    contract identical to buildGridBuckets; grid-stat fields are 0, cellEdge is
    0, maxParticlesPerBucket == groupSize, and centroid is left empty. */
GridBuckets buildObjectBuckets(const cuBQL::vec3d *d_points,
                               size_t n,
                               cuBQL::vec3d outputShift,
                               int groupSize);

/*! Structured (monodisperse) object bucketizer for the TC_SRC_TEMPLATE path:
    same bucket layout as buildObjectBuckets (one contiguous run of groupSize
    input points per bucket, perm = identity, no sort/split), but it does NOT
    materialize the per-point position arrays -- `points`/`points64` stay EMPTY,
    because the treecode reconstructs each source position on the fly from a
    shared reference template + the object's rigid transform. The per-bucket
    AABBs are still computed directly from d_points (a segmented reduce over the
    shifted fp32 boxes), so the resulting BVH is bit-identical to
    buildObjectBuckets. Requires n % groupSize == 0. */
GridBuckets buildStructuredObjectBuckets(const cuBQL::vec3d *d_points,
                                         size_t n,
                                         cuBQL::vec3d outputShift,
                                         int groupSize);

} // namespace util
