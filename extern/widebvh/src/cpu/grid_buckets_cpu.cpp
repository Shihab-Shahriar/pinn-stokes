// SPDX-License-Identifier: Apache-2.0
//
// See grid_buckets_cpu.h. Step-for-step port of buildGridBuckets
// (grid_buckets.cu:261-424) with the Thrust/CUB orchestration replaced by
// std::sort + serial scans + OpenMP loops:
//   radix SortPairs      -> std::sort on (key, idx) with idx tie-break
//                           (restores the stable-sort determinism)
//   RunLengthEncode+scan -> one serial pass over the sorted coarse keys
//   fillBucketRanges     -> plain loop (same chunk math)
//   SegmentedReduce      -> per-bucket fp32 AABB loop
#include "grid_buckets_cpu.h"

#include "../hilbert_sfc.cuh"   // iHilbert21 (host-portable)

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tccpu {
namespace {

// Verbatim from grid_buckets.cu:32-46.
inline uint64_t expandBits21(uint64_t v)
{
  v &= 0x1fffffull;
  v = (v | v << 32) & 0x1f00000000ffffull;
  v = (v | v << 16) & 0x1f0000ff0000ffull;
  v = (v | v <<  8) & 0x100f00f00f00f00full;
  v = (v | v <<  4) & 0x10c30c30c30c30c3ull;
  v = (v | v <<  2) & 0x1249249249249249ull;
  return v;
}

inline uint64_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
  return (expandBits21(x) << 2) | (expandBits21(y) << 1) | expandBits21(z);
}

// Composite grid bucket key, verbatim from MakeCompositeKey<Hilbert=true>
// (grid_buckets.cu:54-92): (coarseCellMorton << fineBits) | within-cell
// Hilbert key of the fractional cell coordinate quantized to B bits/axis.
struct CompositeKeyFn {
  vec3d lower;
  double h;
  uint32_t nx, ny, nz;
  int B;
  int fineBits;

  uint64_t operator()(const vec3d &p) const
  {
    double gx = std::floor((p.x - lower.x) / h);
    double gy = std::floor((p.y - lower.y) / h);
    double gz = std::floor((p.z - lower.z) / h);
    if (gx < 0.0) gx = 0.0;
    if (gy < 0.0) gy = 0.0;
    if (gz < 0.0) gz = 0.0;
    uint32_t cx = (uint32_t)gx;
    uint32_t cy = (uint32_t)gy;
    uint32_t cz = (uint32_t)gz;
    if (cx >= nx) cx = nx - 1;
    if (cy >= ny) cy = ny - 1;
    if (cz >= nz) cz = nz - 1;
    const uint64_t coarse = morton3D(cx, cy, cz);

    uint64_t fine = 0;
    if (B > 0) {
      const double scale = (double)(1u << B);
      const uint32_t qmax = (1u << B) - 1u;
      uint32_t qx = (uint32_t)std::floor(((p.x - lower.x) - (double)cx * h) / h * scale);
      uint32_t qy = (uint32_t)std::floor(((p.y - lower.y) - (double)cy * h) / h * scale);
      uint32_t qz = (uint32_t)std::floor(((p.z - lower.z) - (double)cz * h) / h * scale);
      if (qx > qmax) qx = qmax;
      if (qy > qmax) qy = qmax;
      if (qz > qmax) qz = qmax;
      fine = util::iHilbert21(qx, qy, qz);
    }
    return (coarse << fineBits) | fine;
  }
};

// Port of gridDimForExtent (grid_buckets.cu:213-220).
uint32_t gridDimForExtent(double extent, double h)
{
  const double d = std::ceil(extent / h);
  if (!(d > 1.0)) return 1u;
  if (d > (double)std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("grid dimension overflows uint32_t; increase cell edge");
  return (uint32_t)d;
}

struct KeyIdx {
  uint64_t key;
  uint32_t idx;
};

} // anonymous namespace

double gridHilbertCellEdgeCpu(size_t n, const box3d &bounds, int maxLeaf,
                              double gridHilbertQ)
{
  const double q =
      (gridHilbertQ > 0.0)
          ? gridHilbertQ
          : std::cbrt(std::max(1024.0,
                               (double)n / (double)std::max(1, maxLeaf)));
  const vec3d sz = bounds.size();
  const double rDomain =
      0.5 * std::sqrt(sz.x * sz.x + sz.y * sz.y + sz.z * sz.z);
  return 2.0 * (rDomain / q) / std::sqrt(3.0);
}

GridBucketsCpu buildGridHilbertBucketsCpu(const vec3d *pts, size_t n,
                                          const box3d &bounds,
                                          vec3d outputShift,
                                          double cellEdge, int maxLeaf)
{
  if (n == 0) throw std::runtime_error("no particles to bucket");
  if (!(cellEdge > 0.0)) throw std::runtime_error("cell edge must be > 0");
  if (maxLeaf < 1) throw std::runtime_error("maxLeaf must be >= 1");
  if (n > (size_t)std::numeric_limits<int>::max())
    throw std::runtime_error("particle count exceeds int-sized index range");

  GridBucketsCpu out;
  out.bounds = bounds;
  out.outputShift = outputShift;
  out.cellEdge = cellEdge;
  out.maxParticlesPerBucket = maxLeaf;

  // -- grid dims + key bit budget (grid_buckets.cu:282-301, Hilbert cap 20) --
  const vec3d extent = bounds.size();
  out.nx = gridDimForExtent(extent.x, cellEdge);
  out.ny = gridDimForExtent(extent.y, cellEdge);
  out.nz = gridDimForExtent(extent.z, cellEdge);
  out.totalCells = (uint64_t)out.nx * (uint64_t)out.ny * (uint64_t)out.nz;

  const uint32_t maxDim = std::max(out.nx, std::max(out.ny, out.nz));
  out.coarseBitsPerAxis =
      (maxDim <= 1) ? 1 : (int)std::ceil(std::log2((double)maxDim));
  out.coarseBits = 3 * out.coarseBitsPerAxis;
  if (out.coarseBits > 63)
    throw std::runtime_error("grid too fine for a 64-bit Morton key; increase cell edge");
  out.fineBitsPerAxis = std::min(20, (64 - out.coarseBits) / 3);
  if (out.fineBitsPerAxis < 0) out.fineBitsPerAxis = 0;
  out.fineBits = 3 * out.fineBitsPerAxis;

  // -- composite keys + argsort ------------------------------------------
  const CompositeKeyFn keyFn{bounds.lower, cellEdge,
                             out.nx, out.ny, out.nz,
                             out.fineBitsPerAxis, out.fineBits};
  std::vector<KeyIdx> ki(n);
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)n; ++i)
    ki[i] = KeyIdx{keyFn(pts[i]), (uint32_t)i};

  std::sort(ki.begin(), ki.end(), [](const KeyIdx &a, const KeyIdx &b) {
    return (a.key != b.key) ? (a.key < b.key) : (a.idx < b.idx);
  });

  // -- gather bucket-ordered shifted positions (fp32 + fp64) + perm --------
  // schedule(static): NUMA first-touch spreads these read-shared arrays.
  out.points.resize(n);
  out.points64.resize(n);
  out.perm.resize(n);
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)n; ++i) {
    const uint32_t src = ki[i].idx;
    const vec3d p = pts[src] - outputShift;
    out.perm[i] = src;
    out.points64[i] = p;
    out.points[i] = vec3f((float)p.x, (float)p.y, (float)p.z);
  }

  // -- run-length-encode the coarse keys, chunk runs into buckets ----------
  // (replaces RunLengthEncode + scans + fillBucketRanges; same chunk math as
  // grid_buckets.cu:203-209). Serial: one pass over n plus one per bucket.
  out.occupiedCells = 0;
  size_t runStart = 0;
  for (size_t i = 1; i <= n; ++i) {
    if (i == n || (ki[i].key >> out.fineBits) != (ki[runStart].key >> out.fineBits)) {
      const size_t cnt = i - runStart;
      ++out.occupiedCells;
      const size_t k = (cnt + (size_t)maxLeaf - 1) / (size_t)maxLeaf;
      for (size_t j = 0; j < k; ++j) {
        const size_t b0 = runStart + j * (size_t)maxLeaf;
        const size_t b1 = std::min(runStart + (j + 1) * (size_t)maxLeaf,
                                   runStart + cnt);
        out.begin.push_back((int)b0);
        out.end.push_back((int)b1);
      }
      runStart = i;
    }
  }
  if (out.numBuckets() <= 0) throw std::runtime_error("no buckets built");

  // -- one tight fp32 AABB per bucket (over the SHIFTED fp32 points, like
  //    computeBucketBoxes; the BVH and node geometry derive from these) -----
  const int numBuckets = out.numBuckets();
  out.boxes.resize((size_t)numBuckets);
#pragma omp parallel for schedule(static)
  for (int b = 0; b < numBuckets; ++b) {
    box3f box; // empty
    for (int i = out.begin[b]; i < out.end[b]; ++i)
      box = box.including(box3f(out.points[i]));
    out.boxes[b] = box;
  }

  return out;
}

GridBucketsCpu buildObjectBucketsCpu(const vec3d *pts, size_t n,
                                     const box3d &bounds,
                                     vec3d outputShift, int groupSize)
{
  if (n == 0) throw std::runtime_error("no particles to bucket");
  if (groupSize < 1)
    throw std::runtime_error("buildObjectBucketsCpu: groupSize must be >= 1");
  if (n % (size_t)groupSize != 0)
    throw std::runtime_error(
        "buildObjectBucketsCpu: n (" + std::to_string(n) +
        ") not divisible by groupSize (" + std::to_string(groupSize) +
        "); object mode needs contiguous, evenly divisible groups");
  if (n > (size_t)std::numeric_limits<int>::max())
    throw std::runtime_error("particle count exceeds int-sized index range");

  GridBucketsCpu out;
  out.bounds = bounds;
  out.outputShift = outputShift;
  out.cellEdge = 0.0;                  // no grid layer in this mode
  out.maxParticlesPerBucket = groupSize;

  // Identity permutation: objects keep input order (each object's points are
  // already a contiguous run). schedule(static): NUMA first-touch.
  out.points.resize(n);
  out.points64.resize(n);
  out.perm.resize(n);
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)n; ++i) {
    const vec3d p = pts[i] - outputShift;
    out.perm[i] = (uint32_t)i;
    out.points64[i] = p;
    out.points[i] = vec3f((float)p.x, (float)p.y, (float)p.z);
  }

  const int numBuckets = (int)(n / (size_t)groupSize);
  out.begin.resize((size_t)numBuckets);
  out.end.resize((size_t)numBuckets);
  out.boxes.resize((size_t)numBuckets);
#pragma omp parallel for schedule(static)
  for (int b = 0; b < numBuckets; ++b) {
    out.begin[b] = b * groupSize;
    out.end[b] = (b + 1) * groupSize;
    box3f box; // empty
    for (int i = out.begin[b]; i < out.end[b]; ++i)
      box = box.including(box3f(out.points[i]));
    out.boxes[b] = box;
  }

  return out;
}

} // namespace tccpu
