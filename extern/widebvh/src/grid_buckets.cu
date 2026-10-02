// SPDX-License-Identifier: Apache-2.0
#include "grid_buckets.cuh"
#include "hilbert_sfc.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess)                                                \
      throw std::runtime_error(std::string("CUDA error (") + #call +      \
                               "): " + cudaGetErrorString(_e));           \
  } while (0)

namespace util {
namespace {

__host__ __device__ inline uint64_t expandBits21(uint64_t v)
{
  v &= 0x1fffffull;
  v = (v | v << 32) & 0x1f00000000ffffull;
  v = (v | v << 16) & 0x1f0000ff0000ffull;
  v = (v | v <<  8) & 0x100f00f00f00f00full;
  v = (v | v <<  4) & 0x10c30c30c30c30c3ull;
  v = (v | v <<  2) & 0x1249249249249249ull;
  return v;
}

__host__ __device__ inline uint64_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
  return (expandBits21(x) << 2) | (expandBits21(y) << 1) | expandBits21(z);
}

// Composite grid bucket key: (coarseCellMorton << fineBits) | fineCurve.
// Hilbert=false orders each cell's particles by a fine Morton curve (the
// bit-anchored legacy grid scheme); Hilbert=true orders them along a per-cell
// Hilbert curve (local iHilbert21) so dense-cell chunks stay compact instead of
// Morton-stacked. Everything but the fine term is identical, so the grid path
// is bit-for-bit unchanged.
template <bool Hilbert>
struct MakeCompositeKey {
  cuBQL::vec3d lower;
  double h;
  uint32_t nx, ny, nz;
  int B;
  int fineBits;

  __host__ __device__ uint64_t operator()(const cuBQL::vec3d &p) const
  {
    double gx = floor((p.x - lower.x) / h);
    double gy = floor((p.y - lower.y) / h);
    double gz = floor((p.z - lower.z) / h);
    if (gx < 0.0) gx = 0.0;
    if (gy < 0.0) gy = 0.0;
    if (gz < 0.0) gz = 0.0;
    uint32_t cx = (uint32_t)gx;
    uint32_t cy = (uint32_t)gy;
    uint32_t cz = (uint32_t)gz;
    if (cx >= nx) cx = nx - 1;
    if (cy >= ny) cy = ny - 1;
    if (cz >= nz) cz = nz - 1;
    uint64_t coarse = morton3D(cx, cy, cz);

    uint64_t fine = 0;
    if (B > 0) {
      const double scale = (double)(1u << B);
      const uint32_t qmax = (1u << B) - 1u;
      uint32_t qx = (uint32_t)floor(((p.x - lower.x) - (double)cx * h) / h * scale);
      uint32_t qy = (uint32_t)floor(((p.y - lower.y) - (double)cy * h) / h * scale);
      uint32_t qz = (uint32_t)floor(((p.z - lower.z) - (double)cz * h) / h * scale);
      if (qx > qmax) qx = qmax;
      if (qy > qmax) qy = qmax;
      if (qz > qmax) qz = qmax;
      fine = Hilbert ? iHilbert21(qx, qy, qz) : morton3D(qx, qy, qz);
    }
    return (coarse << fineBits) | fine;
  }
};

// Global Hilbert key over the (cubized) SFC box: normalize each axis to
// [0, 2^21) the way cornerstone's sfc3D does (mx = 2^21 / (upper-lower);
// ix = clamp(floor(x*mx) - lower*mx, 0, 2^21-1)), then iHilbert21. Backs the
// grid-free `hilbert` bucketizer's argsort.
struct MakeHilbertKey {
  cuBQL::vec3d lower;
  cuBQL::vec3d mx;  // per-axis 2^21 / extent

  __host__ __device__ uint64_t operator()(const cuBQL::vec3d &p) const
  {
    constexpr long long qmax = (1 << 21) - 1;
    // sfc3D: truncate the double difference once (floor(x*mx) - xmin*mx).
    long long ix = (long long)(floor(p.x * mx.x) - lower.x * mx.x);
    long long iy = (long long)(floor(p.y * mx.y) - lower.y * mx.y);
    long long iz = (long long)(floor(p.z * mx.z) - lower.z * mx.z);
    if (ix < 0) ix = 0; else if (ix > qmax) ix = qmax;
    if (iy < 0) iy = 0; else if (iy > qmax) iy = qmax;
    if (iz < 0) iz = 0; else if (iz > qmax) iz = qmax;
    return iHilbert21((unsigned)ix, (unsigned)iy, (unsigned)iz);
  }
};

struct CoarseShift {
  int fineBits;
  __host__ __device__ uint64_t operator()(uint64_t key) const
  { return key >> fineBits; }
};

struct CeilDiv {
  int d;
  __host__ __device__ int operator()(int n) const
  { return (n + d - 1) / d; }
};

struct GatherShiftedFloat {
  const cuBQL::vec3d *points;
  cuBQL::vec3d shift;

  __host__ __device__ cuBQL::vec3f operator()(uint32_t idx) const
  {
    const cuBQL::vec3d p = points[idx] - shift;
    return cuBQL::vec3f((float)p.x, (float)p.y, (float)p.z);
  }
};

// fp64 twin of GatherShiftedFloat: same centered frame (shift = bounds.center()),
// no fp32 cast. Backs the fp64-geometry P2M/M2P (accurate source/target coords).
struct GatherShiftedDouble {
  const cuBQL::vec3d *points;
  cuBQL::vec3d shift;

  __host__ __device__ cuBQL::vec3d operator()(uint32_t idx) const
  {
    return points[idx] - shift;
  }
};

struct PointToBox3f {
  __host__ __device__ cuBQL::box3f operator()(const cuBQL::vec3f &p) const
  { return cuBQL::box3f(p); }
};

// fp64 input point -> degenerate shifted fp32 box, for the structured object
// bucketizer's per-bucket AABB reduce straight off d_points (no stored points).
// Matches GatherShiftedFloat's centered-frame cast so the boxes are identical to
// buildObjectBuckets'.
struct PointToShiftedBox3f {
  cuBQL::vec3d shift;
  __host__ __device__ cuBQL::box3f operator()(const cuBQL::vec3d &p) const
  {
    const cuBQL::vec3d q = p - shift;
    return cuBQL::box3f(cuBQL::vec3f((float)q.x, (float)q.y, (float)q.z));
  }
};

struct BoxUnion3f {
  __host__ __device__
  cuBQL::box3f operator()(const cuBQL::box3f &a, const cuBQL::box3f &b) const
  { return a.including(b); }
};

struct Vec3fAdd {
  __host__ __device__
  cuBQL::vec3f operator()(const cuBQL::vec3f &a, const cuBQL::vec3f &b) const
  { return a + b; }
};

struct DivByCount {
  __host__ __device__
  cuBQL::vec3f operator()(const cuBQL::vec3f &s, int c) const
  {
    const float inv = (c > 0) ? 1.0f / (float)c : 0.0f;
    return cuBQL::vec3f(s.x * inv, s.y * inv, s.z * inv);
  }
};

__global__ void fillBucketRanges(int numCells,
                                 const int *cellStart,
                                 const int *cellCount,
                                 const int *firstBucketOfCell,
                                 int maxLeaf,
                                 int *bucketBegin,
                                 int *bucketEnd)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= numCells) return;
  const int start = cellStart[c];
  const int cnt = cellCount[c];
  const int base = firstBucketOfCell[c];
  const int k = (cnt + maxLeaf - 1) / maxLeaf;
  for (int j = 0; j < k; ++j) {
    const int b0 = start + j * maxLeaf;
    int b1 = start + (j + 1) * maxLeaf;
    if (b1 > start + cnt) b1 = start + cnt;
    bucketBegin[base + j] = b0;
    bucketEnd[base + j] = b1;
  }
}

uint32_t gridDimForExtent(double extent, double h)
{
  const double d = std::ceil(extent / h);
  if (!(d > 1.0)) return 1u;
  if (d > (double)std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("grid dimension overflows uint32_t; increase cell edge");
  return (uint32_t)d;
}

// Fixed-size chunk ranges over a sorted particle array: bucket i covers
// [i*maxLeaf, min(n, (i+1)*maxLeaf)).
struct ChunkBegin {
  int maxLeaf;
  __host__ __device__ int operator()(int i) const { return i * maxLeaf; }
};
struct ChunkEnd {
  int maxLeaf, n;
  __host__ __device__ int operator()(int i) const
  {
    const long long e = (long long)(i + 1) * maxLeaf;
    return (e > n) ? n : (int)e;
  }
};

// One tight fp32 AABB per bucket via CUB segmented reduce (shared by the grid
// and Hilbert builders).
void computeBucketBoxes(const thrust::device_vector<cuBQL::vec3f> &points,
                        const thrust::device_vector<int> &begin,
                        const thrust::device_vector<int> &end,
                        thrust::device_vector<cuBQL::box3f> &boxes)
{
  const int numBuckets = (int)begin.size();
  boxes.resize((size_t)numBuckets);
  auto boxIt =
    thrust::make_transform_iterator(devicePtr(points), PointToBox3f{});
  const cuBQL::box3f emptyBox;
  size_t bytes = 0;
  CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
    nullptr, bytes, boxIt, devicePtr(boxes), numBuckets,
    devicePtr(begin), devicePtr(end), BoxUnion3f{}, emptyBox));
  thrust::device_vector<char> tmp(bytes);
  CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
    devicePtr(tmp), bytes, boxIt, devicePtr(boxes), numBuckets,
    devicePtr(begin), devicePtr(end), BoxUnion3f{}, emptyBox));
}

} // anonymous namespace

GridBuckets buildGridBuckets(const cuBQL::vec3d *d_points,
                             size_t n,
                             cuBQL::box3d bounds,
                             cuBQL::vec3d outputShift,
                             double cellEdge,
                             int maxParticlesPerBucket,
                             FineCurve fineCurve)
{
  if (n == 0) throw std::runtime_error("no particles to bucket");
  if (!(cellEdge > 0.0)) throw std::runtime_error("cell edge must be > 0");
  if (maxParticlesPerBucket < 1)
    throw std::runtime_error("maxParticlesPerBucket must be >= 1");
  if (n > (size_t)std::numeric_limits<int>::max())
    throw std::runtime_error("particle count exceeds CUB int-sized API limit");

  GridBuckets out;
  out.bounds = bounds;
  out.outputShift = outputShift;
  out.cellEdge = cellEdge;
  out.maxParticlesPerBucket = maxParticlesPerBucket;

  const cuBQL::vec3d extent = bounds.size();
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
  // Morton fine keys keep the historical 10-bit/axis cap (bit-compat with all
  // grid-mode baselines). The Hilbert fine curve uses the full remaining key
  // budget (up to 20 bits/axis): with big cells a 10-bit lattice under-resolves
  // dense cores (equal keys -> input-order chunks -> stacking returns).
  const int fineCap = (fineCurve == FineCurve::Hilbert) ? 20 : 10;
  out.fineBitsPerAxis = std::min(fineCap, (64 - out.coarseBits) / 3);
  if (out.fineBitsPerAxis < 0) out.fineBitsPerAxis = 0;
  out.fineBits = 3 * out.fineBitsPerAxis;

  thrust::device_ptr<const cuBQL::vec3d> points(d_points);
  thrust::device_vector<uint64_t> key(n);
  if (fineCurve == FineCurve::Hilbert)
    thrust::transform(points, points + n, key.begin(),
                      MakeCompositeKey<true>{bounds.lower, cellEdge,
                                             out.nx, out.ny, out.nz,
                                             out.fineBitsPerAxis, out.fineBits});
  else
    thrust::transform(points, points + n, key.begin(),
                      MakeCompositeKey<false>{bounds.lower, cellEdge,
                                              out.nx, out.ny, out.nz,
                                              out.fineBitsPerAxis, out.fineBits});

  thrust::device_vector<uint32_t> idx(n);
  thrust::sequence(idx.begin(), idx.end());
  thrust::device_vector<uint64_t> keySorted(n);
  thrust::device_vector<uint32_t> idxSorted(n);
  {
    const int endBit = out.coarseBits + out.fineBits;
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, bytes, devicePtr(key), devicePtr(keySorted),
      devicePtr(idx), devicePtr(idxSorted), (int)n, 0, endBit));
    thrust::device_vector<char> tmp(bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      devicePtr(tmp), bytes, devicePtr(key), devicePtr(keySorted),
      devicePtr(idx), devicePtr(idxSorted), (int)n, 0, endBit));
  }

  out.points.resize(n);
  thrust::transform(idxSorted.begin(), idxSorted.end(), out.points.begin(),
                    GatherShiftedFloat{d_points, outputShift});

  // fp64 bucket-ordered positions (same idxSorted + shift), for the fp64-geometry
  // far field. Identical ordering to out.points, so a bucket index selects both.
  out.points64.resize(n);
  thrust::transform(idxSorted.begin(), idxSorted.end(), out.points64.begin(),
                    GatherShiftedDouble{d_points, outputShift});

  // bucket slot -> original input index (for gather/scatter back to caller order)
  out.perm = idxSorted;

  thrust::device_vector<uint64_t> coarse(n);
  thrust::transform(keySorted.begin(), keySorted.end(), coarse.begin(),
                    CoarseShift{out.fineBits});

  thrust::device_vector<uint64_t> uniqueCoarse(n);
  thrust::device_vector<int> cellCount(n);
  thrust::device_vector<int> numRuns(1);
  {
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
      nullptr, bytes, devicePtr(coarse), devicePtr(uniqueCoarse),
      devicePtr(cellCount), devicePtr(numRuns), (int)n));
    thrust::device_vector<char> tmp(bytes);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
      devicePtr(tmp), bytes, devicePtr(coarse), devicePtr(uniqueCoarse),
      devicePtr(cellCount), devicePtr(numRuns), (int)n));
  }
  out.occupiedCells = numRuns[0];
  if (out.occupiedCells <= 0) throw std::runtime_error("no occupied cells");

  thrust::device_vector<int> cellStart(out.occupiedCells);
  thrust::exclusive_scan(cellCount.begin(), cellCount.begin() + out.occupiedCells,
                         cellStart.begin());

  thrust::device_vector<int> perCellBuckets(out.occupiedCells);
  thrust::transform(cellCount.begin(), cellCount.begin() + out.occupiedCells,
                    perCellBuckets.begin(), CeilDiv{maxParticlesPerBucket});

  thrust::device_vector<int> firstBucket(out.occupiedCells);
  thrust::exclusive_scan(perCellBuckets.begin(),
                         perCellBuckets.begin() + out.occupiedCells,
                         firstBucket.begin());

  const long long numBuckets =
    thrust::reduce(perCellBuckets.begin(),
                   perCellBuckets.begin() + out.occupiedCells,
                   (long long)0);
  if (numBuckets <= 0 || numBuckets > (long long)std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("bucket count out of range");

  out.begin.resize((size_t)numBuckets);
  out.end.resize((size_t)numBuckets);
  {
    const int threads = 256;
    const int blocks = (out.occupiedCells + threads - 1) / threads;
    fillBucketRanges<<<blocks, threads>>>(
      out.occupiedCells, devicePtr(cellStart), devicePtr(cellCount),
      devicePtr(firstBucket), maxParticlesPerBucket,
      devicePtr(out.begin), devicePtr(out.end));
    CUDA_CHECK(cudaGetLastError());
  }

  computeBucketBoxes(out.points, out.begin, out.end, out.boxes);

  out.count.resize((size_t)numBuckets);
  thrust::transform(out.end.begin(), out.end.end(),
                    out.begin.begin(), out.count.begin(),
                    thrust::minus<int>());

  thrust::device_vector<cuBQL::vec3f> bucketSum((size_t)numBuckets);
  {
    const cuBQL::vec3f zero(0.f);
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
      nullptr, bytes, devicePtr(out.points), devicePtr(bucketSum),
      (int)numBuckets, devicePtr(out.begin), devicePtr(out.end),
      Vec3fAdd{}, zero));
    thrust::device_vector<char> tmp(bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
      devicePtr(tmp), bytes, devicePtr(out.points), devicePtr(bucketSum),
      (int)numBuckets, devicePtr(out.begin), devicePtr(out.end),
      Vec3fAdd{}, zero));
  }
  out.centroid.resize((size_t)numBuckets);
  thrust::transform(bucketSum.begin(), bucketSum.end(),
                    out.count.begin(), out.centroid.begin(),
                    DivByCount{});

  return out;
}

GridBuckets buildHilbertBuckets(const cuBQL::vec3d *d_points,
                                size_t n,
                                cuBQL::box3d sfcBox,
                                cuBQL::vec3d outputShift,
                                int maxParticlesPerBucket)
{
  if (n == 0) throw std::runtime_error("no particles to bucket");
  if (maxParticlesPerBucket < 1)
    throw std::runtime_error("maxParticlesPerBucket must be >= 1");
  if (n > (size_t)std::numeric_limits<int>::max())
    throw std::runtime_error("particle count exceeds CUB int-sized API limit");

  GridBuckets out;
  out.bounds = sfcBox;
  out.outputShift = outputShift;
  out.cellEdge = 0.0;                  // no grid layer in this mode
  out.maxParticlesPerBucket = maxParticlesPerBucket;

  // Global Hilbert argsort (local iHilbert21, 63-bit keys); every bucket is a
  // contiguous curve segment, so its AABB is spatially compact -- no
  // Morton-chunk stacking.
  const cuBQL::vec3d ext = sfcBox.size();
  const double cube = (double)(1u << 21);   // 2^21 lattice points per axis
  cuBQL::vec3d mx;
  mx.x = (ext.x > 0.0) ? cube / ext.x : 0.0;
  mx.y = (ext.y > 0.0) ? cube / ext.y : 0.0;
  mx.z = (ext.z > 0.0) ? cube / ext.z : 0.0;

  thrust::device_ptr<const cuBQL::vec3d> points(d_points);
  thrust::device_vector<uint64_t> key(n);
  thrust::transform(points, points + n, key.begin(),
                    MakeHilbertKey{sfcBox.lower, mx});

  thrust::device_vector<uint32_t> idx(n);
  thrust::sequence(idx.begin(), idx.end());
  thrust::device_vector<uint64_t> keySorted(n);
  out.perm.resize(n);
  {
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, bytes, devicePtr(key), devicePtr(keySorted),
      devicePtr(idx), devicePtr(out.perm), (int)n, 0, 63));
    thrust::device_vector<char> tmp(bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      devicePtr(tmp), bytes, devicePtr(key), devicePtr(keySorted),
      devicePtr(idx), devicePtr(out.perm), (int)n, 0, 63));
  }

  out.points.resize(n);
  thrust::transform(out.perm.begin(), out.perm.end(), out.points.begin(),
                    GatherShiftedFloat{d_points, outputShift});
  out.points64.resize(n);
  thrust::transform(out.perm.begin(), out.perm.end(), out.points64.begin(),
                    GatherShiftedDouble{d_points, outputShift});

  const size_t numBuckets =
    (n + (size_t)maxParticlesPerBucket - 1) / (size_t)maxParticlesPerBucket;
  out.begin.resize(numBuckets);
  out.end.resize(numBuckets);
  thrust::tabulate(out.begin.begin(), out.begin.end(),
                   ChunkBegin{maxParticlesPerBucket});
  thrust::tabulate(out.end.begin(), out.end.end(),
                   ChunkEnd{maxParticlesPerBucket, (int)n});

  computeBucketBoxes(out.points, out.begin, out.end, out.boxes);

  out.count.resize(numBuckets);
  thrust::transform(out.end.begin(), out.end.end(),
                    out.begin.begin(), out.count.begin(),
                    thrust::minus<int>());
  // centroid left empty: placeholder field with no engine consumer.
  return out;
}

GridBuckets buildObjectBuckets(const cuBQL::vec3d *d_points,
                               size_t n,
                               cuBQL::vec3d outputShift,
                               int groupSize)
{
  if (n == 0) throw std::runtime_error("no particles to bucket");
  if (groupSize < 1)
    throw std::runtime_error("buildObjectBuckets: groupSize must be >= 1");
  if (n % (size_t)groupSize != 0)
    throw std::runtime_error(
      "buildObjectBuckets: n (" + std::to_string(n) +
      ") not divisible by groupSize (" + std::to_string(groupSize) +
      "); object mode needs contiguous, evenly divisible groups");
  if (n > (size_t)std::numeric_limits<int>::max())
    throw std::runtime_error("particle count exceeds CUB int-sized API limit");

  GridBuckets out;
  out.outputShift = outputShift;
  out.cellEdge = 0.0;                  // no grid layer in this mode
  out.maxParticlesPerBucket = groupSize;

  // Identity permutation: objects keep input order. Each object's group is a
  // contiguous run already, so no sort is needed; the source BVH reorders the
  // per-object boxes spatially at build time.
  out.perm.resize(n);
  thrust::sequence(out.perm.begin(), out.perm.end());

  out.points.resize(n);
  thrust::transform(out.perm.begin(), out.perm.end(), out.points.begin(),
                    GatherShiftedFloat{d_points, outputShift});
  out.points64.resize(n);
  thrust::transform(out.perm.begin(), out.perm.end(), out.points64.begin(),
                    GatherShiftedDouble{d_points, outputShift});

  const size_t numBuckets = n / (size_t)groupSize;
  out.begin.resize(numBuckets);
  out.end.resize(numBuckets);
  thrust::tabulate(out.begin.begin(), out.begin.end(),
                   ChunkBegin{groupSize});
  thrust::tabulate(out.end.begin(), out.end.end(),
                   ChunkEnd{groupSize, (int)n});

  computeBucketBoxes(out.points, out.begin, out.end, out.boxes);

  out.count.resize(numBuckets);
  thrust::transform(out.end.begin(), out.end.end(),
                    out.begin.begin(), out.count.begin(),
                    thrust::minus<int>());
  // centroid left empty: placeholder field with no engine consumer.
  return out;
}


GridBuckets buildStructuredObjectBuckets(const cuBQL::vec3d *d_points,
                                         size_t n,
                                         cuBQL::vec3d outputShift,
                                         int groupSize)
{
  if (n == 0) throw std::runtime_error("no particles to bucket");
  if (groupSize < 1)
    throw std::runtime_error("buildStructuredObjectBuckets: groupSize must be >= 1");
  if (n % (size_t)groupSize != 0)
    throw std::runtime_error(
      "buildStructuredObjectBuckets: n (" + std::to_string(n) +
      ") not divisible by groupSize (" + std::to_string(groupSize) + ")");
  if (n > (size_t)std::numeric_limits<int>::max())
    throw std::runtime_error("particle count exceeds CUB int-sized API limit");

  GridBuckets out;
  out.outputShift = outputShift;
  out.cellEdge = 0.0;
  out.maxParticlesPerBucket = groupSize;

  // Identity permutation: object k is the input run [k*groupSize,(k+1)*groupSize).
  out.perm.resize(n);
  thrust::sequence(out.perm.begin(), out.perm.end());

  // points / points64 are deliberately LEFT EMPTY -- the whole point of this
  // bucketizer is to not store the per-point source positions.

  const size_t numBuckets = n / (size_t)groupSize;
  out.begin.resize(numBuckets);
  out.end.resize(numBuckets);
  thrust::tabulate(out.begin.begin(), out.begin.end(), ChunkBegin{groupSize});
  thrust::tabulate(out.end.begin(), out.end.end(), ChunkEnd{groupSize, (int)n});

  // Per-bucket AABBs reduced straight off d_points (shifted fp32 boxes). This is
  // the same box each particle would contribute in buildObjectBuckets, so the
  // BVH built from these boxes is bit-identical -- only the positions are dropped.
  out.boxes.resize(numBuckets);
  auto boxIt = thrust::make_transform_iterator(d_points,
                                               PointToShiftedBox3f{outputShift});
  const cuBQL::box3f emptyBox;
  size_t bytes = 0;
  CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
    nullptr, bytes, boxIt, devicePtr(out.boxes), (int)numBuckets,
    devicePtr(out.begin), devicePtr(out.end), BoxUnion3f{}, emptyBox));
  thrust::device_vector<char> tmp(bytes);
  CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
    devicePtr(tmp), bytes, boxIt, devicePtr(out.boxes), (int)numBuckets,
    devicePtr(out.begin), devicePtr(out.end), BoxUnion3f{}, emptyBox));

  out.count.resize(numBuckets);
  thrust::transform(out.end.begin(), out.end.end(),
                    out.begin.begin(), out.count.begin(),
                    thrust::minus<int>());
  return out;
}

} // namespace util
