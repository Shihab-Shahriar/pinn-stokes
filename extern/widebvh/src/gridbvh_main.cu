// SPDX-License-Identifier: Apache-2.0
//
// Driver: build a BVH whose LEAVES are constrained on BOTH (a) max particles
// per leaf and (b) max leaf edge length. We construct the leaves ("buckets")
// ourselves instead of letting cuBQL choose them:
//
//   1. bucket particles into a uniform grid of cubes (edge h) -> caps max edge,
//   2. order particles inside each occupied cell by a fine Morton code,
//   3. count-chunk each cell's run into contiguous pieces of <= maxLeaf,
//   4. each piece = one primitive: compute its tight AABB (+ placeholder P2M),
//   5. build a cuBQL BVH over the per-bucket boxes with leaf-threshold 1, so
//      every BVH leaf is exactly one of our controlled buckets.
//
// All bulk work runs on the GPU via CUB / thrust. Positions are converted to
// fp32 for the BVH (per request); fp64 is used only for the global bounds so
// the grid origin is accurate before we translate into a small-magnitude frame.
//
// Usage: gridbvh_driver [path-to-bin] [h] [maxLeafParticles]
//   defaults: two_ball/center_timestep_50.bin   10   64
#include "common.cuh"
#include "grid_buckets.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess)                                                \
      throw std::runtime_error(std::string("CUDA error (") + #call +      \
                               "): " + cudaGetErrorString(_e));           \
  } while (0)

namespace {

// ---------------------------------------------------------------------------
// host-side metric helpers
// ---------------------------------------------------------------------------
static double median_sorted(const std::vector<double> &s)
{
  const size_t n = s.size();
  if (n == 0) return 0.0;
  return (n & 1) ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}
static double percentile_sorted(const std::vector<double> &s, double p)
{
  const size_t n = s.size();
  if (n == 0) return 0.0;
  size_t r = (size_t)std::ceil(p / 100.0 * (double)n);
  if (r > 0) --r;
  if (r >= n) r = n - 1;
  return s[r];
}
static double maxEdge(const cuBQL::box3f &b)
{
  const float ex = b.upper.x - b.lower.x;
  const float ey = b.upper.y - b.lower.y;
  const float ez = b.upper.z - b.lower.z;
  return (double)std::max(ex, std::max(ey, ez));
}

} // anonymous namespace

int main(int argc, char **argv)
{
  const std::string path =
    (argc > 1) ? argv[1] : "two_ball/center_timestep_50.bin";
  const double h       = (argc > 2) ? std::atof(argv[2]) : 10.0;
  const int    maxLeaf = (argc > 3) ? std::atoi(argv[3]) : 64;

  try {
    if (h <= 0.0)    throw std::runtime_error("h must be > 0");
    if (maxLeaf < 1) throw std::runtime_error("maxLeafParticles must be >= 1");

    // -- 1. load fp64 particles, upload ----------------------------------
    printf("# loading '%s'\n", path.c_str());
    std::vector<cuBQL::vec3d> hostPts = util::loadPointsFP64(path);
    const size_t N = hostPts.size();
    printf("# loaded %zu particles (fp64); h=%.6g, maxLeaf=%d\n", N, h, maxLeaf);
    if (N == 0) throw std::runtime_error("no particles in input");

    thrust::device_vector<cuBQL::vec3d> d_ptsD(hostPts);

    // -- 2. fp64 bounds -> grid; translate to origin; drop to fp32 -------
    const cuBQL::box3d gb = util::computeBounds(util::devicePtr(d_ptsD), N);
    const cuBQL::vec3d origin = gb.lower;
    const cuBQL::vec3d extent = gb.size();
    printf("# domain box: lower (%.6g, %.6g, %.6g)  size (%.6g, %.6g, %.6g)\n",
           gb.lower.x, gb.lower.y, gb.lower.z, extent.x, extent.y, extent.z);

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::high_resolution_clock::now();

    util::GridBuckets buckets =
      util::buildGridBuckets(util::devicePtr(d_ptsD), N, gb, origin, h, maxLeaf);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t1 = std::chrono::high_resolution_clock::now();

    d_ptsD.clear();
    d_ptsD.shrink_to_fit();

    const long long B_total = (long long)buckets.numBuckets();
    printf("# uniform grid: %u x %u x %u = %llu cells (edge h=%.6g)\n",
           buckets.nx, buckets.ny, buckets.nz,
           (unsigned long long)buckets.totalCells, h);
    printf("# Morton key: coarse=%d bits (%d/axis) + fine=%d bits (%d/axis)\n",
           buckets.coarseBits, buckets.coarseBitsPerAxis,
           buckets.fineBits, buckets.fineBitsPerAxis);

    // -- 9. build the cuBQL BVH over the bucket boxes (leaf-threshold 1) -
    cuBQL::bvh3f bvh;
    cuBQL::gpuBuilder(bvh, util::devicePtr(buckets.boxes), buckets.numBuckets(),
                      cuBQL::BuildConfig(1));
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double msBucket =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double msBuild =
      std::chrono::duration<double, std::milli>(t2 - t1).count();

    // -- 10. verification metrics ---------------------------------------
    std::vector<int>          counts((size_t)B_total);
    std::vector<cuBQL::box3f> boxes((size_t)B_total);
    CUDA_CHECK(cudaMemcpy(counts.data(), util::devicePtr(buckets.count),
                          B_total * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(boxes.data(), util::devicePtr(buckets.boxes),
                          B_total * sizeof(cuBQL::box3f), cudaMemcpyDeviceToHost));

    long long sumParticles = 0;
    int       maxBucket = 0, minBucket = counts.empty() ? 0 : counts[0];
    std::vector<double> cd; cd.reserve(counts.size());
    for (int c : counts) {
      sumParticles += c;
      maxBucket = std::max(maxBucket, c);
      minBucket = std::min(minBucket, c);
      cd.push_back((double)c);
    }
    std::sort(cd.begin(), cd.end());

    std::vector<double> ed; ed.reserve(boxes.size());
    double maxEdgeAll = 0.0; int edgeViolations = 0, zeroVol = 0;
    const double tol = 1e-4 * h;
    for (const auto &b : boxes) {
      const double e = maxEdge(b);
      ed.push_back(e);
      maxEdgeAll = std::max(maxEdgeAll, e);
      if (e > h + tol) ++edgeViolations;
      if (e == 0.0)    ++zeroVol;
    }
    std::sort(ed.begin(), ed.end());

    printf("\n==================== grid-bucketed BVH ====================\n");
    printf("timing            : bucketing %.2f ms | BVH build %.2f ms\n",
           msBucket, msBuild);
    printf("grid cells        : %llu total, %d occupied (%.3f%%)\n",
           (unsigned long long)buckets.totalCells, buckets.occupiedCells,
           100.0 * (double)buckets.occupiedCells / (double)buckets.totalCells);
    printf("buckets (= prims) : %lld\n", B_total);

    printf("\n-- (a) particles per bucket  (max must be <= %d) --\n", maxLeaf);
    printf("  min %d | median %.1f | mean %.2f | p99 %.0f | max %d\n",
           minBucket, median_sorted(cd),
           (double)sumParticles / (double)B_total,
           percentile_sorted(cd, 99.0), maxBucket);
    printf("  >>> constraint (a) %s  (max %d %s %d)\n",
           (maxBucket <= maxLeaf) ? "PASS" : "FAIL",
           maxBucket, (maxBucket <= maxLeaf) ? "<=" : ">", maxLeaf);

    printf("\n-- (b) bucket max edge length  (max must be <= %.6g) --\n", h);
    printf("  min %.6g | median %.6g | mean %.6g | p99 %.6g | max %.6g\n",
           ed.front(), median_sorted(ed),
           std::accumulate(ed.begin(), ed.end(), 0.0) / (double)ed.size(),
           percentile_sorted(ed, 99.0), ed.back());
    printf("  zero-volume buckets : %d\n", zeroVol);
    printf("  edge violations     : %d\n", edgeViolations);
    printf("  >>> constraint (b) %s  (max %.6g %s %.6g)\n",
           (edgeViolations == 0) ? "PASS" : "FAIL",
           maxEdgeAll, (maxEdgeAll <= h + tol) ? "<=" : ">", h);

    printf("\n-- conservation --\n");
    printf("  particles in buckets : %lld  (input %zu)\n", sumParticles, N);
    printf("  >>> conservation %s\n",
           ((size_t)sumParticles == N) ? "PASS" : "FAIL");

    printf("\n-- upper BVH (cuBQL, leaf-threshold 1) --\n");
    printf("  numPrims %u  (%s B_total = %lld)\n", bvh.numPrims,
           (bvh.numPrims == (uint32_t)B_total) ? "==" : "!=", B_total);
    printf("===========================================================\n");

    // Full BVH tree metrics via the shared analyzer. NOTE: each leaf holds
    // exactly one *bucket* primitive (leaf-threshold 1), so "particles per
    // leaf" below counts buckets-per-leaf (expect all == 1) and the leaf
    // volumes are bucket-box volumes, not raw-particle leaves.
    util::TreeMetrics tm = util::computeTreeMetrics(bvh);
    if (tm.numPrims != (uint64_t)B_total)
      printf("! warning: BVH leaves reference %llu prims but B_total=%lld\n",
             (unsigned long long)tm.numPrims, B_total);
    util::printTreeMetrics(tm, /*leafThreshold=*/1);

    cuBQL::cuda::free(bvh);
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
