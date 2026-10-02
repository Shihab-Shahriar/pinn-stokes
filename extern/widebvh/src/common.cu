// SPDX-License-Identifier: Apache-2.0
#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace util {

  // ---------------------------------------------------------------------
  // thrust functors (host+device so thrust can use them either way)
  // ---------------------------------------------------------------------
  namespace {
    struct PointToBox3d {
      __host__ __device__ cuBQL::box3d operator()(const cuBQL::vec3d &p) const
      { return cuBQL::box3d(p); }
    };
    struct BoxUnion3d {
      __host__ __device__
      cuBQL::box3d operator()(const cuBQL::box3d &a, const cuBQL::box3d &b) const
      { return a.including(b); }
    };
    struct Translate {
      cuBQL::vec3d shift;
      __host__ __device__ cuBQL::vec3d operator()(const cuBQL::vec3d &p) const
      { return p - shift; }
    };
    struct ToFloat {
      __host__ __device__ cuBQL::vec3f operator()(const cuBQL::vec3d &p) const
      {
        cuBQL::vec3f r;
        r.x = (float)p.x; r.y = (float)p.y; r.z = (float)p.z;
        return r;
      }
    };
    struct PointToBox3f {
      __host__ __device__ cuBQL::box3f operator()(const cuBQL::vec3f &p) const
      { return cuBQL::box3f(p); }
    };

    // median of an already-sorted vector (interpolated for even counts)
    double median(const std::vector<double> &sorted)
    {
      const size_t n = sorted.size();
      if (n == 0) return 0.0;
      return (n & 1) ? sorted[n / 2]
                     : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }

    // p-th percentile of an already-sorted vector (nearest-rank)
    double percentile(const std::vector<double> &sorted, double p)
    {
      const size_t n = sorted.size();
      if (n == 0) return 0.0;
      size_t rank = (size_t)std::ceil(p / 100.0 * (double)n);
      if (rank > 0) --rank;          // 1-based rank -> 0-based index
      if (rank >= n) rank = n - 1;
      return sorted[rank];
    }
  } // anonymous namespace

  // ---------------------------------------------------------------------
  // I/O
  // ---------------------------------------------------------------------
  std::vector<cuBQL::vec3d> loadPointsFP64(const std::string &path)
  {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
      throw std::runtime_error("could not open input file '" + path + "'");

    const std::streamsize bytes = in.tellg();
    in.seekg(0);

    const size_t stride = 3 * sizeof(double); // x/y/z coordinate blocks
    if (bytes <= 0 || (size_t)bytes % stride != 0)
      throw std::runtime_error("file '" + path +
                               "' size is not a multiple of 24 bytes "
                               "(expected raw fp64 x/y/z coordinate blocks)");

    const size_t n = (size_t)bytes / stride;
    const size_t totalDoubles = 3 * n;
    std::vector<double> raw(totalDoubles);
    in.read(reinterpret_cast<char *>(raw.data()),
            (std::streamsize)(totalDoubles * sizeof(double)));
    if (!in)
      throw std::runtime_error("short read while loading '" + path + "'");

    std::vector<cuBQL::vec3d> pts(n);
    const double *x = raw.data();
    const double *y = raw.data() + n;
    const double *z = raw.data() + 2 * n;
    for (size_t i = 0; i < n; ++i)
      pts[i] = cuBQL::vec3d(x[i], y[i], z[i]);
    return pts;
  }

  // ---------------------------------------------------------------------
  // bulk geometry (thrust)
  // ---------------------------------------------------------------------
  cuBQL::box3d computeBounds(const cuBQL::vec3d *d_points, size_t n)
  {
    thrust::device_ptr<const cuBQL::vec3d> p(d_points);
    // empty box is the identity element for the box-union reduction
    return thrust::transform_reduce(p, p + n,
                                    PointToBox3d(),
                                    cuBQL::box3d(),
                                    BoxUnion3d());
  }

  void translatePoints(cuBQL::vec3d *d_points, size_t n, cuBQL::vec3d shift)
  {
    thrust::device_ptr<cuBQL::vec3d> p(d_points);
    thrust::transform(p, p + n, p, Translate{shift});
  }

  void convertToFloat(const cuBQL::vec3d *d_in, cuBQL::vec3f *d_out, size_t n)
  {
    thrust::device_ptr<const cuBQL::vec3d> in(d_in);
    thrust::device_ptr<cuBQL::vec3f>       out(d_out);
    thrust::transform(in, in + n, out, ToFloat());
  }

  void pointsToBoxes(const cuBQL::vec3f *d_points, cuBQL::box3f *d_boxes, size_t n)
  {
    thrust::device_ptr<const cuBQL::vec3f> in(d_points);
    thrust::device_ptr<cuBQL::box3f>       out(d_boxes);
    thrust::transform(in, in + n, out, PointToBox3f());
  }

  // ---------------------------------------------------------------------
  // tree analysis
  // ---------------------------------------------------------------------
  TreeMetrics computeTreeMetrics(const cuBQL::bvh3f &bvh)
  {
    using Node = cuBQL::bvh3f::Node;

    // The node array is small (~2x #leaves); pull it to the host and walk it.
    std::vector<Node> nodes(bvh.numNodes);
    cudaMemcpy(nodes.data(), bvh.nodes, bvh.numNodes * sizeof(Node),
               cudaMemcpyDeviceToHost);

    TreeMetrics m;
    m.numNodes = bvh.numNodes;
    if (bvh.numNodes > 0) {
      const cuBQL::box3f &rb = nodes[0].bounds;
      m.rootVolume = ((double)rb.upper.x - rb.lower.x)
                   * ((double)rb.upper.y - rb.lower.y)
                   * ((double)rb.upper.z - rb.lower.z);
    }

    long long depthSum = 0;
    // Iterative DFS from the root (node 0). Walking from the root visits
    // only reachable nodes, so the always-unused node 1 is skipped, and we
    // get each leaf's depth for free.
    std::vector<std::pair<uint32_t, int>> stack;
    stack.reserve(64);
    if (bvh.numNodes > 0) stack.push_back({0u, 0});

    while (!stack.empty()) {
      auto [nodeID, depth] = stack.back();
      stack.pop_back();
      const auto &adm = nodes[nodeID].admin;

      if (adm.count != 0) {
        // leaf
        const int sz = (int)adm.count;
        m.numLeaves++;
        m.numPrims += (uint64_t)sz;
        m.leafSizes.push_back(sz);

        // spatial size of this leaf = volume of its bounding box
        const cuBQL::box3f &bb = nodes[nodeID].bounds;
        const double ex = (double)bb.upper.x - bb.lower.x;
        const double ey = (double)bb.upper.y - bb.lower.y;
        const double ez = (double)bb.upper.z - bb.lower.z;
        const double vol = ex * ey * ez;
        m.leafVolumes.push_back(vol);
        m.totalLeafVolume += vol;
        if (vol == 0.0) m.numZeroVolLeaves++;

        if (m.numLeaves == 1) { m.minLeafDepth = m.maxLeafDepth = depth; }
        m.minLeafDepth = std::min(m.minLeafDepth, depth);
        m.maxLeafDepth = std::max(m.maxLeafDepth, depth);
        depthSum += depth;
      } else {
        // inner node: two children at offset and offset+1
        m.numInner++;
        const uint32_t off = (uint32_t)adm.offset;
        stack.push_back({off + 0, depth + 1});
        stack.push_back({off + 1, depth + 1});
      }
    }

    if (!m.leafSizes.empty()) {
      std::sort(m.leafSizes.begin(), m.leafSizes.end());
      const size_t L = m.leafSizes.size();
      m.minLeaf = m.leafSizes.front();
      m.maxLeaf = m.leafSizes.back();
      m.medianLeaf = (L & 1)
        ? (double)m.leafSizes[L / 2]
        : 0.5 * (m.leafSizes[L / 2 - 1] + m.leafSizes[L / 2]);
      m.meanLeaf      = (double)m.numPrims / (double)L;
      m.meanLeafDepth = (double)depthSum / (double)L;

      std::sort(m.leafVolumes.begin(), m.leafVolumes.end());
      m.minLeafVolume    = m.leafVolumes.front();
      m.maxLeafVolume    = m.leafVolumes.back();
      m.medianLeafVolume = median(m.leafVolumes);
      m.p99LeafVolume    = percentile(m.leafVolumes, 99.0);
      m.meanLeafVolume   = m.totalLeafVolume / (double)L;
    }
    return m;
  }

  // ---------------------------------------------------------------------
  // reporting
  // ---------------------------------------------------------------------
  static std::string bar(double frac, int width = 34)
  {
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    return std::string((size_t)std::lround(frac * width), '#');
  }

  void printTreeMetrics(const TreeMetrics &m, int leafThreshold)
  {
    printf("\n==================== BVH tree metrics ====================\n");
    printf("nodes (total)       : %u\n", m.numNodes);
    printf("  inner nodes       : %u\n", m.numInner);
    printf("  leaf  nodes       : %u\n", m.numLeaves);
    printf("particles in leaves : %llu\n", (unsigned long long)m.numPrims);
    printf("leaf depth          : min %d, max %d, mean %.2f\n",
           m.minLeafDepth, m.maxLeafDepth, m.meanLeafDepth);

    if (m.numLeaves == 0) { printf("(no leaves)\n"); return; }

    printf("\n-- (a) leaf occupancy (particles per leaf), threshold=%d --\n",
           leafThreshold);
    printf("  min    : %d\n", m.minLeaf);
    printf("  median : %.1f\n", m.medianLeaf);
    printf("  mean   : %.2f\n", m.meanLeaf);
    printf("  max    : %d\n", m.maxLeaf);
    printf("  >>> max / median = %.2fx  (%d vs %.1f)\n",
           m.medianLeaf > 0 ? m.maxLeaf / m.medianLeaf : 0.0,
           m.maxLeaf, m.medianLeaf);

    // (b) histogram, binned in steps of 8 over [1..64], plus a >64 overflow.
    const int BIN_W = 8;
    const int NBINS = 8;                 // covers leaf sizes 1..64
    std::vector<long long> leafBin(NBINS + 1, 0), primBin(NBINS + 1, 0);
    for (int s : m.leafSizes) {
      int b = (s > NBINS * BIN_W) ? NBINS : (s - 1) / BIN_W;
      if (b < 0) b = 0;
      leafBin[b]++;
      primBin[b] += s;
    }

    printf("\n-- (b) particles-per-leaf distribution --\n");
    printf("  leaf size    #leaves   %%leaves   #particles  %%parts\n");
    for (int b = 0; b <= NBINS; ++b) {
      if (leafBin[b] == 0 && b != NBINS) continue;
      char label[16];
      if (b == NBINS) snprintf(label, sizeof(label), "  >%d", NBINS * BIN_W);
      else snprintf(label, sizeof(label), "%2d-%2d",
                    b * BIN_W + 1, b * BIN_W + BIN_W);
      const double fLeaf = (double)leafBin[b] / (double)m.numLeaves;
      const double fPart = (double)primBin[b] / (double)m.numPrims;
      printf("  %-7s  %9lld  %6.1f%%   %10lld  %5.1f%%  %s\n",
             label, leafBin[b], 100.0 * fLeaf,
             primBin[b], 100.0 * fPart, bar(fLeaf).c_str());
    }

    // (c) leaf spatial size = volume of each leaf's bounding box
    printf("\n-- (c) leaf spatial size (bounding-box volume) --\n");
    printf("  total leaf volume : %.6g\n", m.totalLeafVolume);
    if (m.rootVolume > 0)
      printf("  domain box volume : %.6g  (leaves sum to %.2f%% of it)\n",
             m.rootVolume, 100.0 * m.totalLeafVolume / m.rootVolume);
    printf("  min    : %.6g\n", m.minLeafVolume);
    printf("  median : %.6g\n", m.medianLeafVolume);
    printf("  mean   : %.6g\n", m.meanLeafVolume);
    printf("  p99    : %.6g\n", m.p99LeafVolume);
    printf("  max    : %.6g\n", m.maxLeafVolume);
    if (m.numZeroVolLeaves)
      printf("  (%u leaves have zero volume: single/coincident points)\n",
             m.numZeroVolLeaves);
    printf("==========================================================\n");
  }

} // namespace util
