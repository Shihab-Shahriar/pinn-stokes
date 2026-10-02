// SPDX-License-Identifier: Apache-2.0
//
// Debug tool: run the SAME cloud through the treecode engine twice under two
// different TC_PATH tokens and diff the full output arrays over ALL targets.
// Built to hunt the direct-warpspec 4M single-target corruption: the signed
// per-component sum of differences discriminates a contribution that MOVED
// between targets (sum ~ 0: binding/indexing bug) from one that was CHANGED
// or dropped for one target (sum = that target's error).
//
// usage: dbg_path_diff <mac> <maxLeaf> <cellEdge> <cloud.bin> [pathA] [pathB]

#include "treecode.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/fill.h>

using TC = Treecode<mp::BaryStokes>;

int main(int argc, char **argv)
{
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <mac> <maxLeaf> <cellEdge> <cloud.bin> "
                 "[pathA=split-warp] [pathB=direct-warpspec]\n", argv[0]);
    return 1;
  }
  const float  mac      = (float)std::atof(argv[1]);
  const int    maxLeaf  = std::atoi(argv[2]);
  const double cellEdge = std::atof(argv[3]);
  const char  *cloud    = argv[4];
  const char  *pathA    = argc > 5 ? argv[5] : "split-warp";
  const char  *pathB    = argc > 6 ? argv[6] : "direct-warpspec";

  try {
    CUDA_CHECK(cudaFree(0));

    std::vector<vec3d> hostPts = util::loadPointsFP64(cloud);
    const size_t N = hostPts.size();
    if (!N) throw std::runtime_error("no particles");
    std::printf("cloud=%s N=%zu mac=%g leaf=%d cell=%g\n",
                cloud, N, (double)mac, maxLeaf, cellEdge);

    thrust::device_vector<vec3d> d_pts(hostPts);
    thrust::device_vector<vec3f> d_force(N, vec3f(0.f, 0.f, -9.81f));

    TC::Config cfg;
    cfg.mac = mac;
    cfg.maxLeaf = maxLeaf;
    cfg.cellEdge = cellEdge;

    auto runPath = [&](const char *path, std::vector<double> &out) {
      setenv("TC_PATH", path, 1);
      TC tc(cfg);
      thrust::device_vector<double> d_out((size_t)3 * N);
      tc.apply(util::devicePtr(d_pts), N, util::devicePtr(d_force),
               util::devicePtr(d_out), /*will_reuse_tree=*/false);
      CUDA_CHECK(cudaDeviceSynchronize());
      out.resize((size_t)3 * N);
      CUDA_CHECK(cudaMemcpy(out.data(), util::devicePtr(d_out),
                            out.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
    };

    std::vector<double> uA, uB;
    runPath(pathA, uA);
    runPath(pathB, uB);

    // Outputs are component-major in ORIGINAL target order: u[c*N + i].
    struct Bad { size_t idx; double diff, mag; };
    std::vector<Bad> bad;
    double sum[3] = {0, 0, 0};      // signed sum of (B - A) per component
    double sumAbs = 0, maxDiff = 0;
    size_t nDiff6 = 0, nDiff9 = 0;  // rel diff > 1e-6 / 1e-9
    for (size_t i = 0; i < N; ++i) {
      double d2 = 0, m2 = 0;
      for (int c = 0; c < 3; ++c) {
        const double a = uA[c * N + i], b = uB[c * N + i];
        sum[c] += b - a;
        d2 += (b - a) * (b - a);
        m2 += a * a;
      }
      const double d = std::sqrt(d2), m = std::sqrt(m2);
      sumAbs += d;
      maxDiff = std::max(maxDiff, d);
      const double rel = d / std::max(m, 1e-300);
      if (rel > 1e-9) ++nDiff9;
      if (rel > 1e-6) {
        ++nDiff6;
        bad.push_back({i, d, m});
      }
    }
    std::sort(bad.begin(), bad.end(),
              [](const Bad &x, const Bad &y) { return x.diff > y.diff; });

    std::printf("\npathA=%s pathB=%s\n", pathA, pathB);
    std::printf("targets rel-differing >1e-9: %zu   >1e-6: %zu (of %zu)\n",
                nDiff9, nDiff6, N);
    std::printf("max |diff| = %.6e   sum|diff| = %.6e\n", maxDiff, sumAbs);
    std::printf("signed sum (B-A): (%.6e, %.6e, %.6e)\n",
                sum[0], sum[1], sum[2]);
    const size_t nShow = std::min<size_t>(bad.size(), 20);
    for (size_t k = 0; k < nShow; ++k) {
      const size_t i = bad[k].idx;
      std::printf("  #%2zu idx=%zu |d|=%.6e |u|=%.6e pos=(%.9g,%.9g,%.9g)\n",
                  k, i, bad[k].diff, bad[k].mag,
                  hostPts[i].x, hostPts[i].y, hostPts[i].z);
      for (int c = 0; c < 3; ++c)
        std::printf("      c%d  A=%.17g  B=%.17g\n",
                    c, uA[c * N + i], uB[c * N + i]);
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
