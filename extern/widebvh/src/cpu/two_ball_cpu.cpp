// SPDX-License-Identifier: Apache-2.0
//
// Driver: CPU-only Stokeslet treecode on a two_ball/distros dataset. CPU port
// of src/two_ball.cu minus CUDA, VRAM sampling, and reapply: load fp64
// points, set a uniform gravity force, run TreecodeCpu (build/upward/
// evaluate), validate against an fp64 direct sum on sampled targets, and
// print the timing / error report.
//
// usage: two_ball_cpu <mac> [maxLeaf] [cellEdge] [path] [sampleTargets]
//   cellEdge <= 0 selects the auto grid-hilbert edge (TC_HILBERT_Q honored).
// Threading: OMP_NUM_THREADS / OMP_PROC_BIND=close OMP_PLACES=cores.
#include "treecode_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <omp.h>

namespace {

using namespace tccpu;

void printUsage(const char *exe)
{
  fprintf(stderr,
          "usage: %s <mac> [maxLeaf] [cellEdge] [path] [sampleTargets]\n"
          "  <mac>          MANDATORY multipole-acceptance criterion (e.g. 0.5)\n"
          "  maxLeaf        max particles per bucket (default 256)\n"
          "  cellEdge       grid cell edge; <= 0 => auto grid-hilbert edge "
          "(default -1)\n"
          "  path           dataset (default distros/two_ball_t50.bin)\n"
          "  sampleTargets  direct-reference target count (default 10000)\n",
          exe);
}

// fp64 direct-sum reference for sampled bucket-order targets: port of
// tcval::directSum64 (treecode_validate.cuh:70-94) -- full 1.0/sqrt, fp64
// geometry/forces, component-major double[3*numSample] in sample order.
// Embarrassingly parallel over samples; per-target accumulation order is
// fixed, so the result is thread-count independent.
void directSum64Cpu(const std::vector<vec3d> &pos64,
                    const std::vector<vec3d> &force64,
                    const std::vector<int> &sample,
                    double pref, std::vector<double> &uref)
{
  const long long numSample = (long long)sample.size();
  const size_t n = pos64.size();
  uref.assign((size_t)3 * (size_t)numSample, 0.0);
#pragma omp parallel for schedule(dynamic, 1)
  for (long long i = 0; i < numSample; ++i) {
    const vec3d T = pos64[(size_t)sample[i]];
    double u0 = 0.0, u1 = 0.0, u2 = 0.0;
    for (size_t s = 0; s < n; ++s) {
      const double Rx = T.x - pos64[s].x;
      const double Ry = T.y - pos64[s].y;
      const double Rz = T.z - pos64[s].z;
      const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
      if (r2 == 0.0) continue;               // self / coincident
      const double ir = 1.0 / std::sqrt(r2);
      const vec3d f = force64[s];
      const double q = (Rx * f.x + Ry * f.y + Rz * f.z) * (ir * ir);
      u0 += ir * (f.x + Rx * q);
      u1 += ir * (f.y + Ry * q);
      u2 += ir * (f.z + Rz * q);
    }
    uref[(size_t)0 * numSample + i] = pref * u0;
    uref[(size_t)1 * numSample + i] = pref * u1;
    uref[(size_t)2 * numSample + i] = pref * u2;
  }
}

int runTwoBallCpu(double mac, int maxLeaf, double cellEdge,
                  const std::string &path, int requestedSamples)
{
  try {
    const double pref = prefactor64();

    // -- 1. load fp64 -----------------------------------------------------
    auto t0 = HostClock::now();
    std::vector<vec3d> pts = loadPointsFP64(path);
    const size_t N = pts.size();
    if (N == 0) throw std::runtime_error("no particles in input");
    for (const auto &p : pts)
      if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
        throw std::runtime_error("NaN detected in input data!");
    const double loadMs = elapsed_ms(t0, HostClock::now());

    // -- 2. uniform gravity force in caller order --------------------------
    std::vector<vec3f> force(N, vec3f(0.f, 0.f, -9.81f));

    // -- 3. treecode apply (build + upward + evaluate) ---------------------
    TreecodeCpu::Config cfg;
    cfg.mac = (float)mac;
    cfg.maxLeaf = maxLeaf;
    cfg.cellEdge = cellEdge;
    if (const char *env = std::getenv("TC_HILBERT_Q"))
      cfg.gridHilbertQ = std::atof(env);

    TreecodeCpu tc(cfg);
    std::vector<double> utree((size_t)3 * N);

    t0 = HostClock::now();
    tc.build(pts.data(), N);
    tc.upward(force.data());
    tc.evaluate(utree.data());
    const double applyWallMs = elapsed_ms(t0, HostClock::now());
    const TreecodeCpu::Stats &st = tc.stats();

    // -- 4. sample targets: evenly spaced over the bucket-order set --------
    const int sampleTargetCount =
        (N < (size_t)requestedSamples) ? (int)N : requestedSamples;
    std::vector<int> sample;
    sample.reserve(sampleTargetCount);
    for (int i = 0; i < sampleTargetCount; ++i)
      sample.push_back((int)(((size_t)i * (N - 1)) / (size_t)(sampleTargetCount - 1)));
    const int numSample = (int)sample.size();

    // -- 5. fp64 direct-sum reference ---------------------------------------
    t0 = HostClock::now();
    std::vector<double> uref;
    directSum64Cpu(tc.points64(), tc.forces64(), sample, pref, uref);
    const double refMs = elapsed_ms(t0, HostClock::now());

    // -- 6. error metrics (port of two_ball.cu:294-349): treecode output is
    //    in caller order, the reference in sampled bucket-slot order --------
    const std::vector<uint32_t> &perm = tc.sourcePermutation();
    double absL2 = 0.0, den2 = 0.0, magSum = 0.0;
    std::vector<double> errSq(numSample), refSq(numSample);
    for (int i = 0; i < numSample; ++i) {
      const size_t originalIdx = (size_t)perm[(size_t)sample[i]];
      double mag2 = 0.0, e2 = 0.0;
      for (int c = 0; c < 3; ++c) {
        const double ub = utree[(size_t)c * N + originalIdx];
        const double ud = uref[(size_t)c * numSample + (size_t)i];
        const double err = ub - ud;
        e2 += err * err;
        den2 += ud * ud;
        mag2 += ud * ud;
      }
      errSq[i] = e2;
      refSq[i] = mag2;
      absL2 += e2;
      magSum += std::sqrt(mag2);
    }
    const double num2 = absL2;
    absL2 = std::sqrt(absL2);
    const double relL2 = (den2 > 0.0) ? std::sqrt(num2 / den2) : absL2;
    const double avgVelocityMagnitude = magSum / (double)numSample;

    double errP50 = 0.0, errP90 = 0.0, errP99 = 0.0, errMax = 0.0;
    double top1Share = 0.0, top10Share = 0.0;
    std::vector<size_t> worstOrig;
    {
      std::vector<double> erel(numSample);
      for (int i = 0; i < numSample; ++i)
        erel[i] = std::sqrt(errSq[i] / std::max(refSq[i], 1e-300));
      std::vector<double> s = erel;
      std::sort(s.begin(), s.end());
      auto pct = [&](double p) { return s[(size_t)((double)(numSample - 1) * p)]; };
      errP50 = pct(0.50); errP90 = pct(0.90); errP99 = pct(0.99);
      errMax = s.back();
      std::vector<int> order(numSample);
      for (int i = 0; i < numSample; ++i) order[i] = i;
      std::sort(order.begin(), order.end(),
                [&](int a, int b) { return errSq[a] > errSq[b]; });
      double acc = 0.0;
      for (int k = 0; k < std::min(10, numSample); ++k) {
        acc += errSq[order[k]];
        if (k == 0) top1Share = acc / std::max(num2, 1e-300);
      }
      top10Share = acc / std::max(num2, 1e-300);
      for (int k = 0; k < std::min(5, numSample); ++k)
        worstOrig.push_back((size_t)perm[(size_t)sample[order[k]]]);
    }

    // -- 7. report ----------------------------------------------------------
    printf("treecode-cpu (Stokeslet, barycentric-Lagrange(KITC) moments, "
           "PDEG=%d) | fp32 moments, fp64 accum\n", bary_cpu::PDEG);
    printf("   dataset=%s  N=%zu  buckets=%u  binary_nodes=%u  leaves=%u\n",
           path.c_str(), N, st.numBuckets, st.numNodes, st.numLeaves);
    printf("   maxBucketParticles=%d  cellEdge=%.6g%s  sampleTargets=%d  mu=%g\n",
           maxLeaf, tc.buckets().cellEdge,
           (cellEdge > 0.0) ? "" : " (auto)", numSample, MU);
    printf("   grid=%u x %u x %u (%llu cells), occupied=%d, "
           "coarse=%d bits fine=%d bits\n",
           tc.buckets().nx, tc.buckets().ny, tc.buckets().nz,
           (unsigned long long)tc.buckets().totalCells,
           tc.buckets().occupiedCells,
           tc.buckets().coarseBits, tc.buckets().fineBits);
    printf("   threads=%d  moments=%.1f MB  max_stack_depth=%d\n",
           omp_get_max_threads(), st.nodeM2PMB, st.maxStackDepth);
    printf("   m2p nodes/target=%.2f  near leaves/target=%.2f  "
           "P2P interactions=%lld (avg/target=%.2f)\n",
           (double)st.m2pNodes / (double)N,
           (double)st.p2pLeaves / (double)N,
           st.p2pInteractions,
           (double)st.p2pInteractions / (double)N);
    printf("   force=(0,0,-9.81) uniform   prefactor=1/(8*pi*mu)=%.6g\n\n",
           pref);
    printf("load_fp64=%.3f\n", loadMs);
    printf("  bucketize=%.3f  build_bvh=%.3f\n", st.bucketMs, st.buildBvhMs);
    printf("  apply: prep_force=%.3f  upward=%.3f  traverse_eval=%.3f  "
           "scatter=%.3f  wall=%.3f\n",
           st.prepForcesMs, st.upwardMs, st.travMs, st.scatterMs, applyWallMs);
    printf("  direct_reference (%d targets)=%.3f\n\n", numSample, refMs);
    printf("mac=%.3f  absL2err=%.6e  relL2err=%.6e  avgVelocityMagnitude=%.6e\n",
           mac, absL2, relL2, avgVelocityMagnitude);
    printf("per-target relerr: p50=%.3e p90=%.3e p99=%.3e max=%.3e  "
           "err2 share: top1=%.1f%% top10=%.1f%%\n",
           errP50, errP90, errP99, errMax,
           100.0 * top1Share, 100.0 * top10Share);
    printf("worst targets (orig idx):");
    for (size_t w : worstOrig) printf(" %zu", w);
    printf("\n");
    return 0;

  } catch (const std::exception &e) {
    fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}

} // namespace

int main(int argc, char **argv)
{
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      printUsage(argv[0]);
      return 0;
    }
    positional.push_back(a);
  }
  if (positional.empty()) {
    printUsage(argv[0]);
    return 2;
  }

  const double mac = std::atof(positional[0].c_str());
  const int maxLeaf = (positional.size() > 1) ? std::atoi(positional[1].c_str()) : 256;
  const double cellEdge =
      (positional.size() > 2) ? std::atof(positional[2].c_str()) : -1.0;
  const std::string path =
      (positional.size() > 3) ? positional[3] : "distros/two_ball_t50.bin";
  const int requestedSamples =
      (positional.size() > 4) ? std::atoi(positional[4].c_str()) : 10000;

  if (!(mac > 0.0) || requestedSamples <= 1) {
    printUsage(argv[0]);
    return 2;
  }

  return runTwoBallCpu(mac, maxLeaf, cellEdge, path, requestedSamples);
}
