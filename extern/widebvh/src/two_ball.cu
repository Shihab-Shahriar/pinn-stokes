// SPDX-License-Identifier: Apache-2.0
//
// Driver: Stokeslet treecode on the "two_ball" dataset (two-ball simulation,
// fp64 particle centers). This is the per-distribution slice -- it loads the
// dataset, sets a uniform gravity force, runs the reusable `Treecode` engine
// (treecode.cuh), then validates against a direct sum on sampled targets and
// prints the timing / error report.
//
// To test the same engine on a different particle distribution, copy this file,
// swap the data-loading + forcing, and add one `add_treecode_driver(...)` line
// to CMakeLists.txt. (treecode.cuh must be included by exactly ONE .cu per
// executable -- see the constraint note at the top of that header.)

#include "treecode.cuh"
#include "treecode_validate.cuh"

#include <cassert>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/fill.h>

namespace {

// Multipole policy: BaryStokes by default; the two_ball_cart target compiles
// this same TU with -DTWO_BALL_CARTESIAN to bench the Cartesian Taylor policy,
// and two_ball_sph with -DTWO_BALL_SPHERICAL for the FMM3D-style
// spherical-harmonic policy.
#ifdef TWO_BALL_CARTESIAN
using TwoBallTreecode = Treecode<mp::CartesianStokes>;
#elif defined(TWO_BALL_SPHERICAL)
using TwoBallTreecode = Treecode<mp::SphericalStokes>;
#else
using TwoBallTreecode = Treecode<mp::BaryStokes>;
#endif

template<class TC>
struct RunResult {
  typename TC::Stats apply;
  typename TC::Stats reapply;
  double applyWallMs = 0.0;
  double reapplyWallMs = 0.0;
};

// Peak-VRAM sampler. A background host thread polls cudaMemGetInfo (a cheap
// memory-manager query that does NOT touch the CUDA stream or synchronize the
// device, so it does not perturb the event/wall timings) every ~0.5 ms and
// tracks the minimum free memory over the run. The dominant treecode buffer (the
// near-field pair pipeline, ~40 B/pair) is held for many ms, far longer than the
// poll interval, so the structural peak is never missed and captures every
// allocator (thrust / CUB / cuBQL, all synchronous cudaMalloc on this path).
//   abs   = total - min_free  : true device-occupancy peak (context + co-tenant)
//   delta = free_start - min_free : this process's footprint above the post-init
//                                   baseline (primary metric; cancels a static
//                                   co-tenant, and min-across-repeats filters a
//                                   transient one since the structural peak is
//                                   repeat-invariant)
// minFree has a single writer (the sampler thread) while running; start()/
// stopAndJoin() run on the main thread with join() as the synchronization point,
// so only `stop` needs to be atomic.
struct VramPeakSampler {
  std::thread th;
  std::atomic<bool> stop{false};
  size_t total = 0, freeStart = 0, minFree = ~size_t(0);
  void start() {
    cudaMemGetInfo(&freeStart, &total);   // baseline: context already inited
    minFree = freeStart;
    stop.store(false);
    th = std::thread([this] {
      cudaSetDevice(0);                    // only the pinned GPU is visible
      while (!stop.load(std::memory_order_relaxed)) {
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess && f < minFree) minFree = f;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
      }
    });
  }
  void stopAndJoin() {                     // final sample in case the peak is at the end
    stop.store(true);
    if (th.joinable()) th.join();
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) == cudaSuccess && f < minFree) minFree = f;
  }
  ~VramPeakSampler() { stop.store(true); if (th.joinable()) th.join(); }  // exception-safe
  double peakAbsMB()   const { return (double)(total - minFree) / (1024.0 * 1024.0); }
  double peakDeltaMB() const { return (double)(freeStart - minFree) / (1024.0 * 1024.0); }
  double baselineMB()  const { return (double)(total - freeStart) / (1024.0 * 1024.0); }
};

// The execution path (split/direct-near, thread/warp/...) is chosen only by the
// TC_PATH environment variable; it is not a Config field.
template<class TC>
typename TC::Config makeConfig(float mac, int maxLeaf, double cellEdge)
{
  typename TC::Config cfg;
  cfg.mac        = mac;
  cfg.cellEdge   = cellEdge;
  cfg.maxLeaf    = maxLeaf;
  // TC_ORDER overrides the dispatch order; either way clamp into the policy's
  // supported range (CartesianStokes caps at 4, below the Config default 6).
  if (const char *env = std::getenv("TC_ORDER"))
    cfg.order = std::atoi(env);
  cfg.order = std::max(1, std::min(cfg.order, TC::MAX_ORDER));
  // TC_NEAR_CUTOFF turns on the near-field exclusion (Config::nearCutoff): the
  // treecode then sums only r >= rc, and the direct-sum reference in
  // treecode_validate.cuh applies the same rule -- so the reported relL2err is
  // still pure multipole truncation and is directly comparable to a rc=0 run.
  if (const char *env = std::getenv("TC_NEAR_CUTOFF"))
    cfg.nearCutoff = std::atof(env);
  return cfg;
}

template<class TC>
RunResult<TC> runTreecode(TC &tc,
                          const thrust::device_vector<vec3d> &d_pts,
                          size_t N,
                          const thrust::device_vector<vec3f> &d_force,
                          thrust::device_vector<double> &d_vel,
                          thrust::device_vector<double> &d_velReapply)
{
  RunResult<TC> out;
  d_vel.resize((size_t)3 * N);
  d_velReapply.resize((size_t)3 * N);

  auto stepStart = HostClock::now();
  tc.apply(util::devicePtr(d_pts), N, util::devicePtr(d_force),
           util::devicePtr(d_vel), /*will_reuse_tree=*/true);
  CUDA_CHECK(cudaDeviceSynchronize());
  out.applyWallMs = elapsed_ms(stepStart, HostClock::now());
  out.apply = tc.stats();

  stepStart = HostClock::now();
  tc.reapply(util::devicePtr(d_force), util::devicePtr(d_velReapply));
  CUDA_CHECK(cudaDeviceSynchronize());
  out.reapplyWallMs = elapsed_ms(stepStart, HostClock::now());
  out.reapply = tc.stats();

  return out;
}

void releaseVector(thrust::device_vector<double> &v)
{
  thrust::device_vector<double>().swap(v);
}

double maxAbsDiff(const std::vector<double> &a, const std::vector<double> &b)
{
  assert(a.size() == b.size());
  double out = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    out = std::max(out, std::abs(a[i] - b[i]));
  return out;
}

// Generic per-run timing breakdown. Works for every TC_PATH: travMs is the
// traversal kernel time (M2P emit, interaction list, or merged direct-near) and
// p2pMs is the near-field/scatter time (cached P2P, replay, or scatter-only).
template<class TC>
void printRunTimings(const RunResult<TC> &run)
{
  const typename TC::Stats &a = run.apply;
  const typename TC::Stats &r = run.reapply;
  const double applyMeasured =
      a.bucketMs + a.targetBucketMs + a.prepForcesMs + a.buildBvhMs
      + a.upwardMs + (double)a.travMs + (double)a.p2pMs;
  const double reapplyMeasured =
      r.prepForcesMs + r.upwardMs + (double)r.travMs + (double)r.p2pMs;

  printf("  bucketize=%.3f  target_bucketize=%.3f  build_bvh=%.3f\n",
         a.bucketMs, a.targetBucketMs, a.buildBvhMs);
  printf("  apply: prep_force=%.3f  upward=%.3f  traverse=%.3f  "
         "near_scatter=%.3f  measured_sum=%.3f  wall=%.3f\n",
         a.prepForcesMs, a.upwardMs, (double)a.travMs, (double)a.p2pMs,
         applyMeasured, run.applyWallMs);
  printf("  reapply(force): prep_force=%.3f  upward=%.3f  traverse=%.3f  "
         "near_scatter=%.3f  measured_sum=%.3f  wall=%.3f\n\n",
         r.prepForcesMs, r.upwardMs, (double)r.travMs, (double)r.p2pMs,
         reapplyMeasured, run.reapplyWallMs);
}

void printUsage(const char *exe)
{
  fprintf(stderr,
          "usage: %s <mac> [maxLeaf] [cellEdge] [path] [sampleTargets]\n"
          "  <mac>          MANDATORY multipole-acceptance criterion (e.g. 0.3)\n"
          "  maxLeaf        max particles per bucket (default 32)\n"
          "  cellEdge       uniform grid cell edge length (default 10.0)\n"
          "  path           dataset (default two_ball/center_timestep_50.bin)\n"
          "  sampleTargets  direct-reference target count (default 10000)\n",
          exe);
}

int runTwoBall(double mac, int maxLeaf, double cellEdge,
               const std::string &path, int requestedSamples)
{
  using TC = TwoBallTreecode;
  assert(mac > 0.0);
  assert(cellEdge > 0.0);
  assert(requestedSamples > 1);

  try {
    CUDA_CHECK(cudaFree(0));        // force context init up front, outside all timers
    VramPeakSampler vram;           // baseline captured post-init; peak tracked across the run
    vram.start();
    const float pref = stokes::prefactor();

    // -- 1. load fp64 ---------------------------------------------------
    auto stepStart = HostClock::now();
    std::vector<vec3d> hostPts = util::loadPointsFP64(path);
    const size_t N = hostPts.size();
    if (N == 0) throw std::runtime_error("no particles in input");

    for (const auto& pt : hostPts) {
        if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) {
            throw std::runtime_error("NaN detected in input data!");
        }
    }

    thrust::device_vector<vec3d> d_pts(hostPts);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double loadMs = elapsed_ms(stepStart, HostClock::now());

    // -- 2. source forces in original particle order --------------------
    stepStart = HostClock::now();
    thrust::device_vector<vec3f> d_force(N);
    thrust::fill(d_force.begin(), d_force.end(), vec3f(0.f, 0.f, -9.81f));
    CUDA_CHECK(cudaDeviceSynchronize());
    const double forceInputMs = elapsed_ms(stepStart, HostClock::now());

    typename TC::Config cfg = makeConfig<TC>((float)mac, maxLeaf, cellEdge);

    // -- 3. treecode apply/reapply for ALL N particles, on the single path
    //       selected by TC_PATH (the treecode prints which path ran) --------
    RunResult<TC> run;
    std::vector<double> utreeAll;
    std::vector<double> ureapplyAll;
    {
      TC tc(cfg);
      thrust::device_vector<double> d_vel;
      thrust::device_vector<double> d_velReapply;
      run = runTreecode(tc, d_pts, N, d_force, d_vel, d_velReapply);
      const typename TC::Stats stApply = run.apply;

      // -- 4. sample targets: evenly spaced over the bucket-order target set
      const int sampleTargetCount =
        (N < (size_t)requestedSamples) ? (int)N : requestedSamples;
      assert(sampleTargetCount > 1);               // avoid div-by-zero below
      std::vector<int> sample;
      sample.reserve(sampleTargetCount);
      for (int i = 0; i < sampleTargetCount; ++i) {
        const size_t idx =
            ((size_t)i * (N - 1)) / (size_t)(sampleTargetCount - 1);
        sample.push_back((int)idx);
      }
      const int numSample = (int)sample.size();
      int *d_idx = nullptr;
      CUDA_CHECK(cudaMalloc(&d_idx, numSample * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_idx, sample.data(), numSample * sizeof(int),
                            cudaMemcpyHostToDevice));

      // -- 5. direct-sum reference for the sampled bucket-order targets --
      // fp64-geometry ground truth (tc.points64()): the honest yardstick for the
      // fp64-geometry treecode. The old fp32 directSum shared the treecode's fp32
      // bucket coords, so it was blind to fp32-coordinate quantization error.
      double *d_uref = nullptr;
      CUDA_CHECK(cudaMalloc(&d_uref,
                            (size_t)3 * (size_t)numSample * sizeof(double)));
      tcval::directSum64(tc.points64(), tc.forces(), (int)N, d_idx, numSample,
                         (double)pref, d_uref);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      // -- 6. error metrics: treecode result is in original input order,
      // while directSum sampled bucket slots. Map bucket slot -> original idx.
      std::vector<double> uref((size_t)3 * (size_t)numSample);
      utreeAll.resize((size_t)3 * N);
      ureapplyAll.resize((size_t)3 * N);
      std::vector<uint32_t> sourcePerm(N);
      CUDA_CHECK(cudaMemcpy(uref.data(), d_uref, uref.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(utreeAll.data(), util::devicePtr(d_vel),
                            utreeAll.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ureapplyAll.data(), util::devicePtr(d_velReapply),
                            ureapplyAll.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(sourcePerm.data(), tc.sourcePermutation(),
                            sourcePerm.size() * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

      double absL2 = 0.0, relL2 = 0.0, avgVelocityMagnitude = 0.0;
      double den2 = 0.0, magSum = 0.0;
      std::vector<double> errSq(numSample), refSq(numSample);
      for (int i = 0; i < numSample; ++i) {
        const int bucketIdx = sample[i];
        const size_t originalIdx = (size_t)sourcePerm[(size_t)bucketIdx];
        double mag2 = 0.0, e2 = 0.0;
        for (int c = 0; c < 3; ++c) {
          const double ub = utreeAll[(size_t)c * N + originalIdx];
          const double ud = uref[(size_t)c * (size_t)numSample + (size_t)i];
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
      relL2 = (den2 > 0.0) ? std::sqrt((absL2 * absL2) / den2) : absL2;
      avgVelocityMagnitude = magSum / (double)numSample;

      // Per-target relative-error distribution + tail concentration. On
      // heavy-tailed distros (multi_shells) the aggregate relL2 over a few
      // thousand samples is dominated by a handful of outlier targets, so the
      // scalar above is an unstable estimate; these lines make that visible.
      double errP50 = 0.0, errP90 = 0.0, errP99 = 0.0, errMax = 0.0;
      double top1Share = 0.0, top10Share = 0.0;
      std::vector<size_t> worstOrig;
      {
        std::vector<double> erel(numSample);
        for (int i = 0; i < numSample; ++i)
          erel[i] = std::sqrt(errSq[i] / std::max(refSq[i], 1e-300));
        std::vector<double> s = erel;
        std::sort(s.begin(), s.end());
        auto pct = [&](double p) {
          return s[(size_t)((double)(numSample - 1) * p)];
        };
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
          worstOrig.push_back((size_t)sourcePerm[(size_t)sample[order[k]]]);
      }

      const double reapplyMaxAbs = maxAbsDiff(utreeAll, ureapplyAll);

      printf("treecode (Stokeslet, %s moments, order=%d) | fp32 data, fp64 accum\n",
             TC::multipoleName(), cfg.order);
      printf("   dataset=%s  N=%zu  buckets=%u  binary_nodes=%u  "
             "maxBucketParticles=%d  cellEdge=%.6g  sampleTargets=%d  mu=%g\n",
             path.c_str(), N, stApply.numBuckets, stApply.numNodes,
             maxLeaf, cellEdge, numSample, stokes::MU);
#if TREECODE_STATS
      printf("   traversal leaf depth: min=%d  mean=%.2f  max=%d  "
             "inner=%u  leaves=%u\n",
             stApply.minTraversalLeafDepth, stApply.meanTraversalLeafDepth,
             stApply.maxTraversalLeafDepth, stApply.traversalInner,
             stApply.traversalLeaves);
      printf("   mean leaf interactions/target: %.2f\n",
             (double)stApply.nPairs / (double)N);
#endif
      printf("   grid=%u x %u x %u (%llu cells), occupied=%d, "
             "Morton coarse=%d bits fine=%d bits\n",
             stApply.nx, stApply.ny, stApply.nz,
             (unsigned long long)stApply.totalCells,
             stApply.occupiedCells, stApply.coarseBits, stApply.fineBits);
      printf("   force=(0,0,-9.81) uniform   prefactor=1/(8*pi*mu)=%.6g\n\n",
             pref);
      printf("   near pairs=%lld  P2P interactions=%lld  "
             "(avg near leaves/target=%.2f)\n\n",
             stApply.nPairs, stApply.totalP2P,
             (double)stApply.nPairs / (double)N);
      printf("load_fp64=%.3f  prep_force_input=%.3f\n\n",
             loadMs, forceInputMs);
      printRunTimings(run);
      printf("mac=%.3f  absL2err=%.6e  relL2err=%.6e  "
             "avgVelocityMagnitude=%.6e\n",
             mac, absL2, relL2, avgVelocityMagnitude);
      printf("per-target relerr: p50=%.3e p90=%.3e p99=%.3e max=%.3e  "
             "err2 share: top1=%.1f%% top10=%.1f%%\n",
             errP50, errP90, errP99, errMax,
             100.0 * top1Share, 100.0 * top10Share);
      printf("worst targets (orig idx):");
      for (size_t w : worstOrig) printf(" %zu", w);
      printf("\n");
      printf("reapply_same_force_max_abs_diff=%.6e\n\n", reapplyMaxAbs);

      cudaFree(d_uref);
      cudaFree(d_idx);
      releaseVector(d_vel);
      releaseVector(d_velReapply);
    }

    d_pts.clear();
    d_pts.shrink_to_fit();

    vram.stopAndJoin();
    printf("peak_vram: abs=%.1f MB  delta=%.1f MB  baseline=%.1f MB\n",
           vram.peakAbsMB(), vram.peakDeltaMB(), vram.baselineMB());

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
    std::string a = argv[i];
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
  const int maxLeaf = (positional.size() > 1) ? std::atoi(positional[1].c_str()) : 32;
  const double cellEdge =
      (positional.size() > 2) ? std::atof(positional[2].c_str()) : 10.0;
  const std::string path =
      (positional.size() > 3) ? positional[3] : "two_ball/center_timestep_50.bin";
  const int requestedSamples =
      (positional.size() > 4) ? std::atoi(positional[4].c_str()) : 10000;

  if (!(mac > 0.0) || !(cellEdge > 0.0) || requestedSamples <= 1) {
    printUsage(argv[0]);
    return 2;
  }

  return runTwoBall(mac, maxLeaf, cellEdge, path, requestedSamples);
}
