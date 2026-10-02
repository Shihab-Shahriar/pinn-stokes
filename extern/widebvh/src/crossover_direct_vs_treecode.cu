// SPDX-License-Identifier: Apache-2.0
//
// Crossover experiment: at how many ellipsoids P is a brute-force GPU direct
// Stokeslet sum FASTER than the treecode, on the static 10k-ellipsoid MFS RHS
// case? P is varied by taking prefixes of the frozen arrays (points are
// grouped contiguously by ellipsoid: P*656 sources, P*864 targets), so the
// per-ellipsoid source/target points are never modified. For each P the
// treecode runs first (per mac x TC_PATH combo, apply + reapply); the direct
// sum then runs once per requested kernel (block = one block per target,
// tiled = shared-memory source tiles; it is mac/path-independent) in target
// tiles and is ABORTED as soon as its elapsed time exceeds the slowest
// treecode time for that P, so large P never wastes wall clock. The CROSS
// verdict uses the fastest direct kernel. --direct-bench skips the treecode
// and times the direct kernels alone. Accuracy (relL2 vs an exact sampled
// fp64 direct reference) is computed separately for every combo.

#include "treecode.cuh"
#include "stokes_kernel.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

using XTreecode = Treecode<mp::BaryStokes>;

static constexpr const char *SOURCE_FILE = "mfs_ellipsoid_case/proxy_sources.bin";
static constexpr const char *TARGET_FILE = "mfs_ellipsoid_case/colloc_targets.bin";
static constexpr const char *FORCE_FILE = "mfs_ellipsoid_case/proxy_strengths.bin";
static constexpr const char *CENTERS_FILE = "mfs_ellipsoid_case/centers.bin";

static constexpr size_t NSOURCE = 6560000;
static constexpr size_t NTARGET = 8640000;
static constexpr int SOURCE_GROUP = 656;
static constexpr int TARGET_GROUP = 864;
static constexpr int MAX_P = (int)(NSOURCE / SOURCE_GROUP);

static constexpr double PREF = 0.039788735772973836; // 1/(8*pi), mu=1

struct Options {
  std::vector<int> pvals = {100, 200, 500, 1000, 2000, 5000, 10000};
  std::vector<float> macs = {0.4f, 0.5f};
  std::vector<std::string> paths = {"split-warp", "split-warpspec-atomic",
                                    "direct-warpspec"};
  double cellEdge = 10.0;
  int maxLeaf = 256;
  int warmup = 1;
  int repeat = 2;
  double slack = 0.10;      // direct-abort threshold = maxTreeMs*(1+slack)+floor
  double floorMs = 10.0;
  double directCapMs = -1.0; // >0: hard threshold override
  double tileMs = 20.0;      // target duration of one direct-race tile
  int sampleSlice = 1000;    // accuracy sample stride cap (bench uses 1000)
  int minSamples = 1000;     // lower stride at small P to keep >= this many
  bool selftest = false;
  std::string select = "prefix"; // prefix: file order; radial: P nearest origin
  // Direct-sum kernels to race (the CROSS verdict uses the fastest):
  // block = one block per target, block-stride over sources (L2 reuse only);
  // tiled = one thread per target, shared-memory source tiles (GPU-Gems n-body).
  std::vector<std::string> directKernels = {"block", "tiled"};
  bool directBench = false; // skip the treecode; time the direct kernels only
};

// ---------------------------------------------------------------- CLI ------

static bool startsWith(const std::string &s, const char *prefix)
{
  return s.rfind(prefix, 0) == 0;
}

static std::string valueAfter(const std::string &s, const char *prefix)
{
  return s.substr(std::string(prefix).size());
}

static double parseDouble(const std::string &s, const char *name)
{
  char *end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (!end || *end != '\0')
    throw std::runtime_error(std::string("bad ") + name + ": " + s);
  return v;
}

static int parseInt(const std::string &s, const char *name)
{
  char *end = nullptr;
  const long v = std::strtol(s.c_str(), &end, 10);
  if (!end || *end != '\0')
    throw std::runtime_error(std::string("bad ") + name + ": " + s);
  return (int)v;
}

static std::vector<std::string> splitList(const std::string &s)
{
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    const size_t c = s.find(',', pos);
    const size_t end = (c == std::string::npos) ? s.size() : c;
    if (end > pos) out.push_back(s.substr(pos, end - pos));
    if (c == std::string::npos) break;
    pos = c + 1;
  }
  return out;
}

static void usage(const char *argv0)
{
  std::fprintf(
      stderr,
      "usage: %s [--pvals=100,200,...] [--macs=0.4,0.5]\n"
      "  [--paths=split-warp,split-warpspec-atomic,direct-warpspec]\n"
      "  [--cell=10] [--maxleaf=256] [--warmup=1] [--repeat=2]\n"
      "  [--slack=0.10] [--direct-cap-ms=X] [--tile-ms=20]\n"
      "  [--sample-slice=1000] [--selftest] [--select=prefix|radial]\n"
      "  [--direct-kernel=block,tiled] [--direct-bench]\n",
      argv0);
}

static Options parseArgs(int argc, char **argv)
{
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (startsWith(a, "--pvals=")) {
      opt.pvals.clear();
      for (const auto &t : splitList(valueAfter(a, "--pvals=")))
        opt.pvals.push_back(parseInt(t, "pvals"));
    } else if (startsWith(a, "--macs=")) {
      opt.macs.clear();
      for (const auto &t : splitList(valueAfter(a, "--macs=")))
        opt.macs.push_back((float)parseDouble(t, "macs"));
    } else if (startsWith(a, "--paths=")) {
      opt.paths = splitList(valueAfter(a, "--paths="));
    } else if (startsWith(a, "--cell=")) {
      opt.cellEdge = parseDouble(valueAfter(a, "--cell="), "cell");
    } else if (startsWith(a, "--maxleaf=")) {
      opt.maxLeaf = parseInt(valueAfter(a, "--maxleaf="), "maxleaf");
    } else if (startsWith(a, "--warmup=")) {
      opt.warmup = parseInt(valueAfter(a, "--warmup="), "warmup");
    } else if (startsWith(a, "--repeat=")) {
      opt.repeat = parseInt(valueAfter(a, "--repeat="), "repeat");
    } else if (startsWith(a, "--slack=")) {
      opt.slack = parseDouble(valueAfter(a, "--slack="), "slack");
    } else if (startsWith(a, "--direct-cap-ms=")) {
      opt.directCapMs = parseDouble(valueAfter(a, "--direct-cap-ms="), "cap");
    } else if (startsWith(a, "--tile-ms=")) {
      opt.tileMs = parseDouble(valueAfter(a, "--tile-ms="), "tile-ms");
    } else if (startsWith(a, "--sample-slice=")) {
      opt.sampleSlice = parseInt(valueAfter(a, "--sample-slice="), "slice");
    } else if (a == "--selftest") {
      opt.selftest = true;
    } else if (startsWith(a, "--direct-kernel=")) {
      opt.directKernels = splitList(valueAfter(a, "--direct-kernel="));
      for (const auto &k : opt.directKernels)
        if (k != "block" && k != "tiled")
          throw std::runtime_error("--direct-kernel entries must be "
                                   "block or tiled");
      if (opt.directKernels.empty())
        throw std::runtime_error("--direct-kernel must be non-empty");
    } else if (a == "--direct-bench") {
      opt.directBench = true;
    } else if (startsWith(a, "--select=")) {
      opt.select = valueAfter(a, "--select=");
      if (opt.select != "prefix" && opt.select != "radial")
        throw std::runtime_error("--select must be prefix or radial");
    } else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  if (opt.pvals.empty() || opt.macs.empty() || opt.paths.empty())
    throw std::runtime_error("pvals/macs/paths must be non-empty");
  for (int p : opt.pvals)
    if (p < 1 || p > MAX_P)
      throw std::runtime_error("pvals entries must be in [1, " +
                               std::to_string(MAX_P) + "]");
  if (opt.warmup < 0 || opt.repeat <= 0)
    throw std::runtime_error("warmup must be >= 0 and repeat must be > 0");
  return opt;
}

// ------------------------------------------------------------- loading -----

static void readExact(std::ifstream &in, void *dst, size_t bytes,
                      const std::string &path)
{
  in.read(reinterpret_cast<char *>(dst), (std::streamsize)bytes);
  if (!in) throw std::runtime_error("short read: " + path);
}

static std::vector<vec3d> readSoA3(const std::string &path, size_t n,
                                   size_t fileN = 0)
{
  if (fileN == 0) fileN = n;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);

  std::vector<vec3d> v(n);
  std::vector<double> a(n);
  for (int c = 0; c < 3; ++c) {
    in.seekg((std::streamoff)((size_t)c * fileN * sizeof(double)));
    if (!in) throw std::runtime_error("seek failed: " + path);
    readExact(in, a.data(), n * sizeof(double), path);
    for (size_t i = 0; i < n; ++i)
      v[i][c] = a[i];
  }
  return v;
}

// Ellipsoid ids sorted by ascending distance of their center from the origin
// (centers.bin layout: double cx[P] cy[P] cz[P]). Radial order is nested (the
// P nearest are a prefix of the P' nearest for P < P'), so permuting the
// per-ellipsoid blocks once by this order lets the existing prefix logic
// select exactly the P nearest-origin ellipsoids at every P.
static std::vector<int> radialEllipsoidOrder()
{
  std::ifstream in(CENTERS_FILE, std::ios::binary);
  if (!in) throw std::runtime_error(std::string("cannot open ") + CENTERS_FILE);
  std::vector<double> c((size_t)3 * MAX_P);
  readExact(in, c.data(), c.size() * sizeof(double), CENTERS_FILE);
  std::vector<int> order(MAX_P);
  for (int i = 0; i < MAX_P; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const double ra = c[a] * c[a] + c[MAX_P + a] * c[MAX_P + a] +
                      c[2 * MAX_P + a] * c[2 * MAX_P + a];
    const double rb = c[b] * c[b] + c[MAX_P + b] * c[MAX_P + b] +
                      c[2 * MAX_P + b] * c[2 * MAX_P + b];
    return ra < rb;
  });
  return order;
}

// Reorder per-ellipsoid blocks of `groupSize` points to `order`.
static void permuteGroups(std::vector<vec3d> &v, const std::vector<int> &order,
                          int groupSize)
{
  std::vector<vec3d> out(v.size());
  for (size_t k = 0; k < order.size(); ++k)
    std::copy_n(v.begin() + (size_t)order[k] * groupSize, groupSize,
                out.begin() + k * groupSize);
  v.swap(out);
}

// ------------------------------------------------------------- kernels -----

// Exact fp64 direct sum (full 1.0/sqrt) at every sampleStride-th target: the
// accuracy ground truth. One block per sampled target, block-stride over all
// sources (mfs_ellipsoid_treecode_bench.cu::directSampleKernel).
__global__ void refSampleKernel(const vec3d *__restrict__ src,
                                const vec3d *__restrict__ target,
                                const vec3d *__restrict__ force,
                                double *__restrict__ out,
                                int nSource, int sampleStride, int nSample)
{
  const int sample = blockIdx.x;
  const int tid = threadIdx.x;
  const int targetIdx = sample * sampleStride;
  const vec3d t = target[targetIdx];

  double ux = 0.0, uy = 0.0, uz = 0.0;
  for (int s = tid; s < nSource; s += blockDim.x) {
    const vec3d y = src[s];
    const vec3d f = force[s];
    const double rx = t.x - y.x;
    const double ry = t.y - y.y;
    const double rz = t.z - y.z;
    const double r2 = rx * rx + ry * ry + rz * rz;
    if (r2 == 0.0) continue;
    const double ir = 1.0 / sqrt(r2);
    const double ir3 = ir / r2;
    const double rdf = rx * f.x + ry * f.y + rz * f.z;
    ux += PREF * (f.x * ir + rx * rdf * ir3);
    uy += PREF * (f.y * ir + ry * rdf * ir3);
    uz += PREF * (f.z * ir + rz * rdf * ir3);
  }

  extern __shared__ double sh[];
  double *sx = sh;
  double *sy = sh + blockDim.x;
  double *sz = sh + 2 * blockDim.x;
  sx[tid] = ux;
  sy[tid] = uy;
  sz[tid] = uz;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
      sz[tid] += sz[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[(size_t)0 * nSample + sample] = sx[0];
    out[(size_t)1 * nSample + sample] = sy[0];
    out[(size_t)2 * nSample + sample] = sz[0];
  }
}

__global__ void gatherSampleKernel(const double *__restrict__ full,
                                   double *__restrict__ sampleOut,
                                   int nTarget, int sampleStride, int nSample)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nSample) return;
  const int targetIdx = i * sampleStride;
  sampleOut[(size_t)0 * nSample + i] = full[(size_t)0 * nTarget + targetIdx];
  sampleOut[(size_t)1 * nSample + i] = full[(size_t)1 * nTarget + targetIdx];
  sampleOut[(size_t)2 * nSample + i] = full[(size_t)2 * nTarget + targetIdx];
}

// One tile of the RACE direct sum: one block per target in
// [tileBegin, tileBegin+gridDim.x), block-stride over all sources with the
// fp64-geometry stokes::p2p (rsqrtf seed + fp64 Newton — the fastest honest
// kernel, same near-field precision as the treecode). Unscaled accumulate,
// prefactor applied once at write. Output component-major, stride nTgtTotal.
__global__ void raceTileKernel(const vec3d *__restrict__ src,
                               const vec3d *__restrict__ tgt,
                               const vec3d *__restrict__ force,
                               double *__restrict__ out,
                               int nSource, int tileBegin, int nTgtTotal)
{
  const int t = tileBegin + blockIdx.x;
  const int tid = threadIdx.x;
  const vec3d T = tgt[t];

  double u[3] = {0.0, 0.0, 0.0};
  for (int s = tid; s < nSource; s += blockDim.x)
    stokes::p2p(T, src[s], force[s], u);

  extern __shared__ double sh[];
  double *sx = sh;
  double *sy = sh + blockDim.x;
  double *sz = sh + 2 * blockDim.x;
  sx[tid] = u[0];
  sy[tid] = u[1];
  sz[tid] = u[2];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
      sz[tid] += sz[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[(size_t)0 * nTgtTotal + t] = PREF * sx[0];
    out[(size_t)1 * nTgtTotal + t] = PREF * sy[0];
    out[(size_t)2 * nTgtTotal + t] = PREF * sz[0];
  }
}

// Tiled RACE direct sum (classic GPU-Gems-3 n-body scheme): one THREAD per
// target (RACE_TILE targets per block), the block cooperatively staging
// RACE_TILE-source chunks (positions + forces) in shared memory. Each staged
// source is reused by all RACE_TILE targets of the block, so source traffic
// drops ~RACE_TILE x vs raceTileKernel and the sum stays compute-bound even
// when the source array spills L2 (P >~ 1000, where the one-block-per-target
// kernel goes DRAM-bound). Same fp64-geometry stokes::p2p, unscaled
// accumulate, prefactor at write; no block reduction (each thread owns its
// whole target sum). tileTargets counts targets in this launch, starting at
// tileBegin; inactive tail threads still run the barriers.
static constexpr int RACE_TILE = 256; // == blockDim.x of raceTiledKernel
__global__ void raceTiledKernel(const vec3d *__restrict__ src,
                                const vec3d *__restrict__ tgt,
                                const vec3d *__restrict__ force,
                                double *__restrict__ out,
                                int nSource, int tileBegin, int tileTargets,
                                int nTgtTotal)
{
  __shared__ vec3d shSrc[RACE_TILE];
  __shared__ vec3d shFrc[RACE_TILE];

  const int tid = threadIdx.x;
  const int t = tileBegin + blockIdx.x * RACE_TILE + tid;
  const bool active = (t < tileBegin + tileTargets);
  const vec3d T = tgt[active ? t : tileBegin];

  double u[3] = {0.0, 0.0, 0.0};
  for (int chunk = 0; chunk < nSource; chunk += RACE_TILE) {
    const int m = min(RACE_TILE, nSource - chunk);
    if (tid < m) {
      shSrc[tid] = src[chunk + tid];
      shFrc[tid] = force[chunk + tid];
    }
    __syncthreads();
    if (active) {
#pragma unroll 4
      for (int j = 0; j < m; ++j)
        stokes::p2p(T, shSrc[j], shFrc[j], u);
    }
    __syncthreads();
  }

  if (active) {
    out[(size_t)0 * nTgtTotal + t] = PREF * u[0];
    out[(size_t)1 * nTgtTotal + t] = PREF * u[1];
    out[(size_t)2 * nTgtTotal + t] = PREF * u[2];
  }
}

// ------------------------------------------------------- treecode timing ---

struct TreeTiming {
  double applyWallMs = 0.0;
  float applyCudaMs = 0.0f;
  double reapplyWallMs = 0.0;
  float reapplyCudaMs = 0.0f;
  XTreecode::Stats stats{};        // apply stats
  XTreecode::Stats reapplyStats{}; // reapply stats
};

static TreeTiming runTreecodeOnce(float mac, const Options &opt,
                                  const vec3d *d_source, size_t nSrc,
                                  const vec3d *d_target, size_t nTgt,
                                  const vec3d *d_force, double *d_out)
{
  XTreecode::Config cfg;
  cfg.mac = mac;
  cfg.cellEdge = opt.cellEdge;
  cfg.maxLeaf = opt.maxLeaf;
  cfg.sourceGroupSize = SOURCE_GROUP;
  cfg.targetGroupSize = TARGET_GROUP;
  cfg.skipSameGroup = false;

  XTreecode tc(cfg);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  TreeTiming t;

  CUDA_CHECK(cudaEventRecord(start));
  const auto wall0 = std::chrono::steady_clock::now();
  tc.apply(d_source, nSrc, d_target, nTgt, d_force, d_out,
           /*will_reuse_tree=*/true);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  const auto wall1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventElapsedTime(&t.applyCudaMs, start, stop));
  t.applyWallMs = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
  t.stats = tc.stats();

  CUDA_CHECK(cudaEventRecord(start));
  const auto rwall0 = std::chrono::steady_clock::now();
  tc.reapply(d_force, d_out);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  const auto rwall1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventElapsedTime(&t.reapplyCudaMs, start, stop));
  t.reapplyWallMs =
      std::chrono::duration<double, std::milli>(rwall1 - rwall0).count();
  t.reapplyStats = tc.stats();

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return t;
}

static void printTreeLine(const char *tag, int P, float mac,
                          const std::string &path, int rep, const TreeTiming &t)
{
  const auto &s = t.stats;
  const auto &r = t.reapplyStats;
  std::printf(
      "%s P=%d mac=%.3g path=%s rep=%d apply_cuda_ms=%.3f apply_wall_ms=%.3f "
      "reapply_cuda_ms=%.3f reapply_wall_ms=%.3f m2p_ms=%.3f p2p_ms=%.3f "
      "reapply_m2p_ms=%.3f reapply_p2p_ms=%.3f near_pairs=%lld direct_p2p=%lld "
      "nodes=%u src_buckets=%u tgt_buckets=%u\n",
      tag, P, mac, path.c_str(), rep, t.applyCudaMs, t.applyWallMs,
      t.reapplyCudaMs, t.reapplyWallMs, (double)s.travMs, (double)s.p2pMs,
      (double)r.travMs, (double)r.p2pMs, s.nPairs, s.totalP2P, s.numNodes,
      s.numSourceBuckets, s.numTargetBuckets);
  std::fflush(stdout);
}

// --------------------------------------------------------- direct race -----

struct DirectResult {
  bool done = false;
  size_t targetsDone = 0;
  int tiles = 0;
  double elapsedCudaMs = 0.0;
  double wallMs = 0.0;
  double projectedMs = 0.0; // == elapsedCudaMs when done
  double thresholdMs = 0.0;
};

// One full-occupancy wave of raceTiledKernel, in targets. The tiled kernel
// packs RACE_TILE targets per block, so rate-sized ~20 ms tiles would launch
// only a handful of blocks at large P and starve the GPU; its race tiles are
// therefore whole occupancy waves (uniform per-block work, so waves finish
// together), even when that exceeds tileMs.
static int tiledWaveTargets()
{
  static int wave = 0;
  if (wave == 0) {
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, raceTiledKernel, RACE_TILE, 0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    wave = std::max(1, blocksPerSM) * prop.multiProcessorCount * RACE_TILE;
  }
  return wave;
}

static DirectResult runDirectRace(bool tiled, const vec3d *d_source,
                                  size_t nSrc, const vec3d *d_target,
                                  size_t nTgt, const vec3d *d_force,
                                  double *d_raceOut, double thresholdMs,
                                  double tileMs)
{
  DirectResult res;
  res.thresholdMs = thresholdMs;
  const long wave = tiled ? tiledWaveTargets() : 0;

  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));

  // Seed tile: ~2e9 pairs (~1-2 ms) to measure the rate before sizing tiles
  // (for tiled at least one full wave, so the measured rate is honest).
  const double PAIR_BUDGET0 = 2.0e9;
  double seed = std::max<double>(PAIR_BUDGET0 / (double)nSrc, 256.0);
  seed = tiled ? std::max<double>(seed, (double)wave)
               : std::min<double>(seed, 65536.0);
  int next = (int)std::min<double>(seed, (double)nTgt);
  // Absorb a sub-wave remainder into this tile: a lone tail launch of a few
  // blocks runs at tiny occupancy, while folded into the fat launch the
  // scheduler overlaps it with the last wave's stragglers.
  if (tiled && (long)nTgt - next < wave) next = (int)nTgt;

  const auto wall0 = std::chrono::steady_clock::now();
  while (res.targetsDone < nTgt) {
    CUDA_CHECK(cudaEventRecord(a));
    if (tiled) {
      const int blocks = (next + RACE_TILE - 1) / RACE_TILE;
      raceTiledKernel<<<blocks, RACE_TILE>>>(
          d_source, d_target, d_force, d_raceOut, (int)nSrc,
          (int)res.targetsDone, next, (int)nTgt);
    } else {
      raceTileKernel<<<next, 256, 3 * 256 * sizeof(double)>>>(
          d_source, d_target, d_force, d_raceOut, (int)nSrc,
          (int)res.targetsDone, (int)nTgt);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    res.elapsedCudaMs += (double)ms;
    res.targetsDone += (size_t)next;
    res.tiles++;

    if (res.targetsDone < nTgt && res.elapsedCudaMs > thresholdMs) break;

    const double rate = (double)res.targetsDone / res.elapsedCudaMs; // tgt/ms
    if (tiled) {
      const long waves = std::min<long>(
          64, std::max<long>(1, std::lround(rate * tileMs / (double)wave)));
      const long remaining = (long)(nTgt - res.targetsDone);
      next = (int)std::min<long>(waves * wave, remaining);
      if (remaining - next < wave) next = (int)remaining; // absorb tail
    } else {
      next = (int)std::min<double>(std::max<double>(rate * tileMs, 512.0),
                                   262144.0);
      next = (int)std::min<size_t>((size_t)next, nTgt - res.targetsDone);
    }
    if (next <= 0) break;
  }
  const auto wall1 = std::chrono::steady_clock::now();
  res.wallMs = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
  res.done = (res.targetsDone == nTgt);
  res.projectedMs =
      res.elapsedCudaMs * (double)nTgt / (double)res.targetsDone;

  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return res;
}

// ------------------------------------------------------------ accuracy -----

struct ErrStats {
  double absL2 = 0.0;
  double relL2 = 0.0;
  double relInf = 0.0;
  double avgMag = 0.0;
  double maxAbs = 0.0;
};

static ErrStats computeErr(const std::vector<double> &tree,
                           const std::vector<double> &ref, int nSample)
{
  double num2 = 0.0, den2 = 0.0, maxAbs = 0.0, maxRef = 0.0, magSum = 0.0;
  for (int i = 0; i < nSample; ++i) {
    double mag2 = 0.0;
    for (int c = 0; c < 3; ++c) {
      const double ub = tree[(size_t)c * nSample + i];
      const double ud = ref[(size_t)c * nSample + i];
      const double e = ub - ud;
      num2 += e * e;
      den2 += ud * ud;
      mag2 += ud * ud;
      maxAbs = std::max(maxAbs, std::abs(e));
      maxRef = std::max(maxRef, std::abs(ud));
    }
    magSum += sqrt(mag2);
  }
  ErrStats es;
  es.absL2 = sqrt(num2);
  es.relL2 = (den2 > 0.0) ? sqrt(num2 / den2) : es.absL2;
  es.relInf = (maxRef > 0.0) ? maxAbs / maxRef : maxAbs;
  es.avgMag = magSum / std::max(1, nSample);
  es.maxAbs = maxAbs;
  return es;
}

static std::vector<double> downloadSamples(const double *d_full, int nTgt,
                                           int stride, int nSample,
                                           thrust::device_vector<double> &d_tmp)
{
  gatherSampleKernel<<<(nSample + 255) / 256, 256>>>(
      d_full, thrust::raw_pointer_cast(d_tmp.data()), nTgt, stride, nSample);
  CUDA_CHECK(cudaGetLastError());
  std::vector<double> h((size_t)3 * nSample);
  CUDA_CHECK(cudaMemcpy(h.data(), thrust::raw_pointer_cast(d_tmp.data()),
                        h.size() * sizeof(double), cudaMemcpyDeviceToHost));
  return h;
}

// ---------------------------------------------------------------- main -----

int main(int argc, char **argv)
{
  try {
    const Options opt = parseArgs(argc, argv);

    std::printf("CFG sources=%s targets=%s strengths=%s\n", SOURCE_FILE,
                TARGET_FILE, FORCE_FILE);
    std::printf("CFG source_group=%d target_group=%d cell=%.6g maxleaf=%d "
                "warmup=%d repeat=%d slack=%.3g floor_ms=%.3g tile_ms=%.3g "
                "sample_slice=%d select=%s\n",
                SOURCE_GROUP, TARGET_GROUP, opt.cellEdge, opt.maxLeaf,
                opt.warmup, opt.repeat, opt.slack, opt.floorMs, opt.tileMs,
                opt.sampleSlice, opt.select.c_str());
    {
      const char *b = std::getenv("TC_BVH_BUILDER");
      const char *k = std::getenv("TC_BUCKETIZER");
      cudaDeviceProp prop{};
      CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
      std::printf("CFG tc_bvh_builder=%s tc_bucketizer=%s gpu=%s\n",
                  (b && *b) ? b : "(default)", (k && *k) ? k : "(default)",
                  prop.name);
      std::printf("CFG direct_kernels=");
      for (size_t i = 0; i < opt.directKernels.size(); ++i)
        std::printf("%s%s", i ? "," : "", opt.directKernels[i].c_str());
      std::printf(" direct_bench=%d tiled_wave_targets=%d sms=%d\n",
                  opt.directBench ? 1 : 0, tiledWaveTargets(),
                  prop.multiProcessorCount);
    }
    std::printf("CFG pvals=");
    for (size_t i = 0; i < opt.pvals.size(); ++i)
      std::printf("%s%d", i ? "," : "", opt.pvals[i]);
    std::printf(" macs=");
    for (size_t i = 0; i < opt.macs.size(); ++i)
      std::printf("%s%.3g", i ? "," : "", opt.macs[i]);
    std::printf(" paths=");
    for (size_t i = 0; i < opt.paths.size(); ++i)
      std::printf("%s%s", i ? "," : "", opt.paths[i].c_str());
    std::printf("\n");
    std::fflush(stdout);

    const auto read0 = std::chrono::steady_clock::now();
    std::vector<vec3d> h_source = readSoA3(SOURCE_FILE, NSOURCE);
    std::vector<vec3d> h_target = readSoA3(TARGET_FILE, NTARGET);
    std::vector<vec3d> h_force = readSoA3(FORCE_FILE, NSOURCE);
    const auto read1 = std::chrono::steady_clock::now();
    std::printf("CFG host_read_ms=%.3f\n",
                std::chrono::duration<double, std::milli>(read1 - read0).count());
    std::fflush(stdout);

    if (opt.select == "radial") {
      const std::vector<int> order = radialEllipsoidOrder();
      permuteGroups(h_source, order, SOURCE_GROUP);
      permuteGroups(h_target, order, TARGET_GROUP);
      permuteGroups(h_force, order, SOURCE_GROUP);
    }

    CUDA_CHECK(cudaFree(0));
    thrust::device_vector<vec3d> d_source(h_source.begin(), h_source.end());
    thrust::device_vector<vec3d> d_target(h_target.begin(), h_target.end());
    thrust::device_vector<vec3d> d_force(h_force.begin(), h_force.end());
    thrust::device_vector<double> d_out((size_t)3 * NTARGET);
    thrust::device_vector<double> d_race((size_t)3 * NTARGET);
    CUDA_CHECK(cudaDeviceSynchronize());
    h_source.clear();
    h_target.clear();
    h_force.clear();

    const vec3d *src = thrust::raw_pointer_cast(d_source.data());
    const vec3d *tgt = thrust::raw_pointer_cast(d_target.data());
    const vec3d *force = thrust::raw_pointer_cast(d_force.data());
    double *out = thrust::raw_pointer_cast(d_out.data());
    double *race = thrust::raw_pointer_cast(d_race.data());

    // Process warm-up: one tiny race tile per kernel (context/module load),
    // discarded.
    raceTileKernel<<<32, 256, 3 * 256 * sizeof(double)>>>(
        src, tgt, force, race, (int)NSOURCE, 0, (int)NTARGET);
    CUDA_CHECK(cudaGetLastError());
    raceTiledKernel<<<32, RACE_TILE>>>(src, tgt, force, race, (int)NSOURCE, 0,
                                       32 * RACE_TILE, (int)NTARGET);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    struct ComboResult {
      float mac;
      std::string path;
      double bestApplyMs = 1e300;
      double bestReapplyMs = 1e300;
      std::vector<double> treeSample; // 3*nSample host, component-major
    };

    for (int P : opt.pvals) {
      const size_t nSrc = (size_t)P * SOURCE_GROUP;
      const size_t nTgt = (size_t)P * TARGET_GROUP;
      int stride = (int)std::max<size_t>(
          1, std::min<size_t>((size_t)opt.sampleSlice,
                              nTgt / (size_t)opt.minSamples));
      const int nSample = (int)(nTgt / (size_t)stride);
      std::printf("\nCASE P=%d nsrc=%zu ntgt=%zu sample_stride=%d nsample=%d "
                  "pairs=%.4e\n",
                  P, nSrc, nTgt, stride, nSample,
                  (double)nSrc * (double)nTgt);
      std::fflush(stdout);

      thrust::device_vector<double> d_sample((size_t)3 * nSample);

      std::vector<ComboResult> combos;
      double maxTreeMs = 0.0;
      for (float mac : opt.directBench ? std::vector<float>{} : opt.macs) {
        for (const std::string &path : opt.paths) {
          setenv("TC_PATH", path.c_str(), 1);
          ComboResult cr;
          cr.mac = mac;
          cr.path = path;
          for (int w = 0; w < opt.warmup; ++w) {
            TreeTiming t =
                runTreecodeOnce(mac, opt, src, nSrc, tgt, nTgt, force, out);
            printTreeLine("WARM", P, mac, path, w, t);
          }
          for (int r = 0; r < opt.repeat; ++r) {
            TreeTiming t =
                runTreecodeOnce(mac, opt, src, nSrc, tgt, nTgt, force, out);
            printTreeLine("TREE", P, mac, path, r, t);
            cr.bestApplyMs = std::min(cr.bestApplyMs, (double)t.applyCudaMs);
            cr.bestReapplyMs =
                std::min(cr.bestReapplyMs, (double)t.reapplyCudaMs);
            maxTreeMs = std::max(maxTreeMs, (double)t.applyCudaMs);
            maxTreeMs = std::max(maxTreeMs, (double)t.reapplyCudaMs);
          }
          std::printf("TREEBEST P=%d mac=%.3g path=%s apply_ms=%.3f "
                      "reapply_ms=%.3f\n",
                      P, mac, path.c_str(), cr.bestApplyMs, cr.bestReapplyMs);
          std::fflush(stdout);
          // d_out still holds this combo's result; grab samples before the
          // next combo overwrites it.
          cr.treeSample =
              downloadSamples(out, (int)nTgt, stride, nSample, d_sample);
          combos.push_back(std::move(cr));
        }
      }

      // Exact sampled reference (independent of the race; always computed,
      // before the races so each kernel's completed output can be selftested
      // in place).
      thrust::device_vector<double> d_ref((size_t)3 * nSample);
      cudaEvent_t ra, rb;
      CUDA_CHECK(cudaEventCreate(&ra));
      CUDA_CHECK(cudaEventCreate(&rb));
      CUDA_CHECK(cudaEventRecord(ra));
      refSampleKernel<<<nSample, 256, 3 * 256 * sizeof(double)>>>(
          src, tgt, force, thrust::raw_pointer_cast(d_ref.data()), (int)nSrc,
          stride, nSample);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(rb));
      CUDA_CHECK(cudaEventSynchronize(rb));
      float refMs = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&refMs, ra, rb));
      CUDA_CHECK(cudaEventDestroy(ra));
      CUDA_CHECK(cudaEventDestroy(rb));
      std::vector<double> h_ref((size_t)3 * nSample);
      CUDA_CHECK(cudaMemcpy(h_ref.data(),
                            thrust::raw_pointer_cast(d_ref.data()),
                            h_ref.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
      std::printf("REF P=%d nsample=%d ref_ms=%.3f\n", P, nSample, refMs);

      // Direct race, once per requested kernel: mac/path-independent, after
      // all treecode runs. Threshold covers the SLOWEST treecode variant, so
      // an aborted direct sum lost against every combo. In --direct-bench
      // mode there is no treecode; run to completion unless capped.
      const double thresholdMs =
          (opt.directCapMs > 0.0)
              ? opt.directCapMs
              : (opt.directBench ? 1e18
                                 : maxTreeMs * (1.0 + opt.slack) + opt.floorMs);
      double bestDirectEffMs = 1e300;
      std::string bestDirectKernel;
      bool bestDirectDone = false;
      for (const std::string &dk : opt.directKernels) {
        DirectResult dr = runDirectRace(dk == "tiled", src, nSrc, tgt, nTgt,
                                        force, race, thresholdMs, opt.tileMs);
        const double pairsDone = (double)nSrc * (double)dr.targetsDone;
        std::printf(
            "DIRECT P=%d kernel=%s nsrc=%zu ntgt=%zu threshold_ms=%.3f "
            "status=%s tiles=%d targets_done=%zu frac=%.4f "
            "elapsed_cuda_ms=%.3f elapsed_wall_ms=%.3f projected_ms=%.3f "
            "pairs_per_s=%.4e\n",
            P, dk.c_str(), nSrc, nTgt, dr.thresholdMs,
            dr.done ? "done" : "aborted", dr.tiles, dr.targetsDone,
            (double)dr.targetsDone / nTgt, dr.elapsedCudaMs, dr.wallMs,
            dr.projectedMs, pairsDone / (dr.elapsedCudaMs * 1e-3));
        std::fflush(stdout);

        // Selftest: the completed race output must match the exact reference
        // to ~fp64 (rsqrtf+Newton vs 1.0/sqrt) at the sampled targets.
        if (opt.selftest && dr.done) {
          std::vector<double> h_race =
              downloadSamples(race, (int)nTgt, stride, nSample, d_sample);
          ErrStats es = computeErr(h_race, h_ref, nSample);
          const double relToMag = es.maxAbs / std::max(es.avgMag, 1e-300);
          std::printf("SELFTEST P=%d kernel=%s race_vs_ref_rel_l2=%.3e "
                      "max_abs_over_avg_mag=%.3e %s\n",
                      P, dk.c_str(), es.relL2, relToMag,
                      (relToMag < 1e-9) ? "PASS" : "FAIL");
          if (relToMag >= 1e-9)
            throw std::runtime_error("selftest failed: race kernel " + dk +
                                     " deviates from exact reference");
        }

        const double eff = dr.done ? dr.elapsedCudaMs : dr.projectedMs;
        if (eff < bestDirectEffMs) {
          bestDirectEffMs = eff;
          bestDirectKernel = dk;
          bestDirectDone = dr.done;
        }
      }
      std::printf("DIRECTBEST P=%d kernel=%s eff_ms=%.3f status=%s\n", P,
                  bestDirectKernel.c_str(), bestDirectEffMs,
                  bestDirectDone ? "done" : "aborted");
      std::fflush(stdout);

      for (const ComboResult &cr : combos) {
        ErrStats es = computeErr(cr.treeSample, h_ref, nSample);
        std::printf("ACC P=%d mac=%.3g path=%s nsample=%d rel_l2=%.6e "
                    "rel_linf=%.6e abs_l2=%.6e avg_mag=%.6e\n",
                    P, cr.mac, cr.path.c_str(), nSample, es.relL2, es.relInf,
                    es.absL2, es.avgMag);
      }
      for (const ComboResult &cr : combos) {
        std::printf("CROSS P=%d mac=%.3g path=%s direct_kernel=%s "
                    "direct_eff_ms=%.3f direct_status=%s tree_apply_ms=%.3f "
                    "tree_reapply_ms=%.3f direct_beats_apply=%d "
                    "direct_beats_reapply=%d\n",
                    P, cr.mac, cr.path.c_str(), bestDirectKernel.c_str(),
                    bestDirectEffMs, bestDirectDone ? "done" : "aborted",
                    cr.bestApplyMs, cr.bestReapplyMs,
                    bestDirectEffMs < cr.bestApplyMs ? 1 : 0,
                    bestDirectEffMs < cr.bestReapplyMs ? 1 : 0);
      }
      std::fflush(stdout);
    }

    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
