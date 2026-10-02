// SPDX-License-Identifier: Apache-2.0
//
// Quick treecode benchmark for the exported 10k ellipsoid MFS RHS case.

#include "treecode.cuh"
#include "mfs_case_io.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

using BenchTreecode = Treecode<mp::BaryStokes>;

static constexpr const char *CASE_DIR = "mfs_ellipsoid_case";
static constexpr const char *SOURCE_FILE = "mfs_ellipsoid_case/proxy_sources.bin";
static constexpr const char *TARGET_FILE = "mfs_ellipsoid_case/colloc_targets.bin";
static constexpr const char *FORCE_FILE = "mfs_ellipsoid_case/proxy_strengths.bin";
// Per-object transforms (centers + row-major R) + the shared source template,
// for the structured (TC_SRC_TEMPLATE) reconstruction path.
static constexpr const char *CONFIG_CSV =
    "mfs/ellipsoid_config_p10000_delta0p2_nv34.csv";
static constexpr const char *SOURCE_TEMPLATE = "mfs/points/s_ellipsoid_p1000.txt";

static constexpr size_t NSOURCE = 6560000;
static constexpr size_t NTARGET = 8640000;
static constexpr int SOURCE_GROUP = 656;
static constexpr int TARGET_GROUP = 864;
static constexpr int SAMPLE_SLICE = 1000;

// Structured source (reconstruct from template + per-object transform). Device
// buffers live for the whole run; enabled by the TC_SRC_TEMPLATE env var.
struct StructuredSource {
  bool enabled = false;
  int nPtsPerObj = 0;
  int nObj = 0;
  thrust::device_vector<vec3d>  tmpl;      // template (nPtsPerObj), reference frame
  thrust::device_vector<double> R;         // nObj*9 row-major
  thrust::device_vector<vec3d>  centers;   // nObj input-frame centers
};

static constexpr double DEFAULT_CELL_EDGE = 10.0;
static constexpr int DEFAULT_MAX_LEAF = 256;

// The execution path is chosen only by the TC_PATH environment variable; it is
// no longer a command-line option or a Config field.
struct Options {
  float mac = -1.0f;
  double cellEdge = DEFAULT_CELL_EDGE;
  int maxLeaf = DEFAULT_MAX_LEAF;
  int warmup = 0;
  int repeat = 1;
  bool profile = false;
};

struct BenchSizes {
  size_t nSource;
  size_t nTarget;
  int sourceGroup;
  int targetGroup;
  int sampleSlice;
  int sample;
};

static BenchSizes makeSizes(bool profile)
{
  BenchSizes s;
  s.nSource = profile ? NSOURCE / 4 : NSOURCE;
  s.nTarget = profile ? NTARGET / 4 : NTARGET;
  s.sourceGroup = SOURCE_GROUP;
  s.targetGroup = TARGET_GROUP;
  s.sampleSlice = SAMPLE_SLICE;
  s.sample = (int)(s.nTarget / s.sampleSlice);
  return s;
}

struct Timing {
  double wallMs = 0.0;
  float cudaMs = 0.0f;
  double reapplyWallMs = 0.0;
  float reapplyCudaMs = 0.0f;
  BenchTreecode::Stats stats{};          // apply stats
  BenchTreecode::Stats reapplyStats{};   // reapply stats (cached near field)
  size_t srcGeomBytes = 0;               // treecode-held source geometry
  size_t vramUsedBytes = 0;              // free-mem delta across build+apply
};

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

static void usage(const char *argv0)
{
  std::fprintf(stderr,
               "usage: %s --mac=VALUE [--cell=10] [--maxleaf=256] "
               "[--warmup=0] [--repeat=1] [--profile]\n"
               "  (execution path is selected by the TC_PATH environment "
               "variable; default split-warp)\n",
               argv0);
}

static Options parseArgs(int argc, char **argv)
{
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (startsWith(a, "--mac=") || startsWith(a, "--tc-mac="))
      opt.mac = (float)parseDouble(a.substr(a.find('=') + 1), "mac");
    else if (startsWith(a, "--cell=") || startsWith(a, "--tc-cell="))
      opt.cellEdge = parseDouble(a.substr(a.find('=') + 1), "cell");
    else if (startsWith(a, "--maxleaf=") || startsWith(a, "--tc-maxleaf="))
      opt.maxLeaf = parseInt(a.substr(a.find('=') + 1), "maxleaf");
    else if (startsWith(a, "--warmup="))
      opt.warmup = parseInt(valueAfter(a, "--warmup="), "warmup");
    else if (startsWith(a, "--repeat="))
      opt.repeat = parseInt(valueAfter(a, "--repeat="), "repeat");
    else if (a == "--profile")
      opt.profile = true;
    else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  if (opt.mac < 0.0f) {
    usage(argv[0]);
    throw std::runtime_error("--mac is required and must be >= 0");
  }
  if (opt.warmup < 0 || opt.repeat <= 0)
    throw std::runtime_error("warmup must be >= 0 and repeat must be > 0");
  return opt;
}

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

__global__ void directSampleKernel(const vec3d *__restrict__ src,
                                   const vec3d *__restrict__ target,
                                   const vec3d *__restrict__ force,
                                   double *__restrict__ out,
                                   int nSource, int sampleSlice, int nSample)
{
  const int sample = blockIdx.x;
  const int tid = threadIdx.x;
  const int targetIdx = sample * sampleSlice;
  const vec3d t = target[targetIdx];

  double ux = 0.0, uy = 0.0, uz = 0.0;
  constexpr double pref = 0.039788735772973836; // 1/(8*pi), mu=1
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
    ux += pref * (f.x * ir + rx * rdf * ir3);
    uy += pref * (f.y * ir + ry * rdf * ir3);
    uz += pref * (f.z * ir + rz * rdf * ir3);
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
                                   int nTarget, int sampleSlice, int nSample)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nSample) return;
  const int targetIdx = i * sampleSlice;
  sampleOut[(size_t)0 * nSample + i] = full[(size_t)0 * nTarget + targetIdx];
  sampleOut[(size_t)1 * nSample + i] = full[(size_t)1 * nTarget + targetIdx];
  sampleOut[(size_t)2 * nSample + i] = full[(size_t)2 * nTarget + targetIdx];
}

static void reportError(const vec3d *d_source,
                        const vec3d *d_target,
                        const vec3d *d_force,
                        const double *d_tree,
                        const BenchSizes &sz)
{
  const int nSample = sz.sample;
  thrust::device_vector<double> d_direct((size_t)3 * nSample);
  thrust::device_vector<double> d_tree_sample((size_t)3 * nSample);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  directSampleKernel<<<nSample, 256, 3 * 256 * sizeof(double)>>>(
      d_source, d_target, d_force, thrust::raw_pointer_cast(d_direct.data()),
      (int)sz.nSource, sz.sampleSlice, nSample);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float directMs = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&directMs, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  gatherSampleKernel<<<(nSample + 255) / 256, 256>>>(
      d_tree, thrust::raw_pointer_cast(d_tree_sample.data()),
      (int)sz.nTarget, sz.sampleSlice, nSample);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> direct((size_t)3 * nSample);
  std::vector<double> tree((size_t)3 * nSample);
  CUDA_CHECK(cudaMemcpy(direct.data(), thrust::raw_pointer_cast(d_direct.data()),
                        direct.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(tree.data(), thrust::raw_pointer_cast(d_tree_sample.data()),
                        tree.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));

  double num2 = 0.0, den2 = 0.0, maxAbs = 0.0, maxRef = 0.0, magSum = 0.0;
  for (int i = 0; i < nSample; ++i) {
    double mag2 = 0.0;
    for (int c = 0; c < 3; ++c) {
      const double ub = tree[(size_t)c * nSample + i];
      const double ud = direct[(size_t)c * nSample + i];
      const double e = ub - ud;
      num2 += e * e;
      den2 += ud * ud;
      mag2 += ud * ud;
      maxAbs = std::max(maxAbs, std::abs(e));
      maxRef = std::max(maxRef, std::abs(ud));
    }
    magSum += sqrt(mag2);
  }

  const double absL2 = sqrt(num2);
  const double relL2 = (den2 > 0.0) ? sqrt(num2 / den2) : absL2;
  const double relInf = (maxRef > 0.0) ? maxAbs / maxRef : maxAbs;
  std::printf(
      "direct_sample_targets %d\n"
      "direct_sample_slice   %d\n"
      "direct_sum_ms         %.3f\n"
      "avg_velocity_mag      %.6e\n"
      "abs_l2_error          %.6e\n"
      "rel_l2_error          %.6e\n"
      "rel_linf_error        %.6e\n\n",
      nSample, sz.sampleSlice, directMs, magSum / nSample, absL2, relL2, relInf);
}

static Timing runOnce(const Options &opt,
                      const BenchSizes &sz,
                      const vec3d *d_source,
                      const vec3d *d_target,
                      const vec3d *d_force,
                      double *d_out,
                      const StructuredSource &ss)
{
  BenchTreecode::Config cfg;
  cfg.mac = opt.mac;
  cfg.cellEdge = opt.cellEdge;
  cfg.maxLeaf = opt.maxLeaf;
  cfg.sourceGroupSize = sz.sourceGroup;
  cfg.targetGroupSize = sz.targetGroup;
  cfg.skipSameGroup = false;

  BenchTreecode tc(cfg);
  if (ss.enabled)
    tc.setStructuredSource(thrust::raw_pointer_cast(ss.tmpl.data()),
                           ss.nPtsPerObj,
                           thrust::raw_pointer_cast(ss.R.data()),
                           thrust::raw_pointer_cast(ss.centers.data()), ss.nObj);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Free memory before the treecode allocates anything, so freeBefore-freeAfter
  // is the tree's own footprint (source geometry included).
  size_t freeBefore = 0, totalMem = 0;
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemGetInfo(&freeBefore, &totalMem));

  Timing t;

  // apply (keep the tree for reapply + VRAM query).
  CUDA_CHECK(cudaEventRecord(start));
  const auto wall0 = std::chrono::steady_clock::now();
  tc.apply(d_source, sz.nSource, d_target, sz.nTarget, d_force, d_out,
           /*will_reuse_tree=*/true);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  const auto wall1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventElapsedTime(&t.cudaMs, start, stop));
  t.wallMs = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
  t.stats = tc.stats();

  size_t freeAfter = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeAfter, &totalMem));
  t.vramUsedBytes = (freeBefore > freeAfter) ? (freeBefore - freeAfter) : 0;
  t.srcGeomBytes = tc.sourceGeomBytes();

  // reapply (same forces -> same result; times the cached near field).
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

static void printTiming(const char *label, int i, const Timing &t)
{
  const auto &s = t.stats;
  const auto &r = t.reapplyStats;
  const double pieces = s.bucketMs + s.targetBucketMs + s.buildBvhMs +
                        s.prepForcesMs + s.upwardMs +
                        (double)s.travMs + (double)s.p2pMs;
  std::printf(
      "%s %d\n"
      "total_cuda_ms    %.3f\n"
      "total_wall_ms    %.3f\n"
      "src_bucket_ms    %.3f\n"
      "tgt_bucket_ms    %.3f\n"
      "bvh_ms           %.3f\n"
      "prep_force_ms    %.3f\n"
      "upward_ms        %.3f\n"
      "traverse_m2p_ms  %.3f\n"
      "p2p_ms           %.3f\n"
      "group_ms         %.3f\n"
      "sum_pieces_ms    %.3f\n"
      "reapply_cuda_ms  %.3f\n"
      "reapply_wall_ms  %.3f\n"
      "reapply_m2p_ms   %.3f\n"
      "reapply_p2p_ms   %.3f\n"
      "source_buckets   %u\n"
      "target_buckets   %u\n"
      "nodes            %u\n"
      "near_pairs       %lld\n"
      "direct_p2p       %lld\n"
      "src_geom_bytes   %zu\n"
      "src_geom_MB      %.2f\n"
      "vram_used_MB     %.2f\n\n",
      label, i, t.cudaMs, t.wallMs, s.bucketMs, s.targetBucketMs,
      s.buildBvhMs, s.prepForcesMs, s.upwardMs, (double)s.travMs,
      (double)s.p2pMs, (double)s.groupMs, pieces,
      t.reapplyCudaMs, t.reapplyWallMs, (double)r.travMs, (double)r.p2pMs,
      s.numSourceBuckets, s.numTargetBuckets,
      s.numNodes, s.nPairs, s.totalP2P,
      t.srcGeomBytes, (double)t.srcGeomBytes / (1024.0 * 1024.0),
      (double)t.vramUsedBytes / (1024.0 * 1024.0));
}

int main(int argc, char **argv)
{
  try {
    const Options opt = parseArgs(argc, argv);
    const BenchSizes sz = makeSizes(opt.profile);

    std::printf("case           %s\n", CASE_DIR);
    std::printf("profile        %d\n", opt.profile ? 1 : 0);
    std::printf("sources        %s (%zu)\n", SOURCE_FILE, sz.nSource);
    std::printf("targets        %s (%zu)\n", TARGET_FILE, sz.nTarget);
    std::printf("strengths      %s\n", FORCE_FILE);
    std::printf("mac            %.6g\n", opt.mac);
    std::printf("cell           %.6g\n", opt.cellEdge);
    std::printf("maxleaf        %d\n", opt.maxLeaf);
    {
      const char *tcPath = std::getenv("TC_PATH");
      std::printf("tc_path        %s\n", (tcPath && *tcPath) ? tcPath
                                                             : "split-warp");
    }
    std::printf("warmup         %d\n", opt.warmup);
    std::printf("repeat         %d\n\n", opt.repeat);
    std::fflush(stdout);

    const auto read0 = std::chrono::steady_clock::now();
    std::vector<vec3d> h_source = readSoA3(SOURCE_FILE, sz.nSource, NSOURCE);
    std::vector<vec3d> h_target = readSoA3(TARGET_FILE, sz.nTarget, NTARGET);
    std::vector<vec3d> h_force = readSoA3(FORCE_FILE, sz.nSource, NSOURCE);
    const auto read1 = std::chrono::steady_clock::now();
    std::printf("host_read_ms   %.3f\n",
                std::chrono::duration<double, std::milli>(read1 - read0).count());
    std::fflush(stdout);

    CUDA_CHECK(cudaFree(0));
    const auto up0 = std::chrono::steady_clock::now();
    thrust::device_vector<vec3d> d_source(h_source.begin(), h_source.end());
    thrust::device_vector<vec3d> d_target(h_target.begin(), h_target.end());
    thrust::device_vector<vec3d> d_force(h_force.begin(), h_force.end());
    thrust::device_vector<double> d_out((size_t)3 * sz.nTarget);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto up1 = std::chrono::steady_clock::now();
    std::printf("upload_ms      %.3f\n\n",
                std::chrono::duration<double, std::milli>(up1 - up0).count());
    std::fflush(stdout);

    // ---- structured (TC_SRC_TEMPLATE) source: template + per-object transform ----
    StructuredSource ss;
    if (const char *e = std::getenv("TC_SRC_TEMPLATE");
        e && *e && std::strcmp(e, "0") != 0) {
      ss.enabled = true;
      bool hasRef = false;
      std::vector<mfscase::Row> rows = mfscase::loadCSV(CONFIG_CSV, hasRef);
      ss.nObj = (int)rows.size();
      int nT = 0;
      std::vector<double> tmplFlat = mfscase::loadPointsASCII(SOURCE_TEMPLATE, nT);
      ss.nPtsPerObj = nT;
      std::printf("structured     ON (csv=%s P=%d, template=%s N=%d)\n",
                  CONFIG_CSV, ss.nObj, SOURCE_TEMPLATE, ss.nPtsPerObj);
      if ((size_t)ss.nObj * (size_t)ss.nPtsPerObj != sz.nSource)
        throw std::runtime_error("structured: nObj*nPtsPerObj != nSource "
                                 "(needs the full, non-profile case)");
      if (ss.nPtsPerObj != sz.sourceGroup)
        throw std::runtime_error("structured: template point count != SOURCE_GROUP");

      // Host self-check vs the frozen bin: recon(t=k*N+i) = R_k*tmpl[i] + c_k.
      // Guards CSV row order == bin object order and the R convention.
      double maxAbs = 0.0;
      auto checkT = [&](size_t t) {
        const int k = (int)(t / (size_t)ss.nPtsPerObj);
        const int i = (int)(t % (size_t)ss.nPtsPerObj);
        const double *R = rows[k].R, *c = rows[k].c;
        const double yx = tmplFlat[3 * i], yy = tmplFlat[3 * i + 1],
                     yz = tmplFlat[3 * i + 2];
        const double rx = R[0] * yx + R[1] * yy + R[2] * yz + c[0];
        const double ry = R[3] * yx + R[4] * yy + R[5] * yz + c[1];
        const double rz = R[6] * yx + R[7] * yy + R[8] * yz + c[2];
        const vec3d &p = h_source[t];
        maxAbs = std::max(maxAbs, std::abs(rx - p.x));
        maxAbs = std::max(maxAbs, std::abs(ry - p.y));
        maxAbs = std::max(maxAbs, std::abs(rz - p.z));
      };
      for (int i = 0; i < ss.nPtsPerObj; ++i) checkT((size_t)i);  // all of object 0
      for (size_t t = 0; t < sz.nSource; t += 6553) checkT(t);    // strided sample
      std::printf("struct_recon_max_abs_err %.3e\n", maxAbs);
      std::fflush(stdout);
      if (maxAbs > 1e-6)
        throw std::runtime_error("structured: reconstruction does not match the "
                                 "frozen source bin (ordering/convention mismatch)");

      std::vector<vec3d> h_tmpl(ss.nPtsPerObj);
      for (int i = 0; i < ss.nPtsPerObj; ++i)
        h_tmpl[i] = vec3d(tmplFlat[3 * i], tmplFlat[3 * i + 1], tmplFlat[3 * i + 2]);
      std::vector<vec3d> h_centers(ss.nObj);
      std::vector<double> h_R((size_t)ss.nObj * 9);
      for (int k = 0; k < ss.nObj; ++k) {
        h_centers[k] = vec3d(rows[k].c[0], rows[k].c[1], rows[k].c[2]);
        for (int m = 0; m < 9; ++m) h_R[(size_t)k * 9 + m] = rows[k].R[m];
      }
      ss.tmpl.assign(h_tmpl.begin(), h_tmpl.end());
      ss.R.assign(h_R.begin(), h_R.end());
      ss.centers.assign(h_centers.begin(), h_centers.end());
    } else {
      std::printf("structured     OFF\n");
    }
    std::fflush(stdout);

    h_source.clear();
    h_target.clear();
    h_force.clear();

    const vec3d *src = thrust::raw_pointer_cast(d_source.data());
    const vec3d *tgt = thrust::raw_pointer_cast(d_target.data());
    const vec3d *force = thrust::raw_pointer_cast(d_force.data());
    double *out = thrust::raw_pointer_cast(d_out.data());

    for (int i = 0; i < opt.warmup; ++i)
      printTiming("warmup", i, runOnce(opt, sz, src, tgt, force, out, ss));
    for (int i = 0; i < opt.repeat; ++i)
      printTiming("run", i, runOnce(opt, sz, src, tgt, force, out, ss));

    reportError(src, tgt, force, out, sz);

    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
