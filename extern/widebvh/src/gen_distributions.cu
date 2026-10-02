// SPDX-License-Identifier: Apache-2.0
//
// gen_distributions - synthetic particle-cloud generator for the split-warpspec
// vs split-warp treecode sweep.
//
// Writes two_ball-format binaries: headerless little-endian fp64, structure of
// arrays [x0..xN-1][y0..yN-1][z0..zN-1], N inferred by the reader from
// filesize/24 (see util::loadPointsFP64 in src/common.cu). Each distribution is
// one self-contained host function; main() generates all of them (or one, via
// --only=NAME) into an output directory.
//
// This is a one-shot offline generator, so it is deliberately plain host C++
// (std::mt19937_64 + std::ofstream): no device kernels, no cuBQL/treecode deps.
// It is a .cu only so it lives with the rest of the project sources and builds
// with the same toolchain.
//
// The clouds span a controlled range of *tile-scale heterogeneity* (how much the
// M2P interaction-list length varies between spatially-adjacent targets), which
// is the quantity split-warpspec's consumer-warp load-balancing should exploit:
//
//   uniform_cube   - flat density            -> ~no variance   (floor / control)
//   solid_ball     - one homogeneous volume  -> low variance
//   plummer        - smooth central cusp+halo-> radial contrast (Plummer model)
//   sphere_surface - one homogeneous surface -> low variance, HUGE surface area
//                                               (isolates "surface" from "variance")
//   multi_shells   - many small surfaces      -> high variance  (ellipsoid-MFS analog)
//   clumps         - dense blobs + sparse halo-> high variance  (density contrast)
//   fractal        - cluster-of-clusters      -> max  variance  (multi-scale)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using std::size_t;

// ---------------------------------------------------------------------------
// I/O: write three contiguous fp64 coordinate blocks (x, then y, then z).
// ---------------------------------------------------------------------------
void writeSoA(const std::string &path, const std::vector<double> &x,
              const std::vector<double> &y, const std::vector<double> &z) {
  const size_t n = x.size();
  if (y.size() != n || z.size() != n)
    throw std::runtime_error("writeSoA: x/y/z size mismatch");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    throw std::runtime_error("writeSoA: cannot open '" + path + "'");
  const std::streamsize bytes = (std::streamsize)(n * sizeof(double));
  out.write(reinterpret_cast<const char *>(x.data()), bytes);
  out.write(reinterpret_cast<const char *>(y.data()), bytes);
  out.write(reinterpret_cast<const char *>(z.data()), bytes);
  if (!out)
    throw std::runtime_error("writeSoA: short write to '" + path + "'");
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
struct V3 { double x, y, z; };

// uniform point on the unit sphere (Marsaglia via 3 Gaussians, normalized).
inline V3 randUnit(std::mt19937_64 &rng, std::normal_distribution<double> &gauss) {
  for (;;) {
    const double gx = gauss(rng), gy = gauss(rng), gz = gauss(rng);
    const double r2 = gx * gx + gy * gy + gz * gz;
    if (r2 > 1e-300) {
      const double inv = 1.0 / std::sqrt(r2);
      return {gx * inv, gy * inv, gz * inv};
    }
  }
}

// ---------------------------------------------------------------------------
// distributions
// ---------------------------------------------------------------------------

// 1. uniform in the cube [0,L]^3.
void gen_uniform_cube(std::vector<double> &x, std::vector<double> &y,
                      std::vector<double> &z, size_t N, std::mt19937_64 &rng,
                      double L = 1000.0) {
  x.resize(N); y.resize(N); z.resize(N);
  std::uniform_real_distribution<double> U(0.0, L);
  for (size_t i = 0; i < N; ++i) { x[i] = U(rng); y[i] = U(rng); z[i] = U(rng); }
}

// 2. uniform inside one solid sphere (r = R*u^(1/3) for uniform volume density).
void gen_solid_ball(std::vector<double> &x, std::vector<double> &y,
                    std::vector<double> &z, size_t N, std::mt19937_64 &rng,
                    double R = 500.0, double cx = 500.0, double cy = 500.0,
                    double cz = 500.0) {
  x.resize(N); y.resize(N); z.resize(N);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::normal_distribution<double> gauss(0.0, 1.0);
  for (size_t i = 0; i < N; ++i) {
    const double r = R * std::cbrt(U(rng));
    const V3 u = randUnit(rng, gauss);
    x[i] = cx + r * u.x; y[i] = cy + r * u.y; z[i] = cz + r * u.z;
  }
}

// 2b. Plummer model (astrophysics): radial density rho(r) ∝ (1 + r^2/a^2)^(-5/2),
//     enclosed-mass CDF M(r)/M = r^3/(a^2+r^2)^(3/2). Inverse-CDF sampling with
//     u~U(0,1): r = a / sqrt(u^(-2/3) - 1) (u->0 center, u->1 halo). Origin-
//     centered; a smooth single-scale central concentration with an extended
//     heavy tail. Per spec: a=1 and a hard cutoff at +/-cutoff in each Cartesian
//     axis (rejection-resampled; the tail beyond +/-100 is ~0.02% of mass).
void gen_plummer(std::vector<double> &x, std::vector<double> &y,
                 std::vector<double> &z, size_t N, std::mt19937_64 &rng,
                 double a = 1.0, double cutoff = 100.0) {
  x.resize(N); y.resize(N); z.resize(N);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::normal_distribution<double> gauss(0.0, 1.0);
  for (size_t i = 0; i < N; ++i) {
    for (;;) {
      const double u = U(rng);
      if (u <= 0.0 || u >= 1.0) continue; // r finite, nonzero denom
      const double r = a / std::sqrt(std::pow(u, -2.0 / 3.0) - 1.0);
      const V3 d = randUnit(rng, gauss);
      const double px = r * d.x, py = r * d.y, pz = r * d.z;
      if (std::fabs(px) <= cutoff && std::fabs(py) <= cutoff &&
          std::fabs(pz) <= cutoff) {
        x[i] = px; y[i] = py; z[i] = pz;
        break;
      }
    }
  }
}

// 3. all points on the surface of ONE sphere (uniform on S^2). Extreme
//    surface-to-volume, but homogeneous (every target is equivalent).
void gen_sphere_surface(std::vector<double> &x, std::vector<double> &y,
                        std::vector<double> &z, size_t N, std::mt19937_64 &rng,
                        double R = 500.0, double cx = 500.0, double cy = 500.0,
                        double cz = 500.0) {
  x.resize(N); y.resize(N); z.resize(N);
  std::normal_distribution<double> gauss(0.0, 1.0);
  for (size_t i = 0; i < N; ++i) {
    const V3 u = randUnit(rng, gauss);
    x[i] = cx + R * u.x; y[i] = cy + R * u.y; z[i] = cz + R * u.z;
  }
}

// 4. K small sphere surfaces (thin shells) scattered through the box. Many
//    separated surfaces with voids between them -> high tile-scale variance.
void gen_multi_shells(std::vector<double> &x, std::vector<double> &y,
                      std::vector<double> &z, size_t N, std::mt19937_64 &rng,
                      double L = 1000.0, size_t K = 1000, double r = 9.0) {
  x.resize(N); y.resize(N); z.resize(N);
  std::uniform_real_distribution<double> Uc(r, L - r);
  std::normal_distribution<double> gauss(0.0, 1.0);
  size_t i = 0;
  for (size_t k = 0; k < K; ++k) {
    const double sx = Uc(rng), sy = Uc(rng), sz = Uc(rng);
    // distribute N across shells as evenly as possible (last shells get +1).
    const size_t base = N / K, extra = N % K;
    const size_t cnt = base + (k < extra ? 1 : 0);
    for (size_t j = 0; j < cnt && i < N; ++j, ++i) {
      const V3 u = randUnit(rng, gauss);
      x[i] = sx + r * u.x; y[i] = sy + r * u.y; z[i] = sz + r * u.z;
    }
  }
  // pad if rounding left a few (shouldn't, but be safe).
  for (; i < N; ++i) { x[i] = Uc(rng); y[i] = Uc(rng); z[i] = Uc(rng); }
}

// 5. density contrast: fraction f of points in K tight Gaussian blobs, the rest
//    uniform halo in the box.
void gen_clumps(std::vector<double> &x, std::vector<double> &y,
                std::vector<double> &z, size_t N, std::mt19937_64 &rng,
                double L = 1000.0, size_t K = 50, double sigma = 8.0,
                double f = 0.85) {
  x.resize(N); y.resize(N); z.resize(N);
  std::uniform_real_distribution<double> Uc(3.0 * sigma, L - 3.0 * sigma);
  std::uniform_real_distribution<double> Ubox(0.0, L);
  std::normal_distribution<double> gauss(0.0, 1.0);
  // blob centers
  std::vector<V3> centers(K);
  for (size_t k = 0; k < K; ++k) centers[k] = {Uc(rng), Uc(rng), Uc(rng)};
  const size_t nBlob = (size_t)(f * (double)N);
  std::uniform_int_distribution<size_t> pick(0, K - 1);
  size_t i = 0;
  for (; i < nBlob; ++i) {
    const V3 c = centers[pick(rng)];
    x[i] = c.x + sigma * gauss(rng);
    y[i] = c.y + sigma * gauss(rng);
    z[i] = c.z + sigma * gauss(rng);
  }
  for (; i < N; ++i) { x[i] = Ubox(rng); y[i] = Ubox(rng); z[i] = Ubox(rng); }
}

// 6. hierarchical multi-scale (cluster-of-clusters). Children share their
//    parent's offset, so density is self-similar across scales -> tree depth
//    (and thus M2P-list length) varies enormously from point to point.
void fractal_emit(double cx, double cy, double cz, int level, size_t count,
                  double R0, double s, int b, int D, std::mt19937_64 &rng,
                  std::normal_distribution<double> &gauss, std::vector<double> &x,
                  std::vector<double> &y, std::vector<double> &z) {
  if (count == 0) return;
  if (level >= D || count <= (size_t)b) {
    // leaf: scatter `count` points in a small Gaussian sized by this level.
    const double sigma = R0 * std::pow(s, (double)level);
    for (size_t j = 0; j < count; ++j) {
      x.push_back(cx + sigma * gauss(rng));
      y.push_back(cy + sigma * gauss(rng));
      z.push_back(cz + sigma * gauss(rng));
    }
    return;
  }
  const double radius = R0 * std::pow(s, (double)level);
  for (int c = 0; c < b; ++c) {
    const size_t cnt = count / b + ((size_t)c < (count % b) ? 1 : 0);
    const V3 u = randUnit(rng, gauss);
    fractal_emit(cx + radius * u.x, cy + radius * u.y, cz + radius * u.z,
                 level + 1, cnt, R0, s, b, D, rng, gauss, x, y, z);
  }
}

void gen_fractal(std::vector<double> &x, std::vector<double> &y,
                 std::vector<double> &z, size_t N, std::mt19937_64 &rng,
                 double L = 1000.0, double R0 = 400.0, double s = 0.4, int b = 8,
                 int D = 8) {
  x.clear(); y.clear(); z.clear();
  x.reserve(N); y.reserve(N); z.reserve(N);
  std::normal_distribution<double> gauss(0.0, 1.0);
  fractal_emit(L * 0.5, L * 0.5, L * 0.5, 0, N, R0, s, b, D, rng, gauss, x, y, z);
  // recursion partitions count exactly, so size == N; guard regardless.
  x.resize(N, L * 0.5); y.resize(N, L * 0.5); z.resize(N, L * 0.5);
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------
using GenFn = std::function<void(std::vector<double> &, std::vector<double> &,
                                 std::vector<double> &, size_t,
                                 std::mt19937_64 &)>;

// Parse a size token into a point count. Accepts a bare integer ("1000000") or
// a decimal-SI suffixed form: k/K = 1e3, m/M = 1e6, g/G = 1e9 (so "4M" ->
// 4000000, "400K" -> 400000). The numeric part may be fractional ("1.5M").
size_t parseSize(const std::string &tok) {
  if (tok.empty())
    throw std::runtime_error("parseSize: empty size token");
  double mult = 1.0;
  std::string num = tok;
  switch (tok.back()) {
    case 'k': case 'K': mult = 1e3; num.pop_back(); break;
    case 'm': case 'M': mult = 1e6; num.pop_back(); break;
    case 'g': case 'G': mult = 1e9; num.pop_back(); break;
    default: break;
  }
  size_t consumed = 0;
  const double val = std::stod(num, &consumed);
  if (num.empty() || consumed != num.size() || val < 0.0)
    throw std::runtime_error("parseSize: bad size token '" + tok + "'");
  return (size_t)std::llround(val * mult);
}

// Canonical size suffix for output filenames, matching the existing
// distros/<name>_<suffix>.bin convention (e.g. 2000000 -> "2M", 400000 ->
// "400k"). Falls back to the raw count when it is not a clean SI multiple.
std::string sizeSuffix(size_t N) {
  char buf[32];
  if (N != 0 && N % 1000000 == 0)
    std::snprintf(buf, sizeof buf, "%zuM", N / 1000000);
  else if (N != 0 && N % 1000 == 0)
    std::snprintf(buf, sizeof buf, "%zuk", N / 1000);
  else
    std::snprintf(buf, sizeof buf, "%zu", N);
  return std::string(buf);
}

void reportBBox(const std::string &name, const std::vector<double> &x,
                const std::vector<double> &y, const std::vector<double> &z) {
  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  const std::vector<double> *c[3] = {&x, &y, &z};
  for (int d = 0; d < 3; ++d)
    for (double v : *c[d]) { lo[d] = std::min(lo[d], v); hi[d] = std::max(hi[d], v); }
  std::printf("  %-15s N=%zu  bbox=[%.1f,%.1f]x[%.1f,%.1f]x[%.1f,%.1f]\n",
              name.c_str(), x.size(), lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
}

} // namespace

int main(int argc, char **argv) {
  std::string outdir;
  size_t N = 1000000;
  uint64_t seed = 12345;
  std::string only;

  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--only=", 0) == 0) only = a.substr(7);
    else if (a == "-h" || a == "--help") {
      std::printf("usage: %s <outdir> [N=1000000] [seed=12345] [--only=NAME]\n"
                  "  N accepts SI suffixes: 4M -> 4000000, 400K -> 400000\n"
                  "  outputs are named <name>_<size>.bin (e.g. plummer_4M.bin)\n"
                  "  distributions: uniform_cube solid_ball plummer "
                  "sphere_surface multi_shells clumps fractal\n",
                  argv[0]);
      return 0;
    } else pos.push_back(a);
  }
  if (pos.empty()) {
    std::fprintf(stderr, "error: output directory required (see --help)\n");
    return 1;
  }
  outdir = pos[0];
  if (pos.size() > 1) N = parseSize(pos[1]);
  if (pos.size() > 2) seed = (uint64_t)std::stoull(pos[2]);
  const std::string suffix = sizeSuffix(N);

  // ordered low -> high tile-scale heterogeneity
  const std::vector<std::pair<std::string, GenFn>> dists = {
      {"uniform_cube",   [](auto &x, auto &y, auto &z, size_t n, auto &r) { gen_uniform_cube(x, y, z, n, r); }},
      {"solid_ball",     [](auto &x, auto &y, auto &z, size_t n, auto &r) { gen_solid_ball(x, y, z, n, r); }},
      {"plummer",        [](auto &x, auto &y, auto &z, size_t n, auto &r) { gen_plummer(x, y, z, n, r); }},
      {"sphere_surface", [](auto &x, auto &y, auto &z, size_t n, auto &r) { gen_sphere_surface(x, y, z, n, r); }},
      {"multi_shells",   [](auto &x, auto &y, auto &z, size_t n, auto &r) { gen_multi_shells(x, y, z, n, r); }},
      {"clumps",         [](auto &x, auto &y, auto &z, size_t n, auto &r) { gen_clumps(x, y, z, n, r); }},
      {"fractal",        [](auto &x, auto &y, auto &z, size_t n, auto &r) { gen_fractal(x, y, z, n, r); }},
  };

  std::printf("gen_distributions: outdir=%s N=%zu (%s) seed=%llu%s\n",
              outdir.c_str(), N, suffix.c_str(), (unsigned long long)seed,
              only.empty() ? "" : (" only=" + only).c_str());

  bool matched = false;
  for (const auto &d : dists) {
    if (!only.empty() && d.first != only) continue;
    matched = true;
    // per-distribution independent, reproducible seed
    const uint64_t h = std::hash<std::string>{}(d.first);
    std::mt19937_64 rng(seed + 0x9E3779B97F4A7C15ull * (h | 1ull));
    std::vector<double> x, y, z;
    d.second(x, y, z, N, rng);
    const std::string path = outdir + "/" + d.first + "_" + suffix + ".bin";
    writeSoA(path, x, y, z);
    reportBBox(d.first, x, y, z);
    std::printf("  -> wrote %s\n", path.c_str());
  }
  if (!only.empty() && !matched) {
    std::fprintf(stderr, "error: unknown distribution '%s'\n", only.c_str());
    return 1;
  }
  return 0;
}
