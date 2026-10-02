// SPDX-License-Identifier: Apache-2.0
//
// Pure-host ASCII loaders for the ellipsoid MFS case, factored so the
// treecode-only bench (mfs_ellipsoid_treecode_bench.cu) can read the per-object
// transforms + reference template WITHOUT pulling the heavy PETSc/CUDA
// mfs_broms.cuh. Namespaced (mfscase::) and `inline` so it can be included
// alongside other headers with no symbol clashes. The full MFS solver
// (test_mfs_ellipsoid.cu) keeps its own local copies; this header is additive.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mfscase {

// One ellipsoid's per-object state parsed from the config/mobility CSV.
struct Row {
  double c[3], R[9], f[3], t[3], v[3], w[3];
};

// Auto-detects the two CSV layouts by column count (lifted verbatim from
// test_mfs_ellipsoid.cu):
//   42 cols: mobility  -- center@18, R@21, force@30, torque@33, vel@36, omega@39
//   27 cols: config    -- center@9,  R@12, force@21, torque@24 (no reference vel)
// hasRef = whether reference velocities were present.
inline std::vector<Row> loadCSV(const std::string &path, bool &hasRef)
{
  std::ifstream in(path);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
  std::vector<Row> rows;
  std::string line;
  std::getline(in, line);  // header
  int ncol = 1;
  for (char ch : line) if (ch == ',') ++ncol;
  hasRef = (ncol >= 42);
  const int cOff = hasRef ? 18 : 9;    // center
  const int rOff = hasRef ? 21 : 12;   // rotation r11..r33 (row-major)
  const int fOff = hasRef ? 30 : 21;   // force
  const int tOff = hasRef ? 33 : 24;   // torque
  const int need = hasRef ? 42 : 27;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    for (char &ch : line) if (ch == ',') ch = ' ';
    std::istringstream ss(line);
    std::vector<double> a(need);
    bool ok = true;
    for (int i = 0; i < need; ++i) if (!(ss >> a[i])) { ok = false; break; }
    if (!ok) continue;
    Row r{};
    for (int d = 0; d < 3; ++d) {
      r.c[d] = a[cOff + d];
      r.f[d] = a[fOff + d];
      r.t[d] = a[tOff + d];
      if (hasRef) { r.v[d] = a[36 + d]; r.w[d] = a[39 + d]; }
    }
    for (int e = 0; e < 9; ++e) r.R[e] = a[rOff + e];
    rows.push_back(r);
  }
  return rows;
}

// Load whitespace-separated "x y z" fp64 points. Returns AoS (3*n), sets n.
inline std::vector<double> loadPointsASCII(const std::string &path, int &n)
{
  std::ifstream in(path);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
  std::vector<double> v;
  double x;
  while (in >> x) v.push_back(x);
  if (v.size() % 3 != 0) {
    std::fprintf(stderr, "%s: %zu doubles not divisible by 3\n", path.c_str(),
                 v.size());
    std::exit(1);
  }
  n = (int)(v.size() / 3);
  return v;
}

} // namespace mfscase
