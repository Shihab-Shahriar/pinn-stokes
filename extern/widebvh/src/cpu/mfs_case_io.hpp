// SPDX-License-Identifier: Apache-2.0
//
// Loaders for the widebvh ellipsoid MFS case, adapted from
// ~/programs/widebvh/src/mfs_case_io.cuh (pure-host, copied with permission —
// that repo is read-only for us). Extended with binary SoA loaders for the
// frozen full-scale export in widebvh/mfs_ellipsoid_case/ (DATA_FORMAT.txt:
// little-endian fp64 SoA x[]y[]z[]; the two coordinate files carry a trailing
// int32 ellipsoid_id[] block). All loaders return interleaved-xyz AoS, the
// layout the solver and pvfmm use.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

inline long fileSize(const std::string &path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
  return (long)in.tellg();
}

// Read one SoA component block of n doubles and scatter into AoS slot d.
inline void readSoAComponent(std::ifstream &in, std::vector<double> &aos,
                             size_t n, int d)
{
  std::vector<double> buf(n);
  in.read(reinterpret_cast<char *>(buf.data()), (std::streamsize)(n * sizeof(double)));
  if (!in) { std::fprintf(stderr, "short read (component %d)\n", d); std::exit(1); }
  for (size_t i = 0; i < n; ++i) aos[3 * i + d] = buf[i];
}

// Coordinate export: double x[n] y[n] z[n] + int32 ellipsoid_id[n].
// Returns interleaved AoS (3*n); sets n; fills ids if non-null.
inline std::vector<double> loadSoACoords(const std::string &path, size_t &n,
                                         std::vector<int32_t> *ids = nullptr)
{
  const long sz = fileSize(path);
  if (sz % (3 * (long)sizeof(double) + (long)sizeof(int32_t)) != 0) {
    std::fprintf(stderr, "%s: size %ld not divisible by 28\n", path.c_str(), sz);
    std::exit(1);
  }
  n = (size_t)(sz / 28);
  std::ifstream in(path, std::ios::binary);
  std::vector<double> aos(3 * n);
  for (int d = 0; d < 3; ++d) readSoAComponent(in, aos, n, d);
  if (ids) {
    ids->resize(n);
    in.read(reinterpret_cast<char *>(ids->data()), (std::streamsize)(n * sizeof(int32_t)));
    if (!in) { std::fprintf(stderr, "%s: short read (ids)\n", path.c_str()); std::exit(1); }
  }
  return aos;
}

// Field export (strengths/velocities): double fx[n] fy[n] fz[n], no ids.
inline std::vector<double> loadSoAField(const std::string &path, size_t &n)
{
  const long sz = fileSize(path);
  if (sz % (3 * (long)sizeof(double)) != 0) {
    std::fprintf(stderr, "%s: size %ld not divisible by 24\n", path.c_str(), sz);
    std::exit(1);
  }
  n = (size_t)(sz / 24);
  std::ifstream in(path, std::ios::binary);
  std::vector<double> aos(3 * n);
  for (int d = 0; d < 3; ++d) readSoAComponent(in, aos, n, d);
  return aos;
}

} // namespace mfscase
