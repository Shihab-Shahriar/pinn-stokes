// SPDX-License-Identifier: Apache-2.0
//
// Shared utilities for the CPU-only treecode port (src/cpu/): cuBQL type
// aliases, wall clock, the fp64 point-cloud loader, and the exact Stokeslet
// P2P pair interaction. Plain C++17 + OpenMP -- no CUDA anywhere in src/cpu/.
//
// cuBQL is used header-only: bvh.h pulls in the host builder
// (cuBQL::cpu::spatialMedian, implementation included via
// CUBQL_CPU_BUILDER_IMPLEMENTATION) and compiles under a plain C++ compiler
// (all CUDA includes/attributes are __CUDACC__-guarded).
#pragma once

#include "cuBQL/bvh.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tccpu {

using cuBQL::vec3f;
using cuBQL::vec3d;
using cuBQL::box3f;
using cuBQL::box3d;
using bvh3f = cuBQL::BinaryBVH<float, 3>;

using HostClock = std::chrono::steady_clock;
inline double elapsed_ms(HostClock::time_point a, HostClock::time_point b)
{
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Viscosity + Stokeslet prefactor 1/(8*pi*mu). Deliberate deviation from the
// GPU (stokes_kernel.cuh:23 rounds the prefactor through float): full fp64
// here. Invisible to relL2 because the direct-sum reference uses the same
// value on both sides.
constexpr double MU = 1.0;
inline double prefactor64() { return 1.0 / (8.0 * M_PI * MU); }

// Port of util::loadPointsFP64 (common.cu:72-102): headerless little-endian
// raw fp64 SoA coordinate blocks x[0:N], y[0:N], z[0:N]; N = filesize/24.
inline std::vector<vec3d> loadPointsFP64(const std::string &path)
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
  std::vector<double> raw(3 * n);
  in.read(reinterpret_cast<char *>(raw.data()),
          (std::streamsize)(raw.size() * sizeof(double)));
  if (!in)
    throw std::runtime_error("short read while loading '" + path + "'");

  std::vector<vec3d> pts(n);
  const double *x = raw.data();
  const double *y = raw.data() + n;
  const double *z = raw.data() + 2 * n;
  for (size_t i = 0; i < n; ++i)
    pts[i] = vec3d(x[i], y[i], z[i]);
  return pts;
}

// Direct Oseen pair contribution, fp64 geometry: u += f/r + R (R.f)/r^3 in the
// factored form u_i += (1/r)*(f_i + R_i*q), q = (R.f)/r^2. Port of the fp64
// stokes::p2p overload (stokes_kernel.cuh:74-93). UNSCALED (caller applies the
// 1/(8*pi*mu) prefactor once); skips self/coincident (r2 == 0).
// Deliberate deviation: plain 1.0/sqrt instead of the GPU's rsqrtf seed + fp64
// Newton step (simpler, and exact where the GPU is ~1e-13 relative).
inline void p2p64(const vec3d &T, const vec3d &src, const vec3d &f,
                  double &u0, double &u1, double &u2)
{
  const double Rx = T.x - src.x;
  const double Ry = T.y - src.y;
  const double Rz = T.z - src.z;
  const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
  if (r2 == 0.0) return;                  // skip self / coincident
  const double ir = 1.0 / std::sqrt(r2);
  const double q  = (Rx * f.x + Ry * f.y + Rz * f.z) * (ir * ir);
  u0 = std::fma(ir, std::fma(Rx, q, f.x), u0);
  u1 = std::fma(ir, std::fma(Ry, q, f.y), u1);
  u2 = std::fma(ir, std::fma(Rz, q, f.z), u2);
}

} // namespace tccpu
