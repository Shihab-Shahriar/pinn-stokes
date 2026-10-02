// SPDX-License-Identifier: Apache-2.0
//
// Minimal 3D Hilbert space-filling-curve key, extracted from the cornerstone
// octree library so the grid-hilbert / hilbert bucketizers no longer depend on
// vendored cornerstone-octree.
//
// The iHilbert21 body below is a direct port of cstone::iHilbert<uint64_t>
// (forward encode only, maxTreeLevel hardcoded to 21):
//
//   Cornerstone octree -- Copyright (c) 2024 CSCS, ETH Zurich -- MIT License
//   https://github.com/sekelle/cornerstone-octree
//   Hilbert curve by Sebastian Keller <sebastian.f.keller@gmail.com>, based on
//     Yohei Miki, Masayuki Umemura, "GOTHIC: Gravitational oct-tree code
//     accelerated by hierarchical time step controlling",
//     https://doi.org/10.1016/j.newast.2016.10.007
//
// We only need the forward (integer coords -> key) encoder; the decode, 2D, and
// StrongType layers are intentionally not ported.
#pragma once

#include <cstdint>

// Host-portability: this header is shared with the CPU-only treecode port
// (src/cpu/), where nvcc's __host__/__device__ tokens do not exist.
#ifdef __CUDACC__
#  define HILBERT_SFC_HD __host__ __device__
#else
#  define HILBERT_SFC_HD
#endif

namespace util {

// 21-bit-per-axis 3D Hilbert curve. px,py,pz must lie in [0, 2^21); returns a
// 63-bit Hilbert key. Used for both the grid-hilbert within-cell fine curve
// (inputs quantized to fineBitsPerAxis <= 20 bits) and the global Hilbert sort.
HILBERT_SFC_HD inline uint64_t iHilbert21(unsigned px, unsigned py, unsigned pz)
{
  constexpr int kLevels = 21;
  const unsigned mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};

  uint64_t key = 0;

  for (int level = kLevels - 1; level >= 0; --level) {
    unsigned xi = (px >> level) & 1u;
    unsigned yi = (py >> level) & 1u;
    unsigned zi = (pz >> level) & 1u;

    // append 3 bits to the key
    unsigned octant = (xi << 2) | (yi << 1) | zi;
    key = (key << 3) + mortonToHilbert[octant];

    // turn px, py and pz
    px ^= -(xi & ((!yi) | zi));
    py ^= -((xi & (yi | zi)) | (yi & (!zi)));
    pz ^= -((xi & (!yi) & (!zi)) | (yi & (!zi)));

    if (zi) {
      // cyclic rotation
      unsigned pt = px;
      px = py;
      py = pz;
      pz = pt;
    } else if (!yi) {
      // swap x and z
      unsigned pt = px;
      px = pz;
      pz = pt;
    }
  }

  return key;
}

} // namespace util
