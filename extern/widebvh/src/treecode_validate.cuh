// SPDX-License-Identifier: Apache-2.0
//
// Reusable validation helpers for the Stokeslet treecode drivers: a brute-force
// O(N) direct-sum reference per sampled target and the L2 / relative-L2 error
// metric against it. Every per-distribution driver includes this to check the
// treecode against ground truth on a handful of sampled targets.
//
// These depend only on the shared physical kernel (stokes::p2p), NOT on the
// multipole policy MP nor on the treecode engine -- hence they live here, apart
// from both `Treecode` (treecode.cuh) and the generic `util` layer (common.cuh,
// which is intentionally kept free of any physics dependency).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>

#include <cuda_runtime.h>

#include "cuBQL/math/vec.h"
#include "stokes_kernel.cuh"

namespace tcval {

using cuBQL::vec3f;
using cuBQL::vec3d;
using stokes::p2p;

// Brute-force O(N) reference for each sampled target (fp64 accumulator).
// `potential_direct` is component-major double[3*numTargets]. `pos`/`force` are
// the bucket-contiguous source set; `sampleIdx[t]` indexes into it.
__global__ void directSumKernel(const vec3f *pos, const vec3d *force, int N,
                                const int *sampleIdx, int numTargets,
                                float pref, double *potential_direct)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numTargets) return;
  const vec3f T = pos[sampleIdx[tid]];
  double u[3] = {0.0, 0.0, 0.0};
  for (int s = 0; s < N; ++s)
    p2p(T, pos[s], force[s], u);
  // Prefactor applied once here (p2p now accumulates the unscaled sum).
  const double pd = (double)pref;
  potential_direct[(size_t)0 * (size_t)numTargets + (size_t)tid] = pd * u[0];
  potential_direct[(size_t)1 * (size_t)numTargets + (size_t)tid] = pd * u[1];
  potential_direct[(size_t)2 * (size_t)numTargets + (size_t)tid] = pd * u[2];
}

// Host launcher for the direct-sum reference. `d_pos`/`d_force` come from
// Treecode::points()/forces(); `d_sampleIdx` is a device array of `numTargets`
// indices into the bucket-contiguous set; writes double[3*numTargets] into
// `d_potential_direct`.
inline void directSum(const vec3f *d_pos, const vec3d *d_force, int N,
                      const int *d_sampleIdx, int numTargets, float pref,
                      double *d_potential_direct)
{
  const int block = 128;
  const int grid  = (numTargets + block - 1) / block;
  directSumKernel<<<grid, block>>>(d_pos, d_force, N, d_sampleIdx, numTargets,
                                   pref, d_potential_direct);
}

// fp64-GEOMETRY direct-sum reference: identical physics to directSumKernel but
// the source/target coordinates and the 1/r are fp64 (full sqrt/divide, not the
// rsqrtf-seeded near-field path). This is the honest ground truth for a treecode
// whose geometry is fp64 -- the fp32 `directSum` above shares the treecode's fp32
// bucket coordinates, so it is blind to fp32-coordinate quantization error and
// cannot measure an fp64-geometry accuracy gain. `pos` is the bucket-contiguous
// fp64 source set (Treecode::points64()); `sampleIdx[t]` indexes into it.
__global__ void directSum64Kernel(const vec3d *pos, const vec3d *force, int N,
                                  const int *sampleIdx, int numTargets,
                                  double pref, double *potential_direct)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numTargets) return;
  const vec3d T = pos[sampleIdx[tid]];
  double u[3] = {0.0, 0.0, 0.0};
  for (int s = 0; s < N; ++s) {
    const vec3d S = pos[s];
    const double Rx = T.x - S.x, Ry = T.y - S.y, Rz = T.z - S.z;
    const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
    if (r2 == 0.0) continue;                 // skip self / coincident
    // Same near-field exclusion the treecode applies (stokes::c_nearCut2d), or
    // the reported error would be dominated by the excluded near pairs instead
    // of by the multipole truncation this reference exists to measure. The fp32
    // directSumKernel above gets it for free: it goes through stokes::p2p.
    if (r2 < stokes::c_nearCut2d) continue;
    const double ir  = 1.0 / sqrt(r2);       // full fp64 1/r (ground truth)
    const double ir2 = ir * ir;
    const double fx = force[s].x, fy = force[s].y, fz = force[s].z;
    const double q  = (Rx * fx + Ry * fy + Rz * fz) * ir2;   // (R.f)/r^2
    // Ground truth must use whatever physical kernel the build selected, or the
    // reported relL2err would measure Stokeslet-vs-RPY instead of truncation.
    if constexpr (!stokes::RPY_ON) {
      u[0] += ir * (fx + Rx * q);
      u[1] += ir * (fy + Ry * q);
      u[2] += ir * (fz + Rz * q);
    } else {
      const double ir3 = ir * ir2;
      const double A   = ir + stokes::RPY_C * ir3;             // 1/r + c/r^3
      const double Bq  = (ir - 3.0 * stokes::RPY_C * ir3) * q; // (1/r - 3c/r^3) q
      u[0] += A * fx + Bq * Rx;
      u[1] += A * fy + Bq * Ry;
      u[2] += A * fz + Bq * Rz;
    }
  }
  potential_direct[(size_t)0 * (size_t)numTargets + (size_t)tid] = pref * u[0];
  potential_direct[(size_t)1 * (size_t)numTargets + (size_t)tid] = pref * u[1];
  potential_direct[(size_t)2 * (size_t)numTargets + (size_t)tid] = pref * u[2];
}

inline void directSum64(const vec3d *d_pos, const vec3d *d_force, int N,
                        const int *d_sampleIdx, int numTargets, double pref,
                        double *d_potential_direct)
{
  const int block = 128;
  const int grid  = (numTargets + block - 1) / block;
  directSum64Kernel<<<grid, block>>>(d_pos, d_force, N, d_sampleIdx, numTargets,
                                     pref, d_potential_direct);
}

// Compare treecode result (`potential`, component-major double[3*n], indexed by
// the full particle id) to the direct sum (`potential_direct`, component-major
// double[3*sample_count], indexed by sample slot) on the sampled targets.
inline void velocity_l2_error(const double *potential,
                              const double *potential_direct,
                              const int *sample_indices, int n, int sample_count,
                              double *abs_l2, double *rel_l2,
                              double *avg_velocity_magnitude)
{
  double num2 = 0.0;
  double den2 = 0.0;
  double mag_sum = 0.0;

  for (int i = 0; i < sample_count; ++i) {
    int idx = sample_indices[i];
    double mag2 = 0.0;
    for (int c = 0; c < 3; ++c) {
      double ub = potential[(size_t)c * (size_t)n + (size_t)idx];
      double ud = potential_direct[(size_t)c * (size_t)sample_count + (size_t)i];
      double err = ub - ud;
      num2 += err * err;
      den2 += ud * ud;
      mag2 += ud * ud;
    }
    mag_sum += sqrt(mag2);
  }

  *abs_l2 = sqrt(num2);
  *rel_l2 = (den2 > 0.0) ? sqrt(num2 / den2) : sqrt(num2);
  *avg_velocity_magnitude = mag_sum / (double)sample_count;
}

} // namespace tcval
