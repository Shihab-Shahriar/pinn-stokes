// SPDX-License-Identifier: Apache-2.0
//
// Shared helper for the monodisperse "structured source" treecode path
// (TC_SRC_TEMPLATE): reconstruct a source position from one shared template
// point cloud plus a per-object rigid transform, instead of materializing the
// per-point positions. Used by both the moment build (bary_stokes.cuh) and the
// leaf-centric near-field kernel (treecode.cuh), so it lives in its own tiny
// header included by both. `inline` => no ODR clash across translation units.
//
// Convention matches mfs_broms.cuh::tileRotateTranslate exactly:
//   pos = R * yLocal + c        (R row-major 3x3, left-to-right accumulate)
// Pass the SHIFTED center (c - outputShift) so the result is already in the
// treecode's centered frame (matching points64() = input - bounds.center()).
#pragma once

#include "cuBQL/math/vec.h"

__host__ __device__ __forceinline__
cuBQL::vec3d reconstructSrc64(const double *R9, cuBQL::vec3d y, cuBQL::vec3d c)
{
  return cuBQL::vec3d(R9[0] * y.x + R9[1] * y.y + R9[2] * y.z + c.x,
                      R9[3] * y.x + R9[4] * y.y + R9[5] * y.z + c.y,
                      R9[6] * y.x + R9[7] * y.y + R9[8] * y.z + c.z);
}
