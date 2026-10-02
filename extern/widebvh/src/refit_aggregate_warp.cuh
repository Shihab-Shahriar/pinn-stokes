// SPDX-License-Identifier: Apache-2.0
//
// Project-local variant of cuBQL's refit_aggregate that gives one warp, not one
// thread, to each aggregate callback. This keeps cuBQL vendored code untouched
// while preserving its bottom-up sibling-arrival protocol.
#pragma once

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/builder/cuda/refit.h"

namespace tcgpu {

template<typename T, int D, typename AggregateNodeData,
         typename AggregateFunctor, int WARPS_PER_BLOCK>
__global__ void refitAggregateWarpRun(cuBQL::BinaryBVH<T, D> bvh,
                                      AggregateNodeData *aggregateNodeData,
                                      AggregateFunctor aggregateFct,
                                      uint32_t *refitData)
{
  static_assert(WARPS_PER_BLOCK > 0, "WARPS_PER_BLOCK must be positive");

  extern __shared__ __align__(16) unsigned char sharedBytes[];
  using Shared = typename AggregateFunctor::Shared;
  Shared *shared = reinterpret_cast<Shared *>(sharedBytes);

  const int lane = threadIdx.x & 31;
  const int warpInBlock = threadIdx.x >> 5;
  const int warpID = (int)blockIdx.x * WARPS_PER_BLOCK + warpInBlock;
  int nodeID = warpID;

  if (nodeID == 1 || nodeID >= (int)bvh.numNodes) return;

  if (bvh.nodes[nodeID].admin.count == 0) return;

  const unsigned int mask = 0xffffffffu;
  int parentID = (int)(refitData[nodeID] >> 1);
  Shared &warpShared = shared[warpInBlock];

  while (true) {
    aggregateFct(bvh, aggregateNodeData, nodeID, lane, mask, warpShared);
    __syncwarp(mask);

    // Each lane may have written part of the node aggregate. All lanes must
    // fence before lane 0 publishes this node to its parent via atomicAdd.
    __threadfence();
    __syncwarp(mask);

    if (nodeID == 0) break;

    uint32_t refitBits = 0;
    if (lane == 0) refitBits = atomicAdd(&refitData[parentID], 1u);
    refitBits = __shfl_sync(mask, refitBits, 0);
    if ((refitBits & 1u) == 0u) break;

    nodeID = parentID;
    parentID = (int)(refitBits >> 1);
  }
}

template<int WARPS_PER_BLOCK = 3,
         typename T, int D, typename AggregateNodeData,
         typename AggregateFunctor>
void refit_aggregate_warp(cuBQL::BinaryBVH<T, D> bvh,
                          AggregateNodeData *d_aggregateNodeData,
                          AggregateFunctor aggregateFct,
                          cudaStream_t stream = 0,
                          cuBQL::GpuMemoryResource &memResource =
                              cuBQL::defaultGpuMemResource())
{
  static_assert(WARPS_PER_BLOCK > 0, "WARPS_PER_BLOCK must be positive");
  if (bvh.numNodes == 0) return;

  uint32_t *refitData = nullptr;
  memResource.malloc((void **)&refitData,
                     (size_t)bvh.numNodes * sizeof(*refitData), stream);

  cuBQL::cuda::refit_init<T, D>
      <<<cuBQL::divRoundUp((int)bvh.numNodes, 1024), 1024, 0, stream>>>(
          bvh.nodes, refitData, (int)bvh.numNodes);

  using Shared = typename AggregateFunctor::Shared;
  constexpr int blockThreads = WARPS_PER_BLOCK * 32;
  const int grid = cuBQL::divRoundUp((int)bvh.numNodes, WARPS_PER_BLOCK);
  const size_t sharedBytes = (size_t)WARPS_PER_BLOCK * sizeof(Shared);
  // Above 48KB dynamic shared memory a kernel must opt in explicitly or the
  // launch fails with cudaErrorInvalidValue (hit by BaryStokes at PDEG >= 7:
  // Shared is ~17KB/warp). One-time per instantiation.
  if (sharedBytes > 48 * 1024) {
    static bool attrSet = false;
    if (!attrSet) {
      const cudaError_t rc = cudaFuncSetAttribute(
          (const void *)&refitAggregateWarpRun<T, D, AggregateNodeData,
                                               AggregateFunctor,
                                               WARPS_PER_BLOCK>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);
      if (rc != cudaSuccess)
        fprintf(stderr,
                "refit_aggregate_warp: shared-mem opt-in (%zu B) failed: %s\n",
                sharedBytes, cudaGetErrorString(rc));
      attrSet = true;
    }
  }
  refitAggregateWarpRun<T, D, AggregateNodeData,
                        AggregateFunctor, WARPS_PER_BLOCK>
      <<<grid, blockThreads, sharedBytes, stream>>>(
          bvh, d_aggregateNodeData, aggregateFct, refitData);

  memResource.free((void *)refitData, stream);
}

} // namespace tcgpu
