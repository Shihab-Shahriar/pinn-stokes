// SPDX-License-Identifier: Apache-2.0
//
// leaf_overlap_stats: quantifies how much the source partition's LEAF BOXES
// overlap each other, and what that overlap does to the MAC near field
// (rejected leaves -> P2P pairs), on a point cloud in the engine's shifted fp32
// frame:
//
//   bvh - grid-bucket leaves (buildGridBuckets at cellEdge/maxLeaf, or the
//         grid-hilbert / global-hilbert bucketizers); with cuBQL BuildConfig(1)
//         these bucket AABBs ARE the BVH leaf boxes the treecode traverses, so
//         no BVH build is needed here.
//
// Reported per partition:
//   - occupancy and half-diagonal stats of non-empty leaves
//   - containment multiplicity: how many leaf boxes cover each particle
//     (disjoint partition => exactly 1)
//   - Monte-Carlo volume coverage: sum(leaf volume) / union volume
//   - pairwise AABB intersections, split into same-grid-cell (Morton-chunk
//     stacking) vs cross-cell pairs, plus intersection volume / total volume
//   - MAC near-shell prediction at theta: per target, the number of leaves
//     with halfDiag2 >= theta^2 * dist2(target, boxCenter) (the traversal's
//     reject test) and the sum of their occupancies (predicted P2P
//     interactions/target). Cross-check against the driver's near pairs.
//
// Diagnostic only -- brute-force O(N * leaves) kernels are fine here; this
// never runs in the treecode hot path.
//
// Usage: leaf_overlap_stats <points.bin> <mac> <cellEdge> <maxLeaf>
//   cellEdge > 0 -> grid buckets (Morton fine); == 0 -> global-hilbert buckets;
//   < 0 -> grid-hilbert (auto edge, within-cell Hilbert; TC_HILBERT_Q overrides).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include "common.cuh"
#include "grid_buckets.cuh"

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err__ = (call);                                           \
    if (err__ != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
              cudaGetErrorString(err__), __FILE__, __LINE__);             \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

using cuBQL::vec3d;
using cuBQL::vec3f;
using cuBQL::box3f;
using cuBQL::box3d;

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// Per particle: # leaf boxes containing it, # leaves failing the MAC at theta
// (near leaves), and the occupancy sum of those near leaves (P2P interactions).
__global__ void perParticleKernel(const vec3f *pts, int n,
                                  const box3f *boxes, const int *cnt, int nb,
                                  float theta2,
                                  int *mContain, int *mNear, long long *iNear)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const vec3f p = pts[i];
  int c = 0, nl = 0;
  long long ii = 0;
  for (int b = 0; b < nb; ++b) {
    const box3f bx = boxes[b];
    if (p.x >= bx.lower.x && p.x <= bx.upper.x &&
        p.y >= bx.lower.y && p.y <= bx.upper.y &&
        p.z >= bx.lower.z && p.z <= bx.upper.z)
      ++c;
    const float hx = 0.5f * (bx.upper.x - bx.lower.x);
    const float hy = 0.5f * (bx.upper.y - bx.lower.y);
    const float hz = 0.5f * (bx.upper.z - bx.lower.z);
    const float dx = p.x - 0.5f * (bx.lower.x + bx.upper.x);
    const float dy = p.y - 0.5f * (bx.lower.y + bx.upper.y);
    const float dz = p.z - 0.5f * (bx.lower.z + bx.upper.z);
    const float hd2 = hx * hx + hy * hy + hz * hz;
    const float r2  = dx * dx + dy * dy + dz * dz;
    if (hd2 >= theta2 * r2) {   // same reject test as classifyTraversalNode
      ++nl;
      ii += cnt[b];
    }
  }
  mContain[i] = c;
  mNear[i]    = nl;
  iNear[i]    = ii;
}

// Per leaf i: overlapping partners among j>i, split same-cell/cross-cell,
// plus the summed pairwise intersection volume.
__global__ void pairOverlapKernel(const box3f *boxes, const long long *cellId,
                                  int nb,
                                  int *nSame, int *nCross, double *volOv)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  const box3f a = boxes[i];
  const long long ci = cellId ? cellId[i] : -1;
  int s = 0, x = 0;
  double v = 0.0;
  for (int j = i + 1; j < nb; ++j) {
    const box3f b = boxes[j];
    const float ox = fminf(a.upper.x, b.upper.x) - fmaxf(a.lower.x, b.lower.x);
    const float oy = fminf(a.upper.y, b.upper.y) - fmaxf(a.lower.y, b.lower.y);
    const float oz = fminf(a.upper.z, b.upper.z) - fmaxf(a.lower.z, b.lower.z);
    if (ox > 0.f && oy > 0.f && oz > 0.f) {
      if (cellId && cellId[j] == ci) ++s; else ++x;
      v += (double)ox * (double)oy * (double)oz;
    }
  }
  nSame[i]  = s;
  nCross[i] = x;
  volOv[i]  = v;
}

// Monte-Carlo coverage multiplicity of uniform samples in `domain`.
__device__ __forceinline__ unsigned pcgHash(unsigned x)
{
  x = x * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28) + 4u)) ^ x) * 277803737u;
  return (x >> 22) ^ x;
}

__global__ void mcCoverageKernel(box3f domain, const box3f *boxes, int nb,
                                 int nSamples, int *mult)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nSamples) return;
  const float ux = pcgHash(3u * i + 1u) * (1.f / 4294967296.f);
  const float uy = pcgHash(3u * i + 2u) * (1.f / 4294967296.f);
  const float uz = pcgHash(3u * i + 3u) * (1.f / 4294967296.f);
  const float px = domain.lower.x + ux * (domain.upper.x - domain.lower.x);
  const float py = domain.lower.y + uy * (domain.upper.y - domain.lower.y);
  const float pz = domain.lower.z + uz * (domain.upper.z - domain.lower.z);
  int m = 0;
  for (int b = 0; b < nb; ++b) {
    const box3f bx = boxes[b];
    if (px >= bx.lower.x && px <= bx.upper.x &&
        py >= bx.lower.y && py <= bx.upper.y &&
        pz >= bx.lower.z && pz <= bx.upper.z)
      ++m;
  }
  mult[i] = m;
}

__global__ void gatherShiftedF32Kernel(const vec3d *pts, const unsigned *perm,
                                       int n, vec3d shift, vec3f *out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const vec3d p = pts[perm ? (int)perm[i] : i];
  vec3f o;
  o.x = (float)(p.x - shift.x);
  o.y = (float)(p.y - shift.y);
  o.z = (float)(p.z - shift.z);
  out[i] = o;
}

struct GeIntF {
  int t;
  __host__ __device__ bool operator()(int m) const { return m >= t; }
};

// ---------------------------------------------------------------------------
// host-side analysis of one leaf set
// ---------------------------------------------------------------------------

static double percentile(std::vector<double> v, double p)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * (v.size() - 1);
  const size_t lo  = (size_t)idx;
  const size_t hi  = std::min(lo + 1, v.size() - 1);
  const double w   = idx - lo;
  return v[lo] * (1.0 - w) + v[hi] * w;
}

// cellId: per-leaf grid-cell id (same-cell overlap attribution), or empty.
static void analyzeLeafSet(const char *tag,
                           const thrust::device_vector<box3f> &d_boxes,
                           const thrust::device_vector<int> &d_cnt,
                           const std::vector<long long> &cellIdHost,
                           const thrust::device_vector<vec3f> &d_pts,
                           float mac)
{
  const int nb = (int)d_boxes.size();
  const int n  = (int)d_pts.size();

  thrust::host_vector<box3f> boxes = d_boxes;
  thrust::host_vector<int>   cnt   = d_cnt;

  // occupancy / geometry stats
  std::vector<double> occ(nb), hd(nb);
  double sumVol = 0.0;
  long long nPrims = 0;
  int atCap = 0, maxOcc = 0;
  for (int i = 0; i < nb; ++i) {
    occ[i] = cnt[i];
    nPrims += cnt[i];
    maxOcc = std::max(maxOcc, cnt[i]);
    const double hx = 0.5 * (boxes[i].upper.x - boxes[i].lower.x);
    const double hy = 0.5 * (boxes[i].upper.y - boxes[i].lower.y);
    const double hz = 0.5 * (boxes[i].upper.z - boxes[i].lower.z);
    hd[i] = std::sqrt(hx * hx + hy * hy + hz * hz);
    sumVol += 8.0 * hx * hy * hz;
  }
  for (int i = 0; i < nb; ++i) atCap += (cnt[i] == maxOcc);

  printf("=== %s ===\n", tag);
  printf("leaves=%d  particles=%lld\n", nb, nPrims);
  printf("occupancy: mean=%.1f p50=%.0f max=%d  frac_at_max=%.2f\n",
         (double)nPrims / nb, percentile(occ, 0.5), maxOcc,
         (double)atCap / nb);
  printf("halfDiag:  mean=%.4g p50=%.4g p90=%.4g  sum_leaf_vol=%.6g\n",
         std::accumulate(hd.begin(), hd.end(), 0.0) / nb,
         percentile(hd, 0.5), percentile(hd, 0.9), sumVol);

  // per-particle containment + MAC near shell
  thrust::device_vector<int>       mContain(n), mNear(n);
  thrust::device_vector<long long> iNear(n);
  {
    const int bs = 256, gs = (n + bs - 1) / bs;
    perParticleKernel<<<gs, bs>>>(util::devicePtr(d_pts), n,
                                  util::devicePtr(d_boxes),
                                  util::devicePtr(d_cnt), nb, mac * mac,
                                  util::devicePtr(mContain),
                                  util::devicePtr(mNear),
                                  util::devicePtr(iNear));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  const long long sumContain =
      thrust::reduce(mContain.begin(), mContain.end(), 0LL);
  const int maxContain =
      *thrust::max_element(mContain.begin(), mContain.end());
  const long long nMulti =
      thrust::count_if(mContain.begin(), mContain.end(), GeIntF{2});
  const long long sumNear = thrust::reduce(mNear.begin(), mNear.end(), 0LL);
  const int maxNear = *thrust::max_element(mNear.begin(), mNear.end());
  const long long sumInt = thrust::reduce(iNear.begin(), iNear.end(), 0LL);
  printf("containment multiplicity/particle: mean=%.3f max=%d frac>=2=%.3f\n",
         (double)sumContain / n, maxContain, (double)nMulti / n);
  printf("MAC theta=%.2f near shell (leaf-geometry prediction): "
         "nearLeaves/target mean=%.1f max=%d   p2pInt/target mean=%.0f\n",
         mac, (double)sumNear / n, maxNear, (double)sumInt / n);

  // pairwise overlap
  thrust::device_vector<int>    nSame(nb), nCross(nb);
  thrust::device_vector<double> volOv(nb);
  thrust::device_vector<long long> d_cell;
  const long long *cellPtr = nullptr;
  if (!cellIdHost.empty()) {
    d_cell = cellIdHost;
    cellPtr = util::devicePtr(d_cell);
  }
  {
    const int bs = 128, gs = (nb + bs - 1) / bs;
    pairOverlapKernel<<<gs, bs>>>(util::devicePtr(d_boxes), cellPtr, nb,
                                  util::devicePtr(nSame),
                                  util::devicePtr(nCross),
                                  util::devicePtr(volOv));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  const long long pairsSame =
      thrust::reduce(nSame.begin(), nSame.end(), 0LL);
  const long long pairsCross =
      thrust::reduce(nCross.begin(), nCross.end(), 0LL);
  const double volOverlap = thrust::reduce(volOv.begin(), volOv.end(), 0.0);
  printf("pairwise AABB intersections: pairs=%lld (same-cell=%lld "
         "cross-cell=%lld)  partners/leaf=%.1f  overlapVol/sumVol=%.3f\n",
         pairsSame + pairsCross, pairsSame, pairsCross,
         2.0 * (pairsSame + pairsCross) / nb,
         volOverlap / sumVol);

  // Monte-Carlo union volume (domain = bounding box of the leaf boxes)
  box3f dom;
  dom.lower.x = dom.lower.y = dom.lower.z = FLT_MAX;
  dom.upper.x = dom.upper.y = dom.upper.z = -FLT_MAX;
  for (int i = 0; i < nb; ++i) {
    dom.lower.x = std::min(dom.lower.x, boxes[i].lower.x);
    dom.lower.y = std::min(dom.lower.y, boxes[i].lower.y);
    dom.lower.z = std::min(dom.lower.z, boxes[i].lower.z);
    dom.upper.x = std::max(dom.upper.x, boxes[i].upper.x);
    dom.upper.y = std::max(dom.upper.y, boxes[i].upper.y);
    dom.upper.z = std::max(dom.upper.z, boxes[i].upper.z);
  }
  const int nS = 8 << 20;
  thrust::device_vector<int> mult(nS);
  {
    const int bs = 256, gs = (nS + bs - 1) / bs;
    mcCoverageKernel<<<gs, bs>>>(dom, util::devicePtr(d_boxes), nb, nS,
                                 util::devicePtr(mult));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  const long long covered =
      thrust::count_if(mult.begin(), mult.end(), GeIntF{1});
  const long long sumMult = thrust::reduce(mult.begin(), mult.end(), 0LL);
  const double domVol =
      (double)(dom.upper.x - dom.lower.x) *
      (double)(dom.upper.y - dom.lower.y) *
      (double)(dom.upper.z - dom.lower.z);
  const double unionVol = domVol * (double)covered / nS;
  printf("MC volume: unionVol=%.6g  sumVol/unionVol=%.3f  "
         "(MC sumVol check=%.6g)\n\n",
         unionVol, sumVol / unionVol, domVol * (double)sumMult / nS);
  fflush(stdout);
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s <points.bin> <mac> <cellEdge> <maxLeaf>\n"
            "  cellEdge > 0 -> grid; == 0 -> global-hilbert; "
            "< 0 -> grid-hilbert (auto edge)\n", argv[0]);
    return 1;
  }
  const std::string path = argv[1];
  const float  mac      = (float)atof(argv[2]);
  const double cellEdge = atof(argv[3]);
  const int    maxLeaf  = atoi(argv[4]);

  std::vector<vec3d> host = util::loadPointsFP64(path);
  const size_t n = host.size();
  printf("loaded %zu points from %s  (mac=%.2f)\n\n", n, path.c_str(), mac);

  thrust::device_vector<vec3d> d_pts64 = host;
  const box3d bounds = util::computeBounds(util::devicePtr(d_pts64), n);
  const vec3d shift  = bounds.center();

  // shared shifted fp32 particle array (original order)
  thrust::device_vector<vec3f> d_ptsF(n);
  {
    const int bs = 256, gs = (int)((n + bs - 1) / bs);
    gatherShiftedF32Kernel<<<gs, bs>>>(util::devicePtr(d_pts64), nullptr,
                                       (int)n, shift,
                                       util::devicePtr(d_ptsF));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- BVH-side leaves (== cuBQL leaf boxes at BuildConfig(1)):
  // cellEdge > 0 -> grid buckets (Morton fine); cellEdge == 0 -> Hilbert
  // buckets (global Hilbert sort, fixed-size chunks); cellEdge < 0 ->
  // grid-hilbert (auto cell edge from the R_max formula, within-cell Hilbert;
  // TC_HILBERT_Q overrides q), matching TC_BUCKETIZER=grid-hilbert.
  {
    util::GridBuckets gb;
    std::vector<long long> cellId;
    char tag[128];
    if (cellEdge != 0.0) {
      double edge = cellEdge;
      util::FineCurve fc = util::FineCurve::Morton;
      if (cellEdge < 0.0) {
        double q = std::cbrt(std::max(1024.0,
                                      (double)n / std::max(1, maxLeaf)));
        if (const char *e = getenv("TC_HILBERT_Q")) q = atof(e);
        const vec3d sz = bounds.size();
        const double rDomain =
          0.5 * std::sqrt(sz.x * sz.x + sz.y * sz.y + sz.z * sz.z);
        edge = 2.0 * (rDomain / q) / std::sqrt(3.0);
        fc = util::FineCurve::Hilbert;
        printf("[grid-hilbert: auto cellEdge=%g (q=%g)]\n", edge, q);
      }
      gb = util::buildGridBuckets(util::devicePtr(d_pts64), n, bounds, shift,
                                  edge, maxLeaf, fc);
      const int nb = (int)gb.numBuckets();
      // grid-cell id per bucket from its box center (buckets never span
      // cells), to attribute overlapping pairs to Morton-chunk stacking.
      thrust::host_vector<box3f> hb = gb.boxes;
      cellId.resize(nb);
      for (int i = 0; i < nb; ++i) {
        const double cx = 0.5 * (hb[i].lower.x + hb[i].upper.x) + shift.x;
        const double cy = 0.5 * (hb[i].lower.y + hb[i].upper.y) + shift.y;
        const double cz = 0.5 * (hb[i].lower.z + hb[i].upper.z) + shift.z;
        const long long ix = (long long)((cx - bounds.lower.x) / edge);
        const long long iy = (long long)((cy - bounds.lower.y) / edge);
        const long long iz = (long long)((cz - bounds.lower.z) / edge);
        cellId[i] = (ix * 4096 + iy) * 4096 + iz;
      }
      printf("[grid: %u x %u x %u cells, occupied=%d -> buckets/occupied-cell "
             "mean=%.1f]\n", gb.nx, gb.ny, gb.nz, gb.occupiedCells,
             (double)nb / std::max(gb.occupiedCells, 1));
      snprintf(tag, sizeof(tag), "BVH %s buckets  cell=%g maxLeaf=%d",
               fc == util::FineCurve::Hilbert ? "grid-hilbert" : "grid",
               edge, maxLeaf);
    } else {
      const vec3d sz = bounds.size();
      const double half = 0.5 * std::max(sz.x, std::max(sz.y, sz.z));
      box3d sfcBox;
      sfcBox.lower = shift - vec3d(half);
      sfcBox.upper = shift + vec3d(half);
      gb = util::buildHilbertBuckets(util::devicePtr(d_pts64), n, sfcBox,
                                     shift, maxLeaf);
      printf("[hilbert buckets: %u chunks of <=%d]\n", gb.numBuckets(),
             maxLeaf);
      snprintf(tag, sizeof(tag), "BVH hilbert buckets  chunk=%d", maxLeaf);
    }
    analyzeLeafSet(tag, gb.boxes, gb.count, cellId, d_ptsF, mac);
  }

  return 0;
}
