// SPDX-License-Identifier: Apache-2.0
//
// C-ABI wrapper around Treecode<MP> for the NeMO / pinn-stokes PyTorch
// pipeline. MP is BaryStokes by default; -DNEMO_CARTESIAN=1 builds the same
// wrapper over CartesianStokes instead (libwidebvh_nemo_cart.so), which is the
// A/B variant -- the two expansions reach a given accuracy at different `mac`
// and, for Cartesian, at a runtime `order`.
//
// Why a C ABI and not pybind11: NeMO runs under a conda PyTorch build compiled
// with a different GCC/CUDA than widebvh's blessed toolchain (env.sh: GCC 14.3 +
// CUDA 12.9). Keeping the boundary pure C, statically linking libstdc++/libgcc
// and the CUDA runtime, means no C++ type and no libstdc++ symbol crosses into
// the Python process -- so widebvh keeps its own toolchain (and therefore all of
// its tuning) while still being dlopen-able from torch.
//
// Conventions, all matching the underlying engine:
//   - every pointer marked "device" is a raw CUDA device pointer owned by the
//     caller (in practice a torch tensor's .data_ptr()).
//   - positions are fp64 AoS, i.e. exactly cuBQL::vec3d[n] == a contiguous torch
//     (n,3) float64 tensor.
//   - forces are fp32 AoS == a contiguous torch (n,3) float32 tensor. The engine
//     widens them once into its bucket-contiguous buffer, so passing fp32 costs
//     nothing and halves the traffic from torch.
//   - the velocity output is COMPONENT-major double[3*n] in target input order,
//     i.e. exactly a contiguous torch (3,n) float64 tensor. Transposing that to
//     (n,3) on the torch side is a free stride change.
//   - targets are the sources (NeMO evaluates the mobility at the particles).
//   - all entry points return 0 on success, non-zero on failure, and never let
//     an exception escape; wbnemo_last_error() carries the message.
//
// Everything the caller can tune that is NOT a Config field (execution path, BVH
// builder, bucketizer, pair budget, quiet mode) is read by the engine from the
// environment at wbnemo_create() time -- set those with os.environ before
// creating a handle.
//
// treecode.cuh must be included by exactly ONE .cu per target, and that target
// needs CUDA_SEPARABLE_COMPILATION + CUDA_RESOLVE_DEVICE_SYMBOLS (refit_aggregate
// uses device function pointers). See CMakeLists.txt: add_widebvh_nemo_library.

#include "treecode.cuh"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

namespace {

// Which multipole expansion this build carries. Compile-time because the policy
// is a template parameter of the engine, so each one is a separate .so -- the
// same reason each Chebyshev degree is. `mac` and `order` mean different things
// to the two, and their values do not transfer: see the Python wrapper.
#ifdef NEMO_CARTESIAN
using NemoTreecode = Treecode<mp::CartesianStokes>;
constexpr int NEMO_POLICY_ID = 1;
#if WIDEBVH_FP32_LEVEL > 0
#error "WIDEBVH_FP32_LEVEL is a BaryStokes fast path; the Cartesian library has none"
#endif
#else
using NemoTreecode = Treecode<mp::BaryStokes>;
constexpr int NEMO_POLICY_ID = 0;
#endif

// Last error message, per thread. Sized generously: the engine's messages embed
// file:line and CUDA error strings.
thread_local char g_lastError[1024] = {0};

void setError(const char *what)
{
  std::snprintf(g_lastError, sizeof(g_lastError), "%s", what ? what : "(null)");
}

} // namespace

struct wbnemo_tc {
  NemoTreecode *tc = nullptr;
  bool          reusable = false;   // an apply(reuse=true) has run
};

// The target is built with hidden default visibility so that several degree
// variants of this library can coexist in one process without interposing on
// each other; these are the only symbols that get out.
#define WBNEMO_API __attribute__((visibility("default")))

#define WBNEMO_GUARD(BODY)                                                    \
  try {                                                                       \
    BODY                                                                      \
    return 0;                                                                 \
  } catch (const std::exception &e) {                                         \
    setError(e.what());                                                       \
    return 1;                                                                 \
  } catch (...) {                                                             \
    setError("unknown C++ exception");                                        \
    return 2;                                                                 \
  }

extern "C" {

// Bump whenever any signature below changes; the Python wrapper checks it.
//   3  the original fp64-only engine
//   4  + wbnemo_fp32_level() (call-compatible with 3 otherwise; a v3 caller
//      that only knows the v3 entry points still works, and a v4 caller treats
//      a v3 library as fp32_level 0)
WBNEMO_API int wbnemo_abi_version(void) { return 4; }

// Compile-time build identity, so a caller can log/verify which .so it loaded.
// The degree variants and the Cartesian variant export identical symbol names,
// so a caller that loads more than one MUST check wbnemo_policy()/wbnemo_pdeg()
// after dlopen -- that is what catches an interposition (see the visibility
// note on WBNEMO_API above and the RTLD_LOCAL note in the Python wrapper).
WBNEMO_API int    wbnemo_policy(void) { return NEMO_POLICY_ID; }  // 0 bary, 1 cartesian
WBNEMO_API int    wbnemo_pdeg(void)
{
#ifdef NEMO_CARTESIAN
  return 0;             // no compile-time degree; the ladder is wbnemo_max_order()
#else
  return mp::bary::PDEG;
#endif
}
WBNEMO_API int    wbnemo_max_order(void) { return NemoTreecode::MAX_ORDER; }
WBNEMO_API double wbnemo_radius(void) { return stokes::RPY_A; }   // 0 => plain Stokeslet
// fp32 fast-path level this .so was compiled with (WIDEBVH_FP32_LEVEL, see
// stokes_kernel.cuh): 0 fp64 production, 1 fp32 M2P, 2 + fp32 P2P, 3 + fp32
// upward pass. Same identity role as wbnemo_pdeg(): the levels export the same
// symbol names, so a caller loading more than one must check it after dlopen.
WBNEMO_API int    wbnemo_fp32_level(void) { return stokes::FP32_LEVEL; }

WBNEMO_API const char *wbnemo_last_error(void) { return g_lastError; }

// Attach this library's CUDA runtime to `device`'s primary context. Call AFTER
// torch has initialized CUDA so both runtimes share one context.
WBNEMO_API int wbnemo_device_init(int device)
{
  WBNEMO_GUARD(
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(0));          // force primary-context creation
  )
}

// ---------------------------------------------------------------------------
// Handle lifecycle
//
// mac         : multipole acceptance criterion (node half-diagonal / distance).
// max_leaf    : max particles per bucket == per BVH leaf.
// cell_edge   : legacy uniform-grid cell edge; ignored by the default
//               grid-hilbert bucketizer (which derives it), pass 0 for default.
// near_cutoff : near-field exclusion radius rc. The far field then covers
//               exactly r >= rc; 0 restores the classic whole-sum treecode.
// order       : multipole expansion order, 0 = the engine's default. This is
//               the Cartesian policy's whole accuracy ladder (1..MAX_ORDER=4);
//               BaryStokes ignores it, since its degree is the compile-time
//               PDEG and Config::order survives only for dispatch.
//
// Config is immutable once a tree is built, so changing mac means destroy +
// create. That is cheap -- the handle owns no geometry until the first apply.
// ---------------------------------------------------------------------------
WBNEMO_API int wbnemo_create(wbnemo_tc **out, float mac, int max_leaf, double cell_edge,
                  double near_cutoff, int order)
{
  WBNEMO_GUARD(
    if (out == nullptr) throw std::runtime_error("wbnemo_create: out is null");
    *out = nullptr;

    NemoTreecode::Config cfg;
    cfg.mac         = mac;
    cfg.maxLeaf     = max_leaf;
    cfg.nearCutoff  = near_cutoff;
    if (cell_edge > 0.0) cfg.cellEdge = cell_edge;
    // Checked here rather than left to the engine's assert(), which is compiled
    // out in Release and would then be a silent wrong answer. order == 0 means
    // "policy default", and the Config default (6) is above CartesianStokes's
    // MAX_ORDER, so it has to be clamped rather than passed through.
    if (order != 0) {
      if (order < 1 || order > NemoTreecode::MAX_ORDER)
        throw std::runtime_error(
            "wbnemo_create: order " + std::to_string(order) + " outside 1.." +
            std::to_string(NemoTreecode::MAX_ORDER) + " for policy " +
            NemoTreecode::multipoleName());
      cfg.order = order;
    } else if (cfg.order > NemoTreecode::MAX_ORDER) {
      cfg.order = NemoTreecode::MAX_ORDER;
    }

    std::unique_ptr<wbnemo_tc> h(new wbnemo_tc());
    h->tc = new NemoTreecode(cfg);
    *out = h.release();
  )
}

WBNEMO_API int wbnemo_destroy(wbnemo_tc *h)
{
  WBNEMO_GUARD(
    if (h) {
      delete h->tc;
      delete h;
    }
  )
}

// ---------------------------------------------------------------------------
// Evaluation
//
// pos64   : device const double[3*n] AoS  (== cuBQL::vec3d[n])
// force32 : device const float [3*n] AoS  (== cuBQL::vec3f[n])
// vel_out : device double[3*n], COMPONENT-major, in input order
//
// reuse_tree != 0 keeps the geometry + cached near-pair list alive so
// wbnemo_reapply can reuse them with new forces. NeMO's particles move every
// step, so its default is 0 (rebuild), which also frees the memory-heavy state
// before returning.
// ---------------------------------------------------------------------------
WBNEMO_API int wbnemo_apply(wbnemo_tc *h, const void *pos64, size_t n,
                 const void *force32, void *vel_out, int reuse_tree)
{
  WBNEMO_GUARD(
    if (h == nullptr || h->tc == nullptr)
      throw std::runtime_error("wbnemo_apply: null handle");
    if (pos64 == nullptr || force32 == nullptr || vel_out == nullptr)
      throw std::runtime_error("wbnemo_apply: null device pointer");
    if (n == 0) throw std::runtime_error("wbnemo_apply: n == 0");

    h->tc->apply(static_cast<const vec3d *>(pos64), n,
                 static_cast<const vec3f *>(force32),
                 static_cast<double *>(vel_out),
                 reuse_tree != 0);
    h->reusable = (reuse_tree != 0);
  )
}

WBNEMO_API int wbnemo_reapply(wbnemo_tc *h, const void *force32, void *vel_out)
{
  WBNEMO_GUARD(
    if (h == nullptr || h->tc == nullptr)
      throw std::runtime_error("wbnemo_reapply: null handle");
    if (!h->reusable)
      throw std::runtime_error(
          "wbnemo_reapply: no reusable tree -- call wbnemo_apply with "
          "reuse_tree=1 first");
    if (force32 == nullptr || vel_out == nullptr)
      throw std::runtime_error("wbnemo_reapply: null device pointer");

    h->tc->reapply(static_cast<const vec3f *>(force32),
                   static_cast<double *>(vel_out));
  )
}

// Phase timings (ms) and counters from the most recent apply/reapply, in the
// fixed order documented in the Python wrapper's WBNEMO_STAT_FIELDS.
WBNEMO_API int wbnemo_stats(wbnemo_tc *h, double *out)
{
  WBNEMO_GUARD(
    if (h == nullptr || h->tc == nullptr || out == nullptr)
      throw std::runtime_error("wbnemo_stats: null argument");
    const auto &s = h->tc->stats();
    out[0] = s.bucketMs;
    out[1] = s.targetBucketMs;
    out[2] = s.prepForcesMs;
    out[3] = s.buildBvhMs;
    out[4] = s.upwardMs;
    out[5] = (double)s.travMs;
    out[6] = (double)s.p2pMs;
    out[7] = (double)s.nPairs;
    out[8] = (double)s.numNodes;
    out[9] = (double)s.numSourceBuckets;
  )
}

// ---------------------------------------------------------------------------
// Self-contained smoke test, kept from the Phase-0 ABI gate: exercises the full
// engine (constant-memory moment tables, the separable device-function-pointer
// refit, thrust/CUB, the cuBQL builder) without any Python plumbing, so an
// ABI/interop failure is unambiguous and separable from a wrapper bug.
// ---------------------------------------------------------------------------
WBNEMO_API int wbnemo_smoke(int n, float mac, int maxLeaf, double *out_checksum)
{
  WBNEMO_GUARD(
    if (n <= 0) throw std::runtime_error("wbnemo_smoke: n must be > 0");

    // Deterministic host-side cloud: a jittered cube, no RNG dependency.
    std::vector<vec3d> hostPts((size_t)n);
    uint64_t s = 88172645463325252ull;
    auto rnd = [&s]() {                         // xorshift64, in [0,1)
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      return (double)(s >> 11) * (1.0 / 9007199254740992.0);
    };
    const double L = 100.0;
    for (int i = 0; i < n; ++i)
      hostPts[i] = vec3d(rnd() * L, rnd() * L, rnd() * L);

    thrust::device_vector<vec3d>  d_pts(hostPts);
    thrust::device_vector<vec3f>  d_force((size_t)n, vec3f(0.f, 0.f, -9.81f));
    thrust::device_vector<double> d_vel((size_t)3 * (size_t)n, 0.0);

    NemoTreecode::Config cfg;
    cfg.mac     = mac;
    cfg.maxLeaf = maxLeaf;
    if (cfg.order > NemoTreecode::MAX_ORDER)      // see wbnemo_create
      cfg.order = NemoTreecode::MAX_ORDER;

    NemoTreecode tc(cfg);
    tc.apply(util::devicePtr(d_pts), (size_t)n,
             util::devicePtr(d_force), util::devicePtr(d_vel),
             /*will_reuse_tree=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (out_checksum) {
      // Sum of the z-component block: the only component with a net signal
      // under uniform -z forcing, so a zero here means nothing was computed.
      *out_checksum = thrust::reduce(d_vel.begin() + (size_t)2 * (size_t)n,
                                     d_vel.end(), 0.0);
    }
  )
}

} // extern "C"
