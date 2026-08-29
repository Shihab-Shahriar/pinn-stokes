"""Far-field treecode backed by the widebvh GPU engine.

`WidebvhFMM` is a drop-in replacement for `WarpFMM` (src/treecode.py): same
constructor role, same `get_far_field_vel` / `apply` contracts, same stdout keys.
Only the far field changes; the near field (self + two-body NN + n-body NN over
the hash-grid neighbour list) is inherited unchanged.

Why swap: `WarpFMM` carries a monopole + first-order dipole expansion, which
measures 6.8e-3 relative truncation error against a dense fp64 RPY r>=6 reference
at its production theta=0.3 (artifacts/treecode_symmetry_report.md). widebvh
carries a degree-PDEG barycentric-Lagrange expansion at Chebyshev proxy points
and reaches ~1e-8 at mac 0.5, so the same accuracy is available at a far looser
(and therefore cheaper) acceptance criterion.

The engine is C++/CUDA and lives in its own repo, built with its own toolchain;
we reach it through a C ABI (`libwidebvh_nemo*.so`, see widebvh/src/nemo_capi.cu)
loaded with ctypes. Nothing is copied into this repo.

Two widebvh features exist specifically for this integration:
  * `Config::nearCutoff` -- the far field covers exactly r >= rc, with no node
    reaching inside rc ever accepted, so it is the exact complement of NeMO's
    near-field neighbour list and cannot double count.
  * `WIDEBVH_RPY_A=1` -- the kernel is the Rotne-Prager-Yamakawa mobility of
    unit-radius spheres rather than the bare Stokeslet, matching
    `rpy_far_velocity_*` in src/treecode.py.

Two multipole expansions are available, one .so each, selected with `policy`:

  * `"bary"` (production) -- barycentric Lagrange at Chebyshev proxy points,
    accuracy set by the compile-time degree `pdeg`, tuned with `mac`.
  * `"cart"` -- the analytic Cartesian Taylor expansion, accuracy set by the
    RUNTIME `order` (1..4) and `mac`. See artifacts/cartesian_far_field_report.md.

They are different expansions of the same kernel, so a `mac` calibrated for one
does not transfer to the other; take operating points from the DEFAULT_*
constants below, or from data/*_mac_calibration.csv.

A third axis, `fp32_level`, picks how much of the bary far field runs in
fp32 (0 = the fp64 production kernels). It exists for consumer GPUs, whose fp64
rate is 1/64 of fp32; see FP32_LEVELS below.

Build (once, from the widebvh checkout):

    source ./env.sh
    cmake -S . -B build-nemo -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DWIDEBVH_NEMO_PDEG="3;5;7" -DWIDEBVH_NEMO_FP32_LEVELS="1;2"
    cmake --build build-nemo -j --target widebvh_nemo widebvh_nemo_p5 \
          widebvh_nemo_p3 widebvh_nemo_cart widebvh_nemo_f32l2 \
          widebvh_nemo_p5_f32l2
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import torch

from src.treecode import WarpFMM

# Where the C-ABI libraries live. Override with WIDEBVH_BUILD_DIR.
WIDEBVH_BUILD_DIR = Path(
    os.environ.get(
        "WIDEBVH_BUILD_DIR",
        "/mnt/ffs24/home/khanmd/programs/widebvh/build-nemo",
    )
)

# ABI versions this wrapper speaks (wbnemo_abi_version() in
# widebvh/src/nemo_capi.cu). 3 is the original fp64-only engine; 4 adds
# wbnemo_fp32_level() and is otherwise call-compatible, so both are accepted
# and a v3 library simply implies fp32_level 0.
WBNEMO_ABI_VERSIONS = (3, 4)
WBNEMO_ABI_VERSION = 4

# Precision ladder of the far-field kernels, a compile-time choice in widebvh
# (WIDEBVH_FP32_LEVEL) and therefore a separate .so per level:
#   0  production: fp64 M2P / P2P / upward pass (H200 numbers, bit-identical
#      to the pre-fp32 engine)
#   1  fp32 M2P (barycentric evaluation + per-target accumulation)
#   2  + fp32 P2P (leaf pair sums)
#   3  + fp32 upward pass (P2M / M2M)
# On an fp64-starved consumer card (Ada/Ampere: 1/64 fp32 rate) the two hot
# kernels are ~100% fp64 in the inner loop, so level 2 is where the far field
# stops being fp64-bound. Accuracy stays truncation-dominated (see
# artifacts/consumer_gpu_far_field_report.md). Selected with `fp32_level`, or
# NEMO_FAR_FP32_LEVEL for scripts that do not expose the kwarg.
FP32_LEVELS = (0, 1, 2, 3)
DEFAULT_FP32_LEVEL = int(os.environ.get("NEMO_FAR_FP32_LEVEL", "0"))

def _default_pair_budget_gb() -> float:
    """Cap on the P2P pair-list working set, in GB, sized from the device.

    The engine allocates min(nTarget * AVG_NEAR_LEAVES, budget/40B) pairs as two
    device_vector<int> (widebvh/src/treecode.cuh:3965). At N=1M the first term is
    134M pairs = 1.07 GB, so any budget above ~5 GB is inert -- which is why the
    old advice to drop 12 -> 6 on a small card measured as a 0.5 GiB no-op. Below
    it the budget binds and the buffer shrinks proportionally; 1 GB caps it at
    26.8M pairs, still 3.7x the 7.3M actually emitted at mac 0.8, and saves
    ~0.8 GiB at no time cost.

    Sizing off total VRAM gets a 20 GB card a sane value with no env var while
    leaving the H200 headroom for the oversize-tiling path. Exceeding the cap does
    not OOM, it tiles.
    """
    override = os.environ.get("TC_PAIR_BUDGET_GB")
    if override is not None:
        return float(override)
    try:
        total_gb = torch.cuda.mem_get_info()[1] / (1024 ** 3)
    except Exception:
        return 4.0  # no CUDA context yet; the engine's own floor is 1M pairs
    # ~6% of VRAM: 1.2 GB on a 20 GB card, 8.4 GB on a 140 GB H200.
    return max(1.0, min(12.0, 0.06 * total_gb))


# Resolved ONCE and cached, not per instance: os.environ is also where each
# instance *writes* its own setting below, so a per-instance read would let one
# solver's budget leak into the next -- the same trap TC_HILBERT_Q hit. Resolved
# on first use rather than at import, because querying the device would otherwise
# initialise a CUDA context as a side effect of importing this module.
_PAIR_BUDGET_GB: Optional[float] = None


def default_pair_budget_gb() -> float:
    global _PAIR_BUDGET_GB
    if _PAIR_BUDGET_GB is None:
        _PAIR_BUDGET_GB = _default_pair_budget_gb()
    return _PAIR_BUDGET_GB

# wbnemo_policy() -> which multipole expansion the .so was compiled with.
POLICY_ID = {"bary": 0, "cart": 1}
POLICY_LIB = {"cart": "libwidebvh_nemo_cart.so"}

# Order of the doubles filled by wbnemo_stats().
WBNEMO_STAT_FIELDS = (
    "bucket_ms",
    "target_bucket_ms",
    "prep_forces_ms",
    "build_bvh_ms",
    "upward_ms",
    "traverse_ms",
    "p2p_ms",
    "near_pairs",
    "num_nodes",
    "num_source_buckets",
)

# Operating point, chosen by benchmarks/mac_calibration.py (raw data in
# data/widebvh_mac_calibration.csv).
#
# Across the 3k drop, uniform phi=0.1 at 100k, and both 1M two-drop snapshots,
# under gravity and random unit loading, this configuration keeps the far-field
# error below 3.9e-4 of the total velocity -- 37x tighter than the WarpFMM
# theta=0.3 it replaces (worst case 1.4e-2) and ~200x below the learned near
# field's ~7.5% PRMSE, so it contributes nothing measurable end to end.
#
# PDEG stays 7 even though NeMO's target would nominally allow degree 3 or 5:
# reaching the same accuracy at a lower degree requires a much tighter mac, and
# the resulting near-pair explosion (50-62M pairs at mac 0.4 versus 7-10M at mac
# 0.8, on the 1M cases) costs far more than the cheaper M2P saves. Lowering the
# degree here is a false economy; the numbers are in the CSV.
#
# maxLeaf is flat between 512 and 2048 at this operating point; 1024 is
# widebvh's own tuned value for uniform-like clouds.
DEFAULT_MAC = 0.8
DEFAULT_MAX_LEAF = 1024

# Cartesian A/B operating point, chosen by the same script and the same protocol
# -- match the worst case, which is random loading (raw data in
# data/cartesian_mac_calibration.csv).
#
# Order 4 is the policy's maximum and it is also free: the LB traversal kernel
# is 128 registers with ~zero spill at orders 2, 3 and 4 alike, so lower orders
# only buy a tighter mac for the same money. On uniform100k/random, matching
# bary p7 mac 0.8 (3.85e-4) needs order 2 at mac ~0.155, order 3 at ~0.24, or
# order 4 at 0.33 -- 66 ms, 27 ms and 18.7 ms of far field respectively.
#
# max_leaf is inert here: 128 through 2048 give bit-identical error and pair
# counts at this operating point, so it stays at the shared default.
DEFAULT_CART_ORDER = 4
DEFAULT_CART_MAC = 0.33

# Particles per source bucket the Cartesian policy wants. Its matched-accuracy
# mac is 2.4x tighter than bary's, which puts it in a completely different
# near-pair regime (83M vs 7.3M pairs at N=1M), and the engine's automatic cell
# edge -- q = cbrt(max(1024, n/maxLeaf)), i.e. ~1024 cells however large n gets
# -- was tuned for the sparse one. Left on auto the buckets reach ~800
# particles at 1M and P2P alone costs 277 ms of a 344 ms far field; at the right
# granularity the same point costs 141 ms.
#
# Measured optima (random loading): 76 particles/bucket at uniform100k, 183 at
# uniform750k, 139 at the 1M two-drop. Flat enough in between that one target
# covers the range.
CART_PARTICLES_PER_BUCKET = 150


def cart_hilbert_q(n: int, fill: float = 1.0) -> Optional[float]:
    """Cell-edge quantile (cells per axis) for the Cartesian policy at size `n`.

    Returns None -- the engine's auto rule -- whenever auto is already at least
    this fine, which is the case below ~150k, so small-N behaviour is unchanged.

    `fill` is the fraction of the bounding box the cloud actually occupies: q
    counts cells per axis but only OCCUPIED cells become buckets, so a sparse
    geometry needs a larger q for the same bucket count. 1.0 is right for the
    uniform boxes; the 1M two-drop measures ~0.35 and its callers pass the value
    the sweep found directly.
    """
    q = (n / (CART_PARTICLES_PER_BUCKET * fill)) ** (1.0 / 3.0)
    auto = max(1024.0, n / DEFAULT_MAX_LEAF) ** (1.0 / 3.0)
    return float(q) if q > auto else None

_LIB_CACHE: dict[tuple[str, int, int], ctypes.CDLL] = {}


def library_name(policy: str, pdeg: int, fp32_level: int = 0) -> str:
    """File name of the C-ABI library for a (policy, pdeg, fp32_level).

    Mirrors add_widebvh_nemo_library in widebvh/CMakeLists.txt: PDEG 7 keeps
    the plain name, other degrees get _p<N>, and fp32 levels append _f32l<L>.
    """
    if policy == "cart":
        assert fp32_level == 0, "the Cartesian policy has no fp32 build"
        return POLICY_LIB["cart"]
    base = "libwidebvh_nemo" if pdeg == 7 else f"libwidebvh_nemo_p{pdeg}"
    if fp32_level:
        base += f"_f32l{fp32_level}"
    return base + ".so"


def _load_library(policy: str, pdeg: int, fp32_level: int = 0) -> ctypes.CDLL:
    """dlopen (and cache) the C-ABI library for `policy` (and, for bary,
    `pdeg` and `fp32_level`).

    The multipole policy is a template parameter, PDEG and the fp32 level are
    compile-time constants in widebvh, so each combination is a separate .so.
    """
    assert policy in POLICY_ID, f"unknown policy {policy!r}"
    assert fp32_level in FP32_LEVELS, f"fp32_level must be one of {FP32_LEVELS}"
    key = (policy, pdeg, fp32_level)
    if key in _LIB_CACHE:
        return _LIB_CACHE[key]

    name = library_name(policy, pdeg, fp32_level)
    path = WIDEBVH_BUILD_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with:\n"
            f"  cd {WIDEBVH_BUILD_DIR.parent} && source ./env.sh && "
            f'cmake -S . -B {WIDEBVH_BUILD_DIR.name} -G Ninja '
            f'-DCMAKE_BUILD_TYPE=Release -DWIDEBVH_NEMO_PDEG="3;5;7" '
            f'-DWIDEBVH_NEMO_FP32_LEVELS="1;2" && '
            f"cmake --build {WIDEBVH_BUILD_DIR.name} -j --target "
            f"widebvh_nemo widebvh_nemo_p5 widebvh_nemo_p3 widebvh_nemo_cart "
            f"widebvh_nemo_f32l2 widebvh_nemo_p5_f32l2"
        )

    # RTLD_LOCAL (the default) on purpose: the variants export the same symbol
    # names, so loading them globally lets the first one interpose on the rest
    # and every variant silently behaves like it -- an A/B run would compare a
    # backend against itself. The libraries are also built with hidden
    # visibility for the same reason, and the identity asserts below are the
    # check that both mechanisms actually held.
    lib = ctypes.CDLL(str(path))

    lib.wbnemo_abi_version.restype = ctypes.c_int
    lib.wbnemo_policy.restype = ctypes.c_int
    lib.wbnemo_pdeg.restype = ctypes.c_int
    lib.wbnemo_max_order.restype = ctypes.c_int
    lib.wbnemo_radius.restype = ctypes.c_double
    lib.wbnemo_last_error.restype = ctypes.c_char_p
    lib.wbnemo_device_init.argtypes = [ctypes.c_int]
    lib.wbnemo_device_init.restype = ctypes.c_int
    lib.wbnemo_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_float, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_int,
    ]
    lib.wbnemo_create.restype = ctypes.c_int
    lib.wbnemo_destroy.argtypes = [ctypes.c_void_p]
    lib.wbnemo_destroy.restype = ctypes.c_int
    lib.wbnemo_apply.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,
    ]
    lib.wbnemo_apply.restype = ctypes.c_int
    lib.wbnemo_reapply.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.wbnemo_reapply.restype = ctypes.c_int
    lib.wbnemo_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.wbnemo_stats.restype = ctypes.c_int

    abi = lib.wbnemo_abi_version()
    assert abi in WBNEMO_ABI_VERSIONS, (
        f"{path} has ABI version {abi}, this wrapper expects one of "
        f"{WBNEMO_ABI_VERSIONS}; rebuild the library."
    )
    got_policy = lib.wbnemo_policy()
    assert got_policy == POLICY_ID[policy], (
        f"{path} reports policy id {got_policy}, expected {POLICY_ID[policy]} "
        f"({policy}). Another variant has interposed on its symbols."
    )
    if policy == "bary":
        got = lib.wbnemo_pdeg()
        assert got == pdeg, f"{path} was built with PDEG={got}, expected {pdeg}"
    # ABI 3 predates the fp32 ladder and is level 0 by construction; ABI 4
    # reports its level, and the identity check is what catches a stale or
    # interposed .so (all variants export the same symbol names).
    if abi >= 4:
        lib.wbnemo_fp32_level.restype = ctypes.c_int
        got_level = lib.wbnemo_fp32_level()
    else:
        got_level = 0
    assert got_level == fp32_level, (
        f"{path} was built with WIDEBVH_FP32_LEVEL={got_level}, expected "
        f"{fp32_level}"
    )

    _LIB_CACHE[key] = lib
    return lib


class WidebvhFMM(WarpFMM):
    """WarpFMM with the far field computed by the widebvh treecode.

    Note the second tuning parameter is `mac`, NOT `WarpFMM`'s `theta`. Both are
    opening-angle-like, but they are measured against different node radii
    (widebvh: the node's half-diagonal; Warp: its own BVH node extent), so the
    numerical values are not interchangeable -- passing a WarpFMM theta here
    would silently over-refine. `theta` is therefore rejected outright.

    `policy` picks the multipole expansion: "bary" (production, degree `pdeg`)
    or "cart" (Cartesian Taylor, runtime `order`). A `mac` tuned for one is not
    valid for the other either -- the same acceptance criterion, but a different
    series truncated at a different rate.
    """

    def __init__(
        self,
        near_field_operator,
        mac: float = DEFAULT_MAC,
        *,
        near_field_cutoff: float = 6.0,
        max_leaf: int = DEFAULT_MAX_LEAF,
        pdeg: int = 7,
        fp32_level: int = DEFAULT_FP32_LEVEL,
        policy: str = "bary",
        order: Optional[int] = None,
        device: str = "cuda",
        path: str = "split-warpspec-atomic",
        bvh_builder: str = "sah",
        pair_budget_gb: Optional[float] = None,
        hilbert_q: Optional[float] = None,
        quiet: bool = True,
        theta=None,
    ) -> None:
        if pair_budget_gb is None:
            pair_budget_gb = default_pair_budget_gb()
        if theta is not None:
            raise TypeError(
                "WidebvhFMM takes `mac`, not `theta`. They are different "
                "acceptance criteria and their values do not transfer; pick a "
                "mac from data/widebvh_mac_calibration.csv."
            )

        # Reuse WarpFMM's hash-grid setup and near-field machinery verbatim.
        # `theta`/`leaf_size`/`block_dim` end up unused: the far-field override
        # below never touches the Warp BVH.
        super().__init__(
            near_field_operator=near_field_operator,
            theta=float("nan"),
            near_field_cutoff=near_field_cutoff,
            device=device,
        )

        self.mac = float(mac)
        self.max_leaf = int(max_leaf)
        self.pdeg = int(pdeg)
        self.fp32_level = int(fp32_level)
        self.policy = str(policy)
        # 0 tells the engine "policy default", which it clamps to the policy's
        # MAX_ORDER. Only the Cartesian policy reads it; bary's degree is fixed
        # at compile time, so passing an order there would be a silent no-op.
        self.order = 0 if order is None else int(order)
        assert self.policy == "cart" or self.order == 0, (
            f"policy={self.policy!r} ignores `order`; its accuracy is set by "
            f"pdeg at compile time. Pass policy='cart' to use `order`."
        )

        # The engine reads these from the environment inside its constructor, so
        # they must be set before wbnemo_create. Recorded on the instance so
        # benchmarks can log the exact configuration they measured.
        self.env = {
            "TC_PATH": path,
            "TC_BVH_BUILDER": bvh_builder,
            "TC_PAIR_BUDGET_GB": repr(float(pair_budget_gb)),
            "TC_QUIET": "1" if quiet else "0",
        }
        if hilbert_q is not None:
            self.env["TC_HILBERT_Q"] = repr(float(hilbert_q))
        os.environ.update(self.env)
        # Explicitly cleared, not merely left unset: os.environ persists for the
        # life of the process, so without this a default-constructed instance
        # silently inherits the PREVIOUS solver's cell edge. Invisible in a
        # single run; it corrupts any sweep that mixes tuned and auto instances.
        # The engine rejects a non-positive value outright, so "auto" has to be
        # the absence of the variable rather than a sentinel.
        if hilbert_q is None:
            os.environ.pop("TC_HILBERT_Q", None)

        self._lib = _load_library(self.policy, self.pdeg, self.fp32_level)
        self.radius = self._lib.wbnemo_radius()
        self.max_order = self._lib.wbnemo_max_order()

        # Both CUDA runtimes must sit on the same primary context; torch has
        # already created it by the time WarpFMM.__init__ has run.
        torch.cuda.init()
        self._check(self._lib.wbnemo_device_init(
            torch.device(device).index or 0))

        handle = ctypes.c_void_p()
        self._check(self._lib.wbnemo_create(
            ctypes.byref(handle), ctypes.c_float(self.mac),
            ctypes.c_int(self.max_leaf), ctypes.c_double(0.0),
            ctypes.c_double(float(near_field_cutoff)),
            ctypes.c_int(self.order)))
        self._handle = handle

        self._n_cached = -1
        self._pos64 = None
        self._vel_cm = None
        self._stat_buf = (ctypes.c_double * len(WBNEMO_STAT_FIELDS))()
        self.last_stats: dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------
    def _check(self, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(
                f"widebvh error (rc={rc}): "
                f"{self._lib.wbnemo_last_error().decode(errors='replace')}"
            )

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self._lib.wbnemo_destroy(handle)
            self._handle = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- buffers -----------------------------------------------------------
    def _ensure_buffers(self, n: int, device) -> None:
        """(Re)allocate the two staging buffers when the particle count changes.

        `_pos64` is the fp64 AoS position array the engine wants; `_vel_cm` is
        its component-major fp64 output, allocated as (3, N) so that a plain
        transpose gives the (N, 3) view callers expect with no copy.
        """
        if self._n_cached == n and self._pos64 is not None:
            return
        self._pos64 = torch.empty(n, 3, dtype=torch.float64, device=device)
        self._vel_cm = torch.empty(3, n, dtype=torch.float64, device=device)
        self._n_cached = n

    # -- far field ---------------------------------------------------------
    def get_far_field_vel(
        self,
        positions: torch.Tensor,
        forces: torch.Tensor,
    ) -> torch.Tensor:
        """3D far-field velocities (translational only) for r >= near_field_cutoff.

        positions: (N,3) float32 cuda; forces: (N,3) float32 cuda.
        Returns (N,3) float64 cuda.
        """
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        positions = positions.contiguous()
        forces = forces.contiguous()
        n = positions.shape[0]
        assert forces.shape == (n, 3), f"expected (N,3) forces, got {tuple(forces.shape)}"
        assert forces.dtype == torch.float32, "widebvh takes fp32 forces"

        # widebvh launches on the legacy default stream and does not take a
        # stream argument, so ordering is automatic only while torch is also on
        # stream 0. Otherwise fall back to explicit synchronization.
        stream = torch.cuda.current_stream(device=positions.device)
        needs_sync = stream.cuda_stream != 0

        self._ensure_buffers(n, positions.device)

        start_evt.record(stream=stream)
        self._pos64.copy_(positions)          # fused cast+copy, stays on device
        if needs_sync:
            stream.synchronize()

        self._check(self._lib.wbnemo_apply(
            self._handle,
            ctypes.c_void_p(self._pos64.data_ptr()),
            ctypes.c_size_t(n),
            ctypes.c_void_p(forces.data_ptr()),
            ctypes.c_void_p(self._vel_cm.data_ptr()),
            ctypes.c_int(0),                  # positions move every step
        ))

        if needs_sync:
            torch.cuda.default_stream(positions.device).synchronize()
        end_evt.record(stream=stream)
        end_evt.synchronize()

        elapsed = start_evt.elapsed_time(end_evt)
        print(f"[MobFMM] far-field FMM GPU time: {elapsed:.3f} ms")

        self._check(self._lib.wbnemo_stats(self._handle, self._stat_buf))
        self.last_stats = dict(zip(WBNEMO_STAT_FIELDS, list(self._stat_buf)))
        self.last_stats["far_ms"] = elapsed

        # (3,N) row-major IS the component-major layout the engine wrote, so the
        # transpose to (N,3) is a stride change, not a copy.
        far_vel = self._vel_cm.transpose(0, 1)
        assert far_vel.shape == (n, 3)
        return far_vel
