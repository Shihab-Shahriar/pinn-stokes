"""Health check for the NeMO container. Run it before trusting any measurement.

    docker run --rm --gpus all <user>/nemo:1.0                    # this is the default CMD
    singularity exec --nv --containall <sif> python /workspace/pinn-stokes/docker/smoke_test.py

Three stages, deliberately ordered so a failure localizes itself:

  1. widebvh through ctypes, WITHOUT importing torch, in its own process.
     `wbnemo_smoke` is a self-contained GPU test inside nemo_capi.cu that
     exercises the constant-memory moment tables, the separable
     device-function-pointer refit, thrust/CUB and the cuBQL builder. If this
     fails, the problem is the native build or the driver -- nothing to do with
     Python. It runs subprocessed because it creates and tears down its own CUDA
     context, and doing that before torch initialises CUDA in the same process
     crashes stage 3.
  2. Warp initialises and reports the toolkit/driver pair it resolved.
  3. The real operator: WidebvhFMM over Mob_Nbody_Torch on the baked 50k
     configuration. This also verifies the stock-Warp decision -- every patched
     Warp builtin lives in WarpFMM.get_far_field_vel, which WidebvhFMM
     overrides, so if any were reachable Warp's kernel codegen would fail here.

Exit status is 0 only if all three pass.
"""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ABI 3 = original fp64-only engine; 4 adds wbnemo_fp32_level(). Both accepted.
WBNEMO_ABI_VERSIONS = (3, 4)
H200_REF_CHECKSUM = -2.450400935e+07


def banner(text):
    print(f"\n\033[1m=== {text} ===\033[0m", flush=True)


def stage1_widebvh_ctypes():
    banner("1/3  widebvh C ABI + GPU (no torch)")
    build_dir = Path(os.environ.get("WIDEBVH_BUILD_DIR", "/opt/widebvh/build-nemo"))
    # NEMO_SMOKE_LIB selects a variant (e.g. libwidebvh_nemo_f32l2.so); the
    # default is the production library.
    so = build_dir / os.environ.get("NEMO_SMOKE_LIB", "libwidebvh_nemo.so")
    assert so.exists(), f"no {so} -- WIDEBVH_BUILD_DIR is {build_dir}"

    lib = ctypes.CDLL(str(so))          # RTLD_LOCAL, as src/treecode_widebvh.py does
    for fn, restype in (("wbnemo_abi_version", ctypes.c_int),
                        ("wbnemo_policy", ctypes.c_int),
                        ("wbnemo_pdeg", ctypes.c_int),
                        ("wbnemo_max_order", ctypes.c_int),
                        ("wbnemo_radius", ctypes.c_double),
                        ("wbnemo_last_error", ctypes.c_char_p)):
        getattr(lib, fn).restype = restype
        getattr(lib, fn).argtypes = []

    abi, policy, pdeg = lib.wbnemo_abi_version(), lib.wbnemo_policy(), lib.wbnemo_pdeg()
    assert abi in WBNEMO_ABI_VERSIONS, f"ABI {abi}, expected one of {WBNEMO_ABI_VERSIONS}"
    if abi >= 4:
        lib.wbnemo_fp32_level.restype = ctypes.c_int
        lib.wbnemo_fp32_level.argtypes = []
        fp32_level = lib.wbnemo_fp32_level()
    else:
        fp32_level = 0
    print(f"    abi={abi}  policy={policy} (0=bary)  pdeg={pdeg}  "
          f"max_order={lib.wbnemo_max_order()}  RPY_A={lib.wbnemo_radius()}  "
          f"fp32_level={fp32_level}")
    assert policy == 0, f"policy {policy}, expected 0 (bary) for {so.name}"
    if so.name == "libwidebvh_nemo.so":
        assert pdeg == 7, f"pdeg {pdeg}, expected 7"

    lib.wbnemo_smoke.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_double)]
    lib.wbnemo_smoke.restype = ctypes.c_int
    checksum = ctypes.c_double(0.0)
    rc = lib.wbnemo_smoke(50_000, 0.8, 1024, ctypes.byref(checksum))
    assert rc == 0, f"wbnemo_smoke rc={rc}: {lib.wbnemo_last_error()}"
    # The cloud is a fixed xorshift64 sequence, so this is deterministic per
    # architecture. H200/sm_90 gives -2.450400935e+07; another GPU differing in
    # the last digits is floating point, not a fault. A sign flip or an order of
    # magnitude is a fault.
    print(f"    wbnemo_smoke(n=50000, mac=0.8, maxLeaf=1024) -> checksum "
          f"{checksum.value:.9e}   (H200 reference {H200_REF_CHECKSUM:.9e}, "
          f"PDEG 7 / fp32_level 0)")
    assert checksum.value == checksum.value, "checksum is NaN"
    if pdeg == 7:
        # Level 0 has reproduced the H200 value exactly on sm_89; the fp32
        # levels move it by fp32 rounding (~1e-6..1e-5 relative). Anything
        # beyond 1e-4 is a fault, not precision.
        rel = abs(checksum.value - H200_REF_CHECKSUM) / abs(H200_REF_CHECKSUM)
        tol = 1e-9 if fp32_level == 0 else 1e-4
        assert rel < tol, (f"checksum off by {rel:.3e} relative (tolerance {tol:g} "
                           f"for fp32_level {fp32_level})")


def stage2_warp():
    banner("2/3  Warp")
    import warp as wp
    wp.init()
    # NOT hasattr(wp, "bvh_mp_query") -- that is False even on the fork, because
    # Warp's add_builtin registers into a function table and the wp.-qualified
    # names are only type stubs. Same check as src/treecode.warp_fmm_patch_state.
    patched = None
    for mod in ("warp._src.context", "warp.context"):
        try:
            patched = "bvh_mp_query" in __import__(mod, fromlist=["x"]).builtin_functions
            break
        except (ImportError, AttributeError):
            continue
    print(f"    warp {wp.config.version}   FMM-patched build: {patched}")
    if patched is False:
        print("    (expected: stock warp-lang. `--backend warp` is unavailable; "
              "the widebvh path does not use it.)")


def stage3_end_to_end():
    banner("3/3  WidebvhFMM end to end, N=50,000")
    import torch
    from benchmarks.performance_grand_M import (
        build_far_field, build_near_field, build_forces, load_configuration,
        TMP_DIR)

    props = torch.cuda.get_device_properties(0)
    print(f"    torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"gpu {props.name} sm_{props.major}{props.minor}  "
          f"{props.total_memory / 1024**3:.1f} GB")
    print(f"    NEMO_GIT_SHA={os.environ.get('NEMO_GIT_SHA', '(unset)')}")

    cfg_path = TMP_DIR / "uniform_large_0.1_50000.csv"
    assert cfg_path.exists(), f"missing baked configuration {cfg_path}"
    config = load_configuration(cfg_path)
    n = config.shape[0]

    device = torch.device("cuda")
    positions = torch.as_tensor(config[:, :3], dtype=torch.float32, device=device)
    orientations = torch.as_tensor(config[:, 3:], dtype=torch.float32, device=device)
    forces = torch.as_tensor(build_forces(n, seed=2024), dtype=torch.float32,
                             device=device)
    vis = torch.full((n,), 1.0, dtype=torch.float32, device=device)

    op = build_far_field(build_near_field("nbody", "sphere", 6.0), "widebvh", 6.0)
    print(f"    policy={op.policy} mac={op.mac} pdeg={op.pdeg} "
          f"max_leaf={op.max_leaf} "
          f"pair_budget={os.environ.get('TC_PAIR_BUDGET_GB')} GB")

    vel = op.apply(positions, orientations, forces, vis)
    torch.cuda.synchronize()

    assert vel.shape == (n, 6), f"velocity shape {tuple(vel.shape)}, expected {(n, 6)}"
    assert torch.isfinite(vel).all(), "non-finite velocities"
    speed = vel[:, :3].norm(dim=1)
    print(f"    |U|  mean {speed.mean():.6f}  max {speed.max():.6f}   "
          f"peak VRAM {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    # Forces are zero-mean random (build_forces seed 2024), so a physical answer
    # has a small mean speed and no runaway outlier; the H200 reference is
    # mean 0.294435 / max 0.796436. A broken far field shows up as a huge max.
    assert speed.max() < 1e3, f"implausible max speed {speed.max():.3e}"


def main():
    if "--stage1" in sys.argv:            # subprocess entry point, see module docstring
        stage1_widebvh_ctypes()
        return 0

    failures = []
    rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--stage1"])
    if rc != 0:
        failures.append(f"stage1_widebvh_ctypes: subprocess exited {rc}")

    for stage in (stage2_warp, stage3_end_to_end):
        try:
            stage()
        except Exception as exc:                              # noqa: BLE001
            failures.append(f"{stage.__name__}: {type(exc).__name__}: {exc}")
            print(f"    \033[31mFAILED\033[0m {type(exc).__name__}: {exc}", flush=True)

    print()
    if failures:
        print(f"\033[31m{len(failures)} stage(s) failed\033[0m")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\033[32mall stages passed\033[0m — the image is ready to measure with.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
