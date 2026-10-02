# widebvh — the NeMO far-field treecode (vendored source)

A GPU Barnes–Hut treecode for Stokeslet/RPY summation over non-uniform particle
distributions on an adaptive BVH (cuBQL builder, `cuBQL/`, Apache-2.0, vendored with its
LICENSE). NeMO uses it through the C ABI in `src/nemo_capi.cu` (`wbnemo_*`, ABI version 4),
wrapped by `src/treecode_widebvh.py` (`WidebvhFMM`): degree-7 barycentric-Lagrange
multipoles, multipole acceptance criterion `mac` 0.8, leaf size 1024.

This is widebvh commit 03efcdb (the fp32 engine), plus the `--centers=`/`--wrench=` case
input of `sphere_mfs` that `benchmarks/broms_truth.py` uses. It is a copy, not a submodule:
edit it here.

## Build

Once per GPU architecture, on a machine with that GPU (the architecture is detected):

```sh
bash extern/widebvh/build_nemo.sh                              # natively
bash docker/run_local.sh bash extern/widebvh/build_nemo.sh     # inside the NeMO image
```

The libraries land in `extern/widebvh/build-sm<compute capability>/` (`build-sm89` for an RTX
40xx, `build-sm90` for an H100/H200, ...; gitignored), and `WidebvhFMM` loads the one that
matches the GPU it runs on, so one checkout serves several kinds of GPU. ~3–5 min per
architecture. Without a visible GPU (a login node) name it: `CUDA_ARCHS=90 bash
extern/widebvh/build_nemo.sh`. A build anywhere else is selected with `WIDEBVH_BUILD_DIR`.
The script's header lists the remaining knobs (extra PDEG variants, targets, parallelism).

Natively this needs CUDA 12.x (12.8+ for sm_120), CMake ≥ 3.18 (developed against 4.0),
Ninja or make, a C++17 compiler with OpenMP, and pkg-config. The source has no
architecture-specific code paths (no `__CUDA_ARCH__` version branches, nothing newer than
the sm_70 warp intrinsics), and its largest shared-memory request (the fp64 upward pass,
~51 KB per block) fits Turing's 64 KB, so any GPU from sm_75 on builds and runs it.

Targets: `libwidebvh_nemo.so` (fp64 production kernels), `libwidebvh_nemo_cart.so`
(Cartesian Taylor policy, `NEMO_FAR_FIELD=widebvh-cart`), `libwidebvh_nemo_f32l{1,2,3}.so`
(fp32 fast paths, selected with `WidebvhFMM(fp32_level=)` / `NEMO_FAR_FP32_LEVEL`; level 3
is the default on every GPU, level 0 is byte-identical to the fp64 engine).

`env.sh` is the MSU HPCC module environment (GCC, CUDA, CMake, PETSc paths); elsewhere,
ignore it.

`sphere_mfs` (the Broms MFS solver behind the N >= 300 truths of `benchmarks/broms_truth.py`)
needs PETSc (CUDA build) + OpenBLAS at configure time; without them CMake warns and skips
only that executable. Build it with `WIDEBVH_TARGETS=sphere_mfs bash
extern/widebvh/build_nemo.sh` and point `BROMS_SPHERE_MFS` at the result.
