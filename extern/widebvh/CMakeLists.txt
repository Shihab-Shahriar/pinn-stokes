# SPDX-License-Identifier: Apache-2.0
#
# Top-level project for the widebvh treecode work. Pulls in the vendored
# cuBQL as a subproject (header-only library + precompiled per-type
# builders) and builds our own driver/utility code against it.
cmake_minimum_required(VERSION 3.18)

# cuBQL, when included as a subproject, refuses to configure unless the
# CUDA architecture is chosen up front. Honor the cluster env first so one
# configure can target both A100 (sm_80) and H200 (sm_90); otherwise keep the
# laptop-friendly native default.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  if (DEFINED ENV{WIDEBVH_CUDA_ARCHITECTURES} AND NOT "$ENV{WIDEBVH_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{WIDEBVH_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
  elseif (DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}" CACHE STRING "CUDA architectures" FORCE)
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
  endif()
endif()

project(widebvh_treecode LANGUAGES CXX CUDA)
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# vendored cuBQL (defines the cuBQL interface target + cuBQL_cuda_float3 etc.)
add_subdirectory(cuBQL)

# cuBLAS is used by the treecode engine itself now (the TC_PATH=skel target-
# skeleton path: Gram-matrix DGEMM in the one-time ID build + the per-evaluate
# lift DGEMM), so every treecode driver links CUDA::cublas.
find_package(CUDAToolkit REQUIRED)

# Grid-bucketed, leaf-constrained BVH prototype (CUB/thrust + cuBQL). Builds an
# fp32 BVH whose leaves are buckets we construct ourselves (capped on both
# particle count and edge length).
add_executable(gridbvh_driver
  src/gridbvh_main.cu
  src/grid_buckets.cu
  src/common.cu
  )
target_link_libraries(gridbvh_driver PRIVATE cuBQL_cuda_float3)

# Stokeslet treecode driver factory. The reusable engine lives in the headers
# src/treecode.cuh (+ src/treecode_validate.cuh); each per-distribution driver is
# a single .cu that #includes them. Separable compilation is required for
# refit_aggregate's device-function-pointer mechanism (mirrors sample07). Adding
# a new distribution driver is one line: add_treecode_driver(<name> src/<name>.cu).
function(add_treecode_driver TARGET SRC)
  add_executable(${TARGET}
    ${SRC}
    src/grid_buckets.cu
    src/common.cu
    )
  target_link_libraries(${TARGET} PRIVATE cuBQL_cuda_float3 CUDA::cublas)
  set_target_properties(${TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION   ON
    CUDA_RESOLVE_DEVICE_SYMBOLS   ON
    POSITION_INDEPENDENT_CODE     ON
    CUDA_USE_STATIC_CUDA_RUNTIME  ON
    )
endfunction()

# fp32 data + moments, fp64 velocity accumulation, on the two_ball dataset.
add_treecode_driver(two_ball src/two_ball.cu)

# Same driver against the Cartesian Taylor multipole policy (order <= 4).
add_treecode_driver(two_ball_cart src/two_ball.cu)
target_compile_definitions(two_ball_cart PRIVATE TWO_BALL_CARTESIAN=1)

# Same driver with the Rotne-Prager-Yamakawa regularization of the Stokeslet
# (a = 1). Self-consistency gate for the NeMO library: both the M2P proxy sum and
# the fp64 direct-sum reference switch kernel together, so relL2err here measures
# multipole truncation of the RPY kernel and should track the Stokeslet build.
add_treecode_driver(two_ball_rpy src/two_ball.cu)
target_compile_definitions(two_ball_rpy PRIVATE WIDEBVH_RPY_A=1.0)

# The RPY driver at the fp32 fast-path levels the NeMO fp32 libraries ship
# (stokes_kernel.cuh WIDEBVH_FP32_LEVEL). The directSum64 reference stays fp64,
# so relL2err here = truncation + fp32 arithmetic; against two_ball_rpy at the
# same mac it isolates the arithmetic part. Level 2 exercises p2pAtomicKernel32
# only under TC_PATH=split-warpspec-atomic.
add_treecode_driver(two_ball_rpy_f32l1 src/two_ball.cu)
target_compile_definitions(two_ball_rpy_f32l1 PRIVATE WIDEBVH_RPY_A=1.0 WIDEBVH_FP32_LEVEL=1)
add_treecode_driver(two_ball_rpy_f32l2 src/two_ball.cu)
target_compile_definitions(two_ball_rpy_f32l2 PRIVATE WIDEBVH_RPY_A=1.0 WIDEBVH_FP32_LEVEL=2)

# Both at once: the gate for the Cartesian policy's RPY term (the -c d_i d_j(1/r)
# branch in cartesian_stokes.cuh::m2p). Same self-consistency argument as
# two_ball_rpy -- reference and M2P switch kernel together -- so relL2err here is
# pure multipole truncation and must fall with mac and with TC_ORDER. This is the
# configuration libwidebvh_nemo_cart.so ships.
add_treecode_driver(two_ball_cart_rpy src/two_ball.cu)
target_compile_definitions(two_ball_cart_rpy PRIVATE
  TWO_BALL_CARTESIAN=1 WIDEBVH_RPY_A=1.0)

# Same driver against the FMM3D-style spherical-harmonic policy (order <= 12,
# fp64 moment storage + fp64 far-field geometry).
add_treecode_driver(two_ball_sph src/two_ball.cu)
target_compile_definitions(two_ball_sph PRIVATE TWO_BALL_SPHERICAL=1)

# A/B anchor: same policy with fp32 far-field geometry (SPH_FP32_GEOM), to quantify
# the fp64-geometry accuracy gain against the fp64 reference direct-sum.
add_treecode_driver(two_ball_sph_fp32geom src/two_ball.cu)
target_compile_definitions(two_ball_sph_fp32geom PRIVATE TWO_BALL_SPHERICAL=1 SPH_FP32_GEOM=1)

# Standalone unit tests for the spherical-harmonic policy (P2M/M2P/M2M vs the
# python reference stokes_fmm_moment_demo.py; no treecode engine involved).
add_treecode_driver(sph_selftest src/sph_selftest.cu)

# Direct benchmark for exported ellipsoid MFS source/target/strength binaries.
add_treecode_driver(mfs_ellipsoid_treecode_bench src/mfs_ellipsoid_treecode_bench.cu)

# Crossover experiment: ellipsoid count P at which an abortable brute-force
# direct sum beats the treecode on the static ellipsoid RHS case (P = prefix
# of the frozen 10k-ellipsoid arrays).
add_treecode_driver(crossover_direct_vs_treecode src/crossover_direct_vs_treecode.cu)

# Debug tool: diff two TC_PATH outputs over ALL targets of one cloud (used to
# hunt the direct-warpspec large-N single-target corruption).
add_treecode_driver(dbg_path_diff src/dbg_path_diff.cu)

# Diagnostic (not a hot path): quantifies leaf-AABB overlap of the grid-bucket
# BVH leaves (grid / grid-hilbert / global-hilbert bucketizers) and the
# resulting MAC near-shell size. No treecode.cuh include, so no separable
# compilation needed.
add_executable(leaf_overlap_stats
  src/leaf_overlap_stats.cu
  src/grid_buckets.cu
  src/common.cu
  )
target_link_libraries(leaf_overlap_stats PRIVATE cuBQL_cuda_float3)

# Synthetic particle-distribution generator for the split-warpspec sweep.
# Host-only (no cuBQL/treecode, no separable compilation); writes two_ball-format
# fp64 SoA .bin clouds spanning a range of tile-scale heterogeneity.
add_executable(gen_distributions src/gen_distributions.cu)

# ----------------------------------------------------------------------------
# NeMO / pinn-stokes far-field library: a C-ABI shared object around
# Treecode<BaryStokes>, dlopen'd by ctypes from a conda PyTorch process.
#
# That process is built against a DIFFERENT toolchain (GCCcore 13.3 + CUDA 12.6 +
# torch's own cudart/cublas), so this target deliberately isolates itself:
#   - the interface is pure C (no C++ type crosses the boundary),
#   - libstdc++/libgcc are linked STATICALLY, so the host process's older
#     GLIBCXX/CXXABI cannot break us,
#   - the CUDA runtime is static, so we do not fight torch's cudart version.
# The only remaining shared dependencies are libc, the driver (libcuda.so.1,
# shared by definition) and libcublas (ABI-stable within major 12, and reached
# only by the TC_PATH=skel lift, which NeMO never selects).
#
# WIDEBVH_NEMO_PDEG picks the barycentric Chebyshev degree at compile time
# (bary_stokes.cuh's PDEG). NeMO targets a much looser accuracy than widebvh's
# usual 1e-8, so lower degrees are worth benchmarking; each degree is a separate
# .so because PDEG is a constexpr, not a runtime knob.
# ----------------------------------------------------------------------------
set(WIDEBVH_NEMO_PDEG "7" CACHE STRING
    "Chebyshev degree(s) to build the NeMO far-field library for (list, e.g. 3;5;7)")

# WIDEBVH_NEMO_FP32_LEVELS adds fp32 fast-path variants of every degree above
# (stokes_kernel.cuh WIDEBVH_FP32_LEVEL: 1 fp32 M2P, 2 + fp32 P2P, 3 + fp32
# upward pass), for GPUs whose fp64 rate is 1/64 of fp32 (GeForce Ada/Ampere).
# Empty by default: the H200 production build is level 0 only and unchanged.
# Each level is a separate .so, named libwidebvh_nemo[_p<PDEG>]_f32l<L>.so.
set(WIDEBVH_NEMO_FP32_LEVELS "" CACHE STRING
    "fp32 fast-path level(s) to ALSO build the NeMO library for (list, e.g. 1;2); empty = fp64 only")

# Shared body for every NeMO C-ABI variant. Only the multipole policy and its
# compile-time parameters differ between them, and those arrive through ARGN --
# everything else (sources, RPY radius, visibility, link hardening) must stay
# identical across variants, which is why it lives in one place.
function(add_widebvh_nemo_target TARGET)
  add_library(${TARGET} SHARED
    src/nemo_capi.cu
    src/grid_buckets.cu
    src/common.cu
    )
  target_link_libraries(${TARGET} PRIVATE cuBQL_cuda_float3 CUDA::cublas)
  # RPY radius a = 1: NeMO's far field is the Rotne-Prager-Yamakawa mobility of
  # unit-radius spheres, not the bare Stokeslet. Compile-time because it sits in
  # the innermost P2P/M2P loop; every other target leaves it 0 and is unaffected.
  target_compile_definitions(${TARGET} PRIVATE WIDEBVH_RPY_A=1.0 ${ARGN})
  set_target_properties(${TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION   ON
    CUDA_RESOLVE_DEVICE_SYMBOLS   ON
    POSITION_INDEPENDENT_CODE     ON
    CUDA_USE_STATIC_CUDA_RUNTIME  ON
    # Hide everything except the WBNEMO_API entry points. Without this the
    # variants -- which are the same code with a different policy or constexpr --
    # export identical symbol names, and whichever loads first interposes on the
    # others, so all of them silently behave like the first one loaded. An A/B
    # run that loads bary and cartesian together would then compare bary to bary.
    CXX_VISIBILITY_PRESET        hidden
    CUDA_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN    ON
    )
  # Belt and braces for the same hazard: bind intra-library references locally
  # rather than through the global symbol table.
  target_link_options(${TARGET} PRIVATE
    -static-libstdc++ -static-libgcc -Wl,-Bsymbolic)
endfunction()

function(add_widebvh_nemo_library PDEG)
  if (PDEG STREQUAL "7")
    set(BASE widebvh_nemo)                # default degree keeps the plain name
  else()
    set(BASE widebvh_nemo_p${PDEG})
  endif()
  add_widebvh_nemo_target(${BASE} WIDEBVH_PDEG=${PDEG})   # level 0, unchanged name
  foreach(_lvl IN LISTS WIDEBVH_NEMO_FP32_LEVELS)
    if (_lvl STREQUAL "" OR _lvl STREQUAL "0")
      continue()
    endif()
    add_widebvh_nemo_target(${BASE}_f32l${_lvl}
      WIDEBVH_PDEG=${PDEG} WIDEBVH_FP32_LEVEL=${_lvl})
  endforeach()
endfunction()

foreach(_pdeg IN LISTS WIDEBVH_NEMO_PDEG)
  add_widebvh_nemo_library(${_pdeg})
endforeach()

# The same C ABI over the Cartesian Taylor policy instead of barycentric
# Lagrange, as the A/B far field for NeMO. No PDEG: the Cartesian accuracy
# ladder is the RUNTIME Config::order (1..4), passed through wbnemo_create, so
# one .so covers every order. Gated by two_ball_cart_rpy, which validates the
# same policy + kernel combination against a direct sum.
add_widebvh_nemo_target(widebvh_nemo_cart NEMO_CARTESIAN=1)

# ----------------------------------------------------------------------------
# CPU-only treecode port (src/cpu/): plain C++17 + OpenMP, zero CUDA in the
# target. cuBQL is used header-only (BinaryBVH + the host cpu::spatialMedian
# builder, whose implementation is pulled in by builder/cpu.h); the __host__/
# __device__ decorations compile away without __CUDACC__, so a plain include
# path suffices -- do NOT link the cuBQL interface target (it may carry
# CUDA-flavored interface options).
# ----------------------------------------------------------------------------
option(WIDEBVH_CPU_NATIVE "compile the CPU treecode with -march=native" ON)
set(WIDEBVH_CPU_RSQRT_NR "1" CACHE STRING
    "Newton steps in the experimental vrsqrt14pd M2P (mfs_ellipsoid_cpu_rsqrt)")
set(WIDEBVH_CPU_PDEG "7" CACHE STRING
    "Chebyshev degree of the experimental CPU twin (mfs_ellipsoid_cpu_rsqrt)")
option(WIDEBVH_CPU_M2P_FP32
    "evaluate the far field in fp32 (16 lanes) in mfs_ellipsoid_cpu_rsqrt" OFF)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(two_ball_cpu
  src/cpu/two_ball_cpu.cpp
  src/cpu/treecode_cpu.cpp
  src/cpu/grid_buckets_cpu.cpp
  )
target_include_directories(two_ball_cpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cuBQL
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(two_ball_cpu PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(two_ball_cpu PRIVATE
  -O3 -funroll-loops
  $<$<BOOL:${WIDEBVH_CPU_NATIVE}>:-march=native>)

# A/B twin of two_ball_cpu carrying the experimental M2P kernels, so the
# accuracy of the vrsqrt14/fp32 far field can be gated on the canonical
# self-target distributions (two_ball, plummer) and not just on MFS.
add_executable(two_ball_cpu_rsqrt
  src/cpu/two_ball_cpu.cpp
  src/cpu/treecode_cpu.cpp
  src/cpu/grid_buckets_cpu.cpp
  )
target_include_directories(two_ball_cpu_rsqrt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cuBQL
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(two_ball_cpu_rsqrt PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(two_ball_cpu_rsqrt PRIVATE
  TCCPU_FAST_RSQRT TCCPU_RSQRT_NR=${WIDEBVH_CPU_RSQRT_NR}
  TCCPU_PDEG=${WIDEBVH_CPU_PDEG}
  $<$<BOOL:${WIDEBVH_CPU_M2P_FP32}>:TCCPU_M2P_FP32>)
target_compile_options(two_ball_cpu_rsqrt PRIVATE
  -O3 -funroll-loops
  $<$<BOOL:${WIDEBVH_CPU_NATIVE}>:-march=native>)

# ----------------------------------------------------------------------------
# Broms MFS Stokes mobility solver (fp64). The Stokeslet sum is accelerated by
# the GPU treecode (src/treecode.cuh), so the MFS drivers now link cuBQL and
# need separable compilation (refit_aggregate's device
# function pointers) in addition to PETSc (matrix-free GMRES) + cuBLAS/cuSOLVER
# (one-time dense setup). PETSc is found via pkg-config; point PKG_CONFIG_PATH at
# the PETSc arch's lib/pkgconfig at configure time, e.g.
#   PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig cmake -S . -B build ...
# NOTE: we do NOT force the static CUDA runtime here (unlike add_treecode_driver)
# because PETSc links the shared runtime (-lcudart); matching it avoids a
# duplicate-runtime conflict at link time. treecode.cuh (its multipole modules
# define __device__ function-pointer globals) is included only by the single
# driver TU via mfs_broms.cuh; grid_buckets.cu/common.cu do not include it, so
# there is no duplicate-symbol ODR violation.
# ----------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
if (DEFINED ENV{PETSC_DIR} AND NOT "$ENV{PETSC_DIR}" STREQUAL "")
  list(APPEND PKG_CONFIG_ARGN "--define-variable=PETSC_DIR=$ENV{PETSC_DIR}")
endif()
pkg_check_modules(PETSC IMPORTED_TARGET PETSc)
if (PETSC_FOUND)
  function(add_mfs_driver TARGET SRC)
    add_executable(${TARGET}
      ${SRC}
      src/grid_buckets.cu
      src/common.cu
      )
    target_link_libraries(${TARGET} PRIVATE
      PkgConfig::PETSC CUDA::cublas CUDA::cusolver cuBQL_cuda_float3)
    set_target_properties(${TARGET} PROPERTIES
      CUDA_SEPARABLE_COMPILATION   ON
      CUDA_RESOLVE_DEVICE_SYMBOLS   ON
      POSITION_INDEPENDENT_CODE     ON
      )
  endfunction()

  # Multi-particle test against a reference CSV (shares the solver core in
  # src/mfs_broms.cuh). This source may be absent in stripped-down worktrees.
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/test_mfs_mp.cu")
    add_mfs_driver(test_mfs_mp  src/test_mfs_mp.cu)
  endif()
  # Three-sphere triangle sweep vs. the Broms paper Table 3.
  add_mfs_driver(mfs_triangle src/mfs_triangle.cu)
  # Large-scale (P=1000) triaxial-ellipsoid mobility test vs. the MATLAB
  # StokesMFS3D FMM reference CSV.
  add_mfs_driver(test_mfs_ellipsoid src/test_mfs_ellipsoid.cu)
  # Yan et al. (2020) second-kind BI sphere-suspension mobility solver:
  # KAFMM-style traction/velocity treecode (TC_PATH=skel) + offline self blocks
  # from scripts/gen_sphere_selfblocks.py.
  add_mfs_driver(sphere_mobility src/sphere_mobility.cu)
  # MFS mobility benchmark on the SAME sphere-suspension configurations as
  # sphere_mobility (RSA centers + random wrench), skel-treecode accelerated.
  add_mfs_driver(sphere_mfs src/sphere_mfs.cu)

  # --------------------------------------------------------------------------
  # CPU MFS mobility solver: TreecodeCpu evaluator + PETSc GMRES + OpenBLAS
  # dense one-body setup (port of the pvfmm examples/mfs_ellipsoid.cpp CPU
  # driver). Plain CXX, no CUDA in the target. OpenBLAS via openblas.pc (the
  # module env.sh loads puts it on PKG_CONFIG_PATH); NOT openblas64
  # (INTERFACE64 would silently corrupt the 32-bit-int LAPACK prototypes).
  # Explicit -lopenblas is required: ld does not resolve our dgesdd_/dgemm_
  # from libpetsc's transitive DT_NEEDEDs. libpetsc.so itself carries CUDA
  # DT_NEEDEDs (rpath-baked); the rpath-link below satisfies the link-time
  # closure from the CUDA module's driver stubs, and at RUNTIME on GPU-less
  # nodes put the stubs on LD_LIBRARY_PATH:
  #   export LD_LIBRARY_PATH=$EBROOTCUDA/lib64/stubs:$LD_LIBRARY_PATH
  # (mfs_petsc_cpu.cpp sets -device_enable none, so PETSc never touches a
  # device.)
  # --------------------------------------------------------------------------
  pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)

  add_executable(mfs_ellipsoid_cpu
    src/cpu/mfs_ellipsoid_cpu.cpp
    src/cpu/mfs_petsc_cpu.cpp
    src/cpu/treecode_cpu.cpp
    src/cpu/grid_buckets_cpu.cpp
    )
  target_include_directories(mfs_ellipsoid_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuBQL
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(mfs_ellipsoid_cpu PRIVATE
    OpenMP::OpenMP_CXX PkgConfig::PETSC PkgConfig::OPENBLAS)
  target_compile_options(mfs_ellipsoid_cpu PRIVATE
    -O3 -funroll-loops
    $<$<BOOL:${WIDEBVH_CPU_NATIVE}>:-march=native>)
  if (DEFINED ENV{EBROOTCUDA})
    target_link_options(mfs_ellipsoid_cpu PRIVATE
      "-Wl,-rpath-link,$ENV{EBROOTCUDA}/lib64"
      "-Wl,-rpath-link,$ENV{EBROOTCUDA}/lib64/stubs")
  endif()

  # A/B twin of mfs_ellipsoid_cpu built with the experimental AVX-512
  # vrsqrt14pd + Newton M2P (bary_stokes_cpu.h). Same sources, same flags,
  # one define -- keep it a separate binary so the production one is never
  # affected. TCCPU_RSQRT_NR=1|2 selects the Newton step count.
  add_executable(mfs_ellipsoid_cpu_rsqrt
    src/cpu/mfs_ellipsoid_cpu.cpp
    src/cpu/mfs_petsc_cpu.cpp
    src/cpu/treecode_cpu.cpp
    src/cpu/grid_buckets_cpu.cpp
    )
  target_include_directories(mfs_ellipsoid_cpu_rsqrt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuBQL
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(mfs_ellipsoid_cpu_rsqrt PRIVATE
    OpenMP::OpenMP_CXX PkgConfig::PETSC PkgConfig::OPENBLAS)
  target_compile_definitions(mfs_ellipsoid_cpu_rsqrt PRIVATE
    TCCPU_FAST_RSQRT TCCPU_RSQRT_NR=${WIDEBVH_CPU_RSQRT_NR}
    TCCPU_PDEG=${WIDEBVH_CPU_PDEG}
    $<$<BOOL:${WIDEBVH_CPU_M2P_FP32}>:TCCPU_M2P_FP32>)
  target_compile_options(mfs_ellipsoid_cpu_rsqrt PRIVATE
    -O3 -funroll-loops
    $<$<BOOL:${WIDEBVH_CPU_NATIVE}>:-march=native>)
  if (DEFINED ENV{EBROOTCUDA})
    target_link_options(mfs_ellipsoid_cpu_rsqrt PRIVATE
      "-Wl,-rpath-link,$ENV{EBROOTCUDA}/lib64"
      "-Wl,-rpath-link,$ENV{EBROOTCUDA}/lib64/stubs")
  endif()
else()
  message(WARNING "PETSc not found via pkg-config; skipping MFS targets. "
                  "Set PKG_CONFIG_PATH to $PETSC_DIR/$PETSC_ARCH/lib/pkgconfig.")
endif()
