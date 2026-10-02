// SPDX-License-Identifier: Apache-2.0
//
// Target-skeleton ID build for the TC_PATH=skel path ("target-side template
// lift"). All targets are per-particle surface discretizations that are rigid
// motions of ONE template (monodisperse MFS particles), and the far field a
// particle receives comes only from MAC-accepted nodes, i.e. from sources at
// distance >= r_g / mac from the particle's group-box center. The restriction
// of that source-exterior field space to the particle's B collocation points is
// numerically low-rank, so we compute a rank-revealing interpolative
// decomposition (ID) ONCE on the template:
//
//   - sample the space with Stokeslet fields from proxy sources on spheres of
//     radius r_proxy (= safety * min_g r_g / mac) and 2*r_proxy around the
//     template (sample matrix A, one SCALAR row per collocation point, one
//     column per (proxy, force-dir, velocity-component)),
//   - select N_skel "skeleton" collocation points by pivoted Cholesky on the
//     Gram matrix G = A A^T (identical pivots to Businger-Golub column-pivoted
//     QR on A^T; the heavy G build is one cuBLAS DGEMM, the B x B pivoted
//     Cholesky is a trivial host loop),
//   - form the lift matrix T = G[:,S] (G[S,S] + lam I)^{-1} (the least-squares
//     interpolation from skeleton values to all B points; rows at S are exactly
//     identity, so skeleton targets are reproduced to round-off).
//
// The lift is built for SCALAR functions and applied componentwise. This is
// rotation-safe with no per-particle transform plumbing: for a particle at pose
// (R, c), each world velocity component pulled back to the template frame,
// u_c(R y + c) = sum_j R_cj v_j(y), is a linear combination of template-frame
// field components, and the proxy spheres are rotation-symmetric, so the
// sampled scalar space contains every particle's incoming field components
// regardless of orientation. One (skeleton, T) pair therefore serves all
// particles; the treecode evaluates the far field only at the N_skel skeleton
// targets of each particle and reconstructs the rest with one shared DGEMM.
//
// Host-side, one-time per apply (geometry + mac dependent). Included by
// treecode.cuh only (single TU per executable, same ODR rule as the policies).
#pragma once

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    cublasStatus_t _s = (call);                                              \
    if (_s != CUBLAS_STATUS_SUCCESS)                                         \
      throw std::runtime_error(std::string("cuBLAS error ") +                \
                               std::to_string((int)_s) + " @ " +             \
                               __FILE__ ":" + std::to_string(__LINE__));     \
  } while (0)
#endif

namespace tskel {

// One thread per (point, proxy) pair fills the 9 scalar columns that proxy
// contributes: velocity component c of the (unscaled) Stokeslet field of a unit
// force along d placed at proxy q, sampled at template point i.
// A is col-major B x (9 * nProxy): A[i + (q*9 + d*3 + c) * B].
__global__ void tskelAssembleKernel(const cuBQL::vec3d *pts, int B,
                                    cuBQL::vec3d center,
                                    const cuBQL::vec3d *proxies, int nProxy,
                                    double *A)
{
  const long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= (long long)B * nProxy) return;
  const int i = (int)(w % B);
  const int q = (int)(w / B);

  const cuBQL::vec3d p = pts[i];
  const cuBQL::vec3d y = proxies[q];
  const double Rx = (p.x - center.x) - y.x;
  const double Ry = (p.y - center.y) - y.y;
  const double Rz = (p.z - center.z) - y.z;
  const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
  const double ir = 1.0 / sqrt(r2);
  const double ir3 = ir / r2;
  const double R[3] = {Rx, Ry, Rz};
  for (int d = 0; d < 3; ++d)
    for (int c = 0; c < 3; ++c) {
      const double val = ((c == d) ? ir : 0.0) + R[c] * R[d] * ir3;
      A[(size_t)i + (size_t)(q * 9 + d * 3 + c) * (size_t)B] = val;
    }
}

// Which pair kernel the sample matrix (and therefore the skeleton + lift) is
// built from. Traction samples the target-normal-contracted traction fields of
// the proxy Stokeslets instead of their velocity fields: differentiation
// amplifies high-order modes, so the traction lift needs its own ID (the
// velocity skeleton is NOT reused for traction).
enum class SampleKind { Stokeslet, Traction };

// Traction variant of tskelAssembleKernel: component c of the (unscaled)
// single-layer traction field t_c = R_c (R.e_d)(R.n_i)/r^5 of a unit force
// along d at proxy q, sampled at template point i with target normal n_i.
// Same column layout A[i + (q*9 + d*3 + c) * B].
__global__ void tskelAssembleTractionKernel(const cuBQL::vec3d *pts,
                                            const cuBQL::vec3d *nrm, int B,
                                            cuBQL::vec3d center,
                                            const cuBQL::vec3d *proxies,
                                            int nProxy, double *A)
{
  const long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= (long long)B * nProxy) return;
  const int i = (int)(w % B);
  const int q = (int)(w / B);

  const cuBQL::vec3d p = pts[i];
  const cuBQL::vec3d n = nrm[i];
  const cuBQL::vec3d y = proxies[q];
  const double Rx = (p.x - center.x) - y.x;
  const double Ry = (p.y - center.y) - y.y;
  const double Rz = (p.z - center.z) - y.z;
  const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
  const double ir = 1.0 / sqrt(r2);
  const double ir5 = (ir * ir) * (ir * ir) * ir;
  const double rdn = Rx * n.x + Ry * n.y + Rz * n.z;
  const double R[3] = {Rx, Ry, Rz};
  for (int d = 0; d < 3; ++d)
    for (int c = 0; c < 3; ++c)
      A[(size_t)i + (size_t)(q * 9 + d * 3 + c) * (size_t)B] =
          R[c] * R[d] * rdn * ir5;
}

struct TargetSkeleton {
  int B = 0;
  int nSkel = 0;                 // effective (may be clamped to numerical rank)
  std::vector<int> idx;          // nSkel template point indices
  std::vector<double> T;         // B x nSkel col-major fp64 lift matrix
  double relResid = 0.0;         // sqrt(max remaining diag / max initial diag)
  double buildMs = 0.0;
};

// Fibonacci sphere of n points, radius r, appended to out.
inline void fibonacciShell(int n, double r, std::vector<cuBQL::vec3d> &out)
{
  const double ga = M_PI * (3.0 - std::sqrt(5.0));
  for (int k = 0; k < n; ++k) {
    const double y = 1.0 - 2.0 * (k + 0.5) / (double)n;
    const double rho = std::sqrt(std::max(0.0, 1.0 - y * y));
    const double phi = ga * (double)k;
    out.push_back(cuBQL::vec3d(r * rho * std::cos(phi), r * y,
                               r * rho * std::sin(phi)));
  }
}

// d_pts: device pointer to the B template collocation points (particle 0's
// bucket-order fp64 targets); `center` is that particle's group-box center (the
// same center the group MAC measures from). For SampleKind::Traction, d_nrm
// must point to the B template outward normals (same bucket order).
inline TargetSkeleton buildTargetSkeleton(cublasHandle_t handle,
                                          const cuBQL::vec3d *d_pts, int B,
                                          cuBQL::vec3d center, double rProxy,
                                          int nSkel, int perShell,
                                          SampleKind kind = SampleKind::Stokeslet,
                                          const cuBQL::vec3d *d_nrm = nullptr)
{
  const auto t0 = HostClock::now();
  if (nSkel < 1 || nSkel > B)
    throw std::runtime_error("target skeleton: need 1 <= nSkel <= group size");
  if (kind == SampleKind::Traction && d_nrm == nullptr)
    throw std::runtime_error("target skeleton: traction samples need normals");

  // Proxy sources: two shells (r, 2r); the inner shell bounds the worst
  // MAC-admissible source distance, the outer one conditions the sampled space.
  std::vector<cuBQL::vec3d> h_proxy;
  fibonacciShell(perShell, rProxy, h_proxy);
  fibonacciShell(perShell, 2.0 * rProxy, h_proxy);
  const int nProxy = (int)h_proxy.size();
  const int NC = 9 * nProxy;

  thrust::device_vector<cuBQL::vec3d> d_proxy(h_proxy.begin(), h_proxy.end());
  thrust::device_vector<double> d_A((size_t)B * NC);
  {
    const long long total = (long long)B * nProxy;
    const int block = 128;
    const long long grid = (total + block - 1) / block;
    if (kind == SampleKind::Traction)
      tskelAssembleTractionKernel<<<(unsigned)grid, block>>>(
          d_pts, d_nrm, B, center, thrust::raw_pointer_cast(d_proxy.data()),
          nProxy, thrust::raw_pointer_cast(d_A.data()));
    else
      tskelAssembleKernel<<<(unsigned)grid, block>>>(
          d_pts, B, center, thrust::raw_pointer_cast(d_proxy.data()), nProxy,
          thrust::raw_pointer_cast(d_A.data()));
    CUDA_CHECK(cudaGetLastError());
  }

  // Gram matrix G = A A^T (B x B) -- the only heavy product, done by cuBLAS.
  thrust::device_vector<double> d_G((size_t)B * B);
  {
    const double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, B, B, NC,
                             &one, thrust::raw_pointer_cast(d_A.data()), B,
                             thrust::raw_pointer_cast(d_A.data()), B,
                             &zero, thrust::raw_pointer_cast(d_G.data()), B));
  }
  std::vector<double> G((size_t)B * B);
  CUDA_CHECK(cudaMemcpy(G.data(), thrust::raw_pointer_cast(d_G.data()),
                        G.size() * sizeof(double), cudaMemcpyDeviceToHost));

  // Pivoted Cholesky on a working copy H of G. The pivot order equals the
  // Businger-Golub CPQR pivot order on A^T; the remaining diagonal after k
  // steps is each point's squared interpolation residual in the sampled space.
  std::vector<double> H(G);
  std::vector<int> piv(B);
  for (int i = 0; i < B; ++i) piv[i] = i;
  double diag0 = 0.0;
  for (int i = 0; i < B; ++i) diag0 = std::max(diag0, H[(size_t)i * B + i]);
  const double rankTol = 1e-15 * diag0;

  int k = 0;
  for (; k < nSkel; ++k) {
    int p = k;
    double dmax = H[(size_t)k * B + k];
    for (int j = k + 1; j < B; ++j) {
      const double dj = H[(size_t)j * B + j];
      if (dj > dmax) { dmax = dj; p = j; }
    }
    if (!(dmax > rankTol)) break;              // numerical rank reached
    if (p != k) {
      std::swap(piv[k], piv[p]);
      for (int j = 0; j < B; ++j) std::swap(H[(size_t)k * B + j],
                                            H[(size_t)p * B + j]);
      for (int i = 0; i < B; ++i) std::swap(H[(size_t)i * B + k],
                                            H[(size_t)i * B + p]);
    }
    const double d = std::sqrt(H[(size_t)k * B + k]);
    const double invd = 1.0 / d;
    for (int i = k + 1; i < B; ++i) H[(size_t)i * B + k] *= invd;
    // Full symmetric Schur update of the trailing block (both triangles), so
    // the row/col pivot swaps above stay valid on later steps.
    for (int j = k + 1; j < B; ++j) {
      const double Ljk = H[(size_t)j * B + k];
      if (Ljk == 0.0) continue;
      double *Hrow = &H[(size_t)j * B];
      for (int i = k + 1; i < B; ++i)
        Hrow[i] -= H[(size_t)i * B + k] * Ljk;
    }
  }
  const int nEff = k;
  if (nEff < 1)
    throw std::runtime_error("target skeleton: sample matrix numerically zero");
  double resid = 0.0;
  for (int j = nEff; j < B; ++j)
    resid = std::max(resid, H[(size_t)j * B + j]);

  TargetSkeleton sk;
  sk.B = B;
  sk.nSkel = nEff;
  sk.idx.assign(piv.begin(), piv.begin() + nEff);
  sk.relResid = (diag0 > 0.0) ? std::sqrt(std::max(0.0, resid) / diag0) : 0.0;

  // Lift T = G[:,S] (G[S,S] + lam I)^{-1}: dense Cholesky solve of the small
  // nEff x nEff system against B right-hand sides (all on the ORIGINAL G).
  const double lam = 1e-12 * diag0;
  std::vector<double> M((size_t)nEff * nEff);
  for (int a = 0; a < nEff; ++a)
    for (int b = 0; b < nEff; ++b)
      M[(size_t)a * nEff + b] = G[(size_t)sk.idx[a] * B + sk.idx[b]] +
                                ((a == b) ? lam : 0.0);
  // In-place lower Cholesky of M.
  for (int a = 0; a < nEff; ++a) {
    for (int b = 0; b <= a; ++b) {
      double s = M[(size_t)a * nEff + b];
      for (int c = 0; c < b; ++c)
        s -= M[(size_t)a * nEff + c] * M[(size_t)b * nEff + c];
      if (a == b) {
        if (!(s > 0.0))
          throw std::runtime_error("target skeleton: Gram subblock not SPD "
                                   "(raise lam or lower nSkel)");
        M[(size_t)a * nEff + a] = std::sqrt(s);
      } else {
        M[(size_t)a * nEff + b] = s / M[(size_t)b * nEff + b];
      }
    }
  }
  // Solve (L L^T) X = G[S,:], X is nEff x B; T[i + s*B] = X[s][i].
  sk.T.assign((size_t)B * nEff, 0.0);
  std::vector<double> col(nEff);
  for (int i = 0; i < B; ++i) {
    for (int a = 0; a < nEff; ++a)
      col[a] = G[(size_t)sk.idx[a] * B + i];   // (G[S,:])(:,i)
    for (int a = 0; a < nEff; ++a) {           // forward: L y = rhs
      double s = col[a];
      for (int c = 0; c < a; ++c) s -= M[(size_t)a * nEff + c] * col[c];
      col[a] = s / M[(size_t)a * nEff + a];
    }
    for (int a = nEff - 1; a >= 0; --a) {      // backward: L^T x = y
      double s = col[a];
      for (int c = a + 1; c < nEff; ++c) s -= M[(size_t)c * nEff + a] * col[c];
      col[a] = s / M[(size_t)a * nEff + a];
    }
    for (int s = 0; s < nEff; ++s) sk.T[(size_t)i + (size_t)s * B] = col[s];
  }

  sk.buildMs = elapsed_ms(t0, HostClock::now());
  return sk;
}

} // namespace tskel
