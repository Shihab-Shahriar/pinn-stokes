// SPDX-License-Identifier: Apache-2.0
//
// Driver for the second-kind BI sphere-suspension mobility solver
// (sphere_bim.cuh): N rigid unit spheres, 162 surface points each (p=8 grid
// from scripts/gen_sphere_selfblocks.py), forces/torques -> rigid velocities.
// Off-diagonal operators run on the KAFMM-style traction/velocity treecode
// (TC_PATH=skel) or a brute GPU reference (--brute); --check runs both and
// compares. Contact resolution is deliberately out of scope: this measures the
// MOBILITY APPLICATION only.
//
// Usage:
//   sphere_mobility [--nspheres=100] [--seed=1] [--phi=0.05] [--mac=0.75]
//                   [--data=data/sphere_p8] [--brute] [--check]
//                   [--ksp-rtol=1e-8] [--forces=random[:amp]]
//                   [--near-beta=2.0]   (VSH near-pair correction; 0 = off)
// plus any PETSc options (passed through).
//
// Default operating point (2026-07-22 retune): mac=0.75 + traction skeleton
// rank 96 (TC_TGT_SKEL_TRACTION env default set in sbimSetup). With the VSH
// near correction owning the near field, the 10k sweep showed identical GMRES
// iterations and bit-identical rigidity residual from mac 0.6/128 through
// 0.8/96, with solution deltas ~1e-6 against the 6e-3 p=8 discretization
// floor; 0.75/96 cuts the matvec 426 -> 221 ms. Use --mac / the env to go
// back.

#include "sphere_bim.cuh"

namespace {

struct DriverArgs {
  int nspheres = 100;
  unsigned long seed = 1;
  double phi = 0.05;
  float mac = 0.75f;
  std::string data = "data/sphere_p8";
  bool brute = false;
  bool check = false;
  double kspRtol = 1e-8;
  bool randomForces = false;
  double famp = 1.0;
  double nearBeta = 2.0;
};

bool parseArg(const std::string &a, DriverArgs &d)
{
  auto val = [&](const char *k) -> const char * {
    const size_t n = std::strlen(k);
    return (a.compare(0, n, k) == 0) ? a.c_str() + n : nullptr;
  };
  if (const char *v = val("--nspheres=")) { d.nspheres = std::atoi(v); return true; }
  if (const char *v = val("--seed="))     { d.seed = std::strtoul(v, nullptr, 10); return true; }
  if (const char *v = val("--phi="))      { d.phi = std::atof(v); return true; }
  if (const char *v = val("--mac="))      { d.mac = (float)std::atof(v); return true; }
  if (const char *v = val("--data="))     { d.data = v; return true; }
  if (const char *v = val("--ksp-rtol=")) { d.kspRtol = std::atof(v); return true; }
  if (const char *v = val("--near-beta=")) { d.nearBeta = std::atof(v); return true; }
  if (a == "--brute") { d.brute = true; return true; }
  if (a == "--check") { d.check = true; return true; }
  if (const char *v = val("--forces=")) {
    if (std::strncmp(v, "random", 6) == 0) {
      d.randomForces = true;
      if (v[6] == ':') d.famp = std::atof(v + 7);
      return true;
    }
    std::fprintf(stderr, "unknown --forces value: %s\n", v);
    std::exit(2);
  }
  return false;  // not ours (PETSc option)
}

// Random sequential insertion of N unit spheres, min center distance 2.2
// (surface separation >= 0.2), in a cube sized for volume fraction phi.
// N=1 and N=2 are deterministic special cases (gates 1 and 2).
std::vector<double> makeCenters(int N, unsigned long seed, double phi,
                                double &Lout)
{
  std::vector<double> c;
  Lout = 0.0;
  if (N == 1) { c = {0.0, 0.0, 0.0}; return c; }
  if (N == 2) { c = {0.0, 0.0, 0.0, 2.2, 0.0, 0.0}; return c; }
  const double minD = 2.2 + 1e-9, minD2 = minD * minD;
  const double L = std::cbrt((double)N * (4.0 * M_PI / 3.0) / phi);
  Lout = L;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> U(0.0, L);
  c.reserve((size_t)3 * N);
  // cell list (edge = minD) so RSA stays O(N) -- needed at N ~ 10^5. The
  // accept/reject decisions (and thus the configuration for a given seed) are
  // identical to the old O(N^2) scan.
  const double inv = 1.0 / minD;
  auto key = [](long long x, long long y, long long z) {
    return (x * 73856093LL) ^ (y * 19349663LL) ^ (z * 83492791LL);
  };
  std::unordered_multimap<long long, int> cells;
  cells.reserve((size_t)N * 2);
  long long attempts = 0;
  const long long maxAttempts = 2000000LL * N;
  while ((int)(c.size() / 3) < N) {
    if (++attempts > maxAttempts) {
      std::fprintf(stderr,
                   "RSA insertion failed after %lld attempts (phi=%.3f too "
                   "dense?); lower --phi\n", attempts, phi);
      std::exit(1);
    }
    const double x = U(rng), y = U(rng), z = U(rng);
    const long long cx = (long long)std::floor(x * inv);
    const long long cy = (long long)std::floor(y * inv);
    const long long cz = (long long)std::floor(z * inv);
    bool ok = true;
    for (long long dx = -1; dx <= 1 && ok; ++dx)
      for (long long dy = -1; dy <= 1 && ok; ++dy)
        for (long long dz = -1; dz <= 1 && ok; ++dz) {
          auto range = cells.equal_range(key(cx + dx, cy + dy, cz + dz));
          for (auto it = range.first; it != range.second; ++it) {
            const size_t j = (size_t)3 * it->second;
            const double ex = x - c[j], ey = y - c[j + 1], ez = z - c[j + 2];
            if (ex * ex + ey * ey + ez * ez < minD2) { ok = false; break; }
          }
        }
    if (ok) {
      cells.emplace(key(cx, cy, cz), (int)(c.size() / 3));
      c.push_back(x); c.push_back(y); c.push_back(z);
    }
  }
  return c;
}

std::vector<double> makeWrench(int N, bool random, double amp,
                               unsigned long seed)
{
  std::vector<double> FT((size_t)6 * N, 0.0);
  if (random) {
    std::mt19937_64 rng(seed + 12345);
    std::uniform_real_distribution<double> U(-amp, amp);
    for (double &v : FT) v = U(rng);
  } else {
    for (int k = 0; k < N; ++k) FT[(size_t)6 * k + 2] = -1.0;  // F = (0,0,-1)
  }
  return FT;
}

double relL2(const std::vector<double> &a, const std::vector<double> &ref)
{
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double e = a[i] - ref[i];
    num += e * e;
    den += ref[i] * ref[i];
  }
  return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

// max_b |u_b - (U_j + Omega_j x (x_b - c_j))| -- spectral-consistency sanity.
double rigidityResidual(const SBimContext *m, const std::vector<double> &u,
                        const std::vector<double> &UOm)
{
  double worst = 0.0;
  for (int k = 0; k < m->N; ++k) {
    const double *U = &UOm[(size_t)6 * k];
    const double *Om = U + 3;
    for (int b = 0; b < m->B; ++b) {
      const double *x = &m->h_grid[(size_t)3 * b];  // = point - center
      const double ur[3] = {U[0] + Om[1] * x[2] - Om[2] * x[1],
                            U[1] + Om[2] * x[0] - Om[0] * x[2],
                            U[2] + Om[0] * x[1] - Om[1] * x[0]};
      for (int c = 0; c < 3; ++c)
        worst = std::max(worst,
                         std::abs(u[(size_t)3 * (k * m->B + b) + c] - ur[c]));
    }
  }
  return worst;
}

void reportTiming(const char *tag, const SBimContext *m)
{
  if (!m->timingEnabled) return;
  const double matvecMean =
      m->matvecCount ? m->matvecWallMs / m->matvecCount : 0.0;
  const double offMean =
      m->matvecCount ? m->matvecOffMs / m->matvecCount : 0.0;
  const double nearMean =
      m->matvecCount ? m->matvecNearMs / m->matvecCount : 0.0;
  std::printf(
      "[sbim %s] rhs=%.3f ms (offdiag/tree-build=%.3f)  ksp=%.3f ms "
      "(%d matvecs: mean=%.3f ms, offdiag mean=%.3f ms, near mean=%.3f ms)  "
      "finalS=%.3f ms  extract=%.3f ms  TOTAL(mobility apply)=%.3f ms\n",
      tag, m->rhsWallMs, m->treeBuildMs, m->kspWallMs, m->matvecCount,
      matvecMean, offMean, nearMean, m->finalSMs, m->extractMs,
      m->rhsWallMs + m->kspWallMs + m->finalSMs + m->extractMs);
  std::fflush(stdout);
}

struct SolveResult {
  std::vector<double> UOm;
  std::vector<double> u;
};

SolveResult runSolve(const DriverArgs &d, const std::vector<double> &centers,
                     const std::vector<double> &FT, bool useTreecode,
                     SBimContext **ctxOut)
{
  SBimContext *m = sbimSetup(d.data, d.nspheres, centers, useTreecode, d.mac,
                             d.kspRtol, d.nearBeta);
  SolveResult r;
  sbimSolve(m, FT, r.UOm, &r.u);
  reportTiming(useTreecode ? "treecode" : "brute", m);
  const double rig = rigidityResidual(m, r.u, r.UOm);
  std::printf("[sbim %s] rigidity residual max|u - (U + Omega x r)| = %.3e\n",
              useTreecode ? "treecode" : "brute", rig);
  if (ctxOut) *ctxOut = m; else sbimDestroy(m);
  return r;
}

}  // namespace

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  DriverArgs d;
  for (int i = 1; i < argc; ++i) parseArg(argv[i], d);

  double L = 0.0;
  const std::vector<double> centers = makeCenters(d.nspheres, d.seed, d.phi, L);
  const std::vector<double> FT =
      makeWrench(d.nspheres, d.randomForces, d.famp, d.seed);
  std::printf("[sbim] N=%d spheres, B from %s, phi=%.3f box_L=%.2f seed=%lu "
              "mac=%.2f near-beta=%.2f mode=%s%s\n",
              d.nspheres, d.data.c_str(), d.phi, L, d.seed, (double)d.mac,
              d.nearBeta,
              d.check ? "check(brute+treecode)"
                      : (d.brute ? "brute" : "treecode"),
              d.randomForces ? " forces=random" : " forces=F(0,0,-1)");

  // GPU near-correction selftest vs the Python reference (near_ref.bin):
  // exact side and fused correction, both kernels, three displacements.
  if (d.check && d.nearBeta > 0.0) {
    if (!sbimNearSelftest(d.data)) {
      std::fprintf(stderr, "[near selftest] FAILED\n");
      return 1;
    }
  }

  const bool wantTree = !d.brute && d.nspheres > 1;

  SolveResult tree, brute;
  if (d.check && d.nspheres > 1) {
    SBimContext *mb = nullptr, *mt = nullptr;
    brute = runSolve(d, centers, FT, /*useTreecode=*/false, &mb);
    tree = runSolve(d, centers, FT, /*useTreecode=*/true, &mt);

    // per-apply operator check on the SAME density rho (both ctxs still hold
    // rho from their solves; identical by construction).
    std::vector<double> offT(mt->n), offB(mb->n);
    sbimApplyOffdiag(mt, SBTreecode::KernelKind::Traction, mt->d_rho,
                     mt->d_off, nullptr);
    CUDA_CHECK(cudaMemcpy(offT.data(), mt->d_off, offT.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    sbimApplyOffdiag(mb, SBTreecode::KernelKind::Traction, mb->d_rho,
                     mb->d_off, nullptr);
    CUDA_CHECK(cudaMemcpy(offB.data(), mb->d_off, offB.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    std::printf("[sbim check] traction apply   relL2 = %.3e\n",
                relL2(offT, offB));
    sbimApplyOffdiag(mt, SBTreecode::KernelKind::Stokeslet, mt->d_rho,
                     mt->d_off, nullptr);
    CUDA_CHECK(cudaMemcpy(offT.data(), mt->d_off, offT.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    sbimApplyOffdiag(mb, SBTreecode::KernelKind::Stokeslet, mb->d_rho,
                     mb->d_off, nullptr);
    CUDA_CHECK(cudaMemcpy(offB.data(), mb->d_off, offB.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    std::printf("[sbim check] stokeslet apply  relL2 = %.3e\n",
                relL2(offT, offB));
    std::printf("[sbim check] surface velocity relL2 = %.3e\n",
                relL2(tree.u, brute.u));
    std::printf("[sbim check] U/Omega          relL2 = %.3e\n",
                relL2(tree.UOm, brute.UOm));
    sbimDestroy(mb);
    sbimDestroy(mt);
  } else {
    tree = runSolve(d, centers, FT, wantTree, nullptr);
  }

  const SolveResult &res = tree;

  // print the first few spheres' rigid velocities
  const int nPrint = std::min(d.nspheres, 4);
  for (int k = 0; k < nPrint; ++k) {
    const double *v = &res.UOm[(size_t)6 * k];
    std::printf("[sbim] sphere %d: U=(% .9e, % .9e, % .9e)  "
                "Omega=(% .9e, % .9e, % .9e)\n",
                k, v[0], v[1], v[2], v[3], v[4], v[5]);
  }

  // gate 1: single isolated sphere has the exact mobility U = F/(6*pi*eta*R),
  // Omega = T/(8*pi*eta*R^3).
  if (d.nspheres == 1) {
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
      worst = std::max(worst, std::abs(res.UOm[c] - FT[c] / (6.0 * M_PI)));
      worst = std::max(worst,
                       std::abs(res.UOm[3 + c] - FT[3 + c] / (8.0 * M_PI)));
    }
    std::printf("[sbim gate] single-sphere mobility max abs err = %.3e  %s\n",
                worst, worst < 1e-10 ? "PASS" : "FAIL");
  }

  PetscCall(PetscFinalize());
  return 0;
}
