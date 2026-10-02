// SPDX-License-Identifier: Apache-2.0
//
// MFS mobility benchmark driver for suspensions of rigid UNIT SPHERES, on the
// SAME configurations as the second-kind BIM driver (sphere_mobility.cu):
// identical RSA center generation, identical random wrench, phi/seed defaults.
// Purpose: (1) find how many surface points per sphere the MFS needs to match
// the BIM error level (BIM: 162 pts/sphere, rigidity residual ~1.9e-3 at 10k),
// (2) benchmark the treecode-accelerated mobility solve (TC_PATH=skel) against
// the BIM numbers in results/sphere_bim_results.md.
//
// Clouds are generated in-driver: collocation = (p+1) x (2p+2) Gauss-Legendre
// x uniform grid on the unit sphere (same family as the BIM p=8 grid), sources
// = a coarser grid scaled to radius rsrc. --collocation/--source load ASCII
// clouds instead (e.g. the legacy mfs/points/b_sphere_* anchors).
//
// Error metrics:
//   * N=1 gate: exact single-sphere mobility U=F/(6 pi mu), Omega=T/(8 pi mu).
//   * N=2 gate: method-independent reference (data/sphere_p8/ref_n2.json,
//     exact VSH pair blocks): centers (0,0,0)/(2.2,0,0), F=(0,0,1) both.
//   * --check: brute and treecode solves on identical inputs, relL2 on U/Omega.
//   * --bc-check[=nsub]: fine-grid boundary-condition residual
//     max |u - (U + Omega x r)| on an off-collocation surface grid -- the
//     direct analog of the BIM driver's rigidityResidual. The flow is
//     evaluated as a Stokeslet sum of the final source strengths
//     lam = hat(gamma) - lam0: expanding S_L pinv = I on the solved gamma
//     gives S_self hat + K_M[U6] = gamma = u0 - S_off hat, i.e.
//     S[hat - lam0]|surface = U + Omega x r with the recovered U (and net
//     wrench K_N^T lam = +[F;T], the physically correct far field).
//
// Usage:
//   sphere_mfs [--nspheres=100] [--seed=1] [--phi=0.05] [--p=6] [--psrc=p-1]
//              [--rsrc=0.5] [--collocation=F --source=F] [--brute|--treecode]
//              [--check] [--tc-mac=0.6] [--ksp-rtol=1e-6]
//              [--forces=random[:amp]] [--dump-uom=F] [--ref-uom=F]
//              [--bc-check[=nsub]] [--bc-p=auto]
//              [--centers=F --wrench=F]  (raw little-endian fp64 files, P*3 and
//              P*6 [Fx Fy Fz Tx Ty Tz]; overrides --nspheres/--seed/--phi
//              placement and --forces, rotations stay identity)
// plus any PETSc options (passed through; -ksp_gmres_restart 500 injected
// unless given -- restart 30 stagnates at 10k in the BIM history).

#include "mfs_broms.cuh"

#include <chrono>
#include <random>
#include <unordered_map>

namespace {

struct DriverArgs {
  int nspheres = 100;
  unsigned long seed = 1;
  double phi = 0.05;
  int p = 6;         // collocation grid parameter, M = (p+1)(2p+2)
  int psrc = -1;     // source grid parameter; -1 = p-1
  int nb = 0;        // >0: Fibonacci-lattice collocation with nb points
  int ns = 0;        // Fibonacci source count; 0 = auto round(0.87*nb)
  double rsrc = 0.5; // source shell radius
  std::string bPath, sPath;  // optional file overrides
  bool brute = false;
  bool check = false;
  double kspRtol = 1e-6;
  bool randomForces = false;
  double famp = 1.0;
  float tcMac = 0.6f;
  double tcCell = 10.0;
  int tcMaxLeaf = 256;
  std::string dumpUom, refUom;
  std::string centersPath, wrenchPath;  // raw fp64 case input (P*3, P*6)
  int bcCheck = -1;  // -1 off; 0 = default nsub; >0 explicit nsub
  int bcP = 0;       // 0 = auto: max(8, p+2)
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
  if (const char *v = val("--p="))        { d.p = std::atoi(v); return true; }
  if (const char *v = val("--psrc="))     { d.psrc = std::atoi(v); return true; }
  if (const char *v = val("--nb="))       { d.nb = std::atoi(v); return true; }
  if (const char *v = val("--ns="))       { d.ns = std::atoi(v); return true; }
  if (const char *v = val("--rsrc="))     { d.rsrc = std::atof(v); return true; }
  if (const char *v = val("--collocation=")) { d.bPath = v; return true; }
  if (const char *v = val("--source="))   { d.sPath = v; return true; }
  if (const char *v = val("--tc-mac="))   { d.tcMac = (float)std::atof(v); return true; }
  if (const char *v = val("--tc-cell="))  { d.tcCell = std::atof(v); return true; }
  if (const char *v = val("--tc-maxleaf=")) { d.tcMaxLeaf = std::atoi(v); return true; }
  if (const char *v = val("--ksp-rtol=")) { d.kspRtol = std::atof(v); return true; }
  if (const char *v = val("--dump-uom=")) { d.dumpUom = v; return true; }
  if (const char *v = val("--ref-uom="))  { d.refUom = v; return true; }
  if (const char *v = val("--centers=")) { d.centersPath = v; return true; }
  if (const char *v = val("--wrench="))  { d.wrenchPath = v; return true; }
  if (const char *v = val("--bc-check=")) { d.bcCheck = std::atoi(v); return true; }
  if (const char *v = val("--bc-p="))     { d.bcP = std::atoi(v); return true; }
  if (a == "--bc-check") { d.bcCheck = 0; return true; }
  if (a == "--brute") { d.brute = true; return true; }
  if (a == "--treecode") { d.brute = false; return true; }
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

// Raw little-endian fp64 file of `perRow`-column rows; row count out in `rows`.
std::vector<double> loadRawF64(const std::string &path, int perRow, int &rows)
{
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (bytes <= 0 || bytes % (8L * perRow) != 0) {
    std::fprintf(stderr, "%s: %ld bytes is not a multiple of %d fp64 columns\n",
                 path.c_str(), bytes, perRow);
    std::exit(2);
  }
  std::vector<double> v((size_t)bytes / 8);
  if (std::fread(v.data(), 8, v.size(), f) != v.size()) {
    std::fprintf(stderr, "short read on %s\n", path.c_str());
    std::exit(2);
  }
  std::fclose(f);
  rows = (int)(v.size() / (size_t)perRow);
  return v;
}

// Random sequential insertion of N unit spheres, min center distance 2.2
// (surface separation >= 0.2), in a cube sized for volume fraction phi.
// Copied VERBATIM from sphere_mobility.cu: same (N, seed, phi) => bit-identical
// configuration to the BIM benchmark.
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

// Copied VERBATIM from sphere_mobility.cu (rng stream seed+12345).
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

// (p+1) Gauss-Legendre colatitudes x (2p+2) uniform longitudes on the unit
// sphere; theta ascending from the north pole, colatitude-major -- the same
// grid family as scripts/gen_sphere_selfblocks.py::make_grid (the BIM p=8
// grid). Returns AoS xyz, (p+1)*(2p+2) points.
std::vector<double> makeSphereGrid(int p)
{
  const int nt = p + 1, nphi = 2 * p + 2;
  std::vector<double> x(nt);
  for (int i = 0; i < nt; ++i) {  // GL nodes: Newton on P_nt
    double t = std::cos(M_PI * (i + 0.75) / (nt + 0.5));
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (int j = 0; j < nt; ++j) {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2.0 * j + 1.0) * t * p1 - j * p2) / (j + 1.0);
      }
      const double dp = nt * (t * p0 - p1) / (t * t - 1.0);
      const double dt = p0 / dp;
      t -= dt;
      if (std::fabs(dt) < 1e-15) break;
    }
    x[i] = t;
  }
  std::sort(x.begin(), x.end(), std::greater<double>());  // cos(theta) desc
  std::vector<double> pts;
  pts.reserve((size_t)3 * nt * nphi);
  for (int k = 0; k < nt; ++k) {
    const double ct = x[k], st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
    for (int l = 0; l < nphi; ++l) {
      const double ph = 2.0 * M_PI * l / nphi;
      pts.push_back(st * std::cos(ph));
      pts.push_back(st * std::sin(ph));
      pts.push_back(ct);
    }
  }
  return pts;
}

// Fibonacci (golden-spiral) lattice: n quasi-uniform points on the unit
// sphere. Unlike the GL x uniform grid, this has no polar clustering, which
// matters for the MFS max-norm BC residual (the GL family stalls ~1e-2 at
// close pairs; quasi-uniform clouds like the legacy mfs/points/b_sphere_*
// behave much better per point).
std::vector<double> makeFibonacciGrid(int n)
{
  std::vector<double> pts;
  pts.reserve((size_t)3 * n);
  const double ga = M_PI * (3.0 - std::sqrt(5.0));  // golden angle
  for (int i = 0; i < n; ++i) {
    const double z = 1.0 - (2.0 * i + 1.0) / n;
    const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double ph = ga * i;
    pts.push_back(r * std::cos(ph));
    pts.push_back(r * std::sin(ph));
    pts.push_back(z);
  }
  return pts;
}

// Skel-path env defaults (setenv no-override, like sbimSetup), with the
// TC_TGT_SKEL <= M clamp: the engine's own check throws only inside apply().
void setupTreecodeEnv(int M)
{
  setenv("TC_PATH", "skel", 0);
  setenv("TC_BUCKETIZER", "object", 0);
  setenv("TC_SKEL_P2P", "block", 0);
  char buf[16];
  std::snprintf(buf, sizeof buf, "%d", std::min(96, M));
  setenv("TC_TGT_SKEL", buf, 0);
  const char *path = std::getenv("TC_PATH");
  if (path && std::strcmp(path, "skel") == 0) {
    const char *ns = std::getenv("TC_TGT_SKEL");
    if (ns && std::atoi(ns) > M) {
      std::fprintf(stderr,
                   "TC_TGT_SKEL=%s exceeds collocation points per sphere M=%d; "
                   "set TC_TGT_SKEL <= %d\n", ns, M, M);
      std::exit(2);
    }
  }
  std::printf("[sphere_mfs] env: TC_PATH=%s TC_BUCKETIZER=%s TC_SKEL_P2P=%s "
              "TC_TGT_SKEL=%s\n",
              std::getenv("TC_PATH"), std::getenv("TC_BUCKETIZER"),
              std::getenv("TC_SKEL_P2P"), std::getenv("TC_TGT_SKEL"));
}

// Fine-grid boundary-condition residual max |u - (U + Omega x r)| on an
// off-collocation check grid (first nsub spheres). The MFS flow is the
// Stokeslet sum of the final strengths lam = hat(gamma) - lam0 (see header
// comment); one brute pairwiseStokeslet launch against ALL P*N sources
// (skipSelf=0). Must run before mfsDestroy (uses xsol / d_lam0 / d_Yglob).
double bcResidual(MFSContext *m, const std::vector<double> &centers,
                  const std::vector<double> &UOm, int nsub, int bcP)
{
  const int P = m->P;
  const std::vector<double> g = makeSphereGrid(bcP);
  const int B = (int)(g.size() / 3);
  const int ns = std::min(nsub, P);
  const int nT = ns * B;
  std::vector<double> hX((size_t)3 * nT);
  for (int k = 0; k < ns; ++k)
    for (int b = 0; b < B; ++b)
      for (int c = 0; c < 3; ++c)
        hX[3 * ((size_t)k * B + b) + c] = g[(size_t)3 * b + c] + centers[(size_t)3 * k + c];

  // lam_total = hat(gamma_solution) - lam0
  {
    const PetscScalar *dg = nullptr;
    PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayRead(m->xsol, &dg));
    denseApplyBatched(m, dg, m->d_hatGlob);
    PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(m->xsol, &dg));
  }
  double *d_X = nullptr, *d_u = nullptr, *d_lam = nullptr;
  CUDA_CHECK(cudaMalloc(&d_X, (size_t)3 * nT * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u, (size_t)3 * nT * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_lam, (size_t)P * m->tN * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_X, hX.data(), (size_t)3 * nT * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_u, 0, (size_t)3 * nT * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_lam, m->d_hatGlob, (size_t)P * m->tN * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  const double negone = -1.0;
  CUBLAS_CHECK(cublasDaxpy(m->cublas, P * m->tN, &negone, m->d_lam0, 1, d_lam, 1));

  pairwiseStokeslet<<<grid1d(nT, 128), 128>>>(d_X, m->d_Yglob, d_lam, d_u, nT,
                                              P * m->N, B, m->N, m->pref,
                                              /*skipSelf=*/0);
  CUDA_CHECK(cudaGetLastError());
  std::vector<double> hu((size_t)3 * nT);
  CUDA_CHECK(cudaMemcpy(hu.data(), d_u, (size_t)3 * nT * sizeof(double),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_X); cudaFree(d_u); cudaFree(d_lam);

  double worst = 0.0;
  for (int k = 0; k < ns; ++k) {
    const double *U = &UOm[(size_t)6 * k];
    const double *Om = U + 3;
    for (int b = 0; b < B; ++b) {
      const double *x = &g[(size_t)3 * b];  // = point - center (unit sphere)
      const double ur[3] = {U[0] + Om[1] * x[2] - Om[2] * x[1],
                            U[1] + Om[2] * x[0] - Om[0] * x[2],
                            U[2] + Om[0] * x[1] - Om[1] * x[0]};
      for (int c = 0; c < 3; ++c)
        worst = std::max(worst,
                         std::abs(hu[3 * ((size_t)k * B + b) + c] - ur[c]));
    }
  }
  return worst;
}

struct SolveOut {
  std::vector<double> UOm;  // P*6 [vx vy vz wx wy wz]
  double applyMs = 0.0, kspMs = 0.0, matvecAvgMs = 0.0;
  int its = 0;
  double bcResid = -1.0;
};

SolveOut runSolve(const DriverArgs &d, cublasHandle_t cublas,
                  cusolverDnHandle_t solver, const std::vector<double> &Xref,
                  const std::vector<double> &Yref,
                  const std::vector<double> &centers,
                  const std::vector<double> &F, const std::vector<double> &T,
                  bool useTreecode)
{
  const int P = d.nspheres;
  const int M = (int)(Xref.size() / 3), N = (int)(Yref.size() / 3);
  MFSContext *m = mfsSetup(cublas, solver, P, M, N, Xref.data(), Yref.data(),
                           /*mu=*/1.0);
  if (useTreecode) {
    MFSTreecode<>::Config tcCfg;
    tcCfg.mac = d.tcMac;
    tcCfg.cellEdge = d.tcCell;
    tcCfg.maxLeaf = d.tcMaxLeaf;
    mfsSetTreecode(m, tcCfg, d.kspRtol);
  } else {
    PetscReal atol = 0.0, dtol = 0.0;
    PetscInt maxIts = 0;
    PetscCallAbort(PETSC_COMM_SELF,
                   KSPGetTolerances(m->ksp, nullptr, &atol, &dtol, &maxIts));
    PetscCallAbort(PETSC_COMM_SELF,
                   KSPSetTolerances(m->ksp, d.kspRtol, atol, dtol, maxIts));
  }

  std::vector<double> R((size_t)P * 9, 0.0);  // spheres: identity orientations
  for (int k = 0; k < P; ++k)
    R[(size_t)k * 9] = R[(size_t)k * 9 + 4] = R[(size_t)k * 9 + 8] = 1.0;

  SolveOut r;
  mfsSolve(m, centers.data(), R.data(), F.data(), T.data(), r.UOm);
  r.applyMs = m->stepTiming.totalWallMs;
  r.kspMs = m->stepTiming.kspWallMs;
  r.matvecAvgMs =
      m->matvecCount ? m->matvecTotalTiming.wallMs / m->matvecCount : 0.0;
  PetscInt its = 0;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(m->ksp, &its));
  r.its = (int)its;

  if (d.bcCheck >= 0) {
    // default check grid: dense enough for every M in the sweep; FIXED across
    // configs so max-norm residuals are comparable.
    const int bcP = d.bcP > 0 ? d.bcP : 16;  // 17x34 = 578 check pts/sphere
    const int nsubDefault = P <= 1000 ? P : 1000;
    const int nsub = d.bcCheck > 0 ? d.bcCheck : nsubDefault;
    r.bcResid = bcResidual(m, centers, r.UOm, nsub, bcP);
    std::printf("[sphere_mfs %s] BC residual max|u - (U + Omega x r)| = %.3e "
                "(nsub=%d bc_p=%d)\n",
                useTreecode ? "treecode" : "brute", r.bcResid,
                std::min(nsub, P), bcP);
  }
  mfsDestroy(m);

  const char *tag = useTreecode ? "treecode" : "brute";
  std::printf("[sphere_mfs %s] P=%d M=%d N=%d | apply=%.3f s ksp=%.3f s "
              "matvec_avg=%.3f ms its=%d | bc=%.3e\n",
              tag, P, M, N, r.applyMs / 1e3, r.kspMs / 1e3, r.matvecAvgMs,
              r.its, r.bcResid);
  std::fflush(stdout);
  return r;
}

}  // namespace

int main(int argc, char **argv)
{
  DriverArgs d;
  std::vector<char *> petscArgv{argv[0]};
  bool userRestart = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (parseArg(a, d)) continue;
    if (a.find("ksp_gmres_restart") != std::string::npos) userRestart = true;
    petscArgv.push_back(argv[i]);
  }
  // restart 30 (the mfs_broms default) stagnates at 10k spheres in the BIM
  // history; inject an effectively-full GMRES unless the user chose one.
  static char restartOpt[] = "-ksp_gmres_restart";
  static char restartVal[] = "500";
  if (!userRestart) {
    petscArgv.push_back(restartOpt);
    petscArgv.push_back(restartVal);
  }
  int pArgc = (int)petscArgv.size();
  char **pArgv = petscArgv.data();
  PetscCall(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));

  if (d.psrc < 0) d.psrc = std::max(1, d.p - 1);

  // ---- clouds ----
  std::vector<double> Xref, Yref;
  if (!d.bPath.empty() || !d.sPath.empty()) {
    if (d.bPath.empty() || d.sPath.empty()) {
      std::fprintf(stderr, "--collocation and --source must be given together\n");
      return 2;
    }
    int M = 0, N = 0;
    Xref = loadPointsASCII(d.bPath, M);
    Yref = loadPointsASCII(d.sPath, N);
  } else if (d.nb > 0) {
    if (d.ns <= 0) d.ns = (int)std::lround(0.87 * d.nb);
    Xref = makeFibonacciGrid(d.nb);
    Yref = makeFibonacciGrid(d.ns);
    for (double &v : Yref) v *= d.rsrc;
  } else {
    Xref = makeSphereGrid(d.p);
    Yref = makeSphereGrid(d.psrc);
    for (double &v : Yref) v *= d.rsrc;
  }
  const int M = (int)(Xref.size() / 3), N = (int)(Yref.size() / 3);

  setupTreecodeEnv(M);

  // ---- configuration (identical to sphere_mobility for same N/seed/phi) ----
  double L = 0.0;
  std::vector<double> centers;
  std::vector<double> FT;
  const bool fileCase = !d.centersPath.empty() || !d.wrenchPath.empty();
  if (fileCase) {
    if (d.centersPath.empty() || d.wrenchPath.empty()) {
      std::fprintf(stderr, "--centers and --wrench must be given together\n");
      return 2;
    }
    int Pc = 0, Pw = 0;
    centers = loadRawF64(d.centersPath, 3, Pc);
    FT = loadRawF64(d.wrenchPath, 6, Pw);
    if (Pc != Pw || Pc < 1) {
      std::fprintf(stderr, "--centers has %d rows but --wrench has %d\n", Pc, Pw);
      return 2;
    }
    d.nspheres = Pc;
  } else {
    centers = makeCenters(d.nspheres, d.seed, d.phi, L);
    FT = makeWrench(d.nspheres, d.randomForces, d.famp, d.seed);
  }
  const bool n2Gate = (d.nspheres == 2 && !d.randomForces && !fileCase);
  if (n2Gate) {
    // match the method-independent reference (ref_n2.json): F=(0,0,+1) both.
    FT.assign(12, 0.0);
    FT[2] = FT[8] = 1.0;
  }
  std::vector<double> F((size_t)3 * d.nspheres), T((size_t)3 * d.nspheres);
  for (int k = 0; k < d.nspheres; ++k)
    for (int c = 0; c < 3; ++c) {
      F[(size_t)3 * k + c] = FT[(size_t)6 * k + c];
      T[(size_t)3 * k + c] = FT[(size_t)6 * k + 3 + c];
    }

  const bool wantTree = !d.brute && d.nspheres > 1;
  std::printf("[sphere_mfs] P=%d spheres, M=%d colloc (p=%d), N=%d src "
              "(psrc=%d rsrc=%.3f)%s, n=%lld unknowns, phi=%.3f box_L=%.2f "
              "seed=%lu rtol=%.1e mode=%s%s\n",
              d.nspheres, M, d.p, N, d.psrc, d.rsrc,
              d.bPath.empty() ? "" : " [file clouds]", 3LL * M * d.nspheres,
              d.phi, L, d.seed, d.kspRtol,
              d.check ? "check(brute+treecode)"
                      : (wantTree ? "treecode" : "brute"),
              fileCase ? " case=file"
                       : (d.randomForces
                              ? " forces=random"
                              : (n2Gate ? " forces=F(0,0,+1)" : " forces=F(0,0,-1)")));
  if (wantTree || d.check)
    std::printf("[sphere_mfs] tc: %s mac=%.3f\n", MFSTreecode<>::multipoleName(),
                (double)d.tcMac);

  cublasHandle_t cublas;
  cusolverDnHandle_t solver;
  CUBLAS_CHECK(cublasCreate(&cublas));
  CUSOLVER_CHECK(cusolverDnCreate(&solver));

  SolveOut res;
  if (d.check && d.nspheres > 1) {
    SolveOut brute =
        runSolve(d, cublas, solver, Xref, Yref, centers, F, T, false);
    res = runSolve(d, cublas, solver, Xref, Yref, centers, F, T, true);
    std::printf("[sphere_mfs check] U/Omega relL2 (treecode vs brute) = %.3e\n",
                relL2(res.UOm, brute.UOm));
  } else {
    res = runSolve(d, cublas, solver, Xref, Yref, centers, F, T, wantTree);
  }

  const int nPrint = std::min(d.nspheres, 4);
  for (int k = 0; k < nPrint; ++k) {
    const double *v = &res.UOm[(size_t)6 * k];
    std::printf("[sphere_mfs] sphere %d: U=(% .9e, % .9e, % .9e)  "
                "Omega=(% .9e, % .9e, % .9e)\n",
                k, v[0], v[1], v[2], v[3], v[4], v[5]);
  }

  // gate 1: exact single-sphere mobility.
  if (d.nspheres == 1) {
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
      worst = std::max(worst, std::abs(res.UOm[c] - FT[c] / (6.0 * M_PI)));
      worst = std::max(worst,
                       std::abs(res.UOm[3 + c] - FT[3 + c] / (8.0 * M_PI)));
    }
    std::printf("[sphere_mfs gate] single-sphere mobility max abs err = %.3e\n",
                worst);
  }

  // gate 2: N=2 vs the exact-VSH reference (data/sphere_p8/ref_n2.json).
  if (n2Gate) {
    const double ref[12] = {0, 0, 0.0729127353202748,
                            0, -0.0071566813191359015, 0,
                            0, 0, 0.0729127353202748,
                            0, 0.0071566813191359015, 0};
    double worst = 0.0;
    for (int i = 0; i < 12; ++i)
      worst = std::max(worst, std::abs(res.UOm[i] - ref[i]));
    std::printf("[sphere_mfs gate] two-sphere vs ref_n2 max abs err = %.3e\n",
                worst);
  }

  if (!d.dumpUom.empty()) {
    std::FILE *f = std::fopen(d.dumpUom.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", d.dumpUom.c_str()); return 1; }
    std::fwrite(res.UOm.data(), sizeof(double), res.UOm.size(), f);
    std::fclose(f);
    std::printf("[sphere_mfs] wrote U/Omega (%zu doubles) to %s\n",
                res.UOm.size(), d.dumpUom.c_str());
  }
  if (!d.refUom.empty()) {
    std::FILE *f = std::fopen(d.refUom.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", d.refUom.c_str()); return 1; }
    std::vector<double> ref(res.UOm.size());
    const size_t got = std::fread(ref.data(), sizeof(double), ref.size(), f);
    std::fclose(f);
    if (got != ref.size()) {
      std::fprintf(stderr, "%s: expected %zu doubles, got %zu\n",
                   d.refUom.c_str(), ref.size(), got);
      return 1;
    }
    std::vector<double> aV, aW, rV, rW;
    aV.reserve(res.UOm.size() / 2); rV.reserve(res.UOm.size() / 2);
    for (int k = 0; k < d.nspheres; ++k)
      for (int c = 0; c < 3; ++c) {
        aV.push_back(res.UOm[(size_t)6 * k + c]);
        rV.push_back(ref[(size_t)6 * k + c]);
        aW.push_back(res.UOm[(size_t)6 * k + 3 + c]);
        rW.push_back(ref[(size_t)6 * k + 3 + c]);
      }
    std::printf("[sphere_mfs] U/Omega vs %s: relL2(U)=%.3e relL2(Omega)=%.3e "
                "relL2(all)=%.3e\n",
                d.refUom.c_str(), relL2(aV, rV), relL2(aW, rW),
                relL2(res.UOm, ref));
  }

  cublasDestroy(cublas);
  cusolverDnDestroy(solver);
  if (mfsSkipPetscTeardownForProfiler()) {
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(0);
  }
  PetscCall(PetscFinalize());
  return 0;
}
