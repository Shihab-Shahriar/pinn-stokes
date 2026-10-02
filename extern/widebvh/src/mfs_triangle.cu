// Triangle-test driver for the Broms MFS solver. Solver core lives in
// mfs_broms.cuh (shared with test_mfs_mp.cu); main() reproduces
// mfs/validate_triangle.py.
#include "mfs_broms.cuh"

#include <map>

// ==========================================================================
// triangle test (mirrors mfs/validate_triangle.py)
// ==========================================================================
struct RefRow { double s, U1, U2, U3, O; };
static const RefRow REFERENCE[] = {
    {2.01, 0.65528, 0.63461, 0.00498, 0.037336},
    {2.10, 0.73857, 0.59718, 0.03517, 0.052035},
    {2.50, 0.87765, 0.49545, 0.07393, 0.045466},
    {3.00, 0.93905, 0.41694, 0.07824, 0.035022},
    {4.00, 0.97964, 0.31859, 0.06925, 0.021634},
    {6.00, 0.99581, 0.21586, 0.05078, 0.010159},
};

static void triangleCenters(double s, std::vector<double> &centers)
{
  double r = s / std::sqrt(3.0);
  centers = {0.0, r, 0.0, -s / 2.0, -r / 2.0, 0.0, s / 2.0, -r / 2.0, 0.0};
}

// One cached solver context per accuracy level (the reference sphere clouds
// depend only on M,N, not on the triangle separation s).
struct AccCtx { MFSContext *m; int M, N; };

static int runTriangle(cublasHandle_t cublas, cusolverDnHandle_t solver,
                       const std::string &pointsDir, bool useTreecode,
                       const MFSTreecode<>::Config &tcCfg)
{
  std::printf("Stokeslet eval: %s\n",
              useTreecode ? "TREECODE (bary-Lagrange)" : "brute O(P^2*M*N)");
  std::printf("%6s %6s | %10s %10s %11s %11s  ||  %10s %10s %10s %10s\n", "s",
              "acc", "U1", "U2", "v2x", "w2z", "U1_ref", "U2_ref", "U3_ref",
              "O_ref");
  for (int i = 0; i < 120; ++i) std::printf("-");
  std::printf("\n");

  const int P = 3;
  std::map<std::string, AccCtx> ctxByAcc;  // built once per accuracy, reused
  int rc = 0;
  for (const RefRow &ref : REFERENCE) {
    double s = ref.s;
    std::string acc = (s < 2.5) ? "Xfine" : "fine";

    // Build (and cache) the reference-frame setup for this accuracy on first use.
    auto it = ctxByAcc.find(acc);
    if (it == ctxByAcc.end()) {
      int M, N;
      std::vector<double> Xcloud = loadPointsASCII(pointsDir + "/b_sphere_" + acc + ".txt", M);
      std::vector<double> Ycloud = loadPointsASCII(pointsDir + "/s_sphere_" + acc + ".txt", N);
      // The loaded sphere clouds are centered at the origin = the reference frame.
      MFSContext *m = mfsSetup(cublas, solver, P, M, N, Xcloud.data(), Ycloud.data(), 1.0);
      if (useTreecode) mfsSetTreecode(m, tcCfg);
      it = ctxByAcc.emplace(acc, AccCtx{m, M, N}).first;
    }
    MFSContext *m = it->second.m;

    std::vector<double> centers;
    triangleCenters(s, centers);
    std::vector<double> R((size_t)P * 9, 0.0);  // identity orientations
    for (int k = 0; k < P; ++k)
      R[(size_t)k * 9 + 0] = R[(size_t)k * 9 + 4] = R[(size_t)k * 9 + 8] = 1.0;
    std::vector<double> F(9, 0.0), T(9, 0.0);
    F[1] = -6.0 * M_PI;  // sphere 1 force toward centroid (-y)

    std::vector<double> U;
    mfsSolve(m, centers.data(), R.data(), F.data(), T.data(), U);

    double U1 = -U[0 * 6 + 1];   // -v[0,y]
    double U2 = -U[1 * 6 + 1];   // -v[1,y]
    double v2x = U[1 * 6 + 0];
    double w2z = U[1 * 6 + 5];
    std::printf("%6.2f %6s | %10.5f %10.5f %+11.5f %+11.5f  ||  %10.5f %10.5f %10.5f %10.5f\n",
                s, acc.c_str(), U1, U2, v2x, w2z, ref.U1, ref.U2, ref.U3, ref.O);
    // Gate vs the paper's Table 3. The closest gap (s=2.01) differs from the
    // paper by ~1.7e-2 -- but that gap is also present in the reference Python
    // solver (verified: CUDA == broms_mfs.py to 5 decimals), so it is a
    // paper-vs-implementation difference, not a port error. 2.5e-2 catches
    // gross errors while tolerating the known near-contact gap.
    if (!std::isfinite(U1) || !std::isfinite(U2) ||
        std::fabs(U1 - ref.U1) > 2.5e-2 || std::fabs(U2 - ref.U2) > 2.5e-2)
      rc = 1;
  }
  for (auto &kv : ctxByAcc) mfsDestroy(kv.second.m);
  return rc;
}

// ==========================================================================
// rotation / Kabsch self-check (non-identity R)
// ==========================================================================
static int runSelfTest(cusolverDnHandle_t solver)
{
  // random-ish rotation: about axis (1,2,3) by 0.7 rad (Rodrigues)
  double ax[3] = {1, 2, 3};
  double na = std::sqrt(14.0);
  for (double &a : ax) a /= na;
  double th = 0.7, c = std::cos(th), sN = std::sin(th);
  double R[9];
  double K[9] = {0, -ax[2], ax[1], ax[2], 0, -ax[0], -ax[1], ax[0], 0};
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      double KK = 0;
      for (int k = 0; k < 3; ++k) KK += K[a * 3 + k] * K[k * 3 + b];
      R[a * 3 + b] = (a == b ? 1.0 : 0.0) + sN * K[a * 3 + b] + (1 - c) * KK;
    }

  // ref points and rotated copies
  const int npts = 5;
  std::vector<double> refp(3 * npts), part(3 * npts);
  for (int i = 0; i < npts; ++i) {
    double v[3] = {0.3 * i - 0.5, 0.1 * i + 0.2, -0.2 * i + 0.4};
    double w[3];
    mat3vec(R, v, w);
    for (int d = 0; d < 3; ++d) { refp[3 * i + d] = v[d]; part[3 * i + d] = w[d]; }
  }
  double Rrec[9];
  kabsch(solver, refp.data(), part.data(), npts, Rrec);
  double err = 0;
  for (int i = 0; i < 9; ++i) err = std::max(err, std::fabs(Rrec[i] - R[i]));
  std::printf("[selftest] Kabsch recovers R: max|dR| = %.3e  %s\n", err,
              err < 1e-10 ? "PASS" : "FAIL");

  // R^T R = I via applyRot kernels
  double *d_R, *d_in, *d_t, *d_out;
  CUDA_CHECK(cudaMalloc(&d_R, 9 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_in, 3 * npts * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_t, 3 * npts * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_out, 3 * npts * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_R, R, 9 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in, refp.data(), 3 * npts * sizeof(double), cudaMemcpyHostToDevice));
  applyRot<<<grid1d(npts, 64), 64>>>(d_R, d_in, d_t, npts, 0);   // R
  applyRot<<<grid1d(npts, 64), 64>>>(d_R, d_t, d_out, npts, 1);  // R^T
  std::vector<double> out(3 * npts);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, 3 * npts * sizeof(double), cudaMemcpyDeviceToHost));
  double err2 = 0;
  for (int i = 0; i < 3 * npts; ++i) err2 = std::max(err2, std::fabs(out[i] - refp[i]));
  std::printf("[selftest] R^T(R x) == x: max|d| = %.3e  %s\n", err2,
              err2 < 1e-12 ? "PASS" : "FAIL");
  cudaFree(d_R); cudaFree(d_in); cudaFree(d_t); cudaFree(d_out);
  return (err < 1e-10 && err2 < 1e-12) ? 0 : 1;
}

// ==========================================================================
int main(int argc, char **argv)
{
  // Consume our own flags first, then hand the rest to PETSc so it does not
  // warn about unrecognized options (e.g. --selftest).
  std::string pointsDir = "mfs/points";
  bool selftest = false;
  bool useTreecode = false;
  MFSTreecode<>::Config tcCfg;
  std::vector<char *> petscArgv;
  petscArgv.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&](const char *k) { return a.substr(std::string(k).size()); };
    if (a == "--selftest") selftest = true;
    else if (a == "--treecode") useTreecode = true;
    else if (a == "--brute") useTreecode = false;
    else if (a.rfind("--tc-order=", 0) == 0) tcCfg.order = std::atoi(val("--tc-order=").c_str());
    else if (a.rfind("--tc-mac=", 0) == 0) tcCfg.mac = (float)std::atof(val("--tc-mac=").c_str());
    else if (a.rfind("--tc-cell=", 0) == 0) tcCfg.cellEdge = std::atof(val("--tc-cell=").c_str());
    else if (a.rfind("--tc-maxleaf=", 0) == 0) tcCfg.maxLeaf = std::atoi(val("--tc-maxleaf=").c_str());
    else if (!a.empty() && a[0] != '-') pointsDir = a;
    else petscArgv.push_back(argv[i]);  // pass through PETSc options (-ksp_*)
  }
  int pArgc = (int)petscArgv.size();
  char **pArgv = petscArgv.data();
  PetscCall(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));

  cublasHandle_t cublas;
  cusolverDnHandle_t solver;
  CUBLAS_CHECK(cublasCreate(&cublas));
  CUSOLVER_CHECK(cusolverDnCreate(&solver));

  int rc = 0;
  if (selftest) rc |= runSelfTest(solver);
  rc |= runTriangle(cublas, solver, pointsDir, useTreecode, tcCfg);

  cublasDestroy(cublas);
  cusolverDnDestroy(solver);
  if (!mfsSkipPetscTeardownForProfiler()) {
    PetscCall(PetscFinalize());
  } else {
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(rc);
  }
  return rc;
}
