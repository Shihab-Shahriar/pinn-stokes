// Interface between the pvfmm TU (mfs_ellipsoid.cpp, real MPI) and the PETSc
// TU (mfs_petsc.cpp, MPIUNI build). The two sets of headers cannot coexist in
// one translation unit (PETSc's mpiuni mpi.h #defines MPI_* macros that clash
// with the real <mpi.h>), so this header must include NEITHER.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// Resident set size in GB from /proc/self/status: key = "VmRSS:" (current)
// or "VmHWM:" (peak). Shared by both TUs so RAM shows up in their prints.
inline double mfs_rss_gb(const char *key = "VmRSS:")
{
  std::FILE *f = std::fopen("/proc/self/status", "r");
  if (!f) return 0.0;
  char line[256];
  long kb = 0;
  const size_t klen = std::strlen(key);
  while (std::fgets(line, sizeof line, f))
    if (!std::strncmp(line, key, klen)) { kb = std::atol(line + klen); break; }
  std::fclose(f);
  return (double)kb / (1024.0 * 1024.0);
}

// y = A x; x and y are caller-owned length-n host buffers.
using MFSMatVec = std::function<void(const double *x, double *y)>;

struct MFSSolveParams {
  long   n = 0;          // system size (3*M*P)
  double rtol = 1e-7;    // KSPSetTolerances rtol
  double atol = 0.0;     // KSPSetTolerances atol
  int    restart = 30;   // KSPGMRESSetRestart
  int    maxits = 500;   // KSPSetTolerances maxits
};

struct MFSSolveStats {
  int    iterations = 0;
  double final_rnorm = 0.0;
  std::string reason;                 // KSPConvergedReasonString
  double ksp_wall_sec = 0.0;          // KSPSolve wall time
  std::vector<double> iter_rnorm;     // monitor: rnorm at it = 0..its
  std::vector<double> iter_wall_sec;  // monitor: wall since KSPSolve start
};

void mfs_petsc_init(int *argc, char ***argv);  // PetscInitialize (+ -device_enable none)
void mfs_petsc_finalize();                     // PetscFinalize

// GMRES(restart) + PCNONE + zero initial guess via MatShell around `matvec`,
// mirroring widebvh mfs_broms.cuh:1029-1043. b is length n; the solution is
// written to x_out (length n). Returns 0 on success (PETSc error code else).
int mfs_petsc_gmres_solve(const MFSSolveParams &p, const MFSMatVec &matvec,
                          const double *b, double *x_out, MFSSolveStats &stats);
