// PETSc-only translation unit for the MFS mobility solver. Compiled WITHOUT
// any real-MPI include path: this PETSc build is MPIUNI (--with-mpi=0), whose
// petsc/mpiuni/mpi.h macros clash with OpenMPI's <mpi.h>, so all PETSc usage
// is quarantined here behind the header-free interface in mfs_petsc.hpp.
// KSP configuration mirrors widebvh mfs_broms.cuh:1029-1043 exactly
// (KSPGMRES, restart, PCNONE, rtol/atol/maxits, KSPSetFromOptions, custom
// monitor, zero initial guess, default norm type).
#include <petscksp.h>

#include <cstring>

#include "mfs_petsc.hpp"

namespace {

struct ShellCtx {
  const MFSMatVec *mv;
  MFSSolveStats *stats;
  PetscLogDouble t0;
};

PetscErrorCode MatMult_Shell(Mat A, Vec x, Vec y)
{
  ShellCtx *c;
  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &c));
  const PetscScalar *xa;
  PetscScalar *ya;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(y, &ya));
  (*c->mv)(xa, ya);  // all math (incl. y = x + ...) lives in the pvfmm TU
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(VecRestoreArray(y, &ya));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Monitor(KSP, PetscInt it, PetscReal rnorm, void *ctx)
{
  ShellCtx *c = static_cast<ShellCtx *>(ctx);
  PetscFunctionBeginUser;
  PetscLogDouble now = 0.0;
  PetscCall(PetscTime(&now));
  const double el = (double)(now - c->t0);
  const double dt =
      el - (c->stats->iter_wall_sec.empty() ? 0.0 : c->stats->iter_wall_sec.back());
  c->stats->iter_rnorm.push_back((double)rnorm);
  c->stats->iter_wall_sec.push_back(el);
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
                        "[MFS GMRES] it=%" PetscInt_FMT " rnorm=%.6e iter=%.3f s "
                        "total solve=%.3f s rss=%.1f GB\n",
                        it, (double)rnorm, dt, el, mfs_rss_gb()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solve_impl(const MFSSolveParams &p, const MFSMatVec &matvec,
                          const double *b, double *x_out, MFSSolveStats &stats)
{
  const PetscInt n = (PetscInt)p.n;
  ShellCtx ctx{&matvec, &stats, 0.0};
  Mat A;
  Vec vb, vx;
  KSP ksp;
  PC pc;
  PetscFunctionBeginUser;

  PetscCall(MatCreateShell(PETSC_COMM_SELF, n, n, n, n, &ctx, &A));
  PetscCall(MatShellSetOperation(A, MATOP_MULT, (void (*)(void))MatMult_Shell));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &vb));
  PetscCall(VecDuplicate(vb, &vx));
  {
    PetscScalar *a;
    PetscCall(VecGetArray(vb, &a));
    std::memcpy(a, b, (size_t)n * sizeof(double));
    PetscCall(VecRestoreArray(vb, &a));
  }

  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPGMRESSetRestart(ksp, p.restart));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetTolerances(ksp, p.rtol, p.atol, PETSC_DEFAULT, p.maxits));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PetscTime(&ctx.t0));
  PetscCall(KSPMonitorSet(ksp, Monitor, &ctx, nullptr));

  PetscLogDouble t1 = 0.0;
  PetscCall(KSPSolve(ksp, vb, vx));
  PetscCall(PetscTime(&t1));
  stats.ksp_wall_sec = (double)(t1 - ctx.t0);

  PetscInt its = 0;
  PetscReal rnorm = 0.0;
  const char *reason = nullptr;
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPGetResidualNorm(ksp, &rnorm));
  PetscCall(KSPGetConvergedReasonString(ksp, &reason));
  stats.iterations = (int)its;
  stats.final_rnorm = (double)rnorm;
  stats.reason = reason ? reason : "unknown";

  {
    const PetscScalar *a;
    PetscCall(VecGetArrayRead(vx, &a));
    std::memcpy(x_out, a, (size_t)n * sizeof(double));
    PetscCall(VecRestoreArrayRead(vx, &a));
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&vb));
  PetscCall(VecDestroy(&vx));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

void mfs_petsc_init(int *argc, char ***argv)
{
  // The GPU-less run node satisfies libpetsc's CUDA DT_NEEDEDs via driver
  // stubs; make sure PETSc never initializes a device.
  PetscCallAbort(PETSC_COMM_SELF, PetscOptionsSetValue(nullptr, "-device_enable", "none"));
  PetscCallAbort(PETSC_COMM_SELF, PetscInitialize(argc, argv, nullptr, nullptr));
}

void mfs_petsc_finalize()
{
  PetscCallAbort(PETSC_COMM_SELF, PetscFinalize());
}

int mfs_petsc_gmres_solve(const MFSSolveParams &p, const MFSMatVec &matvec,
                          const double *b, double *x_out, MFSSolveStats &stats)
{
  return (int)solve_impl(p, matvec, b, x_out, stats);
}
