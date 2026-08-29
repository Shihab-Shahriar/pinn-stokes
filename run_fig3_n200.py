"""
Driver for the Figure 3 operator comparison at N=200.

N=200 is the exact overlap point with Figure 4 (`run_exp_different_sizes`, which
sweeps N=20..200), so the n-body column here can be compared directly, at the same
N, against Fig-4's committed n-body curve -- no extrapolation argument needed.

Placed at the repo root so `import benchmarks...` resolves to THIS repo first.

NOTE: the driver writes data/grand_M_acc_uniform_fixed_N.csv (no N in the name), so
the N=300 results were backed up to data/grand_M_acc_uniform_fixed_N_P300.csv first
and are re-merged afterwards.
"""
import benchmarks.accuracy_grand_M as A

assert "pinn-stokes" in A.__file__, f"WRONG MODULE: {A.__file__}"
assert A.NUM_PARTICLES_LIST == [200], A.NUM_PARTICLES_LIST
assert A.UNIFORM_CONFIG is True
print("module:", A.__file__, flush=True)

A.run_experiment_fixed_size_diff_operators(skip_if_csv_exists=False)
