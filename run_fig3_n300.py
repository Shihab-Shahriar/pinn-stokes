"""
Driver for the Figure 3 recompute at a genuine N=300.

Placed at the repo root so that `import benchmarks...` resolves to THIS repo's
`benchmarks/` package first (python puts the script's own directory on sys.path[0]).
Asserts guard against the known module-shadowing bug where an old byte-identical
copy of the codebase on PYTHONPATH silently wins.
"""
import benchmarks.accuracy_grand_M as A

assert "pinn-stokes" in A.__file__, f"WRONG MODULE: {A.__file__}"
assert A.NUM_PARTICLES_LIST == [300], A.NUM_PARTICLES_LIST
assert A.UNIFORM_CONFIG is True
print("module:", A.__file__, flush=True)

# skip_if_csv_exists=False forces a real recompute; the default True would just
# re-read the existing CSV and redraw it.
A.run_experiment_fixed_size_diff_operators(skip_if_csv_exists=False)
