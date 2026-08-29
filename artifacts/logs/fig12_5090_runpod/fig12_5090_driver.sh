#!/bin/bash
# Fig-12 RTX 5090 column: generate the missing large configs, then run
# benchmarks/figure12_grand_M.py one size per process (dynamo recompile_limit),
# each writing its own CSV (the script rewrites all rows of a backend in
# whatever --csv it gets).
cd /workspace/pinn-stokes || exit 1
mkdir -p results
exec >> results/driver.log 2>&1
echo "=== driver start $(date) ==="
python - <<'PY'
from benchmarks.cluster import uniform_cluster_generation_large as g
import os, time
for n in (1000000, 1250000, 1500000, 1750000, 2000000):
    p = f"tmp/uniform_large_0.1_{n}.csv"
    if os.path.exists(p):
        print(p, "exists, skip", flush=True)
        continue
    t = time.time()
    g(0.1, n, seed=0)
    print(f"generated {p} in {time.time()-t:.1f}s", flush=True)
PY
echo "=== configs done rc=$? $(date) ==="
for N in 50000 100000 200000 500000 750000 1000000 1250000 1500000 1750000 2000000; do
  echo "=== SIZE $N start $(date +%T) ==="
  NEMO_FAR_FP32_LEVEL=3 python benchmarks/figure12_grand_M.py --backend widebvh \
    --sizes $N --csv results/fig12_5090_$N.csv > results/run_$N.log 2>&1
  echo "=== SIZE $N rc=$? end $(date +%T) ==="
done
echo ALL_DONE
