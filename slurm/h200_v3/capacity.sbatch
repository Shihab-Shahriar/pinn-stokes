#!/bin/bash
# H200 timing, current NeMO (v3 moments pair + pc8c diagonal, fp16 MLP, switch 8),
# widebvh far field at fp32 level 2. Max-particle capacity: bisection over the lattice side n (N = n^3, phi = 0.1), one process per attempt, 'repeatable' = cold + 3 warm applies with empty_cache (artifacts/max_particles_h200_f32l2_report.md protocol). Default pair budget.
#   sbatch slurm/h200_v3/capacity.sbatch      (from the repo root; see slurm/h200_v3/README.md)
#SBATCH --job-name=h200v3_capacity
#SBATCH --account=hmakmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=amd24
#SBATCH --time=2:59:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
source slurm/h200_v3/env.sh
# keep the slurm .out with the rest of the raw data, whatever happens
trap 'cp -f "slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$RUN_DIR/" 2>/dev/null || true' EXIT
write_manifest
test -f "$WIDEBVH_BUILD_DIR/libwidebvh_nemo_f32l2.so"
echo "host $(hostname) job ${SLURM_JOB_ID} sha $(git rev-parse --short HEAD) raw data -> $RUN_DIR"
OUT=data/max_particles_h200_v3_f32l2.csv
echo "n_side,N,ok,result_line" > "$OUT"
attempt() {   # $1 = n_side; returns 0 if cold + all warm applies succeeded
  local s=$1 rc=0
  python benchmarks/max_particles.py --near-op moments-v3 --n-side "$s" --warm 3 \
      --empty-cache > "$RUN_DIR/n_side_$s.log" 2>&1 || rc=$?
  local line; line=$(grep '^RESULT' "$RUN_DIR/n_side_$s.log" | tail -1 | tr -d '"' || true)
  echo "$s,$((s*s*s)),$([ $rc -eq 0 ] && echo 1 || echo 0),\"$line\"" >> "$OUT"
  echo "n_side=$s rc=$rc  $line"
  return $rc
}
# Published ceiling (baseline, switch 6) is 403^3; switch 8 carries ~2.4x the
# near pairs, so bracket from below: grow lo until a failure, then bisect.
lo=0; hi=0
for s in 200 250 290 320 350 380 403 430; do
  if attempt $s; then lo=$s; else hi=$s; break; fi
done
[ $hi -gt 0 ] || { echo "no failure up to 430 -- extend the bracket"; exit 0; }
[ $lo -gt 0 ] || { echo "n_side 200 already fails"; exit 1; }
while [ $((hi - lo)) -gt 1 ]; do
  mid=$(((lo + hi) / 2))
  if attempt $mid; then lo=$mid; else hi=$mid; fi
done
echo "CAPACITY n_side=$lo N=$((lo*lo*lo)) (first failure at n_side=$hi)" | tee "$RUN_DIR/capacity.txt"
cp "$OUT" "$RUN_DIR/"
echo "finished $(date)"
