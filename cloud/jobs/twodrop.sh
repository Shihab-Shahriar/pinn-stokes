#!/bin/bash
# RunPod job: the 1M two-drop sedimentation (Section 3.4 / Fig 13), 50-step
# aggregate, with the current NeMO: v3 moments pair + pc8c diagonal (fp16 MLP,
# switch 8), widebvh far field at fp32 level $NEMO_FAR_FP32_LEVEL, torch.compile on.
# Same command as slurm/h200_v3/twodrop.sbatch's first half. Two processes, one
# configuration each:
#   timing  the measured run, nothing else on the GPU
#   memory  the same run again under a whole-device nvidia-smi sampler
# Launched by cloud/runpod_bench.py; TWODROP_T_FINAL (default 0.5 = 50 steps)
# is for its --quick rehearsal. TWODROP_NEAR_OP / TWODROP_VARIANT select another
# near operator and its output tag (job twodrop_pc6: moments-v3-pc6 / v3_pc6, the
# moments pair correction on d <= 6 only, as slurm/h200_v3/twodrop_pc6.sbatch).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
trap 'echo $? > "$RESULTS/EXIT"' EXIT

T_FINAL=${TWODROP_T_FINAL:-0.5}
NEAR_OP=${TWODROP_NEAR_OP:-moments-v3}
NAME=twodrop_1M_${TAG}_${TWODROP_VARIANT:-v3}_${F32}
run_twodrop() {   # $1 = output stem
    python benchmarks/two_suspensions_1M.py --near-op "$NEAR_OP" \
        --fp32-level "$NEMO_FAR_FP32_LEVEL" --t-final "$T_FINAL" \
        --log-csv "$RESULTS/$1.csv" > "$RESULTS/$1.log" 2>&1
    grep -a '^@@' "$RESULTS/$1.log"
}

stage widebvh;  build_widebvh
stage manifest; write_manifest

stage timing
run_twodrop "$NAME"

stage memory
idle=$(gpu_mem_idle_mb)
mem_sampler_start "$RESULTS/${NAME}_mem_gpu.csv"
run_twodrop "${NAME}_mem"
mem_sampler_stop
python - "$RESULTS/${NAME}_mem_gpu.csv" "$idle" <<'PY' | tee "$RESULTS/${NAME}_mem_summary.txt"
import csv, sys
path, idle = sys.argv[1], float(sys.argv[2])
rows = [r for r in csv.reader(open(path)) if r and r[0][:1].isdigit()]
used = [float(r[1]) for r in rows]
total = float(rows[0][2])
peak = max(used)
print(f"@@ device memory.used: peak {peak:.0f} MiB of {total:.0f} MiB, idle before the run "
      f"{idle:.0f} MiB -> process footprint {peak - idle:.0f} MiB ({(peak - idle) / 1024:.2f} GiB), "
      f"{len(used)} samples at 2 Hz")
PY
stage done
