#!/bin/bash
# RunPod job: Figure 12, end-to-end scaling of one operator application on the
# uniform phi = 0.1 clouds, N = 50k .. 2M, with the current NeMO (v3 moments pair
# + pc8c diagonal, fp16 MLP, switch 8; widebvh far field at fp32 level
# $NEMO_FAR_FP32_LEVEL; torch.compile on). The protocol of the H200 column
# (data/fig12_scaling_h200_v3_f32l3.csv): benchmarks/figure12_grand_M.py, 6 warmup
# + 6 timed applications, ONE PROCESS PER SIZE (the 9th size in a process hits
# dynamo's recompile_limit and silently runs eager), each writing its own CSV,
# merged at the end.
#
# Inputs: the H200 sweep's exact clouds. 50k..750k are baked into the image
# (/workspace/pinn-stokes.image/tmp), 1M..2M are uploaded by the driver; all are
# checked against tmp/cloud_inputs.md5 (the laptop's = the cluster's) first.
# GeForce idle clocks: a pod's GPU idles at ~180 MHz and `nvidia-smi -lgc` is not
# permitted, so at small N six warmup applications cannot reach boost clocks
# (artifacts/fig12_5090_runpod_report.md); sizes <= 200k get --warmup 400.
# FIG12_SIZES overrides the size list (the driver's --quick).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
trap 'echo $? > "$RESULTS/EXIT"' EXIT

SIZES=${FIG12_SIZES:-50000,100000,200000,500000,750000,1000000,1250000,1500000,1750000,2000000}
NEAR_OP=${FIG12_NEAR_OP:-moments-v3}
NAME=fig12_scaling_${TAG}_${FIG12_VARIANT:-v3}_${F32}

stage inputs
for f in /workspace/pinn-stokes.image/tmp/uniform_large_0.1_*.csv; do
    if [[ -e $f && ! -e tmp/${f##*/} ]]; then cp "$f" tmp/; fi
done
grep -E "_($(tr , '|' <<<"$SIZES"))\.csv" tmp/cloud_inputs.md5 | md5sum -c -

stage widebvh;  build_widebvh
stage manifest; write_manifest
md5sum tmp/uniform_large_0.1_*.csv >> "$RESULTS/manifest.txt"

failed=0
for n in ${SIZES//,/ }; do
    warmup=6; (( n <= 200000 )) && warmup=400
    stage "N=$n warmup=$warmup"
    if python benchmarks/figure12_grand_M.py --backend widebvh --near-op "$NEAR_OP" \
            --sizes "$n" --warmup "$warmup" --csv "$RESULTS/${NAME}_$n.csv" \
            > "$RESULTS/${NAME}_$n.log" 2>&1; then
        grep -a '^@@' "$RESULTS/${NAME}_$n.log"
    else
        echo "N=$n FAILED"; tail -5 "$RESULTS/${NAME}_$n.log"; failed=$((failed + 1))
    fi
done

stage merge
python - "$RESULTS" "$NAME" <<'PY'
import csv, glob, sys
results, name = sys.argv[1:]
rows = []
for f in glob.glob(f"{results}/{name}_*.csv"):
    with open(f) as fh:
        rows += list(csv.DictReader(fh))
rows.sort(key=lambda r: int(r["n"]))
with open(f"{results}/{name}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"@@ merged {len(rows)} sizes -> {name}.csv")
for r in rows:
    print(f"@@ N={int(r['n']):>9,}  total {float(r['total_ms']):8.2f} ms  "
          f"far {float(r['far_ms']):7.2f} ms  {float(r['updates_per_sec']) / 1e6:6.3f} M upd/s")
PY
stage done
exit "$failed"
