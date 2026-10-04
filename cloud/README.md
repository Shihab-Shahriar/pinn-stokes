# Benchmarks on rented RunPod GPUs

`cloud/runpod_bench.py` rents one GPU on RunPod, runs one job of this repo there
in the public `sskhan39/nemo:2.0` image, brings the data back and terminates the
pod. It is the RunPod analogue of `slurm/h200_v3/`: a job is a shell script in
`cloud/jobs/` plus one entry in the driver's `JOBS` table.

```bash
python cloud/runpod_bench.py price                         # $/hr and stock, both clouds
python cloud/runpod_bench.py run twodrop --dry-run         # preflight + the pod spec, no pod
python cloud/runpod_bench.py run twodrop --local --quick   # free rehearsal in local docker
python cloud/runpod_bench.py run twodrop                   # the real run
python cloud/runpod_bench.py list                          # what is running, current $/hr
python cloud/runpod_bench.py terminate --all
```

Stdlib only; needs `git`, `ssh`, `~/.ssh/id_ed25519{,.pub}`, and the API key in
`RUNPOD_API_KEY` or `~/.runpod/config.toml` (`apikey = "..."`, the file
`runpodctl config --apiKey` writes). **Never put the key in the repo.**

## What a run does

1. **Preflight, before anything bills.**
   - `git archive HEAD` (the pod runs HEAD; refuses if tracked code under `src/ benchmarks/ extern/ data/models/ experiments/` differs from it).
   - Price and stock for the job's GPU. `--cloud auto` takes secure cloud if it has stock, else community.
   - Refuses above `--max-price`, on a low balance, or if any pod already exists.
   - Packs the job's `inputs` (e.g. the large configs) before the pod exists, so compression is not billed.
2. **Create the pod.**
   - No pod volume (`volumeInGb 0`): nothing bills after termination, and nothing is mounted over the image's `/workspace`.
   - `allowedCudaVersions` ≥ 12.8, which is what the image needs; `22/tcp` exposed.
   - The start command is `cloud/pod_bootstrap.sh`, passed base64 in the env.
3. **Bootstrap, on the pod.**
   - Saves the image ENV to `/workspace/.container_env`: sshd does not hand it to sessions.
   - Arms the dead-man switch.
   - Installs and starts sshd (the image has none).
4. **Upload.** The archive replaces the image's baked tree at `/workspace/pinn-stokes` (moved to `pinn-stokes.image`); `cloud/` comes from the working copy.
5. **Run.** `cloud/jobs/<job>.sh` runs detached, writing into `/workspace/results`; the driver polls it over ssh.
6. **Fetch and terminate.**
   - `results/` + the pod's bootstrap/watchdog logs land in `artifacts/logs/<tag>_v3_f32l<L>/<job>_<utc>/` (`tmp/runpod_local/…` for `--local`); a full run's CSV is also copied into `data/`.
   - The driver then touches `/workspace/.terminate_now`, so every run ends through the dead-man path and a broken switch shows up at once. It deletes the pod itself if that has not happened within 120 s.

## Money guards

| guard | covers |
|---|---|
| preflight: `--max-price` (0.40 $/hr, or the job's own: 1.10 for the 5090), balance, no existing pods | renting the wrong thing, double launches |
| teardown in `finally` / `atexit` / SIGINT, SIGTERM, SIGHUP | the driver failing or being interrupted |
| `--max-minutes` (60) from creation | a hung job |
| pod-side TTL, `--max-minutes` + 10, via RunPod's pod-scoped `RUNPOD_API_KEY` | the laptop sleeping or dying mid-run |

`--keep` leaves the pod up for inspection; its TTL still applies. If the driver
ever prints `COULD NOT CONFIRM … IS GONE`, run `list` / `terminate` right away.

## Jobs

| job | GPU | what | outputs |
|---|---|---|---|
| `twodrop` | RTX A4500 | §3.4 / Fig 13 1M two-drop, 50 steps, `--near-op moments-v3` + pc8c diag, widebvh fp32 L3 built from `extern/widebvh` on the pod; a timing process, then a memory process under a whole-device `nvidia-smi` sampler | `twodrop_1M_a4500_v3_f32l3{.csv,.log}`, `…_mem{.csv,.log,_gpu.csv,_summary.txt}`, `manifest.txt`, `widebvh_build.log` |
| `twodrop_pc6` | RTX A4500 | the same with `--near-op moments-v3-pc6` (moments pair correction on d ≤ 6 only; a timing variant, not a trained operating point) | `twodrop_1M_a4500_v3_pc6_f32l3*` |
| `fig12` | RTX 5090 | Fig 12 scaling, N = 50k..2M, `benchmarks/figure12_grand_M.py --near-op moments-v3`, fp32 L3, one process per size, `--warmup 400` at ≤ 200k (GeForce idle clocks); the H200 v3 sweep's exact clouds (50k..750k from the image, 1M..2M uploaded, all md5-checked) | `fig12_scaling_5090_v3_f32l3.csv` (+ per-size CSVs and logs) |

Memory: per-PID accounting (`NEMO_DEVICE_MEM=1`) cannot match PIDs inside a
container. The pod rents the whole GPU, so device-wide `memory.used` minus the
idle baseline is the process footprint, including widebvh's raw `cudaMalloc`.

To add a job: write `cloud/jobs/<name>.sh` (source `common.sh`, `trap … EXIT` to
write `$RESULTS/EXIT`, one configuration per process for timing), and add a
`JOBS` entry with its GPU type id, the substring to check in `nvidia-smi`, the
output tag and an estimated duration. For a GeForce card at small N, mind the
idle-clock trap (`--warmup 400` at 50k, `artifacts/fig12_5090_runpod_report.md`).
