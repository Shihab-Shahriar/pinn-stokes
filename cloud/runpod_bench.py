#!/usr/bin/env python3
"""Run a NeMO benchmark on a rented RunPod GPU and bring the data home,
without ever leaving a pod billing.

    python cloud/runpod_bench.py price                         # $/hr and stock
    python cloud/runpod_bench.py run twodrop --dry-run         # preflight only, no pod
    python cloud/runpod_bench.py run twodrop --local --quick   # rehearsal in local docker, $0
    python cloud/runpod_bench.py run twodrop                   # the real run (A4500)
    python cloud/runpod_bench.py list                          # anything still running?
    python cloud/runpod_bench.py terminate --all

What a run does: a pod with the public nemo image boots `cloud/pod_bootstrap.sh`
(sshd + a dead-man switch), receives a `git archive` of HEAD plus this cloud/
tree, runs `cloud/jobs/<job>.sh` detached, and is polled over ssh; the results
come back into artifacts/logs/<tag>_v3_f32l<L>/<job>_<utc>/ and the pod is
terminated. Money guards, outermost first:
  - preflight: price cap (--max-price), balance, no other pod running;
  - teardown in `finally`, atexit and on SIGINT/SIGTERM/SIGHUP;
  - a wall-clock cap from pod creation (--max-minutes);
  - the pod terminates itself at --max-minutes + TTL_MARGIN_MIN even if this
    process, or the laptop, dies (RunPod's pod-scoped RUNPOD_API_KEY). Every
    normal run ends through that path first, so a broken switch shows up.
The API key comes from RUNPOD_API_KEY or ~/.runpod/config.toml (`apikey = "..."`,
runpodctl's file), never from the repo. Stdlib only.
"""
import argparse
import atexit
import datetime as dt
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLOUD = REPO / "cloud"
IMAGE = "sskhan39/nemo:2.0"
REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
# The image wants a driver for CUDA >= 12.8 (NVIDIA_REQUIRE_CUDA); older hosts
# would fail to start the container after pulling 17 GB.
CUDA_VERSIONS = ["12.8", "12.9", "13.0"]
TTL_MARGIN_MIN = 10
USER_AGENT = "nemo-runpod-bench/1.0"
# The pod runs HEAD: tracked edits here would silently not be measured.
CODE_PATHS = ["src", "benchmarks", "extern", "data/models", "experiments"]
START_CMD = ('echo "$NEMO_BOOTSTRAP_B64" | base64 -d > /pod_bootstrap.sh '
             '&& exec bash /pod_bootstrap.sh')

FIG12_SIZES = (50_000, 100_000, 200_000, 500_000, 750_000,
               1_000_000, 1_250_000, 1_500_000, 1_750_000, 2_000_000)

# One entry per pod-side job (script + its env). `tag` names the outputs; `csv`
# is the result that is copied into data/ after a full (non --quick, non --local)
# run. `inputs` (repo-relative) are packed before the pod exists and unpacked into
# its tree after the code; `checksums` go along as tmp/cloud_inputs.md5 for the
# job to verify. `max_price` overrides --max-price's default for that GPU.
JOBS = {
    "twodrop": dict(
        script="cloud/jobs/twodrop.sh",
        gpu="NVIDIA RTX A4500", gpu_match="A4500", tag="a4500", est_min=30,
        quick_env={"TWODROP_T_FINAL": "0.03"},
        csv="twodrop_1M_{tag}_v3_{f32}.csv",
        logdir="artifacts/logs/{tag}_v3_{f32}",
    ),
    # Timing variant, not a trained operating point: the moments pair correction
    # only on d <= 6; 2b NN, diagonal and the far field's cutoff stay at 8.
    "twodrop_pc6": dict(
        script="cloud/jobs/twodrop.sh",
        env={"TWODROP_NEAR_OP": "moments-v3-pc6", "TWODROP_VARIANT": "v3_pc6"},
        gpu="NVIDIA RTX A4500", gpu_match="A4500", tag="a4500", est_min=25,
        quick_env={"TWODROP_T_FINAL": "0.03"},
        csv="twodrop_1M_{tag}_v3_pc6_{f32}.csv",
        logdir="artifacts/logs/{tag}_v3_pc6_{f32}",
    ),
    # Figure 12's RTX 5090 column for the v3 NeMO, on the H200 sweep's exact
    # clouds (data/fig12_scaling_h200_v3_f32l3.csv): 50k..750k are baked into the
    # image, 1M..2M are uploaded, and all ten are checked against the laptop's md5s
    # (= the cluster's, 2026-10-04).
    "fig12": dict(
        script="cloud/jobs/fig12.sh",
        gpu="NVIDIA GeForce RTX 5090", gpu_match="5090", tag="5090", est_min=35,
        max_price=1.10,
        inputs=[f"tmp/uniform_large_0.1_{n}.csv" for n in FIG12_SIZES if n >= 1_000_000],
        checksums=[f"tmp/uniform_large_0.1_{n}.csv" for n in FIG12_SIZES],
        quick_env={"FIG12_SIZES": "50000,1000000"},
        csv="fig12_scaling_{tag}_v3_{f32}.csv",
        logdir="artifacts/logs/{tag}_v3_{f32}",
    ),
}

_LOG = None


def log(msg):
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOG is not None:
        with open(_LOG, "a") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------- RunPod API

def api_key():
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    cfg = Path.home() / ".runpod" / "config.toml"
    m = cfg.exists() and re.search(r'^\s*apikey\s*=\s*"([^"]+)"', cfg.read_text(), re.M)
    assert m, "no RunPod key: set RUNPOD_API_KEY or put apikey = \"...\" in ~/.runpod/config.toml"
    return m.group(1)


def http(method, url, body=None, timeout=60):
    """(status, decoded JSON or text); HTTP errors come back as their status."""
    req = urllib.request.Request(
        url, method=method, data=None if body is None else json.dumps(body).encode(),
        # Cloudflare in front of the API rejects urllib's default User-Agent (403, error 1010).
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                 "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, text = r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode(errors="replace")
    try:
        return status, json.loads(text) if text else None
    except json.JSONDecodeError:
        return status, text


def gql(query):
    status, out = http("POST", GQL, {"query": query})
    assert status == 200 and isinstance(out, dict) and not out.get("errors"), \
        f"GraphQL {status}: {out}"
    return out["data"]


def gpu_offer(gpu):
    """Price and stock of one GPU type on both clouds."""
    q = ('query { myself { clientBalance currentSpendPerHr } '
         'gpuTypes(input: {id: "%s"}) { id displayName memoryInGb securePrice communityPrice '
         'sec: lowestPrice(input: {gpuCount: 1, secureCloud: true}) { stockStatus } '
         'com: lowestPrice(input: {gpuCount: 1, secureCloud: false}) { stockStatus } } }' % gpu)
    d = gql(q)
    assert d["gpuTypes"], f"unknown GPU type {gpu!r}"
    g = d["gpuTypes"][0]
    return dict(me=d["myself"], name=g["displayName"], mem=g["memoryInGb"],
                SECURE=(g["securePrice"], (g["sec"] or {}).get("stockStatus")),
                COMMUNITY=(g["communityPrice"], (g["com"] or {}).get("stockStatus")))


def list_pods():
    status, pods = http("GET", f"{REST}/pods")
    assert status == 200, f"GET /pods {status}: {pods}"
    return pods


def get_pod(pod_id):
    """The pod, or None once it is gone (terminated pods 404)."""
    status, pod = http("GET", f"{REST}/pods/{pod_id}")
    if status == 404 or (status == 200 and pod and pod.get("desiredStatus") == "TERMINATED"):
        return None
    assert status == 200, f"GET /pods/{pod_id} {status}: {pod}"
    return pod


def delete_pod(pod_id, tries=4):
    """Terminate and confirm. True once the pod is confirmed gone."""
    for i in range(tries):
        try:
            status, out = http("DELETE", f"{REST}/pods/{pod_id}")
            if status not in (200, 204, 404):
                log(f"DELETE {pod_id}: {status} {out}")
            for _ in range(6):
                if get_pod(pod_id) is None:
                    return True
                time.sleep(5)
        except Exception as e:                       # network blip: retry
            log(f"terminate {pod_id} attempt {i + 1} failed: {e}")
            time.sleep(10)
    return False


# ---------------------------------------------------------------- backends

class RunPodBackend:
    def __init__(self, args, job):
        self.args, self.job = args, job
        self.pod_id = None
        self.cost_per_hr = 0.0
        self.created = None

    def payload(self, cloud, env):
        return dict(
            name=f"nemo-bench-{self.args.job}-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}",
            imageName=IMAGE, gpuTypeIds=[self.args.gpu], gpuCount=1,
            cloudType=cloud, interruptible=False, supportPublicIp=True,
            allowedCudaVersions=CUDA_VERSIONS, minVCPUPerGPU=4, minRAMPerGPU=16,
            # volumeInGb 0: no pod volume, so nothing bills after termination,
            # and nothing is mounted over the image's /workspace (the API's
            # default is a 20 GB volume there).
            containerDiskInGb=30, volumeInGb=0, ports=["22/tcp"],
            env=env, dockerStartCmd=["bash", "-c", START_CMD])

    def create(self, payload):
        self.created = time.time()
        try:
            status, pod = http("POST", f"{REST}/pods", payload, timeout=120)
        except Exception as e:
            status, pod = None, repr(e)
        if not (isinstance(pod, dict) and pod.get("id")):
            # The request may have failed after RunPod made the pod (a timeout on
            # the response): adopt it by its unique name so teardown finds it.
            time.sleep(10)
            mine = [p for p in list_pods() if p.get("name") == payload["name"]]
            if mine:
                self.pod_id = mine[0]["id"]
            raise RuntimeError(f"pod creation failed ({status}): {pod}")
        self.pod_id = pod["id"]
        self.cost_per_hr = float(pod.get("costPerHr") or 0.0)
        return pod

    def endpoint(self):
        try:
            pod = get_pod(self.pod_id)
        except Exception as e:                       # API blip: try again next poll
            log(f"GET pod failed: {e}")
            return None, None, {}
        if pod is None:
            raise RuntimeError(f"pod {self.pod_id} vanished")
        port = (pod.get("portMappings") or {}).get("22")
        ip = pod.get("publicIp")
        return (ip, port, pod) if ip and port else (None, None, pod)

    def gone(self):
        return get_pod(self.pod_id) is None

    def terminate(self):
        return delete_pod(self.pod_id)

    def describe(self):
        pod = get_pod(self.pod_id) or {}
        pod.pop("env", None)                          # bootstrap blob + public key
        pod.pop("consumerUserId", None)               # the account id; logs get committed
        return pod


class LocalBackend:
    """The same bootstrap and job in a local container: a free rehearsal."""

    def __init__(self, args, job):
        self.args, self.job = args, job
        self.pod_id = None
        self.cost_per_hr = 0.0
        self.created = None

    def create(self, env):
        argv = ["docker", "run", "-d", "--rm", "--gpus", "all", "-p", "127.0.0.1::22",
                "--name", f"nemo-bench-local-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}"]
        for k, v in {**env, "NEMO_LOCAL": "1"}.items():
            argv += ["-e", f"{k}={v}"]
        out = subprocess.run(argv + [IMAGE, "bash", "-c", START_CMD],
                             capture_output=True, text=True, check=True)
        self.pod_id, self.created = out.stdout.strip()[:12], time.time()

    def endpoint(self):
        out = subprocess.run(["docker", "port", self.pod_id, "22/tcp"],
                             capture_output=True, text=True)
        if out.returncode or not out.stdout.strip():
            return None, None, {}
        host, port = out.stdout.split()[0].rsplit(":", 1)
        return host, port, {}

    def gone(self):
        return subprocess.run(["docker", "inspect", self.pod_id],
                              capture_output=True).returncode != 0

    def terminate(self):
        subprocess.run(["docker", "rm", "-f", self.pod_id], capture_output=True)
        return self.gone()

    def describe(self):
        return {"id": self.pod_id, "local": True}


# ---------------------------------------------------------------- ssh

class Remote:
    def __init__(self, host, port):
        self.base = ["ssh", "-p", str(port), "-i", str(SSH_KEY),
                     "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                     "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                     "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
                     "-o", "ServerAliveCountMax=4", f"root@{host}"]

    def run(self, cmd, *, stdin=None, timeout=120, check=True, text=True):
        out = subprocess.run(self.base + [cmd], stdin=stdin, capture_output=True,
                             timeout=timeout, text=text)
        if check and out.returncode:
            raise RuntimeError(f"ssh `{cmd[:80]}` -> {out.returncode}: {out.stderr}")
        return out


def tar_gz_of(path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(path, arcname=".")
    return buf.getvalue()


# ---------------------------------------------------------------- run

def pack_inputs(job, tmpdir):
    """The job's input files + their md5s, as one tar.gz, or None. Packed before
    the pod exists, so the compression is not billed."""
    if not job.get("inputs") and not job.get("checksums"):
        return None
    missing = [f for f in job.get("inputs", []) + job.get("checksums", [])
               if not (REPO / f).exists()]
    assert not missing, f"job inputs missing locally: {missing}"
    md5 = subprocess.run(["md5sum"] + job.get("checksums", []), cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    (tmpdir / "tmp").mkdir(exist_ok=True)
    (tmpdir / "tmp" / "cloud_inputs.md5").write_text(md5)
    out = tmpdir / "inputs.tar.gz"
    subprocess.run(["tar", "-czf", str(out), "-C", str(REPO)] + job.get("inputs", [])
                   + ["-C", str(tmpdir), "tmp/cloud_inputs.md5"], check=True)
    log(f"inputs: {len(job.get('inputs', []))} files + md5s of {len(job.get('checksums', []))}, "
        f"{out.stat().st_size / 2**20:.0f} MB packed")
    return out


def preflight(args, job, backend):
    """Everything that can fail before money is spent. Returns the code tarball."""
    assert SSH_KEY.exists() and SSH_KEY.with_suffix(".pub").exists(), f"no ssh key {SSH_KEY}"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--"] + CODE_PATHS,
        cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    if dirty and not args.allow_dirty:
        sys.exit("the pod runs HEAD, but these tracked files differ from it "
                 "(commit, or pass --allow-dirty):\n" + dirty)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.strip()
    tarball = Path(tempfile.mkdtemp(prefix="nemo-bench-")) / f"pinn-stokes-{sha}.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(tarball), "HEAD"],
                   cwd=REPO, check=True)
    log(f"code: HEAD {sha}, {tarball.stat().st_size / 2**20:.1f} MB archive")
    job["_inputs_tar"] = pack_inputs(job, tarball.parent)
    if args.local:
        return sha, tarball, None

    offer = gpu_offer(args.gpu)
    me = offer["me"]
    log(f"account: balance ${me['clientBalance']:.2f}, spending ${me['currentSpendPerHr']:.3f}/hr now")
    for c in ("SECURE", "COMMUNITY"):
        log(f"{offer['name']} {c.lower()}: ${offer[c][0]}/hr, stock {offer[c][1]}")
    if args.cloud == "auto":
        cloud = "SECURE" if offer["SECURE"][1] else "COMMUNITY"
    else:
        cloud = args.cloud.upper()
    price, stock = offer[cloud]
    assert stock, f"no {cloud.lower()} stock for {args.gpu} right now"
    assert price <= args.max_price, f"${price}/hr exceeds --max-price {args.max_price}"
    worst = price * (args.max_minutes + TTL_MARGIN_MIN) / 60
    log(f"using {cloud.lower()} cloud at ${price}/hr; expected ~${price * job['est_min'] / 60:.2f}, "
        f"capped at ${worst:.2f} (--max-minutes {args.max_minutes} + {TTL_MARGIN_MIN} min TTL margin)")
    assert me["clientBalance"] >= max(1.0, 2 * worst), "balance too low for this run"
    pods = list_pods()
    if pods and not args.allow_concurrent:
        for p in pods:
            log(f"  existing pod {p['id']} {p.get('name')} {p.get('desiredStatus')} ${p.get('costPerHr')}/hr")
        sys.exit("pods already exist on this account (terminate them, or --allow-concurrent)")
    return sha, tarball, cloud


def cmd_run(args):
    global _LOG
    job = JOBS[args.job]
    args.gpu = args.gpu or job["gpu"]
    if args.max_price is None:
        args.max_price = job.get("max_price", 0.40)
    f32 = f"f32l{args.fp32_level}"
    tag = "local" if args.local else job["tag"]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.local:
        out_dir = REPO / "tmp" / "runpod_local" / f"{args.job}_{stamp}"
    else:
        out_dir = REPO / job["logdir"].format(tag=tag, f32=f32) / \
            f"{args.job}{'_quick' if args.quick else ''}_{stamp}"
    backend = (LocalBackend if args.local else RunPodBackend)(args, job)
    sha, tarball, cloud = preflight(args, job, backend)

    env = {"PUBLIC_KEY": SSH_KEY.with_suffix(".pub").read_text().strip(),
           "NEMO_POD_TTL_MIN": str(args.max_minutes + TTL_MARGIN_MIN),
           "NEMO_BOOTSTRAP_B64": __import__("base64").b64encode(
               (CLOUD / "pod_bootstrap.sh").read_bytes()).decode()}
    if args.dry_run:
        p = backend.payload(cloud, {**env, "NEMO_BOOTSTRAP_B64": "<%d chars>" % len(env["NEMO_BOOTSTRAP_B64"])}) \
            if not args.local else {"docker": "run -d --rm --gpus all", "env": list(env)}
        print(json.dumps(p, indent=2))
        log("dry run: no pod created")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG = out_dir / "driver.log"
    log(f"run {args.job} on {'the local GPU (docker)' if args.local else f'{args.gpu} ({cloud.lower()})'}, "
        f"results -> {out_dir.relative_to(REPO)}")

    # ---- create, and from this moment on make sure it is torn down
    torn_down = False

    def teardown(reason):
        nonlocal torn_down
        if torn_down or backend.pod_id is None:
            return
        torn_down = True
        log(f"teardown ({reason}): terminating {backend.pod_id}")
        if backend.terminate():
            log(f"{backend.pod_id} confirmed gone")
        else:
            log(f"!!! COULD NOT CONFIRM {backend.pod_id} IS GONE -- check now: "
                f"python cloud/runpod_bench.py list / terminate {backend.pod_id}")
        report_cost(backend)

    atexit.register(teardown, "exit")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda s, f: sys.exit(128 + s))

    try:
        if args.local:
            backend.create(env)
            log(f"container {backend.pod_id} started")
        else:
            pod = backend.create(backend.payload(cloud, env))
            log(f"pod {backend.pod_id} created, ${backend.cost_per_hr}/hr, "
                f"machine {(pod.get('machine') or {}).get('gpuDisplayName', '?')}")
            if backend.cost_per_hr > args.max_price:
                raise RuntimeError(f"pod bills ${backend.cost_per_hr}/hr > --max-price")
        remote = wait_for_ssh(backend, args)
        rc = drive(backend, remote, args, job, sha, tarball, tag, f32, out_dir)
        finish(backend, remote, args, out_dir)
        torn_down = True
        report_cost(backend)
    except BaseException as e:
        if not isinstance(e, SystemExit) or e.code not in (0, None):
            log(f"aborting: {type(e).__name__}: {e}")
        teardown("abort")
        raise
    if rc == 0 and not args.local and not args.quick and job.get("csv"):
        name = job["csv"].format(tag=tag, f32=f32)
        src = out_dir / "results" / name
        if src.exists():
            shutil.copy2(src, REPO / "data" / name)
            log(f"copied {name} -> data/")
    return rc


def wait_for_ssh(backend, args):
    deadline = backend.created + args.boot_timeout * 60
    last = None
    while time.time() < deadline:
        host, port, pod = backend.endpoint()
        state = (pod.get("desiredStatus"), pod.get("lastStatusChange"), host, port)
        if state != last:
            log(f"status {state[0]}  {state[1] or ''}  ssh {host}:{port}")
            last = state
        if host and port:
            remote = Remote(host, port)
            try:
                if remote.run("test -e /workspace/.bootstrap_done", timeout=30,
                              check=False).returncode == 0:
                    log(f"ssh up at root@{host} -p {port} after "
                        f"{(time.time() - backend.created) / 60:.1f} min")
                    return remote
            except subprocess.TimeoutExpired:
                pass
        time.sleep(15)
    raise RuntimeError(f"no ssh within --boot-timeout {args.boot_timeout} min")


def drive(backend, remote, args, job, sha, tarball, tag, f32, out_dir):
    gpu = remote.run("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap "
                     "--format=csv,noheader").stdout.strip()
    log(f"GPU: {gpu}")
    if not args.local and job["gpu_match"] not in gpu:
        raise RuntimeError(f"expected a {job['gpu_match']}, got {gpu}")
    wd = remote.run("head -1 /workspace/watchdog.log", check=False).stdout.strip()
    log(f"dead-man switch: {wd}")
    if not args.local and "armed" not in wd:
        log("!!! the pod cannot terminate itself; do not leave this driver unattended")

    with open(tarball, "rb") as fh:
        remote.run("cat > /workspace/code.tar.gz", stdin=fh, timeout=1800)
    remote.run("set -e; cd /workspace; "
               "if [ -d pinn-stokes ] && [ ! -d pinn-stokes.image ]; then mv pinn-stokes pinn-stokes.image; fi; "
               "rm -rf pinn-stokes cloud results; mkdir -p pinn-stokes cloud results; "
               f"tar -xzf code.tar.gz -C pinn-stokes; echo {sha} > pinn-stokes/GIT_SHA")
    p = subprocess.run(remote.base + ["tar -xzf - -C /workspace/cloud"],
                       input=tar_gz_of(CLOUD), capture_output=True, timeout=120)
    assert p.returncode == 0, p.stderr
    log(f"uploaded HEAD {sha} and cloud/")
    if job.get("_inputs_tar"):
        t0 = time.time()
        with open(job["_inputs_tar"], "rb") as fh:
            remote.run("tar -xzf - -C /workspace/pinn-stokes", stdin=fh, timeout=3600)
        log(f"uploaded job inputs in {time.time() - t0:.0f} s")

    jenv = {"NEMO_GPU_TAG": tag, "NEMO_GIT_SHA": sha, "NEMO_IMAGE": IMAGE,
            "NEMO_FAR_FP32_LEVEL": str(args.fp32_level),
            **job.get("env", {}), **(job.get("quick_env", {}) if args.quick else {})}
    envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in jenv.items())
    # `;`, not `&&`: `a && b &` would background the whole list in a subshell
    # that keeps ssh's stdout open, and the call would block until the job ends.
    remote.run(f"cd /workspace; {envs} setsid nohup bash /workspace/{job['script']} "
               f"> /workspace/results/job.log 2>&1 < /dev/null & echo started", timeout=60)
    log(f"job {job['script']} started ({envs})")

    hard_stop = backend.created + args.max_minutes * 60 - 120     # leave time to fetch
    fails, last = 0, None
    poll = ("cat /workspace/results/EXIT 2>/dev/null; echo '|'; "
            "grep -a '^=== stage' /workspace/results/job.log | tail -1; echo '|'; "
            "f=$(ls -t /workspace/results/*.log 2>/dev/null | grep -v job.log | head -1); "
            "[ -n \"$f\" ] && grep -a -E '^(Step t=|Total particles|Simulation complete|@@)' \"$f\" | tail -1")
    while True:
        time.sleep(args.poll)
        if time.time() > hard_stop:
            log(f"--max-minutes {args.max_minutes} reached: stopping")
            fetch(remote, out_dir)
            return 124
        try:
            out = remote.run(poll, timeout=60, check=False).stdout
            fails = 0
        except subprocess.TimeoutExpired:
            out = None
        if out is None or "|" not in out:
            fails += 1
            log(f"poll failed ({fails}/5)")
            if fails >= 5:
                raise RuntimeError("lost the pod: 5 polls in a row failed")
            continue
        code, st, step = (x.strip() for x in (out.split("|") + ["", ""])[:3])
        now = f"{st.split(' 20')[0]}  {step}"
        if now != last:
            log(now)
            last = now
        if code:
            log(f"job finished, exit {code}, {(time.time() - backend.created) / 60:.1f} min since creation")
            fetch(remote, out_dir)
            return int(code)


def fetch(remote, out_dir):
    p = subprocess.run(remote.base + [
        "cd /workspace && tar -czf - results bootstrap.log watchdog.log apt.log 2>/dev/null"],
        capture_output=True, timeout=600)
    if not p.stdout:
        log(f"fetch failed: {p.stderr.decode(errors='replace')[:300]}")
        return
    with tarfile.open(fileobj=io.BytesIO(p.stdout), mode="r:gz") as tf:
        tf.extractall(out_dir, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    log(f"fetched {sum(1 for _ in out_dir.rglob('*') if _.is_file())} files into "
        f"{out_dir.relative_to(REPO)}")
    for f in sorted((out_dir / "results").glob("*.log")) + sorted((out_dir / "results").glob("*summary.txt")):
        if f.name == "job.log":                       # repeats the per-run lines
            continue
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("@@"):
                log(f"{f.name}: {line}")


def finish(backend, remote, args, out_dir):
    """Normal end: let the pod terminate itself (proves the dead-man switch),
    then make sure from here."""
    (out_dir / "pod.json").write_text(json.dumps(backend.describe(), indent=2))
    if args.keep:
        log(f"--keep: {backend.pod_id} left running; it terminates itself at the TTL. "
            f"Now: python cloud/runpod_bench.py terminate {backend.pod_id}")
        return
    remote.run("touch /workspace/.terminate_now", check=False, timeout=30)
    t0 = time.time()
    while time.time() - t0 < 120:
        if backend.gone():
            log(f"{backend.pod_id} terminated itself after {time.time() - t0:.0f} s "
                "(dead-man switch works)")
            return
        time.sleep(10)
    log("self-termination did not complete in 120 s; terminating from here")
    if not backend.terminate():
        log(f"!!! COULD NOT CONFIRM {backend.pod_id} IS GONE -- "
            f"python cloud/runpod_bench.py terminate {backend.pod_id}")


def report_cost(backend):
    if backend.created and backend.cost_per_hr:
        hrs = (time.time() - backend.created) / 3600
        log(f"pod lifetime {hrs * 60:.1f} min x ${backend.cost_per_hr}/hr = ~${hrs * backend.cost_per_hr:.3f}")


# ---------------------------------------------------------------- other commands

def cmd_list(args):
    me = gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
    print(f"balance ${me['clientBalance']:.2f}, current spend ${me['currentSpendPerHr']:.3f}/hr")
    pods = list_pods()
    for p in pods:
        print(f"{p['id']}  {p.get('name')}  {p.get('desiredStatus')}  ${p.get('costPerHr')}/hr  "
              f"{p.get('image')}  {p.get('lastStatusChange')}")
    print(f"{len(pods)} pod(s)")


def cmd_terminate(args):
    ids = [p["id"] for p in list_pods()] if args.all else args.pod_ids
    for i in ids:
        print(i, "gone" if delete_pod(i) else "NOT CONFIRMED GONE")


def cmd_price(args):
    for g in args.gpus or sorted({j["gpu"] for j in JOBS.values()}):
        o = gpu_offer(g)
        print(f"{o['name']} ({o['mem']} GB): secure ${o['SECURE'][0]}/hr stock {o['SECURE'][1]}, "
              f"community ${o['COMMUNITY'][0]}/hr stock {o['COMMUNITY'][1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a job on a fresh pod")
    r.add_argument("job", choices=sorted(JOBS))
    r.add_argument("--gpu", default=None, help="RunPod GPU type id (default: the job's)")
    r.add_argument("--cloud", choices=("auto", "secure", "community"), default="auto",
                   help="auto = secure if it has stock, else community (default)")
    r.add_argument("--max-price", type=float, default=None,
                   help="$/hr cap (default: the job's, else 0.40)")
    r.add_argument("--max-minutes", type=int, default=60,
                   help="pod wall-clock cap from creation; the pod's own TTL is this + %d" % TTL_MARGIN_MIN)
    r.add_argument("--boot-timeout", type=int, default=25,
                   help="minutes from creation to a working ssh (the image pull is 17 GB)")
    r.add_argument("--fp32-level", type=int, default=3, help="widebvh fp32 level (default 3)")
    r.add_argument("--poll", type=int, default=30, help="seconds between progress polls")
    r.add_argument("--dry-run", action="store_true", help="preflight and print the pod spec only")
    r.add_argument("--local", action="store_true", help="rehearse in a local docker container, no RunPod")
    r.add_argument("--quick", action="store_true", help="a few steps instead of the full job")
    r.add_argument("--keep", action="store_true", help="leave the pod up at the end (its TTL still applies)")
    r.add_argument("--allow-dirty", action="store_true", help="run HEAD although tracked code files differ")
    r.add_argument("--allow-concurrent", action="store_true", help="start even if other pods exist")
    sub.add_parser("list", help="pods on the account and current spend")
    t = sub.add_parser("terminate", help="terminate pods")
    t.add_argument("pod_ids", nargs="*")
    t.add_argument("--all", action="store_true")
    pr = sub.add_parser("price", help="$/hr and stock")
    pr.add_argument("gpus", nargs="*")
    args = ap.parse_args()
    return {"run": cmd_run, "list": cmd_list, "terminate": cmd_terminate,
            "price": cmd_price}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
