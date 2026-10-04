#!/usr/bin/env bash
#
# Container start command for the RunPod benchmark pods (cloud/runpod_bench.py).
# The driver passes this file base64-encoded in NEMO_BOOTSTRAP_B64 and starts the
# pod with
#   bash -c 'echo "$NEMO_BOOTSTRAP_B64" | base64 -d > /pod_bootstrap.sh && exec bash /pod_bootstrap.sh'
# behind the image's own entrypoint (which creates the JIT cache dirs). It
#   1. saves the image's ENV for ssh sessions, which sshd does not hand on;
#   2. arms a dead-man switch: the pod terminates itself at NEMO_POD_TTL_MIN, or
#      as soon as /workspace/.terminate_now appears, whatever the driver does;
#   3. installs and starts sshd (the nemo image has none), key from PUBLIC_KEY;
#   4. stays alive as PID 1 -- if PID 1 exits, RunPod restarts the container and
#      keeps billing.
# Locally (NEMO_LOCAL=1, the driver's --local rehearsal) there is no RunPod key;
# the switch then makes PID 1 exit, which stops and removes the container.

set -u
W=/workspace
exec > >(tee -a "$W/bootstrap.log") 2>&1
echo "$(date -Is) bootstrap start (pod ${RUNPOD_POD_ID:-local})"

# 1. The image's ENV, minus secrets and this script, for `source` in the jobs.
python - > "$W/.container_env" <<'PY'
import os, shlex
skip = {"RUNPOD_API_KEY", "NEMO_BOOTSTRAP_B64", "PUBLIC_KEY", "PWD", "OLDPWD",
        "SHLVL", "_", "HOSTNAME"}
for k, v in sorted(os.environ.items()):
    if k not in skip and k.isidentifier():
        print(f"export {k}={shlex.quote(v)}")
PY

# 2. Dead-man switch.
terminate_self() {
    python - <<'PY'
import json, os, sys, urllib.request
pod, key = os.environ["RUNPOD_POD_ID"], os.environ["RUNPOD_API_KEY"]
# an explicit User-Agent: Cloudflare in front of the API rejects urllib's (403/1010)
hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
       "User-Agent": "nemo-pod-watchdog/1.0"}
gql = "https://api.runpod.io/graphql"
for method, url, body in (
        ("DELETE", f"https://rest.runpod.io/v1/pods/{pod}", None),
        ("POST", gql, {"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pod}),
        ("POST", gql, {"query": 'mutation { podStop(input: {podId: "%s"}) { id } }' % pod})):
    data = None if body is None else json.dumps(body).encode()
    try:
        req = urllib.request.Request(url, data=data, method=method, headers=hdr)
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode(errors="replace")[:300]
            print(method, url, r.status, text, flush=True)
            if r.status < 300 and '"errors"' not in text:
                sys.exit(0)
    except Exception as e:
        print(method, url, "failed:", e, flush=True)
sys.exit(1)
PY
}

watchdog() {
    local deadline=$(( $(date +%s) + ${NEMO_POD_TTL_MIN:-75} * 60 )) reason
    if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
        echo "$(date -Is) watchdog armed: pod $RUNPOD_POD_ID terminates itself at" \
             "$(date -Is -d "@$deadline") or on $W/.terminate_now"
    elif [[ "${NEMO_LOCAL:-0}" == 1 ]]; then
        echo "$(date -Is) watchdog local: container exits at $(date -Is -d "@$deadline")" \
             "or on $W/.terminate_now"
    else
        echo "$(date -Is) watchdog UNARMED: no RUNPOD_POD_ID/RUNPOD_API_KEY in the environment"
        return
    fi
    while :; do
        reason=""
        [[ -e "$W/.terminate_now" ]] && reason="requested"
        (( $(date +%s) >= deadline )) && reason="TTL ${NEMO_POD_TTL_MIN:-75} min reached"
        if [[ -n "$reason" ]]; then
            echo "$(date -Is) terminating: $reason"
            if [[ "${NEMO_LOCAL:-0}" == 1 && -z "${RUNPOD_POD_ID:-}" ]]; then
                touch "$W/.local_exit"
            else
                terminate_self
            fi
            sleep 60     # still alive a minute later: try again
            continue
        fi
        sleep 10
    done
}
watchdog >> "$W/watchdog.log" 2>&1 &

# 3. sshd, key-only.
echo "$(date -Is) installing openssh-server"
apt-get update -qq > "$W/apt.log" 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    openssh-server >> "$W/apt.log" 2>&1 || { echo "apt-get failed:"; tail -20 "$W/apt.log"; }
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
printf '%s\n' "${PUBLIC_KEY:-}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A > /dev/null
/usr/sbin/sshd -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no \
    && echo "$(date -Is) sshd started" || echo "$(date -Is) sshd FAILED to start"
[[ -d "$W/pinn-stokes" ]] || echo "WARNING: $W/pinn-stokes missing -- is a volume mounted over $W?"
touch "$W/.bootstrap_done"

# 4. Stay PID 1 (a plain `sleep infinity` as PID 1 would ignore the local stop).
while [[ ! -e "$W/.local_exit" ]]; do sleep 5; done
echo "$(date -Is) bootstrap exit"
