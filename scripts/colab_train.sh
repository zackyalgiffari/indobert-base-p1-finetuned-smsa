#!/usr/bin/env bash
# Fine-tune on a Colab GPU via the `colab` CLI, then fetch the results and release the VM.
#
#   scripts/colab_train.sh --smoke
#   scripts/colab_train.sh --push-to-hub zackyalgiffari/indobert-base-p1-smsa-sentiment
#
# Every argument is forwarded to `python -m src.train`.
# Env: GPU (default L4), SESSION (default sentimen-<timestamp>), POLL_SECONDS (default 30),
#      CALL_TIMEOUT (120), LAUNCH_TIMEOUT (300).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-L4}"
SESSION="${SESSION:-sentimen-$(date +%H%M%S)}"
POLL_SECONDS="${POLL_SECONDS:-30}"
CALL_TIMEOUT="${CALL_TIMEOUT:-120}"     # max seconds for one poll/download call
LAUNCH_TIMEOUT="${LAUNCH_TIMEOUT:-300}" # max seconds for the launch call (extract + spawn)
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
if [[ " $* " == *" --smoke "* ]]; then DEST="runs/smoke-$SESSION"; else DEST="results"; fi
TRAIN_ARGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")"

mkdir -p dist "$DEST"
# Pack tracked + untracked-but-not-ignored files (works before the first commit too).
git ls-files -z -co --exclude-standard | tar --null -T - -czf dist/src.tar.gz

echo "[local] creating session $SESSION ($GPU)"
colab new -s "$SESSION" --gpu "$GPU"
trap 'echo "[local] stopping $SESSION"; colab stop -s "$SESSION" || true' EXIT

colab upload -s "$SESSION" dist/src.tar.gz /content/src.tar.gz
if [[ -f "$HF_TOKEN_FILE" ]]; then
  colab upload -s "$SESSION" "$HF_TOKEN_FILE" /content/.hf_token
fi

# `colab exec` can hang after the remote cell finishes when its websocket drops
# silently, so every call is bounded by `timeout`. A launch that "fails" this way
# may still have started the job: the poll loop below is what decides.
timeout "$LAUNCH_TIMEOUT" colab exec -s "$SESSION" --env "TRAIN_ARGS=$TRAIN_ARGS_JSON" <<'PY' \
  || echo "[local] launch call did not return cleanly; polling to confirm the job started"
import json, os, subprocess
env = dict(os.environ, HF_TOKEN_PATH="/content/.hf_token", PYTHONUNBUFFERED="1",
           TOKENIZERS_PARALLELISM="false")
if os.path.exists("/content/.hf_token"):
    os.chmod("/content/.hf_token", 0o600)
subprocess.run(["bash", "-c", "rm -rf /content/app && mkdir -p /content/app && "
                "tar -xzf /content/src.tar.gz -C /content/app"], check=True)
args = json.loads(os.environ["TRAIN_ARGS"])
with open("/content/run.log", "w") as log:
    subprocess.Popen(
        ["bash", "-c", 'bash scripts/remote_job.sh "$@"; echo $? > /content/run.exit', "job", *args],
        cwd="/content/app", env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
print("LAUNCHED", args)
PY

POLL_PY='
import os
lines = open("/content/run.log", errors="replace").read().splitlines() if os.path.exists("/content/run.log") else []
print("\n".join(l[-220:] for l in lines[-4:] if l.strip()))
if os.path.exists("/content/run.exit"):
    print("EXIT=" + open("/content/run.exit").read().strip())
else:
    print("RUNNING" if os.path.exists("/content/run.log") else "NOLOG")
'
nolog=0
while true; do
  sleep "$POLL_SECONDS"
  out="$(printf '%s' "$POLL_PY" | timeout "$CALL_TIMEOUT" colab exec -s "$SESSION" 2>&1 || true)"
  echo "----- $(date +%T)"; echo "$out" | grep -vE '^\s*$' | tail -6
  if [[ "$out" == *NOLOG* ]] && (( ++nolog >= 5 )); then
    echo "[local] job never started (no /content/run.log)"; exit 1
  fi
  if [[ "$out" == *"EXIT="[0-9]* ]]; then
    code="$(grep -o 'EXIT=[0-9]*' <<<"$out" | tail -1 | cut -d= -f2)"
    break
  fi
done

timeout "$CALL_TIMEOUT" colab download -s "$SESSION" /content/run.log "$DEST/run.log" || true
for f in metrics.json classification_report.txt confusion_matrix_validation.png confusion_matrix_test.png; do
  timeout "$CALL_TIMEOUT" colab download -s "$SESSION" "/content/app/results/$f" "$DEST/$f" \
    || echo "[local] missing $f"
done
echo "[local] job exit code $code; artifacts in $DEST/"
exit "$code"
