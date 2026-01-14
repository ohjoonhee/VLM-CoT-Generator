#!/bin/bash
# --- SLURM Directives ---
#SBATCH -J launch
#SBATCH -p cpu-farm
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -o logs/%x_%j.log

set -a
source .env
set +a

slack_notify () {
  local msg="$1"
  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"${msg}\"}" \
    "$SLACK_WEBHOOK_URL" >/dev/null
}

# --- PATH RESOLUTION ---
# 1. Logfile: Match the SBATCH -o pattern manually
LOGFILE="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
# 2. Abs Log Path: Resolve to absolute path for clickable links
ABSLOG="$(readlink -f "$LOGFILE" 2>/dev/null || realpath "$LOGFILE")"
# 3. Script Path: Use the injected variable, fallback to submit script, then unknown
SCRIPT_PATH="${ORIG_SBATCH_SCRIPT:-${SLURM_SUBMIT_SCRIPT:-<unknown>}}"

slack_notify ":rocket: JOB START | ${SLURM_JOB_NAME} | id=${SLURM_JOB_ID} | node=${SLURMD_NODENAME}
script: \`${SCRIPT_PATH}\`
log: \`${ABSLOG}\`
follow: \`tail -n 200 -F ${ABSLOG}\`
status: \`scontrol show job ${SLURM_JOB_ID} -dd\`"

PYTHON_FILE=$1

uv run python $PYTHON_FILE
rc=$?

# Send completion status
if [ $rc -eq 0 ]; then
  slack_notify ":white_check_mark: DONE ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID})"
else
  slack_notify ":x: FAIL ${SLURM_JOB_NAME} (jobid=${SLURM_JOB_ID}) exit=${rc}"
fi

exit $rc
