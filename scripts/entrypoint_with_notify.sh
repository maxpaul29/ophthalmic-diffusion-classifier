#!/bin/bash
# Wraps scripts/run.sh so the container can be started detached (docker compose
# up -d): the full training log is captured to a file, and an email is sent
# with a summary once the run finishes (success or failure), via
# scripts/notify_email.py.
#
# `pipefail` makes $? after the pipeline reflect run.sh's real exit code (not
# tee's), since tee itself essentially never fails.
set -o pipefail

LOG_PATH="/workspace/training.log"

bash scripts/run.sh 2>&1 | tee "$LOG_PATH"
EXIT_CODE=$?

python3 scripts/notify_email.py --exit-code "$EXIT_CODE" --log "$LOG_PATH"

exit "$EXIT_CODE"
