#!/bin/bash
# Wraps scripts/run.sh so the container can be started detached (docker compose
# up -d): the full training log is captured to a file, an hourly email with the
# complete log is sent while training is still running (so a locked/sleeping PC
# doesn't leave you without any progress visibility), and a final email with a
# summary is sent once the run finishes (success or failure), via
# scripts/notification/notify_email.py.
#
# `pipefail` makes $? after the pipeline reflect run.sh's real exit code (not
# tee's), since tee itself essentially never fails.
set -o pipefail

LOG_PATH="/workspace/training.log"
HOURLY_INTERVAL_SECONDS=3600

# Background loop: every hour, email the complete log as an in-progress update.
# Killed via the trap below once training finishes (success, failure, or the
# script being interrupted), so it never outlives the training process.
(
    while true; do
        sleep "$HOURLY_INTERVAL_SECONDS"
        python3 scripts/notification/notify_email.py --exit-code 0 --log "$LOG_PATH" \
            --label "IN PROGRESS (hourly update)" --full
    done
) &
HOURLY_PID=$!
trap 'kill "$HOURLY_PID" 2>/dev/null' EXIT

bash scripts/run.sh 2>&1 | tee "$LOG_PATH"
EXIT_CODE=$?

kill "$HOURLY_PID" 2>/dev/null

python3 scripts/notification/notify_email.py --exit-code "$EXIT_CODE" --log "$LOG_PATH"

exit "$EXIT_CODE"
