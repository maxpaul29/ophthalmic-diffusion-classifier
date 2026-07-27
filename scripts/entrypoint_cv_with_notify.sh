#!/bin/bash
# Wraps scripts/run_drusen_cv.sh so the container can be started detached
# (docker compose run -d ... odc bash scripts/entrypoint_cv_with_notify.sh):
# the full CV log is captured to a file, an hourly email with the complete log
# is sent while the run is still in progress (so a locked/sleeping PC doesn't
# leave you without any progress visibility), and a final email with a summary
# is sent once all folds finish (success or failure), via
# scripts/notify_email.py.
#
# Separate from entrypoint_with_notify.sh on purpose: the normal single-run
# workflow (docker-compose.yml's default command) stays completely unchanged;
# this script is only ever invoked explicitly when you actually want to run
# cross-validation.
#
# `pipefail` makes $? after the pipeline reflect run_drusen_cv.sh's real exit
# code (not tee's), since tee itself essentially never fails.
set -o pipefail

LOG_PATH="/workspace/training.log"
HOURLY_INTERVAL_SECONDS=3600

# Background loop: every hour, email the complete log as an in-progress update.
# Killed via the trap below once the CV run finishes (success, failure, or the
# script being interrupted), so it never outlives the run.
(
    while true; do
        sleep "$HOURLY_INTERVAL_SECONDS"
        python3 scripts/notify_email.py --exit-code 0 --log "$LOG_PATH" \
            --label "IN PROGRESS (hourly update)" --full
    done
) &
HOURLY_PID=$!
trap 'kill "$HOURLY_PID" 2>/dev/null' EXIT

bash scripts/run_drusen_cv.sh 2>&1 | tee "$LOG_PATH"
EXIT_CODE=$?

kill "$HOURLY_PID" 2>/dev/null

python3 scripts/notify_email.py --exit-code "$EXIT_CODE" --log "$LOG_PATH"

exit "$EXIT_CODE"
