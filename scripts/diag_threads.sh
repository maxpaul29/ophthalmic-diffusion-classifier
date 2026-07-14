#!/bin/bash
# One-off diagnostic: dump state + wchan for every thread of a given PID.
# Usage: bash scripts/diag_threads.sh <PID>
PID="${1:-67}"
for t in /proc/"$PID"/task/*/; do
    tid=$(basename "$t")
    state=$(grep State "$t/status")
    wchan=$(cat "$t/wchan" 2>/dev/null)
    echo "TID $tid: $state wchan=$wchan"
done
