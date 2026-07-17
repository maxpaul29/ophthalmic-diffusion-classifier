#!/bin/bash
# List every process directly via /proc, bypassing the (possibly broken/minimal) `ps` tool.
for p in /proc/[0-9]*; do
    pid=$(basename "$p")
    comm=$(cat "$p/comm" 2>/dev/null)
    state=$(grep State "$p/status" 2>/dev/null)
    echo "PID $pid: $comm | $state"
done
