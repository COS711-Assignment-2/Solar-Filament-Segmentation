#!/usr/bin/env bash

run_suffix="${1:-corrected-data}"
log_file="${2:-runs/parallel-${run_suffix}.log}"
runner_pid=""

stop_runner() {
  if [[ -n "$runner_pid" ]]; then
    kill -TERM -- "-$runner_pid" 2>/dev/null || true
    wait "$runner_pid" 2>/dev/null || true
    runner_pid=""
  fi
}

trap stop_runner INT TERM HUP

setsid env PJRT_DEVICE=TPU python3 -u scripts/run_parallel_experiments.py \
  --run-suffix "$run_suffix" >>"$log_file" 2>&1 &
runner_pid=$!
wait "$runner_pid"
status=$?
runner_pid=""
exit "$status"
