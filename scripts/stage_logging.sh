#!/usr/bin/env bash
# Shared logging setup for the maintained split-stage launchers.

# Maintained launchers intentionally pipe combined output through tee.  That
# makes stderr non-TTY even when the run is being watched interactively, so the
# shared default must request the real tqdm surface explicitly.  User choices
# such as plain remain authoritative and --no-progress is handled by Python.
export THINK_BRIDGE_PROGRESS_STYLE="${THINK_BRIDGE_PROGRESS_STYLE:-tqdm}"

prepare_stage_invocation() {
  if (( $# != 2 )); then
    echo "prepare_stage_invocation requires PROJECT_DIR and STAGE" >&2
    return 2
  fi
  local project_dir="$1"
  local stage="$2"
  case "$stage" in
    reasoner-sft) ;;
    *)
      echo "unsupported training stage: $stage" >&2
      return 2
      ;;
  esac

  mkdir -p "$project_dir/invocations"
  local timestamp
  local suffix=0
  local log_name
  timestamp="$(date +%Y%m%d-%H%M%S)"
  while :; do
    if (( suffix == 0 )); then
      log_name="${timestamp}-${stage}.log"
    else
      log_name="${timestamp}-${suffix}-${stage}.log"
    fi
    if (set -o noclobber; : > "$project_dir/$log_name") 2>/dev/null; then
      break
    fi
    ((suffix += 1))
  done

  STAGE_LOG="$project_dir/$log_name"
  RUN_LOCATOR="$project_dir/invocations/${log_name%.log}-$$.txt"
  ln -sfn "$log_name" "$project_dir/latest.log"
  export STAGE_LOG RUN_LOCATOR
}
