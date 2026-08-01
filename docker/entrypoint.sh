#!/usr/bin/env bash
# Pod entrypoint: fetch secrets, run one experiment, publish the result.
#
# Never exits non-zero on a probe failure. "This combination does not work" is the
# result the sweep exists to collect, and a non-zero exit would make the pod look
# broken instead of informative. A failed *upload* is different and does exit
# non-zero — see the end of the file.
#
# The pod always publishes something. A pod that dies without uploading is
# indistinguishable from a pod that was never launched, and that difference is the
# entire content of a support matrix cell. So every check that can end this script
# early runs after the announce, and the one that cannot — a missing resolved
# config, without which there is nothing to name the combination by — is the first
# thing tested.
set -uo pipefail

RESULT_DIR="${TRAINBENCH_RESULT_DIR:-/workspace/result}"
CONFIG_PATH="${RESULT_DIR}/resolved_config.json"
PLAN_PATH="${RESULT_DIR}/resolved_plan.json"
RESULT_PATH="${RESULT_DIR}/result.json"
REPO_DIR="${TRAINBENCH_REPO_DIR:-/workspace/train-comparison}"
# Always bounded. A pod with no deadline bills until a human notices, which is the
# failure this default exists to make impossible; the orchestrator overrides it
# with a value slightly under its own deadline.
DEADLINE_SECONDS="${TRAINBENCH_TIMEOUT_SECONDS:-3600}"
# The grace between SIGTERM and SIGKILL, and — because they are the same quantity —
# the floor on one setting's slice of the budget: under it a process cannot even
# finish its own shutdown, so starting one would file a killed run as though the
# setting itself had failed.
KILL_GRACE_SECONDS=60
MIN_SETTING_SECONDS="${TRAINBENCH_MIN_SETTING_SECONDS:-${KILL_GRACE_SECONDS}}"
# Recorded for a setting the budget ran out before. Distinct from `timeout`'s own
# 124, which means a setting started and was killed partway.
BUDGET_EXHAUSTED=125
mkdir -p "${RESULT_DIR}"

if [[ -z "${TRAINBENCH_CONFIG_JSON:-}" ]]; then
    echo "TRAINBENCH_CONFIG_JSON is not set; the orchestrator must pass the resolved config" >&2
    exit 2
fi
printf '%s' "${TRAINBENCH_CONFIG_JSON}" > "${CONFIG_PATH}"
# The ordered list of settings this pod owns. One entry for a probe; every value
# of one axis for a sweep, because an axis is never split across pods.
printf '%s' "${TRAINBENCH_PLAN_JSON:-[]}" > "${PLAN_PATH}"

# Secrets come from Infisical via the pod's machine identity. The orchestrator
# injects a short-lived token as an env var; the client secret never reaches here.
# A machine-identity token carries no project of its own, so --projectId is
# required whenever it is known — and passing it empty makes the CLI reject the call.
run_with_secrets() {
    if [[ -n "${INFISICAL_TOKEN:-}" ]]; then
        local args=(run "--env=${INFISICAL_ENV:-dev}")
        if [[ -n "${INFISICAL_PROJECT_ID:-}" ]]; then
            args+=("--projectId=${INFISICAL_PROJECT_ID}")
        fi
        infisical "${args[@]}" -- "$@"
    else
        echo "INFISICAL_TOKEN not set; running without secret injection" >&2
        "$@"
    fi
}

publish_failures=0

# publish <config path> <publish_result.py args...>
#
# The config travels as an argument rather than being read from the global,
# because a sweep pod has one per setting: the pod-level config names the
# combination, but a fallback record filed under it would describe the axis
# baseline instead of the setting that produced nothing.
publish() {
    local config_path="$1"
    shift
    if [[ -z "${TRAINBENCH_RESULT_REPO:-}" ]]; then
        echo "TRAINBENCH_RESULT_REPO not set; artifacts stay on pod disk under ${RESULT_DIR}" >&2
        return 0
    fi
    if run_with_secrets "${PYTHON[@]}" "${REPO_DIR}/scripts/publish_result.py" \
        --repo "${TRAINBENCH_RESULT_REPO}" \
        --config "${config_path}" \
        "$@"; then
        return 0
    fi
    # publish_result.py already retried with backoff before returning non-zero, so
    # this is the upload having failed for good. Counted rather than ignored: the
    # pod log is the only channel that survives the pod.
    publish_failures=$((publish_failures + 1))
    echo "PUBLISH FAILED: ${*}" >&2
    return 1
}

# One setting of a sweep, addressed by its position in the plan. Both paths are
# derived here so the loop and the publish block cannot drift onto different files.
setting_config_path() { echo "${RESULT_DIR}/setting-$1.json"; }
setting_result_path() { echo "${RESULT_DIR}/result-$1.json"; }

# What the loop produced, one entry per setting, in plan order. Declared up front:
# under `set -u` the publish block cannot ask an array that was never assigned how
# long it is, and the probe path never enters the loop.
setting_labels=()
setting_notes=()

# Unwrap one plan item into a config `bench.py` will accept, and print the name
# the setting is filed under.
#
# A plan item is `orchestrate.Run.summary()` — {name, role, overrides, config} —
# while `bench.py` validates a `BenchConfig`. Handing it the wrapper is rejected
# outright; handing it `null` would be worse, because an empty config file reads
# as a setting that was configured rather than one that never started. So an item
# without a resolved config fails here, loudly, and stops that setting alone.
#
# The name is printed before the config is checked: a setting that cannot run
# still has to publish a record, and it is named by the same key as every other.
plan_setting() {
    "${PYTHON[@]}" - "${PLAN_PATH}" "$1" "$2" <<'PY'
import json
import sys

plan_path, index, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
item = json.load(open(plan_path))[index]
print(item.get("name") or "")
config = item.get("config")
if not isinstance(config, dict) or not config:
    sys.exit(f"plan item {index} carries no resolved config (keys: {sorted(item)})")
with open(out_path, "w") as handle:
    json.dump(config, handle)
PY
}

# The framework venv, or the image's own interpreter when there is no venv to use.
# Expanded with a default because `set -u` aborts at the expansion of an unset
# variable: written as a bare `cd "${TRAINBENCH_ENV_DIR}" || { ... }` the guard
# never ran, its `${TRAINBENCH_ENV_DIR:-<unset>}` message was unreachable, the exit
# code was 1 instead of 2 — and, sitting above the announce, it took the whole
# upload with it. That is precisely the "died without uploading" case this file
# claims to make impossible.
ENV_DIR="${TRAINBENCH_ENV_DIR:-}"
if [[ -n "${ENV_DIR}" && -d "${ENV_DIR}" ]]; then
    cd "${ENV_DIR}"
    PYTHON=(uv run --frozen python)
    env_dir_usable=1
else
    # Best effort, and it may well fail at `import huggingface_hub`. It is still
    # worth attempting: a combination that reads as never launched costs more than
    # a failed upload that is at least in the log.
    PYTHON=(python3)
    env_dir_usable=0
fi

echo "== announce =="
# Uploaded before any model is touched, so a pod that dies during the image pull
# or the checkpoint download is still on record as having started.
publish "${CONFIG_PATH}" --mode start || true

if (( env_dir_usable == 0 )); then
    reason="TRAINBENCH_ENV_DIR=${TRAINBENCH_ENV_DIR:-<unset>} is not a directory; no framework venv to run in"
    echo "${reason}" >&2
    publish "${CONFIG_PATH}" --mode fallback --result "${RESULT_PATH}" --reason "${reason}" || true
    exit 2
fi

purpose="${TRAINBENCH_PURPOSE:-probe}"
echo "== run (purpose=${purpose}, deadline ${DEADLINE_SECONDS}s) =="
if [[ "${purpose}" == "probe" ]]; then
    run_with_secrets timeout --signal=TERM --kill-after="${KILL_GRACE_SECONDS}" \
        "${DEADLINE_SECONDS}" \
        "${PYTHON[@]}" "${REPO_DIR}/scripts/verify_env.py" \
        --config "${CONFIG_PATH}" \
        --out "${RESULT_PATH}"
    run_status=$?
elif [[ "${purpose}" == "timing" || "${purpose}" == "profile" || "${purpose}" == "quality" ]]; then
    # One process per setting, not a loop inside bench.py. A process that already
    # ran a setting carries its autotune cache, compiled graphs and allocator
    # fragmentation into the next one, and `kernel`/`attn` cannot change after the
    # model exists at all. The plan is the ordered list of settings this pod owns;
    # an axis is never split across pods, so this loop is the whole axis.
    settings=$("${PYTHON[@]}" -c \
        'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${PLAN_PATH}")
    if [[ "${settings}" -eq 0 ]]; then
        echo "purpose '${purpose}' with an empty plan: nothing to measure" >&2
        run_status=2
    else
        run_status=0
        # Every setting of the plan through the axis refusals before the first one
        # measures anything. `bench.py` refuses a setting it cannot apply anyway,
        # but only once that setting's turn comes: a pod whose whole plan is
        # unrunnable would learn it after booting a GPU, pulling an image and
        # downloading a checkpoint, and a sweep would learn about its second
        # setting only after the first had finished. This costs seconds.
        #
        # It runs here rather than in a gate on the audit host because the answer
        # is a property of the image — fla, causal-conv1d and a CUDA runtime are
        # what decide it, and on a laptop the same check inverts (it rejects the
        # `kernel=fla` baseline every pod runs correctly and passes the
        # `kernel=none` setting that dies on a Qwen3.5 image).
        #
        # No secrets: it builds no model and reaches no network, so wrapping it in
        # the Infisical call would only add a way for it to fail.
        echo "-- preflight"
        "${PYTHON[@]}" "${REPO_DIR}/scripts/bench.py" --preflight "${PLAN_PATH}"
        preflight_status=$?
        for i in $(seq 0 $((settings - 1))); do
            setting_config=$(setting_config_path "${i}")
            setting_out=$(setting_result_path "${i}")
            echo "-- setting ${i}/${settings}"
            label=$(plan_setting "${i}" "${setting_config}")
            status=$?
            if [[ -z "${label}" ]]; then
                # An item with no name of its own — a malformed plan. Its position
                # is then all that identifies it, and a record filed under nothing
                # is worse than one filed under a weak name.
                label="setting-${i}"
            fi
            note="exit ${status}"
            if (( preflight_status != 0 )); then
                # Not measured, and every setting says so. Skipping the publish
                # instead would leave the axis looking as though the pod was never
                # launched, which is the one thing this file exists to prevent —
                # and the plan is refused as a whole, so this holds for the
                # settings that would have passed on their own too.
                status=${preflight_status}
                note="preflight refused this pod's plan (exit ${status}); nothing was measured"
                echo "-- setting ${i} not started: ${note}" >&2
            elif (( status == 0 )); then
                # The deadline is the POD's, not each setting's. Handed to `timeout`
                # whole once per setting, an N-setting sweep bills N times the budget
                # and the guard whose entire purpose is bounding the bill stops
                # bounding it. So what is left is shared out among the settings still
                # to run: no setting can consume the axis, and the time a fast setting
                # gives back flows to the ones after it.
                remaining=$(( DEADLINE_SECONDS - SECONDS ))
                (( remaining < 0 )) && remaining=0
                slice=$(( remaining / (settings - i) ))
                if (( slice < MIN_SETTING_SECONDS )); then
                    # Not started, and said so. A setting that vanishes silently is
                    # indistinguishable from one that ran and failed, which is the
                    # reporting failure this file exists to prevent.
                    status=${BUDGET_EXHAUSTED}
                    note="${remaining}s left of the pod's ${DEADLINE_SECONDS}s budget,"
                    note="${note} under the ${MIN_SETTING_SECONDS}s a setting needs to start"
                    echo "-- setting ${i} not started: ${note}" >&2
                else
                    run_with_secrets timeout --signal=TERM --kill-after="${KILL_GRACE_SECONDS}" \
                        "${slice}" \
                        "${PYTHON[@]}" "${REPO_DIR}/scripts/bench.py" \
                        --config "${setting_config}" \
                        --out "${setting_out}"
                    status=$?
                    note="exit ${status}"
                fi
            fi
            # Keep going: one setting failing is a recorded gap in the axis, and
            # abandoning the rest wastes the pod that was booted for all of them.
            [[ ${status} -ne 0 ]] && run_status=${status}
            setting_labels+=("${label}")
            setting_notes+=("${note}")
        done
    fi
else
    # No dangling reference to an entry point this image does not have: the
    # missing capability is recorded as the pod's result.
    echo "no entry point in this image for purpose '${purpose}'" >&2
    run_status=127
fi

echo "== publish =="
# A fallback record is still a record: it carries the commit, the image digest
# and the config, so the combination reads as "ran and produced nothing" rather
# than disappearing from the matrix. That holds per setting too — a sweep that
# published only the settings that worked would lose the ones that did not.
if (( ${#setting_labels[@]} > 0 )); then
    # Every setting the loop reached, each to its own destination. Uploading them
    # all to `result.json` would leave the last one standing and silently drop the
    # rest of the axis; not uploading them at all is what the pod did before.
    for i in "${!setting_labels[@]}"; do
        label="${setting_labels[$i]}"
        setting_out=$(setting_result_path "${i}")
        setting_config=$(setting_config_path "${i}")
        # A setting whose config never got written is named by the pod's own, which
        # at least identifies the combination the missing setting belonged to.
        [[ -s "${setting_config}" ]] || setting_config="${CONFIG_PATH}"
        if [[ -s "${setting_out}" ]]; then
            publish "${setting_config}" --mode result --result "${setting_out}" \
                --label "${label}" || true
        else
            publish "${setting_config}" --mode fallback --result "${setting_out}" \
                --label "${label}" \
                --reason "setting '${label}' produced no result: ${setting_notes[$i]}" || true
        fi
    done
elif [[ -s "${RESULT_PATH}" ]]; then
    publish "${CONFIG_PATH}" --mode result --result "${RESULT_PATH}" || true
else
    publish "${CONFIG_PATH}" --mode fallback --result "${RESULT_PATH}" \
        --reason "no result file after the run (exit ${run_status})" || true
fi

echo "run exited ${run_status}"

# A probe that answered "this does not work" exits 0: that is the result. An upload
# that never landed exits non-zero, because nothing else on the pod records it.
#
# The orchestrator cannot read this code — RunPod reports that a container exited,
# not with what — so it is for whoever opens the pod log. On the orchestrator side
# the signal is unchanged and already correct: the ledger says the pod launched, no
# artifact arrives, and the cell reads 결과 없음(기동됨) rather than success. The
# exit code says *why* that cell is empty; only the log can.
if (( publish_failures > 0 )); then
    echo "${publish_failures} upload(s) failed after retries; nothing reached the results repo" >&2
    exit 1
fi
exit 0
