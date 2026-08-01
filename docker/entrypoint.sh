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

publish() {
    if [[ -z "${TRAINBENCH_RESULT_REPO:-}" ]]; then
        echo "TRAINBENCH_RESULT_REPO not set; artifacts stay on pod disk under ${RESULT_DIR}" >&2
        return 0
    fi
    if run_with_secrets "${PYTHON[@]}" "${REPO_DIR}/scripts/publish_result.py" \
        --repo "${TRAINBENCH_RESULT_REPO}" \
        --config "${CONFIG_PATH}" \
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
publish --mode start || true

if (( env_dir_usable == 0 )); then
    reason="TRAINBENCH_ENV_DIR=${TRAINBENCH_ENV_DIR:-<unset>} is not a directory; no framework venv to run in"
    echo "${reason}" >&2
    publish --mode fallback --result "${RESULT_PATH}" --reason "${reason}" || true
    exit 2
fi

purpose="${TRAINBENCH_PURPOSE:-probe}"
echo "== run (purpose=${purpose}, deadline ${DEADLINE_SECONDS}s) =="
if [[ "${purpose}" == "probe" ]]; then
    run_with_secrets timeout --signal=TERM --kill-after=60 "${DEADLINE_SECONDS}" \
        "${PYTHON[@]}" "${REPO_DIR}/scripts/verify_env.py" \
        --config "${CONFIG_PATH}" \
        --out "${RESULT_PATH}"
    run_status=$?
else
    # No dangling reference to an entry point this image does not have: the
    # missing capability is recorded as the pod's result.
    echo "no entry point in this image for purpose '${purpose}'" >&2
    run_status=127
fi

echo "== publish =="
if [[ -s "${RESULT_PATH}" ]]; then
    publish --mode result --result "${RESULT_PATH}" || true
else
    # A fallback record is still a record: it carries the commit, the image digest
    # and the config, so the combination reads as "ran and produced nothing"
    # rather than disappearing from the matrix.
    publish --mode fallback --result "${RESULT_PATH}" \
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
