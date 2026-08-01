#!/usr/bin/env bash
# Pod entrypoint: fetch secrets, run one experiment, publish the result.
#
# Never exits non-zero on a probe failure. "This combination does not work" is the
# result the sweep exists to collect, and a non-zero exit would make the pod look
# broken instead of informative.
#
# The pod always publishes something. A pod that dies without uploading is
# indistinguishable from a pod that was never launched, and that difference is the
# entire content of a support matrix cell.
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

publish() {
    if [[ -z "${TRAINBENCH_RESULT_REPO:-}" ]]; then
        echo "TRAINBENCH_RESULT_REPO not set; artifacts stay on pod disk under ${RESULT_DIR}" >&2
        return 0
    fi
    run_with_secrets uv run --frozen python "${REPO_DIR}/scripts/publish_result.py" \
        --repo "${TRAINBENCH_RESULT_REPO}" \
        --config "${CONFIG_PATH}" \
        "$@"
}

# `cd ||` and not a bare `cd`: without the env directory there is no venv, and
# every command below would silently run against whatever interpreter the image
# happens to ship.
cd "${TRAINBENCH_ENV_DIR}" || {
    echo "TRAINBENCH_ENV_DIR=${TRAINBENCH_ENV_DIR:-<unset>} is not a directory" >&2
    exit 2
}

echo "== announce =="
# Uploaded before any model is touched, so a pod that dies during the image pull
# or the checkpoint download is still on record as having started.
publish --mode start

purpose="${TRAINBENCH_PURPOSE:-probe}"
echo "== run (purpose=${purpose}, deadline ${DEADLINE_SECONDS}s) =="
if [[ "${purpose}" == "probe" ]]; then
    run_with_secrets timeout --signal=TERM --kill-after=60 "${DEADLINE_SECONDS}" \
        uv run --frozen python "${REPO_DIR}/scripts/verify_env.py" \
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
    publish --mode result --result "${RESULT_PATH}"
else
    # A fallback record is still a record: it carries the commit, the image digest
    # and the config, so the combination reads as "ran and produced nothing"
    # rather than disappearing from the matrix.
    publish --mode fallback --result "${RESULT_PATH}" \
        --reason "no result file after the run (exit ${run_status})"
fi

echo "run exited ${run_status}"
exit 0
