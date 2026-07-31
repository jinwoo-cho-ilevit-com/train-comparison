#!/usr/bin/env bash
# Pod entrypoint: fetch secrets, run one probe, publish the result.
#
# Never exits non-zero on a probe failure. "This combination does not work" is the
# result the sweep exists to collect, and a non-zero exit would make the pod look
# broken instead of informative.
set -uo pipefail

RESULT_DIR="${TRAINBENCH_RESULT_DIR:-/workspace/result}"
CONFIG_PATH="${RESULT_DIR}/resolved_config.json"
RESULT_PATH="${RESULT_DIR}/result.json"
mkdir -p "${RESULT_DIR}"

if [[ -z "${TRAINBENCH_CONFIG_JSON:-}" ]]; then
    echo "TRAINBENCH_CONFIG_JSON is not set; the orchestrator must pass the resolved config" >&2
    exit 2
fi
printf '%s' "${TRAINBENCH_CONFIG_JSON}" > "${CONFIG_PATH}"

# Secrets come from Infisical via the pod's machine identity. The token is
# injected by RunPod as an env var; the client secret never appears in argv.
run_with_secrets() {
    if [[ -n "${INFISICAL_TOKEN:-}" ]]; then
        infisical run --env="${INFISICAL_ENV:-dev}" --projectId="${INFISICAL_PROJECT_ID:-}" -- "$@"
    else
        echo "INFISICAL_TOKEN not set; running without secret injection" >&2
        "$@"
    fi
}

cd "${TRAINBENCH_ENV_DIR}"

echo "== probe =="
run_with_secrets uv run --frozen python /workspace/train-comparison/scripts/verify_env.py \
    --config "${CONFIG_PATH}" \
    --out "${RESULT_PATH}"
probe_status=$?

echo "== publish =="
if [[ -n "${TRAINBENCH_RESULT_REPO:-}" ]]; then
    run_with_secrets uv run --frozen python \
        /workspace/train-comparison/scripts/publish_result.py \
        --result "${RESULT_PATH}" \
        --repo "${TRAINBENCH_RESULT_REPO}"
else
    echo "TRAINBENCH_RESULT_REPO not set; result stays on pod disk at ${RESULT_PATH}" >&2
fi

echo "probe exited ${probe_status}"
exit 0
