#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for running the benchmark harness in FastAPI dev mode.
# Usage:
#   BENCH_DEV_USER=alice BENCH_DEV_TOKEN=alice scripts/bench_dev.sh --duration 20 --visualize
# If BENCH_DEV_USER/TOKEN are omitted we default to "bench".

if [ -n "${BENCH_DEV_USER:-}" ]; then
  DEV_USER="${BENCH_DEV_USER}"
else
  DEV_USER="bench-$(date +%s)-$RANDOM"
fi
DEV_TOKEN="${BENCH_DEV_TOKEN:-$DEV_USER}"

detect_base_url() {
  if [ -n "${BENCH_DEV_BASE_URL:-}" ]; then
    printf '%s' "${BENCH_DEV_BASE_URL}"
    return
  fi

  if ! command -v kubectl >/dev/null 2>&1; then
    printf '%s' "${MNEMOSYNE_FASTAPI_URL:-http://127.0.0.1:8001}"
    return
  fi

  local ns="${BENCH_DEV_NAMESPACE:-default}"
  local svc=""
  local -a svc_candidates=()
  if [ -n "${BENCH_DEV_SERVICE:-}" ]; then
    svc_candidates=("${BENCH_DEV_SERVICE}")
  else
    svc_candidates=("mnemosyne-api" "mnemosyne-fastapi")
  fi

  for candidate in "${svc_candidates[@]}"; do
    if kubectl get svc "${candidate}" -n "${ns}" >/dev/null 2>&1; then
      svc="${candidate}"
      break
    fi
  done

  if [ -z "${svc}" ]; then
    printf '%s' "${MNEMOSYNE_FASTAPI_URL:-http://127.0.0.1:8001}"
    return
  fi

  local base_path
  base_path="$(kubectl get svc "${svc}" -n "${ns}" -o jsonpath='{.metadata.annotations.mnemosyne\.dev/base-path}' 2>/dev/null || true)"
  local target_port
  target_port="$(kubectl get svc "${svc}" -n "${ns}" -o jsonpath='{.spec.ports[?(@.name=="http")].targetPort}' 2>/dev/null || true)"
  if [ -z "${target_port}" ]; then
    target_port="$(kubectl get svc "${svc}" -n "${ns}" -o jsonpath='{.spec.ports[0].targetPort}' 2>/dev/null || true)"
  fi
  if [ -z "${target_port}" ]; then
    target_port=8000
  fi

  local host="${BENCH_DEV_HOST:-127.0.0.1}"
  local local_port="${BENCH_DEV_LOCAL_PORT:-$target_port}"
  local base="http://${host}:${local_port}"
  if [ -n "${base_path}" ]; then
    # Ensure leading slash
    if [[ "${base_path}" != /* ]]; then
      base_path="/${base_path}"
    fi
    base="${base}${base_path}"
  fi

  printf '%s' "${base}"
}

DEV_BASE_URL="$(detect_base_url)"

check_health() {
  local candidate="$1"
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi
  local health="${candidate%/}/health"
  local code
  code="$(curl -sk -o /dev/null -w "%{http_code}" "$health" || true)"
  if [ "$code" = "200" ]; then
    return 0
  fi
  return 1
}

if ! check_health "${DEV_BASE_URL}"; then
  maybe_api="${DEV_BASE_URL%/}/api"
  if check_health "${maybe_api}"; then
    DEV_BASE_URL="${maybe_api}"
  fi
fi

ensure_openapi() {
  if ! command -v curl >/dev/null 2>&1; then
    return
  fi
  local base="$1"
  local openapi_url="${base%/}/openapi.json"
  local code
  code="$(curl -sk -o /dev/null -w "%{http_code}" "$openapi_url" || true)"
  if [ "$code" = "200" ]; then
    DEV_BASE_URL="$base"
    return
  fi

  local api_base="${base%/}/api"
  local api_openapi="${api_base%/}/openapi.json"
  code="$(curl -sk -o /dev/null -w "%{http_code}" "$api_openapi" || true)"
  if [ "$code" = "200" ]; then
    DEV_BASE_URL="$api_base"
  fi
}

ensure_openapi "${DEV_BASE_URL}"

# Unset any parent env vars to ensure our randomized user takes precedence
unset MNEMOSYNE_DEV_USER_ID
unset MNEMOSYNE_DEV_TOKEN

export MNEMOSYNE_DEV_USER_ID="${DEV_USER}"
export MNEMOSYNE_DEV_TOKEN="${DEV_TOKEN}"
export MNEMOSYNE_AUTH__MODE="${MNEMOSYNE_AUTH__MODE:-dev_no_auth}"
export MNEMOSYNE_FASTAPI_URL="${DEV_BASE_URL}"

if [ -n "${BENCH_DEV_BASE_URL:-}" ]; then
  echo "Using BENCH_DEV_BASE_URL=${DEV_BASE_URL}"
else
  echo "Detected FastAPI base URL via kubectl: ${DEV_BASE_URL}"
fi
echo "Running benchmark in dev mode as user '${DEV_USER}' (token '${DEV_TOKEN}')"
uv run scripts/bench_mcp.py "$@"
