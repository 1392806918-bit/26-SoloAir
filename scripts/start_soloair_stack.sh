#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV_NAME="${SOLOAIR_CONDA_ENV:-sherpa_env}"
CONDA_SH="${SOLOAIR_CONDA_SH:-/home/elf/miniconda3/etc/profile.d/conda.sh}"
SYSTEM_PYTHON="${SOLOAIR_SYSTEM_PYTHON:-/usr/bin/python3}"

RTSP_PID=""

cleanup() {
  if [[ -n "${RTSP_PID}" ]] && kill -0 "${RTSP_PID}" 2>/dev/null; then
    kill "${RTSP_PID}" 2>/dev/null || true
    wait "${RTSP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[soloair] starting system RTSP service with ${SYSTEM_PYTHON}"
env -i \
  HOME="${HOME}" \
  USER="${USER:-elf}" \
  LOGNAME="${LOGNAME:-elf}" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  LANG="${LANG:-C.UTF-8}" \
  LC_ALL="${LC_ALL:-C.UTF-8}" \
  "${SYSTEM_PYTHON}" "${PROJECT_ROOT}/scripts/soloair_system_rtsp_server.py" &
RTSP_PID="$!"

sleep 2
if ! kill -0 "${RTSP_PID}" 2>/dev/null; then
  echo "[soloair] system RTSP service exited during startup" >&2
  wait "${RTSP_PID}" || true
  exit 1
fi

echo "[soloair] activating conda env ${CONDA_ENV_NAME}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[soloair] conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

# shellcheck source=/home/elf/miniconda3/etc/profile.d/conda.sh
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

echo "[soloair] starting conda control service"
python "${PROJECT_ROOT}/scripts/soloair_control_service.py"
