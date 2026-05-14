#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-aic_camera_bags}"
DURATION_SECONDS="${2:-30}"
USE_SIM_TIME="${AIC_CAMERA_USE_SIM_TIME:-true}"

timestamp="$(date +"%Y%m%d_%H%M%S_%3N")"
session_dir="${OUTPUT_ROOT}/camera_recording_${timestamp}"

mkdir -p "${session_dir}"

record_camera() {
  local camera_name="$1"
  local bag_dir="${session_dir}/${camera_name}_camera"
  local image_topic="/${camera_name}_camera/image"
  local camera_info_topic="/${camera_name}_camera/camera_info"

  local record_cmd=(
    ros2 bag record
    --storage mcap
    --compression-mode none
    --storage-preset-profile none
    --disable-keyboard-controls
    --node-name "record_${camera_name}_camera_bag"
    --output "${bag_dir}"
    --topics "${image_topic}" "${camera_info_topic}"
  )

  if [[ "${USE_SIM_TIME}" == "true" ]]; then
    record_cmd+=(--use-sim-time)
  fi

  echo "Recording ${camera_name} camera bag to: ${bag_dir}"
  echo "  ${image_topic}"
  echo "  ${camera_info_topic}"

  if [[ "${DURATION_SECONDS}" == "0" ]]; then
    "${record_cmd[@]}"
  else
    set +e
    timeout --signal=INT "${DURATION_SECONDS}s" "${record_cmd[@]}"
    local status="$?"
    set -e
    if [[ "${status}" != "0" && "${status}" != "124" ]]; then
      return "${status}"
    fi
  fi
}

echo "Camera recording session: ${session_dir}"
echo "Compression: none"
echo "Storage: mcap"

pids=()
for camera_name in left center right; do
  record_camera "${camera_name}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" != "0" ]]; then
  echo "One or more camera recordings failed." >&2
  exit "${status}"
fi

echo "Finished camera recording session: ${session_dir}"
