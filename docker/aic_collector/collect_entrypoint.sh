#!/bin/bash
#
# Entrypoint for the aic_collector container.
# Mirrors the ZENOH_CONFIG_OVERRIDE pattern from docker/aic_model/Dockerfile
# so that rmw_zenoh_cpp connects to the Zenoh router started by the eval container.
#
# Usage (via docker-compose.dataset_a.yaml):
#   The container is started automatically by docker compose.
#   To run interactively:
#     docker compose -f docker/docker-compose.dataset_a.yaml \
#       run --rm collector --n_scenes 100 --connector_types sfp
#
set -e

# ─── ROS 2 workspace ─────────────────────────────────────────────────────────
source /ws_aic/install/setup.bash

# ─── Zenoh — connect to the router running in the eval container ──────────────
# With network_mode: service:eval, the router is on localhost:7447.
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
    echo "[collector] WARNING: AIC_ROUTER_ADDR not set, defaulting to localhost:7447"
    AIC_ROUTER_ADDR="localhost:7447"
fi

# Build the same ZENOH_CONFIG_OVERRIDE used by the model container.
# Disabling shared_memory is critical when running across containers.
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'

# ACL support (mirrors model container behaviour; usually disabled for local dev)
should_enable_acl() {
    [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
    if [[ -z "$AIC_MODEL_PASSWD" ]]; then
        echo "[collector] ERROR: AIC_MODEL_PASSWD must be set when AIC_ENABLE_ACL=true"
        exit 1
    fi
    echo "model:$AIC_MODEL_PASSWD" >> /tmp/credentials.txt
    ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
    ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
    ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/tmp/credentials.txt"'
fi

export ZENOH_CONFIG_OVERRIDE
echo "[collector] ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"

# ─── Wait for Gazebo + ROS 2 to be ready ─────────────────────────────────────
# The eval container takes ~15-20 s to advertise the camera topics.
# The collector script has its own 60 s wait, but a small upfront sleep avoids
# Zenoh connection errors during the router's own startup window.
echo "[collector] Waiting ${COLLECTOR_INIT_DELAY:-20}s for eval container to initialize..."
sleep "${COLLECTOR_INIT_DELAY:-20}"

# ─── Pre-activate the lazy TF bridge chain ───────────────────────────────────
# The ros_gz_bridge for /task_board/pose_static -> /scoring/tf has (Lazy 1).
# The ground_truth relay (/scoring/tf -> /tf) is also lazy.
# Both subscribe only when a downstream consumer exists.
# We must open a /tf subscription BEFORE spawning the task board so that
# the bridge is already subscribed when PosePublisher starts its 1 Hz publish.
echo "[collector] Pre-activating lazy TF bridge chain (subscribing to /tf)..."
ros2 topic echo /tf --no-arr > /dev/null 2>&1 &
LAZY_SUB_PID=$!
sleep 3

# ─── Spawn the task board with all required mounts ───────────────────────────
# aic_gz_bringup.launch.py does NOT forward nic_card_mount_N_present / sc_port_N_present
# to spawn_task_board.launch.py, so we spawn it here directly.
# The collector shares eval's network namespace → /gz_server/spawn_entity is reachable.
echo "[collector] Spawning task board with all mounts..."
ros2 launch aic_bringup spawn_task_board.launch.py \
  task_board_x:=0.15 \
  task_board_y:=-0.2 \
  task_board_z:=1.14 \
  task_board_roll:=0.0 \
  task_board_pitch:=0.0 \
  task_board_yaw:=3.1415 \
  nic_card_mount_0_present:=true \
  nic_card_mount_1_present:=true \
  nic_card_mount_2_present:=true \
  nic_card_mount_3_present:=true \
  nic_card_mount_4_present:=true \
  sc_port_0_present:=true \
  sc_port_1_present:=true
echo "[collector] Task board spawned. Waiting 5 s for TF to propagate..."
sleep 5

# Kill the pre-activator; the collector's own TransformListener takes over.
kill "$LAZY_SUB_PID" 2>/dev/null || true

# ─── Run the collection script ────────────────────────────────────────────────
echo "[collector] Starting dataset collection. Args: $*"
exec python3 /workspace/collect_keypoint_dataset.py "$@"
