# ─────────────────────────────────────────────────────────────────────────────
# docker/aic_collector/Dockerfile
#
# Collector container for Dataset A (Keypoint Pose Estimator).
# Extends the official aic_eval image — which already contains ROS 2 Kilted,
# Gazebo Harmonic, cv_bridge, message_filters, tf2_ros and the gz CLI — and
# adds the only missing Python dependency: transforms3d.
#
# Build context must be the repository root (same as the other Dockerfiles):
#   docker compose -f docker/docker-compose.dataset_a.yaml build collector
# ─────────────────────────────────────────────────────────────────────────────

FROM ghcr.io/intrinsic-dev/aic/aic_eval:latest

# transforms3d is not shipped with the aic_eval image.
# All other deps (cv2, rclpy, cv_bridge, message_filters, tf2_ros, numpy) are
# already present in the ROS 2 Kilted workspace at /ws_aic.
# pip is not present in the base image; install it first.
# --break-system-packages is required on Ubuntu 24.04 (PEP 668).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --break-system-packages transforms3d

# Copy the entrypoint and fix CRLF line endings (safe for Windows/WSL checkouts)
COPY docker/aic_collector/collect_entrypoint.sh /collect_entrypoint.sh
RUN chmod +x /collect_entrypoint.sh \
    && sed -i 's/\r$//' /collect_entrypoint.sh

# The collection script is bind-mounted at runtime (/workspace) so that edits
# on the host are picked up without rebuilding the image.

ENTRYPOINT ["/collect_entrypoint.sh"]

# Default: print help.  Override via compose `command:` or docker run args.
CMD ["--help"]
