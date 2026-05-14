#!/usr/bin/env python3
#
# Copyright (C) 2026 – AIC Qualification Phase
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
collect_keypoint_dataset.py — Dataset A collection for the Keypoint Pose Estimator.

Runs as a standalone ROS 2 node against a live Gazebo simulation that has been
started with ``ground_truth:=true`` and ``start_aic_engine:=false``.

For every sampled scene the node:
  1. Randomises the task-board pose and scene lighting via gz service calls.
  2. Waits for the physics to settle.
  3. Captures synchronised images from the three cameras.
  4. Looks up the ground-truth port pose and the gripper/tcp pose via /tf.
  5. Computes the 6-DoF pose of the port relative to the TCP frame.
  6. Projects 3-D keypoints (defined in the port local frame) into each image.
  7. Saves one JSON metadata file + three PNG files per scene.

Launch the simulation first (Terminal 1):
    distrobox enter -r aic_eval -- /entrypoint.sh \\
        gazebo_gui:=false \\
        ground_truth:=true \\
        start_aic_engine:=false

Then run this script (Terminal 2, inside the same distrobox):
    pixi run python scripts/collect_keypoint_dataset.py \\
        --output_dir ~/aic_datasets/dataset_A \\
        --n_scenes 5000 \\
        --connector_types sfp sc \\
        --vary_board_yaw true
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
import rclpy.qos
from cv_bridge import CvBridge
import message_filters
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from transforms3d.euler import mat2euler
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

# ---------------------------------------------------------------------------
# Connector geometry: 3-D keypoints defined in the port's local frame (metres)
# Convention: origin at the port opening face, +Z pointing outward from board.
# ---------------------------------------------------------------------------

_SFP_KEYPOINTS_LOCAL: np.ndarray = np.array(
    [
        [ 0.000,  0.000,  0.000],  # kp0 – opening centre
        [ 0.007,  0.004,  0.000],  # kp1 – top-right corner
        [-0.007,  0.004,  0.000],  # kp2 – top-left corner
        [-0.007, -0.004,  0.000],  # kp3 – bottom-left corner
        [ 0.007, -0.004,  0.000],  # kp4 – bottom-right corner
        [ 0.000,  0.004,  0.000],  # kp5 – top-edge midpoint
        [ 0.000, -0.004,  0.000],  # kp6 – bottom-edge midpoint
        [ 0.007,  0.000,  0.000],  # kp7 – right-edge midpoint
        [-0.007,  0.000,  0.000],  # kp8 – left-edge midpoint
    ],
    dtype=np.float64,
)

_R_SC = 0.0045  # SC port ~9 mm diameter
_SC_KEYPOINTS_LOCAL: np.ndarray = np.array(
    [
        [0.000,   0.000,  0.000],  # kp0 – centre
        [0.000,   _R_SC,  0.000],  # kp1 – 12 o'clock
        [ _R_SC * math.sin(math.pi / 4),  _R_SC * math.cos(math.pi / 4), 0.0],  # kp2
        [ _R_SC,  0.000,  0.000],  # kp3 – 3 o'clock
        [ _R_SC * math.sin(math.pi / 4), -_R_SC * math.cos(math.pi / 4), 0.0],  # kp4
        [0.000,  -_R_SC,  0.000],  # kp5 – 6 o'clock
        [-_R_SC * math.sin(math.pi / 4), -_R_SC * math.cos(math.pi / 4), 0.0],  # kp6
        [-_R_SC,  0.000,  0.000],  # kp7 – 9 o'clock
        [-_R_SC * math.sin(math.pi / 4),  _R_SC * math.cos(math.pi / 4), 0.0],  # kp8
    ],
    dtype=np.float64,
)

KEYPOINTS_LOCAL: Dict[str, np.ndarray] = {
    "sfp": _SFP_KEYPOINTS_LOCAL,
    "sc":  _SC_KEYPOINTS_LOCAL,
}

# ---------------------------------------------------------------------------
# TF frame names for each connector type
# The script tries each in order and uses the first resolvable one.
# Pattern: task_board/<module_name>/<port_name>_link  (from CheatCode.py)
# ---------------------------------------------------------------------------
PORT_FRAMES: Dict[str, List[str]] = {
    "sfp": [
        "task_board/nic_card_mount_0/sfp_port_0_link",
        "task_board/nic_card_mount_1/sfp_port_0_link",
        "task_board/nic_card_mount_2/sfp_port_0_link",
        "task_board/nic_card_mount_3/sfp_port_0_link",
        "task_board/nic_card_mount_4/sfp_port_0_link",
    ],
    # SC Port: model name = sc_port_N (from task_board.urdf.xacro sc_port_N_present:=true),
    # link name = sc_port_link (from aic_assets/models/SC Port/model.sdf).
    # Full TF frame: task_board/sc_port_N/sc_port_link
    "sc": [
        "task_board/sc_port_0/sc_port_link",
        "task_board/sc_port_1/sc_port_link",
    ],
}

# ---------------------------------------------------------------------------
# Nominal task-board pose (trial_1 of sample_config.yaml)
# ---------------------------------------------------------------------------
_BOARD_NOM: Dict[str, float] = {
    "x": 0.15, "y": -0.20, "z": 1.14,
    "roll": 0.0, "pitch": 0.0, "yaw": math.pi,
}

# Randomisation ranges (docs_neurobot/training_architecture.md §2 / sample_config.yaml)
_BOARD_XY_RANGE = 0.10               # ±0.10 m
_BOARD_YAW_RANGE = math.radians(30)  # ±30°

# Light names from aic_description/world/aic.sdf
_LIGHT_NAMES: List[str] = ["enclosure_light", "ceiling_01", "ceiling_02"]
_INTENSITY_RANGE: Tuple[float, float] = (0.3, 2.5)

GZ_WORLD = "aic_world"
TCP_FRAME = "gripper/tcp"
BASE_FRAME = "base_link"


# ---------------------------------------------------------------------------
# Gazebo service helpers (via gz CLI – avoids gz-transport Python bindings)
# ---------------------------------------------------------------------------

def _gz_service(
    service: str,
    reqtype: str,
    reptype: str,
    req: str,
    timeout_ms: int = 2000,
) -> bool:
    """Invoke a Gazebo Transport service through the ``gz service`` CLI.

    Returns True when the call succeeds, False otherwise.  Errors are printed
    to stderr but never raised so that a failed randomisation does not abort
    the collection loop.
    """
    cmd = [
        "gz", "service",
        "-s", service,
        "--reqtype", reqtype,
        "--reptype", reptype,
        "--timeout", str(timeout_ms),
        "--req", req,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_ms / 1000.0 + 3.0,
        )
        if result.returncode != 0:
            print(f"[gz service] {service} failed: {result.stderr.strip()}", flush=True)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[gz service] {service} timed out", flush=True)
        return False


def randomise_task_board(
    rng: np.random.Generator,
    vary_yaw: bool,
    vary_xy: bool,
) -> bool:
    """Move the task board to a random pose within the training ranges.

    Uses ``/world/<world>/set_pose_vector`` (gz.msgs.Pose_V).
    """
    dx = rng.uniform(-_BOARD_XY_RANGE, _BOARD_XY_RANGE) if vary_xy else 0.0
    dy = rng.uniform(-_BOARD_XY_RANGE, _BOARD_XY_RANGE) if vary_xy else 0.0
    dyaw = rng.uniform(-_BOARD_YAW_RANGE, _BOARD_YAW_RANGE) if vary_yaw else 0.0

    x = _BOARD_NOM["x"] + dx
    y = _BOARD_NOM["y"] + dy
    z = _BOARD_NOM["z"]
    yaw = _BOARD_NOM["yaw"] + dyaw

    # transforms3d uses [w, x, y, z] order for quaternions
    w, qx, qy, qz = euler2quat(0.0, 0.0, yaw)

    req = (
        f'pose: [{{name: "task_board", '
        f'position: {{x: {x:.6f}, y: {y:.6f}, z: {z:.6f}}}, '
        f'orientation: {{w: {w:.6f}, x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}}}}}]'
    )
    return _gz_service(
        f"/world/{GZ_WORLD}/set_pose_vector",
        "gz.msgs.Pose_V",
        "gz.msgs.Boolean",
        req,
    )


def randomise_lights(rng: np.random.Generator) -> None:
    """Randomise intensity and white-balance of each scene light."""
    for light_name in _LIGHT_NAMES:
        intensity = float(rng.uniform(_INTENSITY_RANGE[0], _INTENSITY_RANGE[1]))
        # Slight colour temperature variation (near white)
        r = float(rng.uniform(0.85, 1.0))
        g = float(rng.uniform(0.85, 1.0))
        b = float(rng.uniform(0.85, 1.0))
        req = (
            f'name: "{light_name}" '
            f'intensity: {intensity:.4f} '
            f'diffuse: {{r: {r:.4f}, g: {g:.4f}, b: {b:.4f}, a: 1.0}}'
        )
        _gz_service(
            f"/world/{GZ_WORLD}/light_config",
            "gz.msgs.Light",
            "gz.msgs.Boolean",
            req,
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _tf_stamped_to_mat4(tf_stamped) -> np.ndarray:
    """Convert a geometry_msgs/TransformStamped to a 4×4 homogeneous matrix."""
    t = tf_stamped.transform
    # transforms3d quat convention: [w, x, y, z]
    R = quat2mat([t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z])
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R
    mat[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return mat


def _mat4_inv_rigid(mat: np.ndarray) -> np.ndarray:
    """Analytically invert a rigid-body 4×4 matrix (no general inverse)."""
    inv = np.eye(4, dtype=np.float64)
    R = mat[:3, :3]
    t = mat[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ t)
    return inv


def compute_relative_pose_6d(
    port_in_base: np.ndarray,
    tcp_in_base: np.ndarray,
) -> List[float]:
    """Return [dx, dy, dz, dRx, dRy, dRz] of the port expressed in TCP frame."""
    base_in_tcp = _mat4_inv_rigid(tcp_in_base)
    port_in_tcp = base_in_tcp @ port_in_base
    dx, dy, dz = port_in_tcp[:3, 3].tolist()
    rx, ry, rz = mat2euler(port_in_tcp[:3, :3], axes="sxyz")
    return [float(dx), float(dy), float(dz), float(rx), float(ry), float(rz)]


def project_keypoints_to_image(
    kp_in_base: np.ndarray,   # (N, 3) – keypoints in base_link frame
    cam_in_base: np.ndarray,  # 4×4 – camera optical frame pose in base_link
    K: np.ndarray,            # 3×3 – intrinsic matrix
    dist: np.ndarray,         # (D,) – distortion coefficients
) -> np.ndarray:
    """Project 3-D keypoints into a camera image.

    Returns an (N, 2) array of [u, v] pixel coordinates.
    Points that project behind the camera are returned as [-1, -1].
    """
    base_in_cam = _mat4_inv_rigid(cam_in_base)
    R = base_in_cam[:3, :3]
    tvec = base_in_cam[:3, 3]

    rvec, _ = cv2.Rodrigues(R)
    pts = kp_in_base.astype(np.float64)

    projected, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    uv = projected.reshape(-1, 2)  # (N, 2)

    # Invalidate points behind the camera plane
    pts_in_cam = (R @ pts.T).T + tvec
    uv[pts_in_cam[:, 2] <= 0.0] = -1.0

    return uv


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class KeypointCollectorNode(Node):
    """Subscribes to the three cameras + TF and provides image/pose snapshots."""

    def __init__(self) -> None:
        super().__init__("keypoint_dataset_collector")
        self._bridge = CvBridge()
        self._lock = Lock()

        self._imgs: Dict[str, Optional[np.ndarray]] = {
            "left": None, "center": None, "right": None,
        }
        self._cam_infos: Dict[str, Optional[CameraInfo]] = {
            "left": None, "center": None, "right": None,
        }
        self._img_fresh = False

        # TF buffer + listener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Synchronised image subscribers
        subs = [
            message_filters.Subscriber(self, Image, f"/{c}_camera/image")
            for c in ("left", "center", "right")
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=10, slop=0.15
        )
        self._sync.registerCallback(self._on_images)

        # Camera info – one QoS-best-available subscription per camera
        for cam in ("left", "center", "right"):
            self.create_subscription(
                CameraInfo,
                f"/{cam}_camera/camera_info",
                lambda msg, c=cam: self._on_camera_info(c, msg),
                rclpy.qos.qos_profile_sensor_data,
            )

        self.get_logger().info("KeypointCollectorNode ready.")

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_camera_info(self, cam: str, msg: CameraInfo) -> None:
        with self._lock:
            if self._cam_infos[cam] is None:
                self._cam_infos[cam] = msg
                self.get_logger().info(
                    f"Camera info received for '{cam}' "
                    f"(frame: {msg.header.frame_id})"
                )

    def _on_images(self, left: Image, center: Image, right: Image) -> None:
        with self._lock:
            self._imgs["left"]   = self._bridge.imgmsg_to_cv2(left,   "bgr8")
            self._imgs["center"] = self._bridge.imgmsg_to_cv2(center, "bgr8")
            self._imgs["right"]  = self._bridge.imgmsg_to_cv2(right,  "bgr8")
            self._img_fresh = True

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def wait_for_camera_infos(self, timeout_sec: float = 30.0) -> bool:
        """Spin until all three camera infos have been received."""
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._lock:
                if all(v is not None for v in self._cam_infos.values()):
                    return True
        return False

    def capture_images(self, timeout_sec: float = 5.0) -> Optional[Dict[str, np.ndarray]]:
        """Block until a fresh synchronised frame set arrives.

        Returns a dict {left, center, right} → BGR numpy arrays, or None on timeout.
        """
        with self._lock:
            self._img_fresh = False  # force a genuinely new frame

        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            with self._lock:
                if self._img_fresh and all(v is not None for v in self._imgs.values()):
                    return {k: v.copy() for k, v in self._imgs.items()}
        return None

    def find_port_frame(self, connector_type: str) -> Optional[str]:
        """Return the first TF frame for *connector_type* that is currently resolvable."""
        for frame in PORT_FRAMES.get(connector_type, []):
            try:
                self._tf_buffer.lookup_transform(BASE_FRAME, frame, RclpyTime())
                return frame
            except TransformException:
                pass

        # One extra spin to allow the TF tree to propagate, then retry once
        rclpy.spin_once(self, timeout_sec=1.0)
        for frame in PORT_FRAMES.get(connector_type, []):
            try:
                self._tf_buffer.lookup_transform(BASE_FRAME, frame, RclpyTime())
                return frame
            except TransformException:
                pass
        return None

    def lookup_mat4(
        self,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = 5.0,
    ) -> Optional[np.ndarray]:
        """Look up *source_frame* in *target_frame* and return a 4×4 matrix."""
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            try:
                tf_s = self._tf_buffer.lookup_transform(
                    target_frame, source_frame, RclpyTime()
                )
                return _tf_stamped_to_mat4(tf_s)
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn(
            f"TF lookup timeout: '{source_frame}' → '{target_frame}'"
        )
        return None

    def get_camera_info(self, cam: str) -> Optional[CameraInfo]:
        with self._lock:
            return self._cam_infos[cam]


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(args: argparse.Namespace) -> None:  # noqa: C901  (complexity OK here)
    rclpy.init()
    node = KeypointCollectorNode()
    log = node.get_logger()
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    connector_types: List[str] = args.connector_types

    # ------------------------------------------------------------------
    # Wait for the simulation to be ready
    # ------------------------------------------------------------------
    log.info("Waiting for camera infos (simulation must be running with ground_truth:=true)…")
    if not node.wait_for_camera_infos(timeout_sec=60.0):
        log.error(
            "Camera infos not received within 60 s. "
            "Is the simulation running with ground_truth:=true?"
        )
        node.destroy_node()
        rclpy.shutdown()
        return

    # Give the TF tree time to populate ground-truth transforms
    log.info("Camera infos ready – waiting 3 s for TF tree to populate…")
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)
    log.info("Starting collection loop.")

    sample_id = 0
    stats: Dict[str, int] = {ct: 0 for ct in connector_types}

    for scene_idx in range(args.n_scenes):
        # ------------------------------------------------------------------
        # 1. Randomise the scene
        # ------------------------------------------------------------------
        randomise_task_board(rng, vary_yaw=args.vary_board_yaw, vary_xy=True)
        randomise_lights(rng)

        # Allow physics to settle after moving entities
        settle_start = time.time()
        while time.time() - settle_start < args.settle_time:
            rclpy.spin_once(node, timeout_sec=0.05)

        # ------------------------------------------------------------------
        # 2. Collect one sample per requested connector type
        # ------------------------------------------------------------------
        for connector_type in connector_types:
            port_frame = node.find_port_frame(connector_type)
            if port_frame is None:
                log.warn(
                    f"[scene {scene_idx}] No TF frame available for '{connector_type}', skipping."
                )
                continue

            # Pose lookups
            tcp_mat = node.lookup_mat4(BASE_FRAME, TCP_FRAME)
            port_mat = node.lookup_mat4(BASE_FRAME, port_frame)
            if tcp_mat is None or port_mat is None:
                log.warn(
                    f"[scene {scene_idx}] TF lookup failed for '{connector_type}', skipping."
                )
                continue

            # Relative pose: port expressed in TCP frame
            pose_rel = compute_relative_pose_6d(port_mat, tcp_mat)

            # Transform port keypoints to base_link frame
            kp_local = KEYPOINTS_LOCAL[connector_type]               # (N, 3)
            kp_hom   = np.hstack([kp_local, np.ones((len(kp_local), 1))])  # (N, 4)
            kp_base  = (port_mat @ kp_hom.T).T[:, :3]              # (N, 3)

            # Capture a fresh synchronised frame
            images = node.capture_images(timeout_sec=5.0)
            if images is None:
                log.warn(
                    f"[scene {scene_idx}] Image capture timeout for '{connector_type}', skipping."
                )
                continue

            # ------------------------------------------------------------------
            # 3. Project keypoints into every camera
            # ------------------------------------------------------------------
            kp_2d: Dict[str, List[List[float]]] = {}
            for cam in ("left", "center", "right"):
                info = node.get_camera_info(cam)
                if info is None:
                    kp_2d[cam] = [[-1.0, -1.0]] * len(kp_local)
                    continue

                # The camera_info header carries the optical-frame name
                cam_frame = info.header.frame_id
                cam_mat = node.lookup_mat4(BASE_FRAME, cam_frame, timeout_sec=3.0)
                if cam_mat is None:
                    kp_2d[cam] = [[-1.0, -1.0]] * len(kp_local)
                    continue

                K    = np.array(info.k, dtype=np.float64).reshape(3, 3)
                dist = np.array(info.d, dtype=np.float64)
                uv   = project_keypoints_to_image(kp_base, cam_mat, K, dist)
                kp_2d[cam] = uv.tolist()

            # ------------------------------------------------------------------
            # 4. Save images + JSON
            # ------------------------------------------------------------------
            sid = f"{sample_id:07d}"
            img_rel_paths: Dict[str, str] = {}
            for cam, bgr in images.items():
                fname = f"{sid}_{cam}.png"
                cv2.imwrite(str(images_dir / fname), bgr)
                img_rel_paths[cam] = str(Path("images") / fname)

            record = {
                "sample_id":        sample_id,
                "scene_idx":        scene_idx,
                "connector_type":   connector_type,
                "port_frame":       port_frame,
                "images": {
                    "left":   img_rel_paths["left"],
                    "center": img_rel_paths["center"],
                    "right":  img_rel_paths["right"],
                },
                # 2-D projections — (N, 2) lists of [u, v] pixel coords
                "keypoints_left":   kp_2d["left"],
                "keypoints_center": kp_2d["center"],
                "keypoints_right":  kp_2d["right"],
                # 6-D relative pose [dx, dy, dz, dRx, dRy, dRz] (port in TCP frame)
                "pose_relative_tcp": pose_rel,
                # Optional extras for debugging / verification
                "tcp_pose_in_base":  tcp_mat[:3, :].tolist(),
                "port_pose_in_base": port_mat[:3, :].tolist(),
            }

            json_path = output_dir / f"{sid}.json"
            with open(json_path, "w") as fh:
                json.dump(record, fh, indent=2)

            stats[connector_type] += 1
            sample_id += 1

            if sample_id % 50 == 0:
                progress = sample_id / (args.n_scenes * len(connector_types)) * 100
                log.info(
                    f"[{sample_id:6d} samples | {progress:5.1f}%] "
                    f"scene={scene_idx} type={connector_type} "
                    f"port={port_frame}"
                )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log.info(
        f"Collection complete. {sample_id} samples saved to {output_dir}\n"
        + "\n".join(f"  {ct}: {n}" for ct, n in stats.items())
    )
    node.destroy_node()
    rclpy.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where JSON and PNG files will be saved.",
    )
    parser.add_argument(
        "--n_scenes",
        type=int,
        default=5000,
        help="Number of randomised scenes to collect (default: 5000).",
    )
    parser.add_argument(
        "--connector_types",
        nargs="+",
        default=["sfp", "sc"],
        choices=["sfp", "sc"],
        help="Connector types to collect per scene (default: sfp sc).",
    )
    parser.add_argument(
        "--vary_board_yaw",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=True,
        help="Randomise task-board yaw within ±30° (default: true).",
    )
    parser.add_argument(
        "--settle_time",
        type=float,
        default=0.5,
        help="Seconds to wait after each scene change for physics to settle (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible scene sampling (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    collect(_parse_args())
