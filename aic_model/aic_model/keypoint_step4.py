"""Step 4 for the AIC Keypoint Estimator pipeline.

This module turns per-camera 2-D keypoints into a 6-D visual estimate of the
target port pose relative to the robot TCP:

    keypoints 2D -> PnP per camera -> multiview fusion -> pose_relative_tcp

It is intentionally independent from ROS node lifecycle code so it can be used
from NeuroPolicy and from future offline conversion scripts.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np


CAMERAS = ("left", "center", "right")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SFP_KEYPOINTS_LOCAL = np.array(
    [
        [0.000, 0.000, 0.000],
        [0.007, 0.004, 0.000],
        [-0.007, 0.004, 0.000],
        [-0.007, -0.004, 0.000],
        [0.007, -0.004, 0.000],
        [0.000, 0.004, 0.000],
        [0.000, -0.004, 0.000],
        [0.007, 0.000, 0.000],
        [-0.007, 0.000, 0.000],
    ],
    dtype=np.float64,
)

R_SC = 0.0045
SC_KEYPOINTS_LOCAL = np.array(
    [
        [0.000, 0.000, 0.000],
        [0.000, R_SC, 0.000],
        [R_SC * math.sin(math.pi / 4), R_SC * math.cos(math.pi / 4), 0.000],
        [R_SC, 0.000, 0.000],
        [R_SC * math.sin(math.pi / 4), -R_SC * math.cos(math.pi / 4), 0.000],
        [0.000, -R_SC, 0.000],
        [-R_SC * math.sin(math.pi / 4), -R_SC * math.cos(math.pi / 4), 0.000],
        [-R_SC, 0.000, 0.000],
        [-R_SC * math.sin(math.pi / 4), R_SC * math.cos(math.pi / 4), 0.000],
    ],
    dtype=np.float64,
)

KEYPOINTS_LOCAL = {
    "sfp": SFP_KEYPOINTS_LOCAL,
    "sc": SC_KEYPOINTS_LOCAL,
}


@dataclass(frozen=True)
class CameraPoseEstimate:
    camera: str
    port_in_camera: np.ndarray
    port_in_base: np.ndarray
    reprojection_error_px: float
    num_inliers: int


@dataclass(frozen=True)
class Step4PoseEstimate:
    pose_relative_tcp: np.ndarray
    port_in_base: np.ndarray
    camera_estimates: Sequence[CameraPoseEstimate]


def _require_torchvision():
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    models = importlib.import_module("torchvision.models")
    return torch, nn, models


def build_keypoint_model(num_keypoints: int = 9):
    """Build the same MobileNetV3-Small regression model used for training."""
    _, nn, models = _require_torchvision()
    backbone = models.mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[0].in_features
    backbone.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_keypoints * 2),
    )
    return backbone


class KeypointEstimatorBank:
    """Loads one keypoint checkpoint per connector type and runs inference."""

    def __init__(
        self,
        checkpoint_paths: Mapping[str, Path],
        device: str,
    ) -> None:
        self.device = device
        self.models = {}
        self.metadata = {}
        self.torch = None

        for connector_type, checkpoint_path in checkpoint_paths.items():
            checkpoint_path = Path(checkpoint_path).expanduser()
            if not checkpoint_path.exists():
                continue
            self._load_checkpoint(connector_type, checkpoint_path)

    def _load_checkpoint(self, connector_type: str, checkpoint_path: Path) -> None:
        torch, _, _ = _require_torchvision()
        self.torch = torch

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        num_keypoints = int(checkpoint.get("num_keypoints", 9))
        model = build_keypoint_model(num_keypoints=num_keypoints).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.models[connector_type] = model
        self.metadata[connector_type] = {
            "checkpoint": str(checkpoint_path),
            "image_height": int(checkpoint.get("image_height", 256)),
            "image_width": int(checkpoint.get("image_width", 288)),
            "num_keypoints": num_keypoints,
        }

    def available_connector_types(self) -> Sequence[str]:
        return tuple(sorted(self.models.keys()))

    def has_model(self, connector_type: str) -> bool:
        return connector_type in self.models

    def predict(self, connector_type: str, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return keypoints in original image pixel coordinates, shape (9, 2)."""
        if connector_type not in self.models:
            return None

        meta = self.metadata[connector_type]
        image_height = meta["image_height"]
        image_width = meta["image_width"]
        num_keypoints = meta["num_keypoints"]

        resized = cv2.resize(image_rgb, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)

        tensor = self.torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            pred_norm = (
                self.models[connector_type](tensor)
                .detach()
                .cpu()
                .numpy()
                .reshape(num_keypoints, 2)
            )

        orig_height, orig_width = image_rgb.shape[:2]
        pred_xy = pred_norm.copy()
        pred_xy[:, 0] *= float(orig_width)
        pred_xy[:, 1] *= float(orig_height)
        return pred_xy.astype(np.float32)

    def predict_multicamera(
        self,
        connector_type: str,
        images_rgb: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        predictions = {}
        for camera in CAMERAS:
            image = images_rgb.get(camera)
            if image is None:
                continue
            keypoints = self.predict(connector_type, image)
            if keypoints is not None:
                predictions[camera] = keypoints
        return predictions


def camera_info_to_intrinsics(camera_info) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(camera_info.d, dtype=np.float64)
    return K, dist


def transform_stamped_to_mat4(transform_stamped) -> np.ndarray:
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    return translation_quaternion_to_mat4(
        [t.x, t.y, t.z],
        [q.x, q.y, q.z, q.w],
    )


def pose_msg_to_mat4(pose_msg) -> np.ndarray:
    return translation_quaternion_to_mat4(
        [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
        [
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ],
    )


def translation_quaternion_to_mat4(
    translation_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    x, y, z, w = [float(v) for v in quaternion_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    mat[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return mat


def invert_rigid(mat: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    R = mat[:3, :3]
    t = mat[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ t)
    return inv


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-9
    if not singular:
        rx = math.atan2(float(R[2, 1]), float(R[2, 2]))
        ry = math.atan2(float(-R[2, 0]), sy)
        rz = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        rx = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        ry = math.atan2(float(-R[2, 0]), sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def mat4_to_pose6d_relative(source_in_target: np.ndarray) -> np.ndarray:
    xyz = source_in_target[:3, 3]
    rxyz = rotation_matrix_to_euler_xyz(source_in_target[:3, :3])
    return np.concatenate([xyz, rxyz]).astype(np.float32)


def solve_port_pose_in_camera(
    keypoints_xy: np.ndarray,
    connector_type: str,
    K: np.ndarray,
    dist: np.ndarray,
    min_points: int = 6,
) -> Optional[tuple[np.ndarray, float, int]]:
    """Estimate port pose in one camera optical frame using PnP."""
    if connector_type not in KEYPOINTS_LOCAL:
        raise ValueError(f"Unsupported connector_type: {connector_type!r}")

    object_points = KEYPOINTS_LOCAL[connector_type]
    image_points = np.asarray(keypoints_xy, dtype=np.float64)
    if image_points.shape != (len(object_points), 2):
        return None

    valid = (
        np.isfinite(image_points).all(axis=1)
        & (image_points[:, 0] >= 0.0)
        & (image_points[:, 1] >= 0.0)
    )
    if int(valid.sum()) < min_points:
        return None

    obj = object_points[valid]
    img = image_points[valid]

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        K,
        dist,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=8.0,
        iterationsCount=100,
        confidence=0.99,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        num_inliers = len(obj)
    else:
        num_inliers = int(len(inliers)) if inliers is not None else len(obj)
        if hasattr(cv2, "solvePnPRefineLM") and num_inliers >= min_points:
            refine_idx = inliers.reshape(-1) if inliers is not None else np.arange(len(obj))
            rvec, tvec = cv2.solvePnPRefineLM(obj[refine_idx], img[refine_idx], K, dist, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    port_in_camera = np.eye(4, dtype=np.float64)
    port_in_camera[:3, :3] = R
    port_in_camera[:3, 3] = tvec.reshape(3)

    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(projected.reshape(-1, 2) - img, axis=1)
    return port_in_camera, float(err.mean()), num_inliers


def fuse_port_poses_in_base(
    estimates: Sequence[CameraPoseEstimate],
) -> Optional[np.ndarray]:
    if not estimates:
        return None

    errors = np.array([max(e.reprojection_error_px, 1e-3) for e in estimates], dtype=np.float64)
    inliers = np.array([max(e.num_inliers, 1) for e in estimates], dtype=np.float64)
    weights = inliers / errors
    weights /= weights.sum()

    translations = np.stack([e.port_in_base[:3, 3] for e in estimates], axis=0)
    fused_t = (translations * weights[:, None]).sum(axis=0)

    rotvecs = []
    for estimate in estimates:
        rotvec, _ = cv2.Rodrigues(estimate.port_in_base[:3, :3])
        rotvecs.append(rotvec.reshape(3))
    fused_rotvec = (np.stack(rotvecs, axis=0) * weights[:, None]).sum(axis=0)
    fused_R, _ = cv2.Rodrigues(fused_rotvec)

    fused = np.eye(4, dtype=np.float64)
    fused[:3, :3] = fused_R
    fused[:3, 3] = fused_t
    return fused


def estimate_pose_relative_tcp_from_multiview(
    keypoints_by_camera: Mapping[str, np.ndarray],
    connector_type: str,
    camera_infos: Mapping[str, object],
    camera_in_base_by_camera: Mapping[str, np.ndarray],
    tcp_in_base: np.ndarray,
) -> Optional[Step4PoseEstimate]:
    camera_estimates = []

    for camera in CAMERAS:
        keypoints = keypoints_by_camera.get(camera)
        camera_info = camera_infos.get(camera)
        camera_in_base = camera_in_base_by_camera.get(camera)
        if keypoints is None or camera_info is None or camera_in_base is None:
            continue

        K, dist = camera_info_to_intrinsics(camera_info)
        solved = solve_port_pose_in_camera(keypoints, connector_type, K, dist)
        if solved is None:
            continue

        port_in_camera, reprojection_error_px, num_inliers = solved
        port_in_base = camera_in_base @ port_in_camera
        camera_estimates.append(
            CameraPoseEstimate(
                camera=camera,
                port_in_camera=port_in_camera,
                port_in_base=port_in_base,
                reprojection_error_px=reprojection_error_px,
                num_inliers=num_inliers,
            )
        )

    fused_port_in_base = fuse_port_poses_in_base(camera_estimates)
    if fused_port_in_base is None:
        return None

    port_in_tcp = invert_rigid(tcp_in_base) @ fused_port_in_base
    pose_relative_tcp = mat4_to_pose6d_relative(port_in_tcp)
    return Step4PoseEstimate(
        pose_relative_tcp=pose_relative_tcp,
        port_in_base=fused_port_in_base,
        camera_estimates=tuple(camera_estimates),
    )
