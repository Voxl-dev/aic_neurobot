#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import time
import json
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from typing import Callable, Dict, Any, List
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from geometry_msgs.msg import Twist, Vector3
from tf2_ros import Buffer, TransformException, TransformListener

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from geometry_msgs.msg import Wrench

# LeRobot & Safetensors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from aic_example_policies.ros.keypoint_step4 import (
    KeypointEstimatorBank,
    estimate_pose_relative_tcp_from_multiview,
    pose_msg_to_mat4,
    transform_stamped_to_mat4,
)


class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parent_node = parent_node
        camera_output_dir = os.environ.get("AIC_CAMERA_OUTPUT_DIR")
        self.camera_data_dir = (
            Path(camera_output_dir) if camera_output_dir else Path.cwd() / "aic_camera_data"
        )
        self.camera_frame_index = 0
        self.step4_failure_count = 0
        for camera_name in ("left", "center", "right"):
            (self.camera_data_dir / camera_name).mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"Camera images will be saved to {self.camera_data_dir}")

        # -------------------------------------------------------------------------
        # 1. Configuration & Weights Loading
        # -------------------------------------------------------------------------
        repo_id = "grkw/aic_act_policy"

        # Path to your checkpoint folder
        policy_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["config.json", "model.safetensors", "*.safetensors"],
            )
        )

        # Load Config Manually (Fixes 'Draccus' error by removing unknown 'type' field)
        with open(policy_path / "config.json", "r") as f:
            config_dict = json.load(f)
            if "type" in config_dict:
                del config_dict["type"]

        config = draccus.decode(ACTConfig, config_dict)

        # Load Policy Architecture & Weights
        self.policy = ACTPolicy(config)
        model_weights_path = policy_path / "model.safetensors"
        self.policy.load_state_dict(load_file(model_weights_path))
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info(f"ACT Policy loaded on {self.device} from {policy_path}")

        # -------------------------------------------------------------------------
        # 2. Normalization Stats Loading
        # -------------------------------------------------------------------------
        stats_path = (
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        stats = load_file(stats_path)

        # Helper to extract and shape stats for broadcasting
        def get_stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        # Image Stats (1, 3, 1, 1) for broadcasting against (Batch, Channel, Height, Width)
        self.img_stats = {
            "left": {
                "mean": get_stat("observation.images.left_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.left_camera.std", (1, 3, 1, 1)),
            },
            "center": {
                "mean": get_stat("observation.images.center_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.center_camera.std", (1, 3, 1, 1)),
            },
            "right": {
                "mean": get_stat("observation.images.right_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.right_camera.std", (1, 3, 1, 1)),
            },
        }
        print(f"Image stats: {self.img_stats}")

        # Robot State Stats (1, 26)
        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        print(f"Robot state mean: {self.state_mean}")
        print(f"Robot state std: {self.state_std}")
        self.expected_state_dim = int(self.state_mean.shape[1])

        # Action Stats (1, 7) - Used for Un-normalization
        self.action_mean = get_stat("action.mean", (1, -1))
        self.action_std = get_stat("action.std", (1, -1))
        print(f"Action mean: {self.action_mean}")
        print(f"Action std: {self.action_std}")

        # Config
        self.image_scaling = 0.25  # Must match AICRobotAICControllerConfig
        self.keypoint_step4_enabled = False
        self.keypoint_estimator = None
        self.tf_buffer = None
        self.tf_listener = None
        self._configure_keypoint_step4()

        self.get_logger().info("Normalization statistics loaded successfully.")

    def _configure_keypoint_step4(self) -> None:
        """Optionally enable keypoint -> pose_relative_tcp -> 32D state."""
        enable_env = os.environ.get("AIC_ENABLE_KEYPOINT_STEP4", "1").lower()
        if enable_env in ("0", "false", "no", "off"):
            self.get_logger().info("Keypoint Step 4 disabled by AIC_ENABLE_KEYPOINT_STEP4.")
            return

        if self.expected_state_dim < 32:
            self.get_logger().warn(
                "Keypoint Step 4 is available but the loaded ACT normalizer expects "
                f"{self.expected_state_dim} state dims, not 32. Running legacy state."
            )
            return

        checkpoint_paths = {
            "sfp": Path(
                os.environ.get(
                    "AIC_KEYPOINT_SFP_CHECKPOINT",
                    "outputs/keypoint_estimator/sfp/best_sfp.pt",
                )
            ),
            "sc": Path(
                os.environ.get(
                    "AIC_KEYPOINT_SC_CHECKPOINT",
                    "outputs/keypoint_estimator/sc/best_sc.pt",
                )
            ),
        }
        self.keypoint_estimator = KeypointEstimatorBank(
            checkpoint_paths=checkpoint_paths,
            device=str(self.device),
        )
        available = self.keypoint_estimator.available_connector_types()
        if not available:
            self.get_logger().warn(
                "Keypoint Step 4 expected a 32D state, but no keypoint checkpoints "
                "were found. The last 6 state dims will be zero until checkpoints exist."
            )
        else:
            self.get_logger().info(
                f"Keypoint Step 4 loaded checkpoints for: {', '.join(available)}"
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.parent_node)
        self.keypoint_step4_enabled = True

    @staticmethod
    def _ros_image_to_numpy(raw_img) -> np.ndarray:
        """Convert a ROS Image message into a uint8 HWC numpy array."""
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        return img_np

    @staticmethod
    def _img_to_tensor(
        raw_img,
        device: torch.device,
        scale: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Converts ROS Image -> Resized -> Permuted -> Normalized Tensor."""
        # 1. Bytes to Numpy (H, W, C)
        img_np = RunACT._ros_image_to_numpy(raw_img)

        # 2. Resize
        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        # 3. To Tensor -> Permute (HWC -> CHW) -> Float -> Div(255) -> Batch Dim
        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

        # 4. Normalize (Apply Mean/Std)
        # Formula: (x - mean) / std
        return (tensor - mean) / std

    def save_camera_images(self, obs_msg: Observation) -> None:
        """Save the raw camera images from the current observation."""
        timestamp_sec = (
            obs_msg.center_image.header.stamp.sec
            + obs_msg.center_image.header.stamp.nanosec / 1e9
        )

        for camera_name, raw_img in (
            ("left", obs_msg.left_image),
            ("center", obs_msg.center_image),
            ("right", obs_msg.right_image),
        ):
            img_np = self._ros_image_to_numpy(raw_img)
            filename = f"{self.camera_frame_index:06d}_{timestamp_sec:.9f}.png"
            image_path = self.camera_data_dir / camera_name / filename
            cv2.imwrite(str(image_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            metadata = {
                "frame_index": self.camera_frame_index,
                "camera": camera_name,
                "timestamp_sec": timestamp_sec,
                "height": raw_img.height,
                "width": raw_img.width,
                "encoding": raw_img.encoding,
                "path": str(image_path),
            }
            with open(self.camera_data_dir / "metadata.jsonl", "a") as f:
                f.write(json.dumps(metadata) + "\n")

        self.camera_frame_index += 1

    def _lookup_camera_poses_in_base(self, obs_msg: Observation) -> Dict[str, np.ndarray]:
        """Return camera optical frame poses in base_link, keyed by camera name."""
        if self.tf_buffer is None:
            return {}

        camera_infos = {
            "left": obs_msg.left_camera_info,
            "center": obs_msg.center_camera_info,
            "right": obs_msg.right_camera_info,
        }
        camera_poses = {}
        for camera_name, camera_info in camera_infos.items():
            frame_id = camera_info.header.frame_id
            if not frame_id:
                continue
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    "base_link",
                    frame_id,
                    RclpyTime(),
                )
            except TransformException:
                continue
            camera_poses[camera_name] = transform_stamped_to_mat4(tf_msg)
        return camera_poses

    def _estimate_visual_pose_relative_tcp(
        self,
        obs_msg: Observation,
        connector_type: str,
    ) -> np.ndarray:
        """Estimate 6D port pose in TCP frame from RGB keypoints.

        Returns zeros when Step 4 is unavailable. This keeps runtime behavior
        deterministic while the keypoint checkpoints are still being produced.
        """
        fallback = np.zeros(6, dtype=np.float32)
        if not self.keypoint_step4_enabled or self.keypoint_estimator is None:
            return fallback
        if not self.keypoint_estimator.has_model(connector_type):
            return fallback

        images_rgb = {
            "left": self._ros_image_to_numpy(obs_msg.left_image),
            "center": self._ros_image_to_numpy(obs_msg.center_image),
            "right": self._ros_image_to_numpy(obs_msg.right_image),
        }
        keypoints_by_camera = self.keypoint_estimator.predict_multicamera(
            connector_type,
            images_rgb,
        )
        camera_infos = {
            "left": obs_msg.left_camera_info,
            "center": obs_msg.center_camera_info,
            "right": obs_msg.right_camera_info,
        }
        camera_poses = self._lookup_camera_poses_in_base(obs_msg)
        tcp_in_base = pose_msg_to_mat4(obs_msg.controller_state.tcp_pose)

        estimate = estimate_pose_relative_tcp_from_multiview(
            keypoints_by_camera=keypoints_by_camera,
            connector_type=connector_type,
            camera_infos=camera_infos,
            camera_in_base_by_camera=camera_poses,
            tcp_in_base=tcp_in_base,
        )
        if estimate is None:
            self.step4_failure_count += 1
            if self.step4_failure_count <= 3 or self.step4_failure_count % 20 == 0:
                self.get_logger().warn(
                    "Keypoint Step 4 could not estimate pose_relative_tcp; "
                    "using zeros for the 6 visual state dims."
                )
            return fallback

        return estimate.pose_relative_tcp.astype(np.float32)

    def prepare_observations(
        self,
        obs_msg: Observation,
        task: Task = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert ROS Observation message into dictionary of normalized tensors."""

        # --- Process Cameras ---
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.device,
                self.image_scaling,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.device,
                self.image_scaling,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.device,
                self.image_scaling,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

        # --- Process Robot State ---
        # Base proprioceptive state (26 dims), then optional Step 4 visual pose
        # appends pose_relative_tcp [dx, dy, dz, dRx, dRy, dRz] to reach 32 dims.
        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity

        base_state_np = np.array(
            [
                # TCP Position (3)
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                # TCP Orientation (4)
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                # TCP Linear Vel (3)
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                # TCP Angular Vel (3)
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                # TCP Error (6)
                *obs_msg.controller_state.tcp_error,
                # Joint Positions (7)
                *obs_msg.joint_states.position[:7],
            ],
            dtype=np.float32,
        )

        if self.expected_state_dim >= 32:
            connector_type = "sfp"
            if task is not None and task.port_type in ("sfp", "sc"):
                connector_type = task.port_type
            visual_pose_rel = self._estimate_visual_pose_relative_tcp(
                obs_msg,
                connector_type=connector_type,
            )
            state_np = np.concatenate([base_state_np, visual_pose_rel]).astype(np.float32)
            if self.expected_state_dim > state_np.shape[0]:
                pad = np.zeros(self.expected_state_dim - state_np.shape[0], dtype=np.float32)
                state_np = np.concatenate([state_np, pad]).astype(np.float32)
        else:
            state_np = base_state_np

        # Normalize State
        raw_state_tensor = (
            torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        )
        obs["observation.state"] = (raw_state_tensor - self.state_mean) / self.state_std

        return obs

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(f"RunACT.insert_cable() enter. Task: {task}")

        start_time = time.time()

        # Run inference for 30 seconds
        while time.time() - start_time < 30.0:
            loop_start = time.time()

            # 1. Get & Process Observation
            observation_msg = get_observation()

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            self.save_camera_images(observation_msg)
            obs_tensors = self.prepare_observations(observation_msg, task=task)

            # 2. Model Inference
            with torch.inference_mode():
                # returns shape [1, 7] (first action of chunk)
                normalized_action = self.policy.select_action(obs_tensors)

            # 3. Un-normalize Action
            # Formula: (norm * std) + mean
            raw_action_tensor = (normalized_action * self.action_std) + self.action_mean

            # 4. Extract and Command
            # raw_action_tensor is [1, 7], taking [0] gives vector of 7
            action = raw_action_tensor[0].cpu().numpy()

            self.get_logger().info(f"Action: {action}")

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]), y=float(action[1]), z=float(action[2])
                ),
                angular=Vector3(
                    x=float(action[3]), y=float(action[4]), z=float(action[5])
                ),
            )
            motion_update = self.set_cartesian_twist_target(twist)
            move_robot(motion_update=motion_update)
            send_feedback("in progress...")

            # Maintain control rate (approx 4Hz loop = 0.25s sleep)
            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.25 - elapsed))

        self.get_logger().info("RunACT.insert_cable() exiting...")
        return True

    def set_cartesian_twist_target(self, twist: Twist, frame_id: str = "base_link"):
        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = twist
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()

        motion_update_msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update_msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()

        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)
        )

        motion_update_msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )

        return motion_update_msg
