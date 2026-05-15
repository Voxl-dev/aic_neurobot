"""
Production ACT insertion policy for AIC Challenge.

ACT handles ALL motion (approach + insertion). InsertionFSM detects states
(CONTACT, EXPLORE, SUCCESS, RETRACT). BayesianEstimator drives FSM confidence.

Env vars:
  AIC_POLICY_CHECKPOINT  Dir with ACT checkpoint (config.json + model.safetensors)
                         Default: models/act_policy
  AIC_KP_SC              SC keypoint checkpoint .pt
                         Default: models/keypoints/best_sc.pt
  AIC_KP_SFP             SFP keypoint checkpoint .pt
                         Default: models/keypoints/best_sfp.pt
  AIC_IMG_H              Image height fed to ACT (must match training)  Default: 240
  AIC_IMG_W              Image width fed to ACT  (must match training)  Default: 320
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from geometry_msgs.msg import Twist, Vector3, Wrench
from safetensors.torch import load_file as _load_safetensors
from std_msgs.msg import Header

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.bayesian_estimator import Axia80BayesianEstimator
from aic_model.insertion_state_machine import InsertionFSM, State
from aic_model.keypoint_step4 import KeypointEstimatorBank
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


def _import_act_classes():
    """Import ACTPolicy/ACTConfig, trying the production API path first."""
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        return ACTPolicy, ACTConfig
    except ImportError:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
        from lerobot.common.policies.act.configuration_act import ACTConfig
        return ACTPolicy, ACTConfig


class NeuroPolicy(Policy):
    """ACT-based cable insertion policy.

    Approach and insertion are both driven by ACT inference. The InsertionFSM
    monitors F/T signals for CONTACT, SUCCESS (click: dFz + Tz spike), and
    overforce (RETRACT). BayesianEstimator provides alignment confidence.
    """

    _STIFFNESS = [100.0, 100.0, 100.0,  50.0,  50.0,  50.0]
    _DAMPING   = [ 40.0,  40.0,  40.0,  15.0,  15.0,  15.0]

    def __init__(self, parent_node):
        super().__init__(parent_node)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._img_h  = int(os.environ.get("AIC_IMG_H", "240"))
        self._img_w  = int(os.environ.get("AIC_IMG_W", "320"))

        ckpt_dir = Path(os.environ.get("AIC_POLICY_CHECKPOINT", "models/act_policy"))
        self.get_logger().info(f"NeuroPolicy: loading ACT from {ckpt_dir}")
        self._act = self._load_act(ckpt_dir)

        kp_paths = {
            "sc":  Path(os.environ.get("AIC_KP_SC",  "models/keypoints/best_sc.pt")),
            "sfp": Path(os.environ.get("AIC_KP_SFP", "models/keypoints/best_sfp.pt")),
        }
        self._kp = KeypointEstimatorBank(checkpoint_paths=kp_paths, device=str(self._device))
        self.get_logger().info(
            f"NeuroPolicy: keypoints loaded for {self._kp.available_connector_types()}"
        )

        bayesian_dir = Path(os.environ.get("AIC_BAYESIAN_DIR", "/opt/aic/bayesiano"))
        csv_path     = bayesian_dir / "calibration_results.csv"
        try:
            self._bayesian = Axia80BayesianEstimator.from_project_files(
                calibration_csv=csv_path,
                data_dir=bayesian_dir,
            )
            self.get_logger().info(f"NeuroPolicy: Bayesian loaded from {bayesian_dir}")
        except Exception as e:
            self.get_logger().warn(f"Bayesian calibration not found ({e}), using built-in defaults")
            self._bayesian = Axia80BayesianEstimator()

        self.get_logger().info(
            f"NeuroPolicy ready — device={self._device}  "
            f"img={self._img_h}x{self._img_w}"
        )

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_act(self, ckpt_dir: Path):
        ACTPolicy, ACTConfig = _import_act_classes()

        # Prefer from_pretrained (restores normalization stats automatically)
        if hasattr(ACTPolicy, "from_pretrained"):
            try:
                return ACTPolicy.from_pretrained(str(ckpt_dir)).eval().to(self._device)
            except Exception as e:
                self.get_logger().warn(f"from_pretrained failed ({e}), using manual load")

        # Manual load following RunACT.py pattern (compatible with older lerobot)
        with open(ckpt_dir / "config.json") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("type", None)

        try:
            import draccus
            config = draccus.decode(ACTConfig, cfg_dict)
        except (ImportError, Exception):
            import dataclasses
            valid_keys = {f.name for f in dataclasses.fields(ACTConfig)}
            config = ACTConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

        policy = ACTPolicy(config)
        policy.load_state_dict(
            _load_safetensors(str(ckpt_dir / "model.safetensors")),
            strict=False,
        )
        return policy.eval().to(self._device)

    # ── Observation helpers ───────────────────────────────────────────────────

    def _ft_array(self, obs) -> np.ndarray:
        w = obs.wrist_wrench.wrench
        return np.array(
            [w.force.x, w.force.y, w.force.z,
             w.torque.x, w.torque.y, w.torque.z],
            dtype=float,
        )

    def _now_sec(self) -> float:
        return self.time_now().nanoseconds * 1e-9

    @staticmethod
    def _raw_rgb(raw_img) -> np.ndarray:
        return np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )

    def _build_state(self, obs, connector_type: str) -> np.ndarray:
        """Build 32D state vector matching bag_to_lerobot.py format.

        Layout: [arm_joints×6 | gripper×1 | kp_norm_flat×18 | tcp_pos×3 | tcp_quat×4]
        """
        arm  = np.array(obs.joint_states.position[:6], dtype=np.float32)
        grip = float(obs.joint_states.position[6]) if len(obs.joint_states.position) > 6 else 0.0

        center_rgb = self._raw_rgb(obs.center_image)
        kp = self._kp.predict(connector_type, center_rgb)
        if kp is None:
            kp = np.zeros((9, 2), dtype=np.float32)

        # Normalize keypoint pixel coords to [0, 1]
        kp_norm = kp.copy()
        kp_norm[:, 0] /= max(float(obs.center_image.width),  1.0)
        kp_norm[:, 1] /= max(float(obs.center_image.height), 1.0)

        p = obs.controller_state.tcp_pose.position
        q = obs.controller_state.tcp_pose.orientation
        return np.concatenate([
            arm, [grip],
            kp_norm.flatten(),
            [p.x, p.y, p.z],
            [q.x, q.y, q.z, q.w],
        ]).astype(np.float32)

    def _img_tensor(self, raw_img) -> torch.Tensor:
        arr = self._raw_rgb(raw_img)
        if (raw_img.height, raw_img.width) != (self._img_h, self._img_w):
            arr = cv2.resize(arr, (self._img_w, self._img_h), interpolation=cv2.INTER_AREA)
        return (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self._device)
        )

    def _act_batch(self, obs, connector_type: str) -> dict:
        state_t = (
            torch.from_numpy(self._build_state(obs, connector_type))
            .unsqueeze(0)
            .to(self._device)
        )
        return {
            "observation.images.center": self._img_tensor(obs.center_image),
            "observation.images.left":   self._img_tensor(obs.left_image),
            "observation.images.right":  self._img_tensor(obs.right_image),
            "observation.state":         state_t,
        }

    # ── Motion helpers ────────────────────────────────────────────────────────

    def _make_twist_update(self, action_np: np.ndarray) -> MotionUpdate:
        mu = MotionUpdate()
        mu.header = Header(
            frame_id="base_link",
            stamp=self.get_clock().now().to_msg(),
        )
        mu.velocity = Twist(
            linear=Vector3(
                x=float(action_np[0]), y=float(action_np[1]), z=float(action_np[2])
            ),
            angular=Vector3(
                x=float(action_np[3]), y=float(action_np[4]), z=float(action_np[5])
            ),
        )
        mu.target_stiffness = np.diag(self._STIFFNESS).flatten()
        mu.target_damping   = np.diag(self._DAMPING).flatten()
        mu.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        mu.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        mu.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return mu

    def _send_twist(self, move_robot: MoveRobotCallback, action_np: np.ndarray) -> None:
        try:
            move_robot(motion_update=self._make_twist_update(action_np))
        except Exception as ex:
            self.get_logger().warn(f"_send_twist error: {ex}")

    # ── Main entry point ──────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        conn = getattr(task, "port_type", "sfp")
        if conn not in ("sfp", "sc"):
            conn = "sfp"

        self.get_logger().info(
            f"NeuroPolicy: cable={task.cable_name} port={task.port_name} type={conn}"
        )

        self._act.reset()
        self._bayesian.reset()
        fsm = InsertionFSM()
        t0         = self._now_sec()
        time_limit = float(task.time_limit) if task.time_limit > 0 else InsertionFSM.TIMEOUT
        _ZERO      = np.zeros(6, dtype=np.float32)

        while not fsm.done:
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.033)
                continue

            t  = self._now_sec() - t0
            if t >= time_limit:
                self.get_logger().warn(f"NeuroPolicy: time_limit {time_limit:.0f}s exceeded")
                break

            ft = self._ft_array(obs)

            try:
                estimate  = self._bayesian.update(ft)
                confident = estimate.confident
            except Exception:
                confident = True  # fail-open: let FSM run without Bayesian

            state = fsm.step(ft, t, bayesian_confident=confident)
            send_feedback(f"fsm:{state.name} Fz={ft[2]:.1f}N t={t:.1f}s")

            if fsm.done:
                break

            if state == State.RETRACT:
                # Overforce or explore timeout — stop, reset chunk, retry
                self._act.reset()
                self._bayesian.reset()
                self._send_twist(move_robot, _ZERO)
                self.sleep_for(0.5)
            else:
                try:
                    batch = self._act_batch(obs, conn)
                    with torch.inference_mode():
                        action = self._act.select_action(batch)
                    self._send_twist(move_robot, action.squeeze().cpu().numpy())
                except Exception as ex:
                    self.get_logger().warn(f"ACT inference error: {ex}")

            self.sleep_for(0.1)  # 10 Hz

        # Always stop the robot before returning
        self._send_twist(move_robot, _ZERO)

        result = fsm.succeeded
        self.get_logger().info(
            f"NeuroPolicy: {'SUCCESS' if result else 'FAILED'}  retries={fsm.retries}"
        )
        return result
