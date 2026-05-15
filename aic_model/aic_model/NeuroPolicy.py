import numpy as np

from aic_model.insertion_state_machine import InsertionFSM, State
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException


class NeuroPolicy(Policy):
    """Two-phase insertion policy.

    Phase 1 (approach): uses TF ground truth to descend toward the port.
                        Only works with ground_truth:=true.
                        TODO(Juanse): replace with ACT inference for eval.
    Phase 2 (insertion): FSM driven by Axia80 F/T at 30 Hz.
                         TODO(Alma): wire bayesian_confident from BayesianEstimator.
    """

    APPROACH_Z_START = 0.15   # m above port to start descent
    APPROACH_Z_STOP  = 0.005  # m above port to stop and hand off to FSM
    APPROACH_STEP    = 0.001  # m per step
    APPROACH_DT      = 0.05   # s between steps
    STABILIZE_SEC    = 2.0

    def __init__(self, parent_node):
        self._task = None
        self._fsm  = None
        super().__init__(parent_node)

    def _wait_for_tf(self, target: str, source: str, timeout: float = 10.0) -> bool:
        start = self.time_now()
        attempt = 0
        while (self.time_now() - start) < Duration(seconds=timeout):
            try:
                self._parent_node._tf_buffer.lookup_transform(target, source, Time())
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(f"Waiting for TF {source} -> {target}...")
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(f"TF {source} not available after {timeout}s")
        return False

    def _ft_array(self, obs) -> np.ndarray:
        w = obs.wrist_wrench.wrench
        return np.array([w.force.x, w.force.y, w.force.z,
                         w.torque.x, w.torque.y, w.torque.z], dtype=float)

    def _now_sec(self) -> float:
        return self.time_now().nanoseconds * 1e-9

    def _gripper_pose(self) -> Pose | None:
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time())
            t, r = tf.transform.translation, tf.transform.rotation
            return Pose(position=Point(x=t.x, y=t.y, z=t.z),
                        orientation=Quaternion(w=r.w, x=r.x, y=r.y, z=r.z))
        except TransformException:
            return None

    def _approach_to_port(self, move_robot, send_feedback) -> bool:
        port_frame  = f"task_board/{self._task.target_module_name}/{self._task.port_name}_link"
        cable_frame = f"{self._task.cable_name}/{self._task.plug_name}_link"

        for frame in [port_frame, cable_frame]:
            if not self._wait_for_tf("base_link", frame):
                self.get_logger().error(
                    "TF unavailable. Run with ground_truth:=true for training.")
                return False

        try:
            port_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time())
        except TransformException as ex:
            self.get_logger().error(f"Port TF lookup failed: {ex}")
            return False

        pos, rot = port_tf.transform.translation, port_tf.transform.rotation
        current  = self._gripper_pose()
        if current is None:
            return False

        send_feedback("approach: moving above port")
        target_z = pos.z + self.APPROACH_Z_START
        for i in range(60):
            frac = (i + 1) / 60.0
            self.set_pose_target(move_robot, Pose(
                position=Point(
                    x=frac * pos.x + (1 - frac) * current.position.x,
                    y=frac * pos.y + (1 - frac) * current.position.y,
                    z=frac * target_z + (1 - frac) * current.position.z),
                orientation=Quaternion(w=rot.w, x=rot.x, y=rot.y, z=rot.z)))
            self.sleep_for(self.APPROACH_DT)

        send_feedback("approach: descending to port")
        z = pos.z + self.APPROACH_Z_START
        while z > pos.z + self.APPROACH_Z_STOP:
            z -= self.APPROACH_STEP
            self.set_pose_target(move_robot, Pose(
                position=Point(x=pos.x, y=pos.y, z=z),
                orientation=Quaternion(w=rot.w, x=rot.x, y=rot.y, z=rot.z)))
            self.sleep_for(self.APPROACH_DT)

        return True

    def _retract_up(self, move_robot, base_pose):
        ref = self._gripper_pose() or base_pose
        if ref is None:
            return
        retract = Pose(
            position=Point(x=ref.position.x, y=ref.position.y, z=ref.position.z + 0.02),
            orientation=ref.orientation)
        for _ in range(10):
            self.set_pose_target(move_robot, retract)
            self.sleep_for(0.05)

    def _run_insertion_fsm(self, get_observation, move_robot, send_feedback) -> bool:
        self._fsm  = InsertionFSM()
        t_start    = self._now_sec()
        base_pose  = self._gripper_pose()

        while not self._fsm.done:
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.033)
                continue

            t     = self._now_sec() - t_start
            ft    = self._ft_array(obs)
            # TODO(Alma): replace True with bayesian_estimator.confident
            state = self._fsm.step(ft, t, bayesian_confident=True)

            send_feedback(f"fsm:{state.name} Fz={ft[2]:.1f}N t={t:.1f}s")

            if state == State.RETRACT:
                self._retract_up(move_robot, base_pose)
                self.sleep_for(0.5)
            elif state in (State.CONTACT, State.EXPLORE):
                delta = self._fsm.get_explore_delta(t)
                if base_pose is not None and np.any(delta != 0):
                    self.set_pose_target(move_robot, Pose(
                        position=Point(
                            x=base_pose.position.x + delta[0],
                            y=base_pose.position.y + delta[1],
                            z=base_pose.position.z),
                        orientation=base_pose.orientation))

            self.sleep_for(0.033)

        return self._fsm.succeeded

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"NeuroPolicy: cable={task.cable_name} port={task.port_name}")
        self._task = task

        send_feedback("phase 1: approach")
        if not self._approach_to_port(move_robot, send_feedback):
            return False

        send_feedback("phase 2: insertion FSM")
        success = self._run_insertion_fsm(get_observation, move_robot, send_feedback)

        if success:
            self.sleep_for(self.STABILIZE_SEC)

        self.get_logger().info(
            f"NeuroPolicy: {'success' if success else 'failed'} "
            f"retries={self._fsm.retries}")
        return success
