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
"""
FTCalibrationSampler — Genera bags de calibración F/T con offsets XY controlados.

Posiciona el cable tip en posiciones fijas respecto al puerto, mantiene cada
posición HOLD_SECONDS segundos y avanza al siguiente offset. Al finalizar la
secuencia deja el robot en la posición home.

Flujo de grabación (idéntico a Dataset B, training_architecture.md §3.2):

  Terminal 1 — Simulación con ground truth:
    distrobox enter -r aic_eval -- /entrypoint.sh \\
      gazebo_gui:=false ground_truth:=true \\
      start_aic_engine:=true shutdown_on_aic_engine_exit:=true \\
      model_discovery_timeout_seconds:=60

  Terminal 2 — Tare del sensor antes de arrancar:
    pixi run ros2 service call \\
      /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger

  Terminal 3 — Grabar (solo los topics que usa la calibración):
    pixi run ros2 bag record \\
      -o ~/aic_datasets/ft_calib_$(date +%Y%m%d_%H%M%S) \\
      /fts_broadcaster/wrench /joint_states /tf /tf_static

  Terminal 4 — Ejecutar esta policy:
    pixi run ros2 run aic_model aic_model --ros-args \\
      -p use_sim_time:=true \\
      -p policy:=aic_example_policies.ros.FTCalibrationSampler

El bag resultante puede procesarse con tools/calibrate_ft.py para obtener
GAIN_X, GAIN_Y, GAIN_Rz y los BIAS correspondientes.

Offsets que recorre (frame base_link, plano XY):
  1. free_space  — tip a 12 cm sobre el puerto, sin contacto  → ancora BIAS
  2. y_plus_15mm — 15 mm en +Y con contacto
  3. y_plus_10mm — 10 mm en +Y con contacto
  4. y_plus_5mm  —  5 mm en +Y con contacto
  5. y_nominal   — posición nominal, sin offset lateral
  6. y_minus_5mm —  5 mm en −Y con contacto
  7. y_minus_10mm— 10 mm en −Y con contacto
  8. x_plus_5mm  —  5 mm en +X con contacto
  9. x_minus_5mm —  5 mm en −X con contacto

9 posiciones × 30 s = ~4.5 min de simulación total.
"""

from __future__ import annotations

from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

# --------------------------------------------------------------------------- #
# Parámetros ajustables                                                        #
# --------------------------------------------------------------------------- #

HOLD_SECONDS = 30
"""Segundos que el robot mantiene cada posición de offset."""

APPROACH_Z_OFFSET = 0.12
"""Altura de seguridad sobre el puerto para la fase de aproximación (metros)."""

CONTACT_Z_OFFSET = 0.002
"""Z sobre el puerto para las posiciones de contacto. Positivo = sobre la
entrada del puerto; negativo = dentro del puerto. 2 mm deja el tip en la
boca del conector con margen para la deformación del cable."""

INTERP_STEPS = 80
"""Pasos de interpolación para cada movimiento (a 0.05 s/paso = 4 s de movimiento)."""

# (label, dx_m, dy_m, drz_rad, z_offset_override)
# drz_rad: offset angular alrededor del eje Z del puerto (para calibrar GAIN_Rz)
# z_offset_override: None → usa CONTACT_Z_OFFSET; float → lo sobreescribe
OFFSET_SEQUENCE: list[tuple[str, float, float, float, float | None]] = [
    ("free_space",    0.000,  0.000, 0.000, APPROACH_Z_OFFSET),
    ("y_plus_15mm",   0.000, +0.015, 0.000, None),
    ("y_plus_10mm",   0.000, +0.010, 0.000, None),
    ("y_plus_5mm",    0.000, +0.005, 0.000, None),
    ("y_nominal",     0.000,  0.000, 0.000, None),
    ("y_minus_5mm",   0.000, -0.005, 0.000, None),
    ("y_minus_10mm",  0.000, -0.010, 0.000, None),
    ("x_plus_5mm",   +0.005,  0.000, 0.000, None),
    ("x_minus_5mm",  -0.005,  0.000, 0.000, None),
]


# --------------------------------------------------------------------------- #
# Policy                                                                       #
# --------------------------------------------------------------------------- #

class FTCalibrationSampler(Policy):
    """Secuencia de offsets controlados para calibración del estimador Bayesiano."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("FTCalibrationSampler.__init__()")

    # ----------------------------------------------------------------------- #
    # TF helpers                                                               #
    # ----------------------------------------------------------------------- #

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame, source_frame, Time()
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Esperando transform '{source_frame}' → '{target_frame}' "
                        f"(¿ground_truth:=true activo?)"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' no disponible tras {timeout_sec} s"
        )
        return False

    def _lookup_xyz(self, target_frame: str, source_frame: str) -> tuple:
        """Retorna (x, y, z, qw, qx, qy, qz) de source_frame en target_frame."""
        tf = self._parent_node._tf_buffer.lookup_transform(
            target_frame, source_frame, Time()
        )
        t = tf.transform.translation
        r = tf.transform.rotation
        return (t.x, t.y, t.z, r.w, r.x, r.y, r.z)

    # ----------------------------------------------------------------------- #
    # Pose computation                                                         #
    # ----------------------------------------------------------------------- #

    def _compute_gripper_pose(
        self,
        port_xyz: tuple,
        port_quat_wxyz: tuple,
        dx: float,
        dy: float,
        z_offset: float,
    ) -> Pose:
        """
        Calcula la pose del gripper/tcp que coloca el cable tip en:
            target_tip = port_xyz + (dx, dy, 0) + (0, 0, z_offset)

        La orientación del gripper se alinea completamente con el puerto
        (slerp = 1.0), igual que CheatCode en fase de inserción.
        """
        # Posición actual del gripper y del tip en base_link
        gx, gy, gz, gqw, gqx, gqy, gqz = self._lookup_xyz("base_link", "gripper/tcp")
        tx, ty, tz, *_ = self._lookup_xyz("base_link", self._tip_frame)

        # Offset rígido gripper → tip (asumido constante para la pose de aproximación)
        off_x = gx - tx
        off_y = gy - ty
        off_z = gz - tz

        # Posición objetivo del gripper para que el tip quede en target
        target_gx = port_xyz[0] + dx + off_x
        target_gy = port_xyz[1] + dy + off_y
        target_gz = port_xyz[2] + z_offset + off_z

        # Orientación: interpola totalmente hacia la orientación del puerto
        q_port = port_quat_wxyz                         # (w, x, y, z)
        q_gripper = (gqw, gqx, gqy, gqz)
        q_target = quaternion_slerp(q_gripper, q_port, 1.0)

        return Pose(
            position=Point(x=target_gx, y=target_gy, z=target_gz),
            orientation=Quaternion(
                w=q_target[0], x=q_target[1], y=q_target[2], z=q_target[3]
            ),
        )

    def _move_to_pose(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        steps: int = INTERP_STEPS,
    ) -> None:
        """Envía la pose repetidamente para que el controlador de impedancia converja."""
        for _ in range(steps):
            self.set_pose_target(move_robot=move_robot, pose=pose)
            self.sleep_for(0.05)

    # ----------------------------------------------------------------------- #
    # insert_cable — punto de entrada del engine                              #
    # ----------------------------------------------------------------------- #

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"FTCalibrationSampler.insert_cable() task={task}"
        )

        port_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link"
        )
        self._tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        # Esperar a que ground_truth publique los TF necesarios
        for frame in [port_frame, self._tip_frame, "gripper/tcp"]:
            if not self._wait_for_tf("base_link", frame):
                self.get_logger().error(
                    f"No se pudo obtener el frame '{frame}'. "
                    "¿Está activo ground_truth:=true?"
                )
                return False

        try:
            px, py, pz, pqw, pqx, pqy, pqz = self._lookup_xyz(
                "base_link", port_frame
            )
        except TransformException as ex:
            self.get_logger().error(f"Error obteniendo pose del puerto: {ex}")
            return False

        port_xyz = (px, py, pz)
        port_quat = (pqw, pqx, pqy, pqz)

        self.get_logger().info(
            f"Puerto '{port_frame}' en base_link: "
            f"xyz=({px:.4f}, {py:.4f}, {pz:.4f})"
        )

        # ------------------------------------------------------------------- #
        # Secuencia de offsets                                                 #
        # ------------------------------------------------------------------- #
        for idx, (label, dx, dy, _drz, z_override) in enumerate(OFFSET_SEQUENCE):
            z_off = z_override if z_override is not None else CONTACT_Z_OFFSET
            total = len(OFFSET_SEQUENCE)

            send_feedback(
                f"[{idx+1}/{total}] offset=({dx*1000:.0f}mm, {dy*1000:.0f}mm) "
                f"z={z_off:.3f}m  hold={HOLD_SECONDS}s  [{label}]"
            )
            self.get_logger().info(
                f"--- OFFSET [{idx+1}/{total}]: {label} "
                f"dx={dx*1000:.1f}mm  dy={dy*1000:.1f}mm  z={z_off:.3f}m ---"
            )

            # Si no es free_space, primero retrae a altura segura
            if z_override is None:
                try:
                    safe_pose = self._compute_gripper_pose(
                        port_xyz, port_quat, dx=0.0, dy=0.0,
                        z_offset=APPROACH_Z_OFFSET,
                    )
                    self._move_to_pose(move_robot, safe_pose, steps=40)
                except TransformException as ex:
                    self.get_logger().warn(f"TF fallido en retracción segura: {ex}")

            # Mover al offset objetivo
            try:
                target_pose = self._compute_gripper_pose(
                    port_xyz, port_quat, dx=dx, dy=dy, z_offset=z_off,
                )
                self._move_to_pose(move_robot, target_pose, steps=INTERP_STEPS)
            except TransformException as ex:
                self.get_logger().warn(
                    f"TF fallido moviendo a '{label}', saltando: {ex}"
                )
                continue

            # Mantener posición y seguir enviando la pose (el controlador de
            # impedancia necesita comandos continuos para mantener la posición)
            self.get_logger().info(
                f"Manteniendo '{label}' durante {HOLD_SECONDS} s..."
            )
            hold_steps = int(HOLD_SECONDS / 0.05)
            for step in range(hold_steps):
                try:
                    self.set_pose_target(move_robot=move_robot, pose=target_pose)
                except TransformException as ex:
                    self.get_logger().warn(f"TF warn en hold step {step}: {ex}")
                self.sleep_for(0.05)

            self.get_logger().info(f"'{label}' completado.")

        # ------------------------------------------------------------------- #
        # Retorno a posición segura final                                      #
        # ------------------------------------------------------------------- #
        self.get_logger().info("Secuencia completa. Retrayendo a altura segura...")
        try:
            safe_pose = self._compute_gripper_pose(
                port_xyz, port_quat, dx=0.0, dy=0.0,
                z_offset=APPROACH_Z_OFFSET,
            )
            self._move_to_pose(move_robot, safe_pose, steps=60)
        except TransformException as ex:
            self.get_logger().warn(f"TF fallido en retracción final: {ex}")

        self.get_logger().info("FTCalibrationSampler.insert_cable() finalizado.")
        return True
