#!/usr/bin/env python3
"""
bag_to_lerobot.py — Convierte bags ROS 2 al formato HuggingFace LeRobotDataset.

Topics requeridos en cada bag:
  /center_camera/image     sensor_msgs/Image  (cámara principal para keypoints)
  /left_camera/image       sensor_msgs/Image
  /right_camera/image      sensor_msgs/Image
  /joint_states            sensor_msgs/JointState
  /tf                      tf2_msgs/TFMessage
  /tf_static               tf2_msgs/TFMessage  [recomendado]

State vector [32]:
  [0:6]   shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3  (rad)
  [6]     gripper/left_finger_joint  (m)
  [7:25]  9 keypoints × (u, v) normalizado [0,1] desde center camera
  [25:28] TCP position (x, y, z) en base_link  (m)
  [28:32] TCP quaternion (qx, qy, qz, qw) en base_link

Action vector [6]:
  [vx, vy, vz, wx, wy, wz]  Cartesian twist en base_link  (m/s, rad/s)
  Derivado de diferencias finitas entre poses TCP consecutivas.

Ejemplos:

  # Dataset A – CheatCode (detecta tipo de conector desde /tf):
  pixi run python fine_tune_act_v2/fine_tune_act/scripts/bag_to_lerobot.py \\
    --bags_dir data/bags_cheatcode \\
    --output   data/dataset_lerobot \\
    --repo_id  aic_team/sfp_sc_insertion \\
    --ckpt_sc  fine_tune_act_v2/keypoints/keypoint_sc_1h/best_sc.pt \\
    --ckpt_sfp fine_tune_act_v2/keypoints/keypoint_sfp_1h/best_sfp.pt

  # Agregar Dataset C – teleop al mismo dataset:
  pixi run python fine_tune_act_v2/fine_tune_act/scripts/bag_to_lerobot.py \\
    --bags_dir data/bags_teleop \\
    --output   data/dataset_lerobot \\
    --repo_id  aic_team/sfp_sc_insertion \\
    --ckpt_sc  fine_tune_act_v2/keypoints/keypoint_sc_1h/best_sc.pt \\
    --ckpt_sfp fine_tune_act_v2/keypoints/keypoint_sfp_1h/best_sfp.pt \\
    --append
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Topic names
# ═══════════════════════════════════════════════════════════════════════════

TOPIC_CENTER = "/center_camera/image"
TOPIC_LEFT   = "/left_camera/image"
TOPIC_RIGHT  = "/right_camera/image"
TOPIC_JOINTS = "/joint_states"
TOPIC_TF     = "/tf"
TOPIC_TF_STATIC = "/tf_static"

ARM_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_JOINT = "gripper/left_finger_joint"

TCP_FRAME  = "gripper/tcp"
BASE_FRAME = "base_link"

# State/action layout
STATE_DIM  = 32   # [6 joints, 1 gripper, 18 kp_coords, 3 tcp_pos, 4 tcp_quat]
ACTION_DIM = 6    # [vx, vy, vz, wx, wy, wz]

STATE_MODE_LEGACY = "legacy_keypoints"
STATE_MODE_STEP4 = "step4_pose"

# Image size stored in the dataset (must match act_aic.yaml input_shapes)
STORE_H = 480
STORE_W = 640

# Synchronization: max time difference to accept a match (ns)
SYNC_TOL_NS = int(0.15 * 1e9)  # 150 ms

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFAULT_CAMERA_K = [
    1236.6316680908203, 0.0, 576.0,
    0.0, 1236.6314697265625, 512.0,
    0.0, 0.0, 1.0,
]
_DEFAULT_CAMERA_D = [0.0, 0.0, 0.0, 0.0, 0.0]
_DEFAULT_CAMERA_FRAMES = {
    "left": "left_camera/optical",
    "center": "center_camera/optical",
    "right": "right_camera/optical",
}


# ═══════════════════════════════════════════════════════════════════════════
# Quaternion / transform math
# ═══════════════════════════════════════════════════════════════════════════

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """q1 ⊗ q2  (x, y, z, w convention)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) → 3×3 rotation matrix."""
    q = q / (np.linalg.norm(q) + 1e-12)
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion (x, y, z, w)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _make_H(trans: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build 4×4 homogeneous matrix from (translation[3], quaternion_xyzw[4])."""
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = _quat_to_mat(quat)
    H[:3, 3]  = trans
    return H


def _invert_H(H: np.ndarray) -> np.ndarray:
    """Invert a homogeneous rigid-body transform."""
    R = H[:3, :3]
    t = H[:3, 3]
    Hi = np.eye(4, dtype=np.float64)
    Hi[:3, :3] = R.T
    Hi[:3, 3]  = -R.T @ t
    return Hi


def _decompose_H(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (translation[3], quaternion_xyzw[4]) from 4×4 matrix."""
    return H[:3, 3].copy(), _mat_to_quat(H[:3, :3])


def _angular_velocity(q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
    """Estimate angular velocity [wx, wy, wz] from two successive quaternions."""
    # dq = q1 ⊗ inv(q0)
    dq = _quat_mul(q1, np.array([-q0[0], -q0[1], -q0[2], q0[3]], dtype=np.float64))
    dq = dq / (np.linalg.norm(dq) + 1e-12)
    half_angle = np.arccos(np.clip(abs(dq[3]), 0.0, 1.0))
    if half_angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    sin_ha = np.sin(half_angle) + 1e-12
    axis = dq[:3] / sin_ha
    if dq[3] < 0:
        axis = -axis
    return axis * (2 * half_angle) / dt


def _load_step4_helpers():
    repo_root = Path(__file__).resolve().parents[3]
    policies_root = repo_root / "aic_example_policies"
    if str(policies_root) not in sys.path:
        sys.path.insert(0, str(policies_root))
    from aic_example_policies.ros.keypoint_step4 import (
        estimate_pose_relative_tcp_from_multiview,
    )
    return estimate_pose_relative_tcp_from_multiview


def _default_camera_infos() -> Dict[str, object]:
    infos = {}
    for camera, frame_id in _DEFAULT_CAMERA_FRAMES.items():
        infos[camera] = SimpleNamespace(
            header=SimpleNamespace(frame_id=frame_id),
            height=1024,
            width=1152,
            distortion_model="plumb_bob",
            d=list(_DEFAULT_CAMERA_D),
            k=list(_DEFAULT_CAMERA_K),
        )
    return infos


def _infer_keypoints_pixels(model: "KeypointInferencer", image_bgr: np.ndarray) -> np.ndarray:
    kp_norm = model.infer(image_bgr)
    h, w = image_bgr.shape[:2]
    kp_xy = kp_norm.astype(np.float32).copy()
    kp_xy[:, 0] *= float(w)
    kp_xy[:, 1] *= float(h)
    return kp_xy


def _lookup_camera_poses_in_base(
    tf_tree: "TFTree",
    ts: int,
    camera_infos: Dict[str, object],
) -> Dict[str, np.ndarray]:
    camera_poses = {}
    for camera, info in camera_infos.items():
        frame_id = getattr(getattr(info, "header", None), "frame_id", None)
        if not frame_id:
            continue
        tf_result = tf_tree.lookup(BASE_FRAME, frame_id, ts)
        if tf_result is None:
            continue
        camera_poses[camera] = _make_H(*tf_result)
    return camera_poses


# ═══════════════════════════════════════════════════════════════════════════
# TF tree
# ═══════════════════════════════════════════════════════════════════════════

class TFTree:
    """TF tree mínimo para procesamiento offline de bags ROS 2."""

    def __init__(self) -> None:
        # (parent, child) → (translation[3], quaternion_xyzw[4])
        self._static: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        # (parent, child) → list of (ts_ns, translation[3], quaternion_xyzw[4])
        self._dynamic: Dict[Tuple[str, str], List[Tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)

    def add_static(self, parent: str, child: str,
                   trans: np.ndarray, quat: np.ndarray) -> None:
        self._static[(parent, child)] = (trans.copy(), quat.copy())

    def add_dynamic(self, ts_ns: int, parent: str, child: str,
                    trans: np.ndarray, quat: np.ndarray) -> None:
        entry = (ts_ns, trans.copy(), quat.copy())
        lst = self._dynamic[(parent, child)]
        bisect.insort(lst, entry)

    def _get_edge(self, parent: str, child: str,
                  ts_ns: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        key = (parent, child)
        if key in self._static:
            return self._static[key]
        if key in self._dynamic:
            lst = self._dynamic[key]
            if not lst:
                return None
            idx = bisect.bisect_right(lst, (ts_ns,)) - 1
            idx = max(idx, 0)
            return lst[idx][1], lst[idx][2]
        return None

    def lookup(self, target: str, source: str,
               ts_ns: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retorna (translation[3], quaternion_xyzw[4]) de *source* expresado
        en el frame *target*, usando BFS sobre el árbol TF.
        """
        if source == target:
            return np.zeros(3, dtype=np.float64), np.array([0., 0., 0., 1.])

        # Construir mapa de adyacencia
        fwd: Dict[str, List[str]] = defaultdict(list)  # parent → children
        inv: Dict[str, List[str]] = defaultdict(list)  # child  → parents

        for (p, c) in self._static:
            if c not in fwd[p]:
                fwd[p].append(c)
            if p not in inv[c]:
                inv[c].append(p)
        for (p, c) in self._dynamic:
            if self._dynamic[(p, c)]:
                if c not in fwd[p]:
                    fwd[p].append(c)
                if p not in inv[c]:
                    inv[c].append(p)

        # BFS desde target acumulando H(target, nodo_actual)
        visited = {target}
        queue: List[Tuple[str, np.ndarray]] = [(target, np.eye(4, dtype=np.float64))]

        while queue:
            node, H_accum = queue.pop(0)

            # Aristas hacia adelante: node → child
            for child in fwd.get(node, []):
                if child in visited:
                    continue
                visited.add(child)
                result = self._get_edge(node, child, ts_ns)
                if result is None:
                    continue
                H_new = H_accum @ _make_H(*result)
                if child == source:
                    return _decompose_H(H_new)
                queue.append((child, H_new))

            # Aristas hacia atrás: node ← parent (traversal inverso)
            for parent in inv.get(node, []):
                if parent in visited:
                    continue
                visited.add(parent)
                result = self._get_edge(parent, node, ts_ns)
                if result is None:
                    continue
                H_new = H_accum @ _invert_H(_make_H(*result))
                if parent == source:
                    return _decompose_H(H_new)
                queue.append((parent, H_new))

        return None

    def detect_connector_type(self) -> Optional[str]:
        """Detecta tipo de conector buscando frames conocidos en el árbol TF."""
        all_children = set()
        for (_, c) in self._static:
            all_children.add(c)
        for (_, c) in self._dynamic:
            all_children.add(c)

        has_sfp = False
        has_sc = False
        for frame in all_children:
            if "sfp" in frame.lower():
                has_sfp = True
            if "sc_tip" in frame.lower() or "sc_module" in frame.lower():
                has_sc = True
        if has_sfp and not has_sc:
            return "sfp"
        if has_sc and not has_sfp:
            return "sc"
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Keypoint inferencer
# ═══════════════════════════════════════════════════════════════════════════

class KeypointInferencer:
    """Carga un checkpoint de keypoints y corre inferencia en imágenes."""

    def __init__(self, checkpoint_path: Path, device: Optional[str] = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        train_mod = self._load_train_module(repo_root)
        torch_mod, _, _, _, _ = train_mod.require_torch()

        self.device = device or ("cuda" if torch_mod.cuda.is_available() else "cpu")
        self._install_pathlib_checkpoint_compat()
        ckpt = torch_mod.load(str(checkpoint_path), map_location=self.device,
                              weights_only=False)

        self.img_h = int(ckpt.get("image_height", 256))
        self.img_w = int(ckpt.get("image_width", 288))
        self.n_kp  = int(ckpt.get("num_keypoints", 9))
        self.connector_type = ckpt.get("connector_type", "unknown")

        self.model = train_mod.build_model(num_keypoints=self.n_kp,
                                           pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self._torch = torch_mod

    @staticmethod
    def _load_train_module(repo_root: Path):
        module_path = repo_root / "scripts" / "train_keypoint_estimator.py"
        if not module_path.exists():
            raise FileNotFoundError(f"No se encontró: {module_path}")
        spec = importlib.util.spec_from_file_location("aic_train_kp", module_path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _install_pathlib_checkpoint_compat() -> None:
        """Compatibilidad para checkpoints pickleados con pathlib._local."""
        if "pathlib._local" in sys.modules:
            return
        import pathlib
        import types

        if sys.platform != "win32":
            pathlib.WindowsPath = pathlib.PosixPath

        compat = types.ModuleType("pathlib._local")
        for name in (
            "Path",
            "PosixPath",
            "WindowsPath",
            "PurePath",
            "PurePosixPath",
            "PureWindowsPath",
        ):
            setattr(compat, name, getattr(pathlib, name))
        sys.modules["pathlib._local"] = compat
        setattr(pathlib, "_local", compat)

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Retorna keypoints normalizados [0,1] con shape (n_kp, 2) = (u_norm, v_norm).
        image_bgr: numpy uint8 HWC BGR.
        """
        resized = cv2.resize(image_bgr, (self.img_w, self.img_h),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = self._torch.from_numpy(
            np.transpose(rgb, (2, 0, 1)).astype(np.float32)
        ).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            pred = self.model(tensor).detach().cpu().numpy()

        return pred.reshape(self.n_kp, 2)  # already normalized [0,1]


# ═══════════════════════════════════════════════════════════════════════════
# Bag reader helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_typestore():
    """Retorna el typestore de rosbags más completo disponible."""
    from rosbags.typesys import get_typestore
    for store_name in ("ROS2_HUMBLE", "ROS2_GALACTIC", "ROS2_FOXY", "LATEST"):
        try:
            from rosbags.typesys import Stores
            store = getattr(Stores, store_name, None)
            if store is not None:
                return get_typestore(store)
        except Exception:
            continue
    # Fallback: try without explicit store
    try:
        return get_typestore()
    except Exception:
        pass
    raise RuntimeError(
        "No se pudo inicializar el typestore de rosbags. "
        "Asegúrate de tener rosbags>=0.9.0 instalado."
    )


def _image_msg_to_bgr(msg) -> np.ndarray:
    """Convierte sensor_msgs/Image a numpy uint8 HWC BGR."""
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = msg.encoding.lower()

    if "rgb8" in enc or "rgb" in enc:
        img = raw.reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif "bgr8" in enc or "bgr" in enc:
        return raw.reshape(msg.height, msg.width, 3).copy()
    elif "mono8" in enc:
        mono = raw.reshape(msg.height, msg.width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    elif "bayer" in enc:
        mono = raw.reshape(msg.height, msg.width)
        return cv2.cvtColor(mono, cv2.COLOR_BAYER_BG2BGR)
    else:
        # Intenta interpretar como RGB
        img = raw.reshape(msg.height, msg.width, -1)
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        raise ValueError(f"Encoding de imagen no soportado: {msg.encoding}")


def _nearest_before(timestamps: List[int], t: int, tol_ns: int) -> int:
    """
    Retorna el índice del timestamp más cercano a t dentro de la tolerancia.
    timestamps debe estar ordenado ascendentemente.
    """
    if not timestamps:
        return -1
    idx = bisect.bisect_right(timestamps, t) - 1
    if idx < 0:
        idx = 0
    # Busca el más cercano (puede estar justo después de t)
    best_idx = idx
    best_diff = abs(timestamps[idx] - t)
    if idx + 1 < len(timestamps):
        diff_next = abs(timestamps[idx + 1] - t)
        if diff_next < best_diff:
            best_idx = idx + 1
            best_diff = diff_next
    if best_diff > tol_ns:
        return -1
    return best_idx


# ═══════════════════════════════════════════════════════════════════════════
# Indexador de bag: agrupa mensajes por topic
# ═══════════════════════════════════════════════════════════════════════════

class BagIndex:
    """Lee un bag ROS 2 completo y almacena mensajes en memoria por topic."""

    def __init__(self, bag_path: Path) -> None:
        from rosbags.rosbag2 import Reader

        self.timestamps: Dict[str, List[int]] = defaultdict(list)
        self.messages:   Dict[str, List]      = defaultdict(list)
        self.tf_tree = TFTree()

        typestore = _get_typestore()

        required_topics = {TOPIC_CENTER, TOPIC_LEFT, TOPIC_RIGHT,
                           TOPIC_JOINTS, TOPIC_TF}

        with Reader(str(bag_path)) as reader:
            available = {c.topic for c in reader.connections}
            missing = required_topics - available
            if missing:
                raise ValueError(
                    f"Bag {bag_path.name} no tiene topics requeridos: {missing}\n"
                    f"Topics disponibles: {sorted(available)}"
                )

            conn_filter = [c for c in reader.connections
                           if c.topic in required_topics | {TOPIC_TF_STATIC}]

            for conn, ts_ns, rawdata in reader.messages(connections=conn_filter):
                topic = conn.topic
                try:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                except Exception:
                    continue

                if topic in (TOPIC_TF, TOPIC_TF_STATIC):
                    is_static = (topic == TOPIC_TF_STATIC)
                    for tf in msg.transforms:
                        parent = tf.header.frame_id
                        child  = tf.child_frame_id
                        t = np.array([
                            tf.transform.translation.x,
                            tf.transform.translation.y,
                            tf.transform.translation.z,
                        ], dtype=np.float64)
                        q = np.array([
                            tf.transform.rotation.x,
                            tf.transform.rotation.y,
                            tf.transform.rotation.z,
                            tf.transform.rotation.w,
                        ], dtype=np.float64)
                        if is_static:
                            self.tf_tree.add_static(parent, child, t, q)
                        else:
                            self.tf_tree.add_dynamic(ts_ns, parent, child, t, q)
                else:
                    self.timestamps[topic].append(ts_ns)
                    self.messages[topic].append(msg)

        # Cada lista de timestamps debe ser monotónicamente creciente;
        # rosbags garantiza orden cronológico, pero lo verificamos
        for topic in self.timestamps:
            ts = self.timestamps[topic]
            msgs = self.messages[topic]
            if not ts:
                continue
            # Ordena por timestamp por si hay mensajes desordenados
            pairs = sorted(zip(ts, msgs), key=lambda x: x[0])
            self.timestamps[topic] = [p[0] for p in pairs]
            self.messages[topic]   = [p[1] for p in pairs]


# ═══════════════════════════════════════════════════════════════════════════
# Procesamiento de un bag → lista de frames
# ═══════════════════════════════════════════════════════════════════════════

def _build_joint_dict(msg) -> Dict[str, float]:
    return dict(zip(msg.name, msg.position))


def process_bag(
    bag_path: Path,
    kp_sc:  KeypointInferencer,
    kp_sfp: KeypointInferencer,
    connector_type: Optional[str],
    target_fps: int,
    state_mode: str = STATE_MODE_LEGACY,
    camera_infos: Optional[Dict[str, object]] = None,
) -> Optional[List[dict]]:
    """
    Procesa un bag completo y retorna una lista de dicts con las claves:
      center, left, right  → numpy uint8 (STORE_H, STORE_W, 3)
      state                → numpy float32 (STATE_DIM,)
      tcp_pos              → numpy float64 (3,)
      tcp_quat             → numpy float64 (4,)

    Retorna None si el bag no tiene suficientes datos.
    """
    print(f"  Indexando {bag_path.name} ...", flush=True)
    try:
        idx = BagIndex(bag_path)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        return None

    # Detectar tipo de conector
    ct = connector_type
    if ct is None or ct == "auto":
        ct = idx.tf_tree.detect_connector_type()
    if ct is None:
        # Intentar desde el nombre del bag
        name_lower = bag_path.name.lower()
        if "sfp" in name_lower:
            ct = "sfp"
        elif "sc" in name_lower:
            ct = "sc"
        else:
            print(f"  [WARN] No se pudo detectar tipo de conector para {bag_path.name}; "
                  f"usando sfp")
            ct = "sfp"

    kp_model = kp_sfp if ct == "sfp" else kp_sc
    estimate_step4 = None
    if state_mode == STATE_MODE_STEP4:
        camera_infos = camera_infos or _default_camera_infos()
        estimate_step4 = _load_step4_helpers()

    center_ts = idx.timestamps[TOPIC_CENTER]
    if not center_ts:
        print(f"  [SKIP] No hay imágenes de center camera")
        return None

    # Submuestrear a target_fps usando la cámara center como reloj maestro
    dt_target_ns = int(1e9 / target_fps)
    selected_indices: List[int] = []
    last_selected_ts = center_ts[0] - dt_target_ns
    for i, ts in enumerate(center_ts):
        if (ts - last_selected_ts) >= dt_target_ns:
            selected_indices.append(i)
            last_selected_ts = ts

    if len(selected_indices) < 4:
        print(f"  [SKIP] Solo {len(selected_indices)} frames después de submuestreo")
        return None

    joints_ts = idx.timestamps[TOPIC_JOINTS]
    left_ts   = idx.timestamps[TOPIC_LEFT]
    right_ts  = idx.timestamps[TOPIC_RIGHT]

    frames = []
    for i in selected_indices:
        ts = center_ts[i]
        center_msg = idx.messages[TOPIC_CENTER][i]

        # Sincronizar joint states
        ji = _nearest_before(joints_ts, ts, SYNC_TOL_NS)
        if ji < 0:
            continue
        jdict = _build_joint_dict(idx.messages[TOPIC_JOINTS][ji])

        # Verificar que tengamos los joints necesarios
        if not all(j in jdict for j in ARM_JOINT_ORDER):
            continue

        # Joints del brazo (6)
        arm_joints = np.array([jdict[j] for j in ARM_JOINT_ORDER], dtype=np.float32)
        gripper_val = float(jdict.get(GRIPPER_JOINT, 0.0))

        # TCP pose desde TF
        tf_result = idx.tf_tree.lookup(BASE_FRAME, TCP_FRAME, ts)
        if tf_result is None:
            continue
        tcp_pos, tcp_quat = tf_result
        tcp_pos  = tcp_pos.astype(np.float32)
        tcp_quat = tcp_quat.astype(np.float32)

        # Imagen center → keypoints
        center_bgr = _image_msg_to_bgr(center_msg)
        center_resized = cv2.resize(center_bgr, (STORE_W, STORE_H),
                                    interpolation=cv2.INTER_LINEAR)

        # Imágenes left y right
        li = _nearest_before(left_ts, ts, SYNC_TOL_NS)
        ri = _nearest_before(right_ts, ts, SYNC_TOL_NS)
        if li < 0 or ri < 0:
            continue
        left_bgr  = _image_msg_to_bgr(idx.messages[TOPIC_LEFT][li])
        right_bgr = _image_msg_to_bgr(idx.messages[TOPIC_RIGHT][ri])
        left_resized  = cv2.resize(left_bgr,  (STORE_W, STORE_H), interpolation=cv2.INTER_LINEAR)
        right_resized = cv2.resize(right_bgr, (STORE_W, STORE_H), interpolation=cv2.INTER_LINEAR)

        state = None
        visual_pose_rel = None
        if state_mode == STATE_MODE_STEP4:
            keypoints_by_camera = {
                "left": _infer_keypoints_pixels(kp_model, left_bgr),
                "center": _infer_keypoints_pixels(kp_model, center_bgr),
                "right": _infer_keypoints_pixels(kp_model, right_bgr),
            }
            camera_poses = _lookup_camera_poses_in_base(idx.tf_tree, ts, camera_infos)
            tcp_in_base = _make_H(tcp_pos.astype(np.float64), tcp_quat.astype(np.float64))
            estimate = estimate_step4(
                keypoints_by_camera=keypoints_by_camera,
                connector_type=ct,
                camera_infos=camera_infos,
                camera_in_base_by_camera=camera_poses,
                tcp_in_base=tcp_in_base,
            )
            if estimate is None:
                continue
            visual_pose_rel = estimate.pose_relative_tcp.astype(np.float32)
            if (
                not np.isfinite(visual_pose_rel).all()
                or np.linalg.norm(visual_pose_rel[:3]) > 1.0
                or np.max(np.abs(visual_pose_rel[3:])) > 4.0
            ):
                continue
        else:
            kp_norm = kp_model.infer(center_bgr)  # (9, 2) normalizado [0,1]
            kp_flat = kp_norm.flatten().astype(np.float32)  # (18,)
            state = np.concatenate([
                arm_joints,          # [0:6]
                [gripper_val],       # [6]
                kp_flat,             # [7:25]
                tcp_pos,             # [25:28]
                tcp_quat,            # [28:32]
            ]).astype(np.float32)
            assert len(state) == STATE_DIM, f"state dim = {len(state)}"

        frames.append({
            "ts": ts,
            "center": center_resized,   # BGR uint8 (H, W, 3)
            "left":   left_resized,
            "right":  right_resized,
            "state":  state,
            "tcp_pos":  tcp_pos,
            "tcp_quat": tcp_quat,
            "arm_joints": arm_joints,
            "gripper_val": gripper_val,
            "visual_pose_rel": visual_pose_rel,
            "connector_type": ct,
        })

    if len(frames) < 4:
        print(f"  [SKIP] Solo {len(frames)} frames válidos (con TCP lookup y sync)")
        return None

    # Calcular acciones como velocidad Cartesiana (diferencias finitas)
    dt = 1.0 / target_fps
    for i, f in enumerate(frames):
        if i < len(frames) - 1:
            dp = frames[i+1]["tcp_pos"]  - f["tcp_pos"]
            dw = _angular_velocity(f["tcp_quat"], frames[i+1]["tcp_quat"], dt)
        else:
            dp = frames[-1]["tcp_pos"]  - frames[-2]["tcp_pos"]
            dw = _angular_velocity(frames[-2]["tcp_quat"], frames[-1]["tcp_quat"], dt)
        f["action"] = np.concatenate([dp / dt, dw]).astype(np.float32)
        if state_mode == STATE_MODE_STEP4:
            base_state = np.concatenate([
                f["tcp_pos"],                         # TCP position (3)
                f["tcp_quat"],                        # TCP quaternion (4)
                f["action"][:3],                      # TCP linear velocity (3)
                f["action"][3:],                      # TCP angular velocity (3)
                np.zeros(6, dtype=np.float32),        # TCP error unavailable in this bag
                f["arm_joints"],                      # Arm joints (6)
                [f["gripper_val"]],                   # Gripper (1)
            ]).astype(np.float32)
            f["state"] = np.concatenate([
                base_state,
                f["visual_pose_rel"],                 # pose_relative_tcp (6)
            ]).astype(np.float32)
            assert len(f["state"]) == STATE_DIM, f"state dim = {len(f['state'])}"

    print(f"  OK  {len(frames)} frames  conector={ct}  state_mode={state_mode}", flush=True)
    return frames


# ═══════════════════════════════════════════════════════════════════════════
# LeRobotDataset wrapper
# ═══════════════════════════════════════════════════════════════════════════

def _get_dataset(repo_id: str, output: Path, fps: int,
                 append: bool) -> object:
    """Crea o abre un LeRobotDataset."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.center": {
            "dtype": "video",
            "shape": (STORE_H, STORE_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.left": {
            "dtype": "video",
            "shape": (STORE_H, STORE_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.right": {
            "dtype": "video",
            "shape": (STORE_H, STORE_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": None,
        },
    }

    if append and output.exists() and (output / "meta" / "info.json").exists():
        print(f"Abriendo dataset existente en {output} para agregar episodios ...")
        return LeRobotDataset(repo_id=repo_id, root=output)

    print(f"Creando nuevo LeRobotDataset en {output} ...")
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output,
        features=features,
        robot_type="ur5e",
        use_videos=True,
    )


def _add_episode(dataset, frames: List[dict], task_description: str) -> None:
    """Agrega un episodio al dataset."""
    for f in frames:
        # lerobot espera imágenes como RGB uint8 HWC
        center_rgb = cv2.cvtColor(f["center"], cv2.COLOR_BGR2RGB)
        left_rgb   = cv2.cvtColor(f["left"],   cv2.COLOR_BGR2RGB)
        right_rgb  = cv2.cvtColor(f["right"],  cv2.COLOR_BGR2RGB)

        dataset.add_frame({
            "observation.images.center": center_rgb,
            "observation.images.left":   left_rgb,
            "observation.images.right":  right_rgb,
            "observation.state": f["state"],
            "action":            f["action"],
            "task":              task_description,
        })

    try:
        dataset.save_episode()
    except TypeError:
        dataset.save_episode(task=task_description)


# ═══════════════════════════════════════════════════════════════════════════
# Recolección de bag paths
# ═══════════════════════════════════════════════════════════════════════════

def _collect_bag_paths(directory: Path) -> List[Path]:
    """
    Retorna la lista de paths de bags ROS 2 en un directorio.
    Un bag es un subdirectorio que contiene metadata.yaml, o un archivo .mcap.
    """
    bags = []
    if not directory.exists():
        return bags

    # Bags como directorios (formato .db3 o .mcap con metadata.yaml)
    for sub in sorted(directory.iterdir()):
        if sub.is_dir() and (sub / "metadata.yaml").exists():
            bags.append(sub)

    # Bags como archivos .mcap directos
    for mcap in sorted(directory.glob("*.mcap")):
        bags.append(mcap)

    return bags


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convierte bags ROS 2 a formato HuggingFace LeRobotDataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bags_dir",   type=Path, required=True,
                   help="Directorio con bags del Dataset A (CheatCode) o Dataset C (teleop).")
    p.add_argument("--output",     type=Path, required=True,
                   help="Directorio de salida para el LeRobotDataset.")
    p.add_argument("--repo_id",    type=str, required=True,
                   help="Repo ID de HuggingFace (ej: aic_team/sfp_sc_insertion).")
    p.add_argument("--ckpt_sc",    type=Path, required=True,
                   help="Checkpoint del modelo keypoint SC (.pt).")
    p.add_argument("--ckpt_sfp",   type=Path, required=True,
                   help="Checkpoint del modelo keypoint SFP (.pt).")
    p.add_argument("--fps",        type=int, default=10,
                   help="FPS objetivo del dataset (default: 10).")
    p.add_argument("--connector_type", choices=["sfp", "sc", "auto"],
                   default="auto",
                   help="Tipo de conector. 'auto' detecta desde /tf o nombre del bag.")
    p.add_argument("--device",     type=str, default=None,
                   help="Dispositivo para inferencia de keypoints (cuda/cpu).")
    p.add_argument("--state_mode", choices=[STATE_MODE_LEGACY, STATE_MODE_STEP4],
                   default=STATE_MODE_LEGACY,
                   help=(
                       "Formato del vector de estado. "
                       "legacy_keypoints: 6 joints + gripper + keypoints center + TCP pose. "
                       "step4_pose: 26D proprioceptivo + pose_relative_tcp [6D] usando Step 4."
                   ))
    p.add_argument("--append",     action="store_true",
                   help="Si el dataset ya existe, agrega episodios en lugar de sobreescribir.")
    p.add_argument("--max_bags",   type=int, default=None,
                   help="Procesar solo los primeros N bags (útil para pruebas).")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    print("=" * 68)
    print("  bag_to_lerobot — Conversión ROS 2 bags → LeRobotDataset")
    print("=" * 68)

    # Verificar checkpoints
    for ck, name in [(args.ckpt_sc, "SC"), (args.ckpt_sfp, "SFP")]:
        if not ck.exists():
            print(f"[ERROR] Checkpoint {name} no existe: {ck}")
            return 1

    print(f"Cargando modelos de keypoints ...")
    kp_sc  = KeypointInferencer(args.ckpt_sc,  device=args.device)
    kp_sfp = KeypointInferencer(args.ckpt_sfp, device=args.device)
    print(f"  SC  → {args.ckpt_sc.name}  device={kp_sc.device}")
    print(f"  SFP → {args.ckpt_sfp.name}  device={kp_sfp.device}")
    camera_infos = _default_camera_infos() if args.state_mode == STATE_MODE_STEP4 else None
    if args.state_mode == STATE_MODE_STEP4:
        print("  Step 4 activo: usando camera_info fijo de la simulación AIC actual")

    # Recolectar bags
    bags = _collect_bag_paths(args.bags_dir)
    if not bags:
        print(f"[ERROR] No se encontraron bags en {args.bags_dir}")
        return 1
    if args.max_bags is not None:
        bags = bags[:args.max_bags]
    print(f"\nBags encontrados: {len(bags)}  (en {args.bags_dir})")

    # Inicializar dataset
    dataset = _get_dataset(args.repo_id, args.output, args.fps, args.append)

    # Procesar cada bag
    t0 = time.time()
    n_ok = 0
    n_skip = 0
    ct_arg = args.connector_type if args.connector_type != "auto" else None

    for bag_idx, bag_path in enumerate(bags):
        print(f"\n[{bag_idx+1}/{len(bags)}] {bag_path.name}")
        frames = process_bag(
            bag_path,
            kp_sc=kp_sc,
            kp_sfp=kp_sfp,
            connector_type=ct_arg,
            target_fps=args.fps,
            state_mode=args.state_mode,
            camera_infos=camera_infos,
        )
        if frames is None:
            n_skip += 1
            continue

        ct_ep = frames[0]["connector_type"]
        task_desc = f"cable insertion {ct_ep}"
        _add_episode(dataset, frames, task_description=task_desc)
        n_ok += 1

        elapsed = time.time() - t0
        eta = elapsed / n_ok * (len(bags) - bag_idx - 1) if n_ok > 0 else 0
        print(f"  episodio {n_ok} guardado  "
              f"(elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min)")

    if n_ok == 0:
        print("\n[ERROR] No se procesó ningún bag con éxito.")
        return 1

    print(f"\nConsolidando dataset ({n_ok} episodios, {n_skip} skipped) ...")
    if hasattr(dataset, "consolidate"):
        try:
            dataset.consolidate(run_compute_stats=True)
        except Exception as e:
            print(f"  [WARN] consolidate() falló: {e}")
            print(f"  El dataset puede estar incompleto pero los episodios están guardados.")
    else:
        print("  consolidate() no aplica para esta versión de LeRobot.")

    elapsed_total = time.time() - t0
    print()
    print("=" * 68)
    print(f"  COMPLETADO — {n_ok} episodios en {elapsed_total/60:.1f} min")
    print(f"  Dataset: {args.output}")
    print(f"  Siguiente paso:")
    print(f"    bash fine_tune_act_v2/fine_tune_act/scripts/run_finetune.sh")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
