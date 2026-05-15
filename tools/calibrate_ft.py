#!/usr/bin/env python3
"""
Calibración F/T del Axia80 — regresión lineal entre offsets del EE y lecturas del wrench.

Procesa dos tipos de fuente y combina sus datos points para una regresión unificada:

  FUENTE A — Bags separados (bag_trial_1, 2, 3):
    Cada bag = 1 punto de datos. El bag completo se promedia.
    Lógica original intacta.

  FUENTE B — Bag multi-fase (ft_calib_*):
    Un solo bag con 9 fases secuenciales (FTCalibrationSampler).
    Se detectan las ventanas estables por velocidad de joints y cada
    ventana estable = 1 punto de datos independiente.

  REGRESIÓN UNIFICADA:
    Todos los puntos de ambas fuentes se combinan para una sola regresión
    F = GAIN * offset + BIAS con mayor cobertura de offsets y mejor R².

Uso:
  # Procesar solo bags originales:
  python tools/calibrate_ft.py

  # Agregar bag multi-fase (ya extraído a CSV):
  python tools/calibrate_ft.py --calib extracted_eval_bags_csv/ft_calib_20260514_204052

  # Extraer bag multi-fase antes de procesar (requiere distrobox/ROS2):
  #   python tools/extract_eval_bags_to_csv.py /tmp/ft_calib_TIMESTAMP
  #   python tools/calibrate_ft.py --calib extracted_eval_bags_csv/ft_calib_TIMESTAMP
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Tuple

# --------------------------------------------------------------------------- #
# Configuración                                                                #
# --------------------------------------------------------------------------- #

TRIALS = [
    "bag_trial_1_20260512_190629_607",
    "bag_trial_2_20260512_190737_082",
    "bag_trial_3_20260512_190908_253",
]

ROOT = Path(__file__).resolve().parents[1]

# Parámetros de detección de fases estables (Fuente B)
STABLE_VEL_THRESHOLD = 0.008   # rad/s — velocidad máxima de joint para considerarlo estable
MIN_STABLE_SEC       = 15.0    # duración mínima de una fase estable para incluirla
TRIM_SEC             = 4.0     # segundos a recortar al inicio y fin de cada fase (settling)
MIN_WRENCH_SAMPLES   = 100     # mínimo de muestras de wrench en una ventana para ser válida

# Frames TF necesarios para ambos tipos de conector
NEEDED_TF_PARENTS = {
    "aic_world",
    "cable_0", "cable_1",
    "task_board",
    "task_board/sc_port_0",
    "task_board/nic_card_mount_0",
}

# Configuración por tipo de conector
CONNECTOR_CONFIGS = {
    "sfp": {
        "tip_suffix": "sfp_tip_link",
        "port_chain": [
            ("aic_world",                    "task_board"),
            ("task_board",                   "task_board/nic_card_mount_0"),
            ("task_board/nic_card_mount_0",  "task_board/nic_card_mount_0/sfp_port_0_link_entrance"),
        ],
    },
    "sc": {
        "tip_suffix": "sc_tip_link",
        "port_chain": [
            ("aic_world",          "task_board"),
            ("task_board",         "task_board/sc_port_0"),
            ("task_board/sc_port_0", "task_board/sc_port_0/sc_port_base_link_entrance"),
        ],
    },
}

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]   # (x, y, z, w)


# --------------------------------------------------------------------------- #
# Quaternion / SE3 helpers (stdlib only)                                       #
# --------------------------------------------------------------------------- #

def quat_mult(q1: Quat, q2: Quat) -> Quat:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    qv: Quat = (v[0], v[1], v[2], 0.0)
    qc: Quat = (-q[0], -q[1], -q[2], q[3])
    r = quat_mult(quat_mult(q, qv), qc)
    return (r[0], r[1], r[2])


def compose_tf(t1: Vec3, q1: Quat, t2: Vec3, q2: Quat) -> Tuple[Vec3, Quat]:
    r = quat_rotate(q1, t2)
    return (t1[0]+r[0], t1[1]+r[1], t1[2]+r[2]), quat_mult(q1, q2)


def quat_to_yaw(q: Quat) -> float:
    """Extrae yaw (Rz) de un quaternion (x,y,z,w)."""
    x, y, z, w = q
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))


# --------------------------------------------------------------------------- #
# TF helpers — versión con timestamps para soporte de ventanas temporales      #
# --------------------------------------------------------------------------- #

def load_tf_timed(csv_path: Path) -> dict:
    """
    Carga TF filtrando solo los parents en NEEDED_TF_PARENTS.
    Retorna {(parent, child): [(timestamp_ns, tx, ty, tz, qx, qy, qz, qw), ...]}.
    """
    table: dict = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["frame_id"] not in NEEDED_TF_PARENTS:
                continue
            key = (row["frame_id"], row["child_frame_id"])
            table[key].append((
                int(row["timestamp_ns"]),
                float(row["translation_x"]),
                float(row["translation_y"]),
                float(row["translation_z"]),
                float(row["rotation_x"]),
                float(row["rotation_y"]),
                float(row["rotation_z"]),
                float(row["rotation_w"]),
            ))
    return dict(table)


def mean_tf_window(
    entries: list,
    t_start: int | None = None,
    t_end:   int | None = None,
) -> Tuple[Vec3, Quat] | None:
    """
    Promedia transforms. Si t_start/t_end son None usa todas las entradas.
    Cada entry es (timestamp_ns, tx, ty, tz, qx, qy, qz, qw).
    """
    if t_start is not None or t_end is not None:
        entries = [
            e for e in entries
            if (t_start is None or e[0] >= t_start)
            and (t_end   is None or e[0] <= t_end)
        ]
    if not entries:
        return None
    n = len(entries)
    tx = sum(e[1] for e in entries) / n
    ty = sum(e[2] for e in entries) / n
    tz = sum(e[3] for e in entries) / n
    qx = sum(e[4] for e in entries) / n
    qy = sum(e[5] for e in entries) / n
    qz = sum(e[6] for e in entries) / n
    qw = sum(e[7] for e in entries) / n
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
    return (tx, ty, tz), (qx/norm, qy/norm, qz/norm, qw/norm)


def chain_tf_window(
    timed_table: dict,
    chain: list,
    t_start: int | None = None,
    t_end:   int | None = None,
) -> Tuple[Vec3, Quat] | None:
    """Compone una cadena de transforms, opcionalmente filtrada por ventana temporal."""
    t: Vec3 = (0.0, 0.0, 0.0)
    q: Quat = (0.0, 0.0, 0.0, 1.0)
    for key in chain:
        if key not in timed_table:
            return None
        tf = mean_tf_window(timed_table[key], t_start, t_end)
        if tf is None:
            return None
        t2, q2 = tf
        t, q = compose_tf(t, q, t2, q2)
    return t, q


# --------------------------------------------------------------------------- #
# Detección de tipo de conector                                                #
# --------------------------------------------------------------------------- #

def detect_connector(timed_table: dict) -> Tuple[str | None, str | None]:
    """
    Detecta el prefijo del cable (cable_0/cable_1) y el tipo de conector (sfp/sc).
    Retorna (cable_prefix, connector_type) o (None, None) si no se puede determinar.
    """
    for prefix in ("cable_0", "cable_1"):
        if ("aic_world", prefix) not in timed_table:
            continue
        for ctype, cfg in CONNECTOR_CONFIGS.items():
            tip_key = (prefix, f"{prefix}/{cfg['tip_suffix']}")
            if tip_key in timed_table:
                return prefix, ctype
    return None, None


# --------------------------------------------------------------------------- #
# Cálculo de offset EE → puerto                                               #
# --------------------------------------------------------------------------- #

def compute_ee_offset(
    timed_table: dict,
    cable_prefix: str,
    connector_type: str,
    t_start: int | None = None,
    t_end:   int | None = None,
) -> dict:
    """
    Calcula offset = tip_link − port_entrance en frame aic_world.
    Funciona para una ventana temporal específica o para todo el bag (t_start=None).
    """
    cfg = CONNECTOR_CONFIGS[connector_type]
    tip_chain = [
        ("aic_world", cable_prefix),
        (cable_prefix, f"{cable_prefix}/{cfg['tip_suffix']}"),
    ]
    tip_tf  = chain_tf_window(timed_table, tip_chain,      t_start, t_end)
    port_tf = chain_tf_window(timed_table, cfg["port_chain"], t_start, t_end)

    if tip_tf is None or port_tf is None:
        return {"offset_x": None, "offset_y": None, "offset_rz": None,
                "note": f"missing chain (connector={connector_type})"}

    p_tip,  q_tip  = tip_tf
    p_port, q_port = port_tf
    return {
        "offset_x":  p_tip[0] - p_port[0],
        "offset_y":  p_tip[1] - p_port[1],
        "offset_rz": quat_to_yaw(q_tip) - quat_to_yaw(q_port),
        "p_tip":     p_tip,
        "p_port":    p_port,
        "note":      "ok",
    }


# --------------------------------------------------------------------------- #
# FUENTE A — bags separados (lógica original)                                  #
# --------------------------------------------------------------------------- #

def load_wrench_stats(trial_dir: Path) -> dict:
    """Media y std de force_x, force_y, torque_z para todo el bag."""
    fx, fy, tz = [], [], []
    with open(trial_dir / "fts_broadcaster__wrench.csv") as f:
        for row in csv.DictReader(f):
            fx.append(float(row["force_x"]))
            fy.append(float(row["force_y"]))
            tz.append(float(row["torque_z"]))
    return {
        "force_x_mean": mean(fx), "force_x_std": stdev(fx),
        "force_y_mean": mean(fy), "force_y_std": stdev(fy),
        "torque_z_mean": mean(tz), "torque_z_std": stdev(tz),
        "n": len(fx),
    }


def load_single_bag_point(trial_dir: Path, label: str) -> dict | None:
    """
    Procesa un bag completo como un único punto de datos.
    (Lógica original para bag_trial_1/2/3.)
    """
    wrench = load_wrench_stats(trial_dir)
    timed  = load_tf_timed(trial_dir / "tf.csv")
    cable_prefix, connector_type = detect_connector(timed)

    if cable_prefix is None:
        print(f"  [warn] {label}: no se detectó frame de cable en TF")
        ee = {"offset_x": None, "offset_y": None, "offset_rz": None}
    else:
        ee = compute_ee_offset(timed, cable_prefix, connector_type)

    return {
        "label":          label,
        "source":         "single_bag",
        "connector_type": connector_type,
        "wrench":         wrench,
        "ee":             ee,
    }


# --------------------------------------------------------------------------- #
# FUENTE B — bag multi-fase (FTCalibrationSampler)                             #
# --------------------------------------------------------------------------- #

def detect_stable_phases(trial_dir: Path) -> list[Tuple[int, int]]:
    """
    Analiza joint_states.csv y devuelve [(t_start_ns, t_end_ns), ...] de cada
    ventana donde el robot estuvo estático, recortando TRIM_SEC en cada extremo.
    """
    # ts → max |velocity| de los joints del brazo (excluye gripper)
    ts_maxvel: dict[int, float] = {}
    with open(trial_dir / "joint_states.csv") as f:
        for row in csv.DictReader(f):
            if "gripper" in row["name"]:
                continue
            t   = int(row["timestamp_ns"])
            vel = abs(float(row["velocity"]))
            if t not in ts_maxvel:
                ts_maxvel[t] = vel
            else:
                ts_maxvel[t] = max(ts_maxvel[t], vel)

    sorted_ts = sorted(ts_maxvel)
    stable_windows: list[Tuple[int, int]] = []
    window_start: int | None = None

    for t in sorted_ts:
        if ts_maxvel[t] < STABLE_VEL_THRESHOLD:
            if window_start is None:
                window_start = t
        else:
            if window_start is not None:
                duration = (t - window_start) / 1e9
                if duration >= MIN_STABLE_SEC:
                    t0 = window_start + int(TRIM_SEC * 1e9)
                    t1 = t            - int(TRIM_SEC * 1e9)
                    if t1 > t0:
                        stable_windows.append((t0, t1))
                window_start = None

    # Última ventana
    if window_start is not None and sorted_ts:
        t = sorted_ts[-1]
        duration = (t - window_start) / 1e9
        if duration >= MIN_STABLE_SEC:
            t0 = window_start + int(TRIM_SEC * 1e9)
            t1 = t            - int(TRIM_SEC * 1e9)
            if t1 > t0:
                stable_windows.append((t0, t1))

    return stable_windows


def load_wrench_window(
    trial_dir: Path,
    wrench_rows: list,
    t_start: int,
    t_end: int,
) -> dict | None:
    """Stats de wrench para la ventana temporal dada."""
    fx, fy, tz = [], [], []
    for row in wrench_rows:
        t = int(row["timestamp_ns"])
        if t_start <= t <= t_end:
            fx.append(float(row["force_x"]))
            fy.append(float(row["force_y"]))
            tz.append(float(row["torque_z"]))
    if len(fx) < MIN_WRENCH_SAMPLES:
        return None
    return {
        "force_x_mean": mean(fx), "force_x_std": stdev(fx) if len(fx) > 1 else 0.0,
        "force_y_mean": mean(fy), "force_y_std": stdev(fy) if len(fy) > 1 else 0.0,
        "torque_z_mean": mean(tz), "torque_z_std": stdev(tz) if len(tz) > 1 else 0.0,
        "n": len(fx),
    }


def load_multiphase_points(trial_dir: Path, bag_label: str) -> list[dict]:
    """
    Procesa un bag multi-fase (FTCalibrationSampler) y devuelve una lista
    de dicts, uno por cada fase estable detectada.
    """
    print(f"\n  Cargando TF (puede tardar unos segundos para bags largos)...")
    timed  = load_tf_timed(trial_dir / "tf.csv")
    cable_prefix, connector_type = detect_connector(timed)

    if cable_prefix is None:
        print(f"  [warn] {bag_label}: no se detectó frame de cable — omitiendo")
        return []

    print(f"  Conector detectado: {connector_type} ({cable_prefix})")

    # Cargar wrench completo una sola vez
    wrench_rows: list[dict] = []
    with open(trial_dir / "fts_broadcaster__wrench.csv") as f:
        wrench_rows = list(csv.DictReader(f))

    # Detectar fases estables
    phases = detect_stable_phases(trial_dir)
    print(f"  Fases estables detectadas: {len(phases)}")

    points: list[dict] = []
    for idx, (t0, t1) in enumerate(phases):
        dur = (t1 - t0) / 1e9
        wrench = load_wrench_window(trial_dir, wrench_rows, t0, t1)
        if wrench is None:
            print(f"    Fase {idx+1}: insuficientes muestras wrench — omitida")
            continue

        ee = compute_ee_offset(timed, cable_prefix, connector_type, t0, t1)
        if ee["offset_x"] is None:
            print(f"    Fase {idx+1}: TF incompleto — omitida")
            continue

        label = f"{bag_label}/fase_{idx+1}"
        print(
            f"    Fase {idx+1} ({dur:.1f}s): "
            f"offset=({ee['offset_x']*1000:+.1f}mm, {ee['offset_y']*1000:+.1f}mm) "
            f"Fx={wrench['force_x_mean']:+.3f}N  Fy={wrench['force_y_mean']:+.3f}N  "
            f"n_wrench={wrench['n']}"
        )
        points.append({
            "label":          label,
            "source":         "multiphase_bag",
            "connector_type": connector_type,
            "phase_idx":      idx,
            "wrench":         wrench,
            "ee":             ee,
        })

    return points


# --------------------------------------------------------------------------- #
# Regresión lineal                                                             #
# --------------------------------------------------------------------------- #

def linreg_gain(offsets: list[float], forces: list[float]) -> float:
    """Regresión forzada por el origen: F = GAIN * offset."""
    num = sum(o * f for o, f in zip(offsets, forces))
    den = sum(o * o for o in offsets)
    return num / den if abs(den) > 1e-12 else float("nan")


def linreg_full(
    offsets: list[float], forces: list[float]
) -> Tuple[float, float, float]:
    """Regresión con intercepto: F = GAIN * offset + BIAS. Retorna (gain, bias, r2)."""
    n = len(offsets)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mo, mf = mean(offsets), mean(forces)
    ss_of = sum((o - mo) * (f - mf) for o, f in zip(offsets, forces))
    ss_oo = sum((o - mo) ** 2 for o in offsets)
    if abs(ss_oo) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    gain  = ss_of / ss_oo
    bias  = mf - gain * mo
    ss_res = sum((f - (gain*o + bias))**2 for o, f in zip(offsets, forces))
    ss_tot = sum((f - mf)**2 for f in forces)
    r2 = 1.0 - ss_res / ss_tot if abs(ss_tot) > 1e-12 else float("nan")
    return gain, bias, r2


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--calib", metavar="DIR", type=Path, action="append", default=[],
        help="Directorio de bag multi-fase extraído (puede repetirse). "
             "Ej: --calib extracted_eval_bags_csv/ft_calib_20260514_204052",
    )
    p.add_argument(
        "--no-trials", action="store_true",
        help="Omitir los bags de FUENTE A (bag_trial_1/2/3). "
             "Útil para evaluar solo los datos de FTCalibrationSampler.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_points: list[dict] = []

    # ------------------------------------------------------------------- #
    # FUENTE A — bags separados originales                                 #
    # ------------------------------------------------------------------- #
    print("=" * 70)
    print("FUENTE A — BAGS SEPARADOS (bag_trial_1/2/3)")
    print("=" * 70)

    trials_active = [] if args.no_trials else TRIALS
    for trial in trials_active:
        trial_dir = ROOT / trial
        if not trial_dir.exists():
            print(f"[warn] No encontrado: {trial_dir}")
            continue
        pt = load_single_bag_point(trial_dir, trial)
        wrench = pt["wrench"]
        ee     = pt["ee"]
        print(f"\n{trial}  [{pt['connector_type'] or '?'}]")
        print(f"  n_wrench={wrench['n']}")
        print(f"  force_x : {wrench['force_x_mean']:+.4f} ± {wrench['force_x_std']:.4f} N")
        print(f"  force_y : {wrench['force_y_mean']:+.4f} ± {wrench['force_y_std']:.4f} N")
        print(f"  torque_z: {wrench['torque_z_mean']:+.5f} ± {wrench['torque_z_std']:.5f} N·m")
        if ee["offset_x"] is not None:
            print(f"  offset_x : {ee['offset_x']*1000:+.2f} mm")
            print(f"  offset_y : {ee['offset_y']*1000:+.2f} mm")
            print(f"  offset_rz: {math.degrees(ee['offset_rz']):+.3f}°")
        else:
            print(f"  [warn] offset no calculable: {ee.get('note','')}")
        all_points.append(pt)

    # ------------------------------------------------------------------- #
    # FUENTE B — bags multi-fase (FTCalibrationSampler)                   #
    # ------------------------------------------------------------------- #
    if args.calib:
        print("\n" + "=" * 70)
        print("FUENTE B — BAGS MULTI-FASE (FTCalibrationSampler)")
        print("=" * 70)
        for calib_dir in args.calib:
            calib_dir = calib_dir if calib_dir.is_absolute() else ROOT / calib_dir
            if not calib_dir.exists():
                print(f"[warn] Directorio no encontrado: {calib_dir}")
                continue
            print(f"\n{calib_dir.name}")
            pts = load_multiphase_points(calib_dir, calib_dir.name)
            all_points.extend(pts)
    else:
        print("\n[info] Sin bags multi-fase. Usa --calib DIR para agregar datos de FTCalibrationSampler.")

    # ------------------------------------------------------------------- #
    # Regresión unificada                                                  #
    # ------------------------------------------------------------------- #
    valid = [p for p in all_points if p["ee"].get("offset_x") is not None]

    print("\n" + "=" * 70)
    print(f"REGRESIÓN UNIFICADA — {len(valid)} puntos de datos")
    print("=" * 70)

    if len(valid) < 3:
        print(f"[warn] Solo {len(valid)} puntos válidos. Se necesitan al menos 3.")
        _fallback_bias(all_points)
        return

    # Tabla de puntos
    print(f"\n  {'Fuente':<45} {'off_x mm':>9} {'off_y mm':>9} {'Fx N':>8} {'Fy N':>8}")
    print(f"  {'-'*45} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")
    for p in valid:
        ee = p["ee"]
        w  = p["wrench"]
        print(
            f"  {p['label']:<45} "
            f"{ee['offset_x']*1000:>+9.2f} {ee['offset_y']*1000:>+9.2f} "
            f"{w['force_x_mean']:>+8.3f} {w['force_y_mean']:>+8.3f}"
        )

    offsets_x  = [p["ee"]["offset_x"]      for p in valid]
    offsets_y  = [p["ee"]["offset_y"]      for p in valid]
    offsets_rz = [p["ee"]["offset_rz"]     for p in valid]
    forces_x   = [p["wrench"]["force_x_mean"] for p in valid]
    forces_y   = [p["wrench"]["force_y_mean"] for p in valid]
    torques_z  = [p["wrench"]["torque_z_mean"] for p in valid]

    gain_x,  bias_x,  r2_x  = linreg_full(offsets_x,  forces_x)
    gain_y,  bias_y,  r2_y  = linreg_full(offsets_y,  forces_y)
    gain_rz, bias_rz, r2_rz = linreg_full(offsets_rz, torques_z)

    print(f"\n  GAIN_X  = {gain_x:+.4f} N/m      (BIAS_X  = {bias_x:+.4f} N,      R²={r2_x:.4f})")
    print(f"  GAIN_Y  = {gain_y:+.4f} N/m      (BIAS_Y  = {bias_y:+.4f} N,      R²={r2_y:.4f})")
    print(f"  GAIN_Rz = {gain_rz:+.4f} N·m/rad  (BIAS_Rz = {bias_rz:+.5f} N·m,  R²={r2_rz:.4f})")

    print("\n  Enviar a Alma:")
    print(f"    GAIN_X  = {gain_x:.4f}   # N/m  {'✓ confiable' if r2_x > 0.8 else '⚠ bajo R²'}")
    print(f"    GAIN_Y  = {gain_y:.4f}   # N/m  {'✓ confiable' if r2_y > 0.8 else '⚠ bajo R²'}")
    print(f"    GAIN_Rz = {gain_rz:.4f}   # N·m/rad  {'✓' if r2_rz > 0.8 else '⚠ bajo R², considerar 0.0'}")
    print(f"    BIAS_X  = {bias_x:.4f}   # N")
    print(f"    BIAS_Y  = {bias_y:.4f}   # N (incluye peso del gripper)")
    print(f"    BIAS_Rz = {bias_rz:.5f}  # N·m")

    # Exportar CSV
    out_path = ROOT / "tools" / "calibration_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "value", "unit", "r2", "n_points"])
        writer.writeheader()
        writer.writerows([
            {"param": "GAIN_X",  "value": f"{gain_x:.6f}",  "unit": "N/m",     "r2": f"{r2_x:.4f}",  "n_points": len(valid)},
            {"param": "GAIN_Y",  "value": f"{gain_y:.6f}",  "unit": "N/m",     "r2": f"{r2_y:.4f}",  "n_points": len(valid)},
            {"param": "GAIN_Rz", "value": f"{gain_rz:.6f}", "unit": "N·m/rad", "r2": f"{r2_rz:.4f}", "n_points": len(valid)},
            {"param": "BIAS_X",  "value": f"{bias_x:.6f}",  "unit": "N",       "r2": "",             "n_points": ""},
            {"param": "BIAS_Y",  "value": f"{bias_y:.6f}",  "unit": "N",       "r2": "",             "n_points": ""},
            {"param": "BIAS_Rz", "value": f"{bias_rz:.6f}", "unit": "N·m",     "r2": "",             "n_points": ""},
        ])
    print(f"\n  [ok] Resultados exportados → {out_path}")
    print(f"  [ok] {len(valid)} puntos de datos: {sum(1 for p in valid if p['source']=='single_bag')} bags separados "
          f"+ {sum(1 for p in valid if p['source']=='multiphase_bag')} fases de calibración")


def _fallback_bias(data: list[dict]) -> None:
    all_fx = [d["wrench"]["force_x_mean"] for d in data]
    all_fy = [d["wrench"]["force_y_mean"] for d in data]
    all_tz = [d["wrench"]["torque_z_mean"] for d in data]
    print(f"\n  BIAS_X  = {mean(all_fx):+.4f} N")
    print(f"  BIAS_Y  = {mean(all_fy):+.4f} N")
    print(f"  BIAS_Rz = {mean(all_tz):+.5f} N·m")
    print("\n  [!] GAIN no calculable — usar GAIN=1.0 y aplicar solo el BIAS.")


if __name__ == "__main__":
    main()
