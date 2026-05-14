#!/usr/bin/env python3
"""Create compact diagnostic plots from extracted AIC bag CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aic-plots")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def as_float(value: str, default: float = 0.0) -> float:
    if value == "":
        return default
    return float(value)


def read_joint_series(path: Path) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"time": [], "position": [], "velocity": [], "effort": []}
    )
    first_ts = None
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            timestamp = int(row["timestamp_ns"])
            first_ts = timestamp if first_ts is None else first_ts
            name = row["name"]
            series[name]["time"].append((timestamp - first_ts) / 1e9)
            series[name]["position"].append(as_float(row["position"], math.nan))
            series[name]["velocity"].append(as_float(row["velocity"], math.nan))
            series[name]["effort"].append(as_float(row["effort"], math.nan))
    return series


def read_wrench_series(path: Path) -> dict[str, list[float]]:
    data = defaultdict(list)
    first_ts = None
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            timestamp = int(row["timestamp_ns"])
            first_ts = timestamp if first_ts is None else first_ts
            fx = as_float(row["force_x"])
            fy = as_float(row["force_y"])
            fz = as_float(row["force_z"])
            tx = as_float(row["torque_x"])
            ty = as_float(row["torque_y"])
            tz = as_float(row["torque_z"])
            data["time"].append((timestamp - first_ts) / 1e9)
            data["force_x"].append(fx)
            data["force_y"].append(fy)
            data["force_z"].append(fz)
            data["force_mag"].append(math.sqrt(fx * fx + fy * fy + fz * fz))
            data["torque_x"].append(tx)
            data["torque_y"].append(ty)
            data["torque_z"].append(tz)
            data["torque_mag"].append(math.sqrt(tx * tx + ty * ty + tz * tz))
    return data


def quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conjugate(q: tuple[float, float, float, float]):
    x, y, z, w = q
    return (-x, -y, -z, w)


def rotate_vec(q: tuple[float, float, float, float], v: tuple[float, float, float]):
    qv = (v[0], v[1], v[2], 0.0)
    rotated = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return rotated[:3]


def compose_transform(a, b):
    at, aq = a
    bt, bq = b
    rbt = rotate_vec(aq, bt)
    return (
        (at[0] + rbt[0], at[1] + rbt[1], at[2] + rbt[2]),
        quat_multiply(aq, bq),
    )


def row_transform(row):
    return (
        (
            as_float(row["translation_x"]),
            as_float(row["translation_y"]),
            as_float(row["translation_z"]),
        ),
        (
            as_float(row["rotation_x"]),
            as_float(row["rotation_y"]),
            as_float(row["rotation_z"]),
            as_float(row["rotation_w"], 1.0),
        ),
    )


def compose_path(graph, source: str, target: str, seen: set[str] | None = None):
    if source == target:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    seen = seen or set()
    seen.add(source)
    for child, transform in graph.get(source, {}).items():
        if child in seen:
            continue
        suffix = compose_path(graph, child, target, seen)
        if suffix is not None:
            return compose_transform(transform, suffix)
    return None


def read_tf_summary(path: Path):
    world_cable_by_name = defaultdict(dict)
    cable_sfp_tip_by_name = defaultdict(dict)
    graph = defaultdict(dict)
    first_ts = None

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            timestamp = int(row["timestamp_ns"])
            first_ts = timestamp if first_ts is None else first_ts
            parent = row["frame_id"]
            child = row["child_frame_id"]
            transform = row_transform(row)

            if child not in graph[parent]:
                graph[parent][child] = transform
            if parent == "aic_world" and child.startswith("cable_"):
                world_cable_by_name[child][timestamp] = transform
            elif parent.startswith("cable_") and child == f"{parent}/sfp_tip_link":
                cable_sfp_tip_by_name[parent][timestamp] = transform

    cable_name = ""
    common_timestamps = []
    for candidate in sorted(world_cable_by_name):
        common = sorted(
            set(world_cable_by_name[candidate]).intersection(cable_sfp_tip_by_name[candidate])
        )
        if len(common) > len(common_timestamps):
            cable_name = candidate
            common_timestamps = common

    tip = {"time": [], "x": [], "y": [], "z": []}
    cable = {"time": [], "x": [], "y": [], "z": []}
    if first_ts is not None:
        for timestamp in common_timestamps:
            cable_transform = world_cable_by_name[cable_name][timestamp]
            tip_transform = compose_transform(
                cable_transform, cable_sfp_tip_by_name[cable_name][timestamp]
            )
            t_cable, _ = cable_transform
            t_tip, _ = tip_transform
            rel_time = (timestamp - first_ts) / 1e9
            cable["time"].append(rel_time)
            cable["x"].append(t_cable[0])
            cable["y"].append(t_cable[1])
            cable["z"].append(t_cable[2])
            tip["time"].append(rel_time)
            tip["x"].append(t_tip[0])
            tip["y"].append(t_tip[1])
            tip["z"].append(t_tip[2])

    targets = {}
    for target in (
        "task_board/nic_card_mount_0/sfp_port_0_link_entrance",
        "task_board/nic_card_mount_0/sfp_port_1_link_entrance",
        "task_board/sc_port_0/sc_port_base_link_entrance",
    ):
        transform = compose_path(graph, "aic_world", target)
        if transform is not None:
            targets[target] = transform[0]

    return {"cable_name": cable_name, "cable": cable, "sfp_tip": tip, "targets": targets}


def save_joint_plot(series, key: str, title: str, ylabel: str, output: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in sorted(series):
        ax.plot(series[name]["time"], series[name][key], linewidth=1.2, label=name)
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_wrench_plot(wrench, output: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(wrench["time"], wrench["force_x"], label="Fx")
    axes[0].plot(wrench["time"], wrench["force_y"], label="Fy")
    axes[0].plot(wrench["time"], wrench["force_z"], label="Fz")
    axes[0].plot(wrench["time"], wrench["force_mag"], color="black", linewidth=1.5, label="|F|")
    axes[0].axhline(20.0, color="red", linestyle="--", linewidth=1, label="20 N")
    axes[0].set_title("Wrist force")
    axes[0].set_ylabel("force [N]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(wrench["time"], wrench["torque_x"], label="Tx")
    axes[1].plot(wrench["time"], wrench["torque_y"], label="Ty")
    axes[1].plot(wrench["time"], wrench["torque_z"], label="Tz")
    axes[1].plot(wrench["time"], wrench["torque_mag"], color="black", linewidth=1.5, label="|T|")
    axes[1].set_title("Wrist torque")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("torque [Nm]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_trajectory_plot(tf_summary, output: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    cable = tf_summary["cable"]
    tip = tf_summary["sfp_tip"]
    if cable["x"]:
        ax.plot(cable["x"], cable["y"], alpha=0.55, label=f"{tf_summary['cable_name']} origin")
        ax.scatter(cable["x"][0], cable["y"][0], marker="o", s=45, label="cable start")
        ax.scatter(cable["x"][-1], cable["y"][-1], marker="x", s=55, label="cable end")
    if tip["x"]:
        ax.plot(tip["x"], tip["y"], linewidth=2, label="sfp_tip")
        ax.scatter(tip["x"][0], tip["y"][0], marker="o", s=45, label="sfp_tip start")
        ax.scatter(tip["x"][-1], tip["y"][-1], marker="x", s=55, label="sfp_tip end")
    for name, point in tf_summary["targets"].items():
        ax.scatter(point[0], point[1], marker="*", s=140, label=name.split("/")[-1])
    ax.set_title("XY trajectory from TF")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_z_plot(tf_summary, output: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in ("cable", "sfp_tip"):
        data = tf_summary[label]
        if data["time"]:
            ax.plot(data["time"], data["z"], label=label)
    for name, point in tf_summary["targets"].items():
        ax.axhline(point[2], linestyle="--", linewidth=1, label=f"{name.split('/')[-1]} z")
    ax.set_title("Cable / tip height from TF")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_bag(bag_dir: Path, output_root: Path) -> None:
    output_dir = output_root / bag_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    joints = read_joint_series(bag_dir / "joint_states.csv")
    wrench = read_wrench_series(bag_dir / "fts_broadcaster__wrench.csv")
    tf_summary = read_tf_summary(bag_dir / "tf.csv")

    save_joint_plot(
        joints,
        "position",
        f"{bag_dir.name}: joint positions",
        "position [rad or m]",
        output_dir / "01_joint_positions.png",
    )
    save_joint_plot(
        joints,
        "effort",
        f"{bag_dir.name}: joint efforts",
        "effort [Nm or N]",
        output_dir / "02_joint_efforts.png",
    )
    save_wrench_plot(wrench, output_dir / "03_wrist_wrench.png")
    save_trajectory_plot(tf_summary, output_dir / "04_tf_xy_trajectory.png")
    save_z_plot(tf_summary, output_dir / "05_tf_z_over_time.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("extracted_eval_bags_csv"),
        help="Directory created by extract_eval_bags_to_csv.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_bag_summary_plots"),
        help="Directory where PNG plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_root = args.input if args.input.is_absolute() else root / args.input
    output_root = args.output if args.output.is_absolute() else root / args.output
    output_root.mkdir(parents=True, exist_ok=True)

    bag_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not bag_dirs:
        raise FileNotFoundError(f"No extracted bag directories found in {input_root}")

    for bag_dir in bag_dirs:
        plot_bag(bag_dir, output_root)
        print(f"[ok] wrote plots for {bag_dir.name}")
    print(f"[done] plots written to {output_root}")


if __name__ == "__main__":
    main()
