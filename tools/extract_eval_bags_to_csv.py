#!/usr/bin/env python3
"""Extract selected AIC rosbag topics to one CSV per topic and bag."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Callable

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


DEFAULT_BAGS = [
    "aic_results_from_eval/bag_trial_1_20260512_190629_607",
    "aic_results_from_eval/bag_trial_2_20260512_190737_082",
    "aic_results_from_eval/bag_trial_3_20260512_190908_253",
]

TARGET_TOPICS = {
    "/joint_states",
    "/fts_broadcaster/wrench",
    "/tf",
    "/tf_static",
}


def open_bag(bag_dir: Path) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)
    return reader


def topic_type_map(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def stamp_fields(stamp) -> tuple[int, int]:
    return stamp.sec, stamp.nanosec


def write_joint_state(writer: csv.DictWriter, bag_name: str, timestamp_ns: int, msg) -> int:
    sec, nsec = stamp_fields(msg.header.stamp)
    count = 0
    for index, name in enumerate(msg.name):
        writer.writerow(
            {
                "bag": bag_name,
                "timestamp_ns": timestamp_ns,
                "msg_stamp_sec": sec,
                "msg_stamp_nanosec": nsec,
                "joint_index": index,
                "name": name,
                "position": msg.position[index] if index < len(msg.position) else "",
                "velocity": msg.velocity[index] if index < len(msg.velocity) else "",
                "effort": msg.effort[index] if index < len(msg.effort) else "",
            }
        )
        count += 1
    return count


def write_wrench(writer: csv.DictWriter, bag_name: str, timestamp_ns: int, msg) -> int:
    sec, nsec = stamp_fields(msg.header.stamp)
    writer.writerow(
        {
            "bag": bag_name,
            "timestamp_ns": timestamp_ns,
            "msg_stamp_sec": sec,
            "msg_stamp_nanosec": nsec,
            "frame_id": msg.header.frame_id,
            "force_x": msg.wrench.force.x,
            "force_y": msg.wrench.force.y,
            "force_z": msg.wrench.force.z,
            "torque_x": msg.wrench.torque.x,
            "torque_y": msg.wrench.torque.y,
            "torque_z": msg.wrench.torque.z,
        }
    )
    return 1


def write_tf(writer: csv.DictWriter, bag_name: str, timestamp_ns: int, msg) -> int:
    count = 0
    for transform in msg.transforms:
        sec, nsec = stamp_fields(transform.header.stamp)
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        writer.writerow(
            {
                "bag": bag_name,
                "timestamp_ns": timestamp_ns,
                "msg_stamp_sec": sec,
                "msg_stamp_nanosec": nsec,
                "frame_id": transform.header.frame_id,
                "child_frame_id": transform.child_frame_id,
                "translation_x": translation.x,
                "translation_y": translation.y,
                "translation_z": translation.z,
                "rotation_x": rotation.x,
                "rotation_y": rotation.y,
                "rotation_z": rotation.z,
                "rotation_w": rotation.w,
            }
        )
        count += 1
    return count


WRITERS: dict[str, tuple[list[str], Callable[[csv.DictWriter, str, int, object], int]]] = {
    "/joint_states": (
        [
            "bag",
            "timestamp_ns",
            "msg_stamp_sec",
            "msg_stamp_nanosec",
            "joint_index",
            "name",
            "position",
            "velocity",
            "effort",
        ],
        write_joint_state,
    ),
    "/fts_broadcaster/wrench": (
        [
            "bag",
            "timestamp_ns",
            "msg_stamp_sec",
            "msg_stamp_nanosec",
            "frame_id",
            "force_x",
            "force_y",
            "force_z",
            "torque_x",
            "torque_y",
            "torque_z",
        ],
        write_wrench,
    ),
    "/tf": (
        [
            "bag",
            "timestamp_ns",
            "msg_stamp_sec",
            "msg_stamp_nanosec",
            "frame_id",
            "child_frame_id",
            "translation_x",
            "translation_y",
            "translation_z",
            "rotation_x",
            "rotation_y",
            "rotation_z",
            "rotation_w",
        ],
        write_tf,
    ),
    "/tf_static": (
        [
            "bag",
            "timestamp_ns",
            "msg_stamp_sec",
            "msg_stamp_nanosec",
            "frame_id",
            "child_frame_id",
            "translation_x",
            "translation_y",
            "translation_z",
            "rotation_x",
            "rotation_y",
            "rotation_z",
            "rotation_w",
        ],
        write_tf,
    ),
}


def safe_topic_name(topic: str) -> str:
    return topic.strip("/").replace("/", "__")


def extract_bag(bag_dir: Path, output_root: Path) -> dict[str, int]:
    reader = open_bag(bag_dir)
    types = topic_type_map(reader)
    missing = sorted(TARGET_TOPICS.difference(types))
    if missing:
        print(f"[warn] {bag_dir.name}: missing topics: {', '.join(missing)}")

    msg_types = {topic: get_message(types[topic]) for topic in TARGET_TOPICS if topic in types}
    bag_output = output_root / bag_dir.name
    bag_output.mkdir(parents=True, exist_ok=True)

    files = {}
    csv_writers = {}
    for topic in sorted(TARGET_TOPICS.intersection(types)):
        fieldnames, _ = WRITERS[topic]
        csv_path = bag_output / f"{safe_topic_name(topic)}.csv"
        csv_file = csv_path.open("w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        files[topic] = csv_file
        csv_writers[topic] = writer

    row_counts: Counter[str] = Counter()
    try:
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            if topic not in csv_writers:
                continue
            msg = deserialize_message(data, msg_types[topic])
            _, write_fn = WRITERS[topic]
            row_counts[topic] += write_fn(csv_writers[topic], bag_dir.name, timestamp_ns, msg)
    finally:
        for csv_file in files.values():
            csv_file.close()

    with (bag_output / "summary.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=["bag", "topic", "rows"])
        writer.writeheader()
        for topic in sorted(TARGET_TOPICS):
            writer.writerow({"bag": bag_dir.name, "topic": topic, "rows": row_counts[topic]})

    return dict(row_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bags",
        nargs="*",
        type=Path,
        help="Bag directories. Defaults to the three aic_results_from_eval trials.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extracted_eval_bags_csv"),
        help="Output directory for extracted CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    bags = args.bags or [root / bag for bag in DEFAULT_BAGS]
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    for bag in bags:
        bag_dir = bag if bag.is_absolute() else root / bag
        if not (bag_dir / "metadata.yaml").exists():
            raise FileNotFoundError(f"Not a rosbag directory: {bag_dir}")
        counts = extract_bag(bag_dir, output)
        rendered_counts = ", ".join(f"{topic}={counts.get(topic, 0)}" for topic in sorted(TARGET_TOPICS))
        print(f"[ok] {bag_dir.name}: {rendered_counts}")
    print(f"[done] CSV files written to {output}")


if __name__ == "__main__":
    main()
