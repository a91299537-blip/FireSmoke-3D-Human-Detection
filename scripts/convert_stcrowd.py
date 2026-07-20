#!/usr/bin/env python3
"""Convert the official STCrowd release to a KITTI-compatible layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


CALIBRATION = (
    "P0: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "P1: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "P3: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "R0_rect: 1 0 0 0 1 0 0 0 1\n"
    "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
    "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stcrowd-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-sequence-start",
        type=int,
        default=41,
        help="Sequences at or above this identifier are assigned to validation.",
    )
    return parser.parse_args()


def prepare_directories(output_dir: Path) -> Path:
    training_dir = output_dir / "training"
    for name in ("velodyne", "label_2", "calib"):
        (training_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "ImageSets").mkdir(parents=True, exist_ok=True)
    return training_dir


def label_line(item: dict) -> str:
    position = item["position"]
    box = item["boundingbox"]
    length, width, height = box["x"], box["y"], box["z"]
    x_cam = -position["y"]
    y_cam = -position["z"] + height / 2.0
    z_cam = position["x"]
    rotation_y = -item["rotation"] - np.pi / 2
    return (
        f"Pedestrian 0.0 {item['occlusion']} -10.0 0.0 0.0 50.0 50.0 "
        f"{height:.4f} {width:.4f} {length:.4f} "
        f"{x_cam:.4f} {y_cam:.4f} {z_cam:.4f} {rotation_y:.4f}\n"
    )


def main() -> None:
    args = parse_args()
    source_root = args.stcrowd_root.resolve()
    output_dir = args.output_dir.resolve()
    annotation_dir = source_root / "anno"
    if not annotation_dir.is_dir():
        raise SystemExit(f"Missing STCrowd annotation directory: {annotation_dir}")

    training_dir = prepare_directories(output_dir)
    split_ids: dict[str, list[str]] = {"train": [], "val": []}
    frame_index = 0

    annotation_files = sorted(
        (path for path in annotation_dir.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    for annotation_file in annotation_files:
        sequence_id = int(annotation_file.stem)
        split = "val" if sequence_id >= args.validation_sequence_start else "train"
        data = json.loads(annotation_file.read_text(encoding="utf-8"))

        for frame in data.get("frames", []):
            source_points = source_root / Path(frame["frame_name"])
            if not source_points.is_file():
                continue

            sample_id = f"{frame_index:06d}"
            split_ids[split].append(sample_id)
            shutil.copy2(source_points, training_dir / "velodyne" / f"{sample_id}.bin")
            (training_dir / "calib" / f"{sample_id}.txt").write_text(
                CALIBRATION, encoding="ascii"
            )

            labels = [
                label_line(item)
                for item in frame.get("items", [])
                if item.get("category") == "person"
            ]
            (training_dir / "label_2" / f"{sample_id}.txt").write_text(
                "".join(labels), encoding="ascii"
            )
            frame_index += 1

    for split, identifiers in split_ids.items():
        (output_dir / "ImageSets" / f"{split}.txt").write_text(
            "\n".join(identifiers), encoding="ascii"
        )

    print(
        f"Converted {frame_index} frames "
        f"({len(split_ids['train'])} train, {len(split_ids['val'])} validation)."
    )


if __name__ == "__main__":
    main()

