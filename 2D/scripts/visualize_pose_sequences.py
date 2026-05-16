#!/usr/bin/env python3
"""Visualize tracked two-character pose sequences on top of the source video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROLE_COLORS = {
    "character_A": (50, 205, 50),
    "character_B": (30, 144, 255),
}

SKELETONS = {
    17: [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ],
    26: [
        (17, 0),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (17, 18),
        (18, 5),
        (18, 6),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (18, 19),
        (19, 11),
        (19, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 24),
        (15, 20),
        (15, 22),
        (16, 25),
        (16, 21),
        (16, 23),
    ],
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_output_root = project_root / "outputs" / "two_character_rtmpose"
    parser = argparse.ArgumentParser(
        description="Overlay tracked 2D keypoints and role labels onto the source video."
    )
    parser.add_argument(
        "--sequence-json",
        default=str(default_output_root / "pose_sequences.json"),
    )
    parser.add_argument(
        "--metadata-json",
        default=str(default_output_root / "metadata.json"),
    )
    parser.add_argument(
        "--output-video",
        default=str(default_output_root / "pose_review_overlay.mp4"),
    )
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--use-smoothed", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=1280)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=3)
    parser.add_argument("--fps", type=float, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_video_path(args: argparse.Namespace, metadata: dict) -> Path:
    if args.video_path:
        return Path(args.video_path).resolve()
    if metadata.get("video_path"):
        return Path(metadata["video_path"]).resolve()
    raise SystemExit("No video path provided and metadata.json does not include one.")


def resolve_frame_points(record: dict, use_smoothed: bool) -> Optional[np.ndarray]:
    key = "smoothed_keypoints" if use_smoothed and record.get("smoothed_keypoints") is not None else "raw_keypoints"
    points = record.get(key)
    if points is None:
        return None

    rows: List[List[float]] = []
    for point in points:
        if point[0] is None or point[1] is None:
            rows.append([np.nan, np.nan])
        else:
            rows.append([float(point[0]), float(point[1])])
    arr = np.asarray(rows, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def resize_frame(frame: np.ndarray, resize_width: Optional[int]) -> Tuple[np.ndarray, float]:
    if not resize_width or frame.shape[1] <= resize_width:
        return frame, 1.0
    scale = resize_width / frame.shape[1]
    resized = cv2.resize(frame, (resize_width, int(round(frame.shape[0] * scale))))
    return resized, scale


def draw_record(
    canvas: np.ndarray,
    record: dict,
    keypoint_count: int,
    scale: float,
    use_smoothed: bool,
    point_radius: int,
    line_thickness: int,
) -> None:
    role = record["role"]
    color = ROLE_COLORS.get(role, (255, 255, 255))

    bbox = record.get("tracker_bbox")
    if bbox is not None:
        x1, y1, x2, y2 = [int(round(v * scale)) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)

        label = role
        if record.get("tracker_id") is not None:
            label += f" | track {record['tracker_id']}"
        if record.get("source_track_id_changed"):
            label += " | ID changed"
        if not record.get("visible"):
            label += " | missing"

        cv2.putText(
            canvas,
            label,
            (x1, max(22, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    points = resolve_frame_points(record, use_smoothed=use_smoothed)
    if points is None:
        return

    skeleton = SKELETONS.get(keypoint_count, [])
    scaled_points = points * scale

    for start, end in skeleton:
        if start >= len(scaled_points) or end >= len(scaled_points):
            continue
        start_point = scaled_points[start]
        end_point = scaled_points[end]
        if np.isnan(start_point).any() or np.isnan(end_point).any():
            continue
        cv2.line(
            canvas,
            (int(round(start_point[0])), int(round(start_point[1]))),
            (int(round(end_point[0])), int(round(end_point[1]))),
            color,
            line_thickness,
            cv2.LINE_AA,
        )

    for idx, point in enumerate(scaled_points):
        if np.isnan(point).any():
            continue
        radius = point_radius + 1 if idx in (17, 18, 19) else point_radius
        cv2.circle(
            canvas,
            (int(round(point[0])), int(round(point[1]))),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    sequence_json = Path(args.sequence_json).resolve()
    metadata_json = Path(args.metadata_json).resolve()
    output_video = Path(args.output_video).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    payload = load_json(sequence_json)
    metadata = load_json(metadata_json) if metadata_json.exists() else payload.get("metadata", {})
    video_path = pick_video_path(args, metadata)

    roles: Dict[str, List[dict]] = payload["roles"]
    role_names = list(roles.keys())
    if not role_names:
        raise SystemExit("No role records found in pose_sequences.json.")

    frame_count = min(len(records) for records in roles.values())
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)

    keypoint_count = metadata.get("keypoint_count")
    if keypoint_count is None:
        sample = next(
            (
                rec
                for records in roles.values()
                for rec in records
                if rec.get("raw_keypoints") is not None
            ),
            None,
        )
        keypoint_count = len(sample["raw_keypoints"]) if sample else 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps or (source_fps if source_fps and source_fps > 0 else 24.0)

    ok, first_frame = cap.read()
    if not ok:
        raise SystemExit(f"Video is empty or unreadable: {video_path}")

    preview_frame, scale = resize_frame(first_frame, args.resize_width)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (preview_frame.shape[1], preview_frame.shape[0]),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break

        canvas, scale = resize_frame(frame, args.resize_width)
        for role_name in role_names:
            record = roles[role_name][frame_index]
            draw_record(
                canvas=canvas,
                record=record,
                keypoint_count=keypoint_count,
                scale=scale,
                use_smoothed=args.use_smoothed,
                point_radius=args.point_radius,
                line_thickness=args.line_thickness,
            )

        status = f"frame {frame_index}"
        if args.use_smoothed:
            status += " | smoothed"
        cv2.putText(
            canvas,
            status,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)

    cap.release()
    writer.release()
    print(f"Saved overlay video to: {output_video}")


if __name__ == "__main__":
    main()
