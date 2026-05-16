#!/usr/bin/env python3
"""Two-in-One style two-person 3D reconstruction pipeline.

This module implements the engineering path described in the project report:
2D observations -> WHAM-compatible initialization -> shared world alignment ->
Two-in-One pair representation -> occlusion masks -> rule-based masked repair ->
contact/physics refinement -> asset-oriented exports.

The code can consume real WHAM/SMPL outputs later through the same motion schema.
When WHAM is not installed, the default path uses deterministic monocular
kinematic lifting from the project 2D observations so the whole asset pipeline is
testable end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROLE_NAMES = ("character_A", "character_B")

HALPE26_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
]

NAME_TO_INDEX = {name: idx for idx, name in enumerate(HALPE26_NAMES)}

SKELETON_EDGES = [
    (19, 18),
    (18, 17),
    (17, 0),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (18, 5),
    (18, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (19, 11),
    (19, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (15, 20),
    (15, 22),
    (15, 24),
    (16, 21),
    (16, 23),
    (16, 25),
]

PARENTS = {
    18: 19,
    17: 18,
    0: 17,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 18,
    6: 18,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 19,
    12: 19,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    20: 15,
    22: 15,
    24: 15,
    21: 16,
    23: 16,
    25: 16,
}

LEFT_JOINTS = {5, 7, 9, 11, 13, 15, 20, 22, 24, 1, 3}
RIGHT_JOINTS = {6, 8, 10, 12, 14, 16, 21, 23, 25, 2, 4}
FOOT_JOINTS = [15, 16, 20, 21, 22, 23, 24, 25]
ANKLE_JOINTS = [15, 16]
WRIST_JOINTS = [9, 10]
TORSO_JOINTS = [5, 6, 11, 12, 18, 19]

ANTHRO_BONE_FRACTIONS = {
    (19, 18): 0.30,
    (18, 17): 0.13,
    (17, 0): 0.08,
    (0, 1): 0.035,
    (0, 2): 0.035,
    (1, 3): 0.06,
    (2, 4): 0.06,
    (18, 5): 0.12,
    (18, 6): 0.12,
    (5, 7): 0.19,
    (7, 9): 0.17,
    (6, 8): 0.19,
    (8, 10): 0.17,
    (19, 11): 0.09,
    (19, 12): 0.09,
    (11, 13): 0.25,
    (13, 15): 0.25,
    (12, 14): 0.25,
    (14, 16): 0.25,
    (15, 20): 0.07,
    (15, 22): 0.06,
    (15, 24): 0.04,
    (16, 21): 0.07,
    (16, 23): 0.06,
    (16, 25): 0.04,
}

ROLE_COLORS = {
    "character_A": (70, 210, 80),
    "character_B": (60, 150, 255),
}


@dataclass
class TwoDTrack:
    role: str
    stable_track_id: int
    keypoints: np.ndarray  # [T, J, 3], x/y/conf in pixels
    valid: np.ndarray  # [T, J]
    bbox_xyxy: np.ndarray  # [T, 4]
    visible: np.ndarray  # [T]
    recovered: np.ndarray  # [T]
    keypoint_source: List[List[str]]


@dataclass
class Pair2DData:
    metadata: dict
    tracks: Dict[str, TwoDTrack]

    @property
    def frame_count(self) -> int:
        return min(track.keypoints.shape[0] for track in self.tracks.values())

    @property
    def fps(self) -> float:
        return float(self.metadata.get("fps") or 30.0)

    @property
    def width(self) -> int:
        return int(self.metadata.get("frame_width") or self.metadata.get("width") or 1)

    @property
    def height(self) -> int:
        return int(self.metadata.get("frame_height") or self.metadata.get("height") or 1)


@dataclass
class Motion3D:
    role: str
    stable_track_id: int
    joints: np.ndarray  # [T, J, 3], meters, Y-up
    confidence: np.ndarray  # [T, J]
    visible_mask: np.ndarray  # [T, J]
    root: np.ndarray  # [T, 3]
    yaw: np.ndarray  # [T]
    foot_contact: np.ndarray  # [T, 2]
    source: str


@dataclass
class WhamSubjectResult:
    subject_id: str
    frame_ids: np.ndarray
    joints_world: Optional[np.ndarray]
    trans_world: Optional[np.ndarray]
    pose_world: Optional[np.ndarray]
    pose_body: Optional[np.ndarray]
    betas: Optional[np.ndarray]
    source_path: str
    source_format: str


@dataclass
class WhamResultBundle:
    subjects: Dict[str, WhamSubjectResult]
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="Two-in-One two-person 3D reconstruction pipeline.")
    parser.add_argument("--video", "--video-path", dest="video_path", default=str(project_root / "videos" / "video_2.mp4"))
    parser.add_argument("--output-root", "--output", dest="output_root", default=str(script_dir.parent / "results" / "two_in_one_alignment" / "video_2"))
    parser.add_argument("--two-d-root", default=None, help="Existing 2D output folder containing wham_2d_observations.json.")
    parser.add_argument("--force-2d", action="store_true", help="Re-run the 2D pipeline even if 2D output exists.")
    parser.add_argument("--skip-2d", action="store_true", help="Require --two-d-root or existing output_root/2d.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-2d-preview", action="store_true", default=True)
    parser.add_argument("--no-preview", action="store_true", help="Skip 3D preview video.")
    parser.add_argument(
        "--reconstruction-backend",
        choices=("auto", "fallback", "wham"),
        default="auto",
        help="Use real WHAM, the deterministic fallback, or WHAM when configured and fallback otherwise.",
    )
    parser.add_argument("--wham-root", default=None, help="Path to the official WHAM checkout.")
    parser.add_argument("--wham-python", default=None, help="Python executable from the WHAM environment.")
    parser.add_argument("--wham-output-root", default=None, help="Folder where WHAM intermediate/output files are stored.")
    parser.add_argument("--wham-results-path", default=None, help="Existing WHAM result file (.npz/.pkl/.pth) to import.")
    parser.add_argument(
        "--wham-track-map",
        default=None,
        help="Map project roles to WHAM subject ids, e.g. character_A=0,character_B=1.",
    )
    parser.add_argument("--wham-force", action="store_true", help="Re-run WHAM even if cached WHAM output exists.")
    parser.add_argument(
        "--wham-local-only",
        action="store_true",
        help="Run WHAM without global SLAM/world trajectory estimation.",
    )
    parser.add_argument("--wham-calib", default=None, help="Camera calibration file passed to WHAM.")
    parser.add_argument("--wham-visualize", action="store_true", help="Ask WHAM to render its mesh visualization.")
    parser.add_argument("--wham-run-smplify", action="store_true", help="Ask WHAM to run Temporal SMPLify.")
    parser.add_argument("--body-height", type=float, default=1.70)
    parser.add_argument("--focal-scale", type=float, default=1.20)
    parser.add_argument("--min-keypoint-score", type=float, default=0.20)
    parser.add_argument("--short-gap", type=int, default=8)
    parser.add_argument("--smooth-alpha", type=float, default=0.45)
    parser.add_argument("--bone-enforce-iterations", type=int, default=2)
    parser.add_argument("--ground-contact-height", type=float, default=0.045)
    parser.add_argument("--ground-contact-speed", type=float, default=0.20)
    parser.add_argument("--root-min-distance", type=float, default=0.30)
    parser.add_argument("--contact-distance", type=float, default=0.22)
    parser.add_argument("--contact-min-frames", type=int, default=4)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite_point(point: Sequence[float]) -> bool:
    return len(point) >= 2 and np.isfinite(float(point[0])) and np.isfinite(float(point[1]))


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if not np.isfinite(box_a).all() or not np.isfinite(box_b).all():
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-8 else 0.0


def load_2d_observations(path: Path, max_frames: Optional[int] = None) -> Pair2DData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    tracks: Dict[str, TwoDTrack] = {}
    frame_count = max_frames
    if frame_count is None:
        frame_count = min(len(track.get("frames", [])) for track in payload.get("tracks", []))

    for track_payload in payload.get("tracks", []):
        role = str(track_payload.get("role"))
        frames = track_payload.get("frames", [])[:frame_count]
        keypoints = np.zeros((len(frames), len(HALPE26_NAMES), 3), dtype=np.float32)
        valid = np.zeros((len(frames), len(HALPE26_NAMES)), dtype=bool)
        bbox = np.full((len(frames), 4), np.nan, dtype=np.float32)
        visible = np.zeros((len(frames),), dtype=bool)
        recovered = np.zeros((len(frames),), dtype=bool)
        sources: List[List[str]] = []

        for idx, frame in enumerate(frames):
            points = frame.get("keypoints_halpe26") or []
            point_count = min(len(points), len(HALPE26_NAMES))
            for joint_idx in range(point_count):
                point = points[joint_idx]
                if len(point) >= 3:
                    keypoints[idx, joint_idx] = [float(point[0]), float(point[1]), float(point[2])]
            mask = frame.get("valid_keypoint_mask") or [False] * len(HALPE26_NAMES)
            valid[idx, : min(len(mask), len(HALPE26_NAMES))] = [bool(value) for value in mask[: len(HALPE26_NAMES)]]
            bbox_xyxy = frame.get("bbox_xyxy")
            if bbox_xyxy is not None and len(bbox_xyxy) >= 4:
                bbox[idx] = np.asarray([float(value) for value in bbox_xyxy[:4]], dtype=np.float32)
            visible[idx] = bool(frame.get("visible"))
            recovered[idx] = bool(frame.get("is_recovered"))
            source = frame.get("keypoint_source") or ["missing"] * len(HALPE26_NAMES)
            sources.append([str(value) for value in source[: len(HALPE26_NAMES)]])

        keypoints[..., 2] = np.where(np.isfinite(keypoints[..., 2]), keypoints[..., 2], 0.0)
        valid &= keypoints[..., 2] >= 0.0
        tracks[role] = TwoDTrack(
            role=role,
            stable_track_id=int(track_payload.get("stable_track_id", len(tracks))),
            keypoints=keypoints,
            valid=valid,
            bbox_xyxy=bbox,
            visible=visible,
            recovered=recovered,
            keypoint_source=sources,
        )

    missing = [role for role in ROLE_NAMES if role not in tracks]
    if missing:
        raise ValueError(f"Missing required roles in 2D observations: {missing}")
    return Pair2DData(metadata=metadata, tracks=tracks)


def run_2d_pipeline(args: argparse.Namespace, output_root: Path) -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    two_d_root = ensure_dir(output_root / "2d")
    observations_path = two_d_root / "wham_2d_observations.json"
    if observations_path.exists() and not args.force_2d:
        return observations_path
    if args.skip_2d:
        raise SystemExit(f"2D observations not found and --skip-2d was set: {observations_path}")

    video_path = Path(args.video_path).resolve()
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--video",
        str(video_path),
        "--output",
        str(two_d_root),
    ]
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.no_2d_preview:
        cmd.append("--no-preview")
    print("Running 2D pipeline:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root / "2D"), check=True)
    return observations_path


def normalize_subject_id(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def to_numpy_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object and array.shape == ():
        array = np.asarray(array.item())
    return array


def first_array(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in payload and payload[key] is not None:
            return to_numpy_array(payload[key])
    return None


def parse_wham_track_map(mapping: Optional[str]) -> Dict[str, str]:
    if not mapping:
        return {}
    result: Dict[str, str] = {}
    for item in mapping.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --wham-track-map entry: {item}")
        role, subject_id = item.split("=", 1)
        role = role.strip()
        if role not in ROLE_NAMES:
            raise ValueError(f"Unknown role in --wham-track-map: {role}")
        result[role] = normalize_subject_id(subject_id.strip())
    return result


def load_pickle_like(path: Path) -> Any:
    try:
        import joblib  # type: ignore

        return joblib.load(str(path))
    except ModuleNotFoundError:
        import pickle

        try:
            with path.open("rb") as fp:
                return pickle.load(fp)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load WHAM pickle/PTH result at {path}. Install joblib in this "
                "environment or pass the bridge-generated project_wham_output.npz."
            ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Could not load WHAM pickle/PTH result at {path}. Install joblib in this "
            "environment or pass the bridge-generated project_wham_output.npz."
        ) from exc


def load_project_wham_npz(path: Path) -> WhamResultBundle:
    archive = np.load(path, allow_pickle=True)
    files = set(archive.files)

    metadata: Dict[str, Any] = {"source_path": str(path), "source_format": "project_npz"}
    if "metadata_json" in files:
        try:
            metadata.update(json.loads(str(archive["metadata_json"].item())))
        except Exception:
            metadata["metadata_json_error"] = "Could not parse metadata_json from npz."

    if "subject_ids" not in files:
        raise ValueError(f"WHAM npz does not contain subject_ids: {path}")

    subjects: Dict[str, WhamSubjectResult] = {}
    for raw_subject_id in archive["subject_ids"].tolist():
        subject_id = normalize_subject_id(raw_subject_id)
        prefix = f"subject_{subject_id}_"
        frame_ids = archive[f"{prefix}frame_ids"].astype(np.int64) if f"{prefix}frame_ids" in files else None
        if frame_ids is None:
            raise ValueError(f"WHAM npz is missing {prefix}frame_ids")
        subjects[subject_id] = WhamSubjectResult(
            subject_id=subject_id,
            frame_ids=frame_ids.reshape(-1),
            joints_world=archive[f"{prefix}joints_world"].astype(np.float32)
            if f"{prefix}joints_world" in files
            else None,
            trans_world=archive[f"{prefix}trans_world"].astype(np.float32)
            if f"{prefix}trans_world" in files
            else None,
            pose_world=archive[f"{prefix}pose_world"].astype(np.float32)
            if f"{prefix}pose_world" in files
            else None,
            pose_body=archive[f"{prefix}pose_body"].astype(np.float32)
            if f"{prefix}pose_body" in files
            else None,
            betas=archive[f"{prefix}betas"].astype(np.float32) if f"{prefix}betas" in files else None,
            source_path=str(path),
            source_format="project_npz",
        )
    return WhamResultBundle(subjects=subjects, metadata=metadata)


def load_official_wham_pickle(path: Path) -> WhamResultBundle:
    raw = load_pickle_like(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected WHAM pickle/PTH to contain a subject dict: {path}")

    subjects: Dict[str, WhamSubjectResult] = {}
    for raw_subject_id, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        subject_id = normalize_subject_id(raw_subject_id)
        frame_ids = first_array(payload, ("frame_ids", "frame_id", "frames"))
        trans_world = first_array(payload, ("trans_world", "trans", "poses_root_world_trans"))
        pose_world = first_array(payload, ("pose_world", "poses_world"))
        pose_body = first_array(payload, ("poses_body", "pose_body"))
        betas = first_array(payload, ("betas", "shape"))
        joints_world = first_array(
            payload,
            (
                "joints_world",
                "joints3d_world",
                "smpl_joints_world",
                "joints_3d_world",
                "joints",
                "joints3d",
            ),
        )

        if frame_ids is None:
            lengths = [
                len(value)
                for value in (joints_world, trans_world, pose_world, pose_body)
                if value is not None and value.ndim > 0
            ]
            if not lengths:
                continue
            frame_ids = np.arange(min(lengths), dtype=np.int64)

        subjects[subject_id] = WhamSubjectResult(
            subject_id=subject_id,
            frame_ids=frame_ids.astype(np.int64).reshape(-1),
            joints_world=joints_world.astype(np.float32) if joints_world is not None else None,
            trans_world=trans_world.astype(np.float32) if trans_world is not None else None,
            pose_world=pose_world.astype(np.float32) if pose_world is not None else None,
            pose_body=pose_body.astype(np.float32) if pose_body is not None else None,
            betas=betas.astype(np.float32) if betas is not None else None,
            source_path=str(path),
            source_format="official_pickle",
        )

    if not subjects:
        raise ValueError(f"No WHAM subjects could be read from {path}")

    return WhamResultBundle(
        subjects=subjects,
        metadata={"source_path": str(path), "source_format": "official_pickle"},
    )


def load_wham_results(path: Path) -> WhamResultBundle:
    if not path.exists():
        raise FileNotFoundError(f"WHAM result file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_project_wham_npz(path)
    if suffix in {".pkl", ".pth"}:
        return load_official_wham_pickle(path)
    raise ValueError(f"Unsupported WHAM result format: {path}")


def run_external_wham(
    observations_path: Path,
    output_root: Path,
    video_path: Path,
    args: argparse.Namespace,
) -> Path:
    if not args.wham_root:
        raise RuntimeError("Set --wham-root to the official WHAM checkout, or pass --wham-results-path.")

    wham_root = Path(args.wham_root).resolve()
    if not wham_root.exists():
        raise FileNotFoundError(f"WHAM root not found: {wham_root}")

    wham_python = Path(args.wham_python).resolve() if args.wham_python else Path(sys.executable)
    bridge_script = Path(__file__).resolve().with_name("run_wham_with_observations.py")
    wham_output_root = Path(args.wham_output_root).resolve() if args.wham_output_root else output_root.parent / "wham"
    expected_result = wham_output_root / video_path.stem / "project_wham_output.npz"
    if expected_result.exists() and not args.wham_force:
        return expected_result

    cmd = [
        str(wham_python),
        str(bridge_script),
        "--wham-root",
        str(wham_root),
        "--video",
        str(video_path),
        "--observations",
        str(observations_path),
        "--output-root",
        str(wham_output_root),
    ]
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.wham_force:
        cmd.append("--force")
    if args.wham_local_only:
        cmd.append("--local-only")
    if args.wham_calib:
        cmd.extend(["--calib", str(Path(args.wham_calib).resolve())])
    if args.wham_visualize:
        cmd.append("--visualize")
    if args.wham_run_smplify:
        cmd.append("--run-smplify")

    print("Running WHAM bridge:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(wham_root), check=True)
    if not expected_result.exists():
        raise RuntimeError(f"WHAM bridge finished but did not produce: {expected_result}")
    return expected_result


def load_or_run_wham(
    observations_path: Path,
    output_root: Path,
    video_path: Path,
    args: argparse.Namespace,
) -> WhamResultBundle:
    if args.wham_results_path:
        result_path = Path(args.wham_results_path).resolve()
    else:
        result_path = run_external_wham(observations_path, output_root, video_path, args)
    bundle = load_wham_results(result_path)
    bundle.metadata.setdefault("result_path", str(result_path))
    bundle.metadata["local_only"] = bool(args.wham_local_only)
    return bundle


def median_bbox_height(track: TwoDTrack) -> float:
    heights = track.bbox_xyxy[:, 3] - track.bbox_xyxy[:, 1]
    heights = heights[np.isfinite(heights) & (heights > 5.0)]
    return float(np.median(heights)) if len(heights) else 300.0


def root_uv_for_frame(track: TwoDTrack, frame_idx: int) -> np.ndarray:
    keypoints = track.keypoints[frame_idx]
    valid = track.valid[frame_idx]
    for joint_idx in (19,):
        if valid[joint_idx] and keypoints[joint_idx, 2] > 0:
            return keypoints[joint_idx, :2].astype(np.float32)
    hip_indices = [idx for idx in (11, 12) if valid[idx] and keypoints[idx, 2] > 0]
    if hip_indices:
        return keypoints[hip_indices, :2].mean(axis=0).astype(np.float32)
    bbox = track.bbox_xyxy[frame_idx]
    if np.isfinite(bbox).all():
        return np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.58], dtype=np.float32)
    return np.zeros((2,), dtype=np.float32)


def estimate_bone_lengths(track: TwoDTrack, body_height: float) -> Dict[Tuple[int, int], float]:
    bbox_height = median_bbox_height(track)
    meters_per_px = body_height / max(1.0, bbox_height)
    lengths: Dict[Tuple[int, int], float] = {}
    for parent, child in SKELETON_EDGES:
        observed = []
        for frame_idx in range(track.keypoints.shape[0]):
            if not (track.valid[frame_idx, parent] and track.valid[frame_idx, child]):
                continue
            conf = min(track.keypoints[frame_idx, parent, 2], track.keypoints[frame_idx, child, 2])
            if conf < 0.35:
                continue
            delta = track.keypoints[frame_idx, child, :2] - track.keypoints[frame_idx, parent, :2]
            length = float(np.linalg.norm(delta)) * meters_per_px
            if np.isfinite(length) and length > 0.01:
                observed.append(length)
        prior = ANTHRO_BONE_FRACTIONS.get((parent, child), 0.08) * body_height
        if observed:
            observed_median = float(np.median(observed)) * 1.06
            lengths[(parent, child)] = float(np.clip(max(prior, observed_median), prior * 0.65, prior * 1.85))
        else:
            lengths[(parent, child)] = prior
    return lengths


def joint_depth_sign(joint_idx: int) -> float:
    if joint_idx in LEFT_JOINTS:
        return 1.0
    if joint_idx in RIGHT_JOINTS:
        return -1.0
    return 0.0


def lift_track_to_3d(
    track: TwoDTrack,
    data: Pair2DData,
    body_height: float,
    focal_scale: float,
    min_keypoint_score: float,
) -> Motion3D:
    frame_count, joint_count, _ = track.keypoints.shape
    frame_diag = math.hypot(float(data.width), float(data.height))
    focal_px = max(1.0, frame_diag * focal_scale)
    center = np.asarray([data.width * 0.5, data.height * 0.5], dtype=np.float32)
    bbox_height = np.maximum(1.0, track.bbox_xyxy[:, 3] - track.bbox_xyxy[:, 1])
    fallback_height = median_bbox_height(track)
    bbox_height = np.where(np.isfinite(bbox_height), bbox_height, fallback_height)
    bone_lengths = estimate_bone_lengths(track, body_height)

    joints = np.full((frame_count, joint_count, 3), np.nan, dtype=np.float32)
    confidence = np.clip(track.keypoints[..., 2], 0.0, 1.0).astype(np.float32)
    visible_mask = (confidence >= min_keypoint_score) & track.valid
    root = np.zeros((frame_count, 3), dtype=np.float32)

    for frame_idx in range(frame_count):
        root_uv = root_uv_for_frame(track, frame_idx)
        scale = body_height / max(1.0, float(bbox_height[frame_idx]))
        z_cam = focal_px * body_height / max(1.0, float(bbox_height[frame_idx]))
        root_x = (float(root_uv[0]) - float(center[0])) / focal_px * z_cam
        root_z = z_cam
        root[frame_idx] = [root_x, 0.0, root_z]

        local = np.full((joint_count, 3), np.nan, dtype=np.float32)
        local[19] = [0.0, 0.0, 0.0]
        for joint_idx in range(joint_count):
            if not visible_mask[frame_idx, joint_idx]:
                continue
            point = track.keypoints[frame_idx, joint_idx, :2]
            local[joint_idx, 0] = (float(point[0]) - float(root_uv[0])) * scale
            local[joint_idx, 1] = -(float(point[1]) - float(root_uv[1])) * scale
            local[joint_idx, 2] = 0.0

        # Traverse repeatedly because some parents may appear later in the edge list.
        for _ in range(3):
            for child, parent in PARENTS.items():
                if child >= joint_count or parent >= joint_count:
                    continue
                if not np.isfinite(local[child, :2]).all() or not np.isfinite(local[parent, :2]).all():
                    continue
                if not np.isfinite(local[parent, 2]):
                    continue
                edge = (parent, child)
                target_length = bone_lengths.get(edge, ANTHRO_BONE_FRACTIONS.get(edge, 0.08) * body_height)
                dxy = local[child, :2] - local[parent, :2]
                dxy_len = float(np.linalg.norm(dxy))
                z_delta = math.sqrt(max(0.0, target_length * target_length - dxy_len * dxy_len))
                local[child, 2] = local[parent, 2] + joint_depth_sign(child) * z_delta * 0.55

        for joint_idx in range(joint_count):
            if np.isfinite(local[joint_idx]).all():
                joints[frame_idx, joint_idx] = root[frame_idx] + local[joint_idx]

    joints = fill_missing_motion(joints, visible_mask, max_gap=24)
    root = joints[:, 19, :].copy()
    yaw = estimate_root_yaw(joints)
    return Motion3D(
        role=track.role,
        stable_track_id=track.stable_track_id,
        joints=joints,
        confidence=confidence,
        visible_mask=visible_mask,
        root=root,
        yaw=yaw,
        foot_contact=np.zeros((frame_count, 2), dtype=bool),
        source="monocular_kinematic_lift_wham_fallback",
    )


def smpl24_to_halpe26(joints: np.ndarray) -> np.ndarray:
    if joints.ndim == 4 and joints.shape[0] == 1:
        joints = joints[0]
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected WHAM joints to have shape [T, J, 3], got {joints.shape}")
    if joints.shape[1] == len(HALPE26_NAMES):
        return joints.astype(np.float32)
    if joints.shape[1] < 24:
        raise ValueError(
            "WHAM result has too few joints for SMPL24->HALPE26 mapping. "
            f"Got {joints.shape[1]} joints."
        )

    # WHAM's SMPL regressor follows the standard 24-joint SMPL order.
    mapping = {
        0: 15,  # nose -> head
        1: 15,  # left_eye -> head
        2: 15,  # right_eye -> head
        3: 15,  # left_ear -> head
        4: 15,  # right_ear -> head
        5: 16,  # left_shoulder
        6: 17,  # right_shoulder
        7: 18,  # left_elbow
        8: 19,  # right_elbow
        9: 20,  # left_wrist
        10: 21,  # right_wrist
        11: 1,  # left_hip
        12: 2,  # right_hip
        13: 4,  # left_knee
        14: 5,  # right_knee
        15: 7,  # left_ankle
        16: 8,  # right_ankle
        17: 15,  # head
        18: 12,  # neck
        19: 0,  # pelvis/root
        20: 10,  # left_big_toe
        21: 11,  # right_big_toe
        22: 10,  # left_small_toe
        23: 11,  # right_small_toe
        24: 7,  # left_heel
        25: 8,  # right_heel
    }
    output = np.zeros((joints.shape[0], len(HALPE26_NAMES), 3), dtype=np.float32)
    for halpe_idx, smpl_idx in mapping.items():
        output[:, halpe_idx] = joints[:, smpl_idx]
    return output


def scatter_wham_series(
    series: Optional[np.ndarray],
    frame_ids: np.ndarray,
    frame_count: int,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if series is None:
        return None, np.zeros((frame_count,), dtype=bool)
    series = np.asarray(series, dtype=np.float32)
    if series.ndim >= 3 and series.shape[0] == 1 and series.shape[1] == len(frame_ids):
        series = series[0]
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    count = min(len(frame_ids), len(series))
    if count == 0:
        shape = (frame_count,) + tuple(series.shape[1:])
        return np.full(shape, np.nan, dtype=np.float32), np.zeros((frame_count,), dtype=bool)

    shape = (frame_count,) + tuple(series.shape[1:])
    timeline = np.full(shape, np.nan, dtype=np.float32)
    frame_ids = np.asarray(frame_ids[:count], dtype=np.int64).reshape(-1)
    valid = (frame_ids >= 0) & (frame_ids < frame_count)
    if np.any(valid):
        timeline[frame_ids[valid]] = series[:count][valid]
    return timeline, valid_frame_mask(frame_ids[valid], frame_count)


def valid_frame_mask(frame_ids: np.ndarray, frame_count: int) -> np.ndarray:
    mask = np.zeros((frame_count,), dtype=bool)
    if len(frame_ids):
        mask[np.asarray(frame_ids, dtype=np.int64)] = True
    return mask


def motion_from_wham_subject(
    role: str,
    subject: WhamSubjectResult,
    data: Pair2DData,
    args: argparse.Namespace,
) -> Motion3D:
    frame_count = data.frame_count
    track = data.tracks[role]
    confidence = np.clip(track.keypoints[:frame_count, :, 2], 0.0, 1.0).astype(np.float32)
    visible_mask = (confidence >= args.min_keypoint_score) & track.valid[:frame_count]

    trans_world, trans_valid = scatter_wham_series(subject.trans_world, subject.frame_ids, frame_count)
    joints_source = "wham_smpl_joints_world"
    wham_valid = trans_valid.copy()

    if subject.joints_world is not None:
        mapped_joints = smpl24_to_halpe26(subject.joints_world)
        joints, joints_valid = scatter_wham_series(mapped_joints, subject.frame_ids, frame_count)
        if joints is None:
            raise ValueError(f"WHAM subject {subject.subject_id} did not provide usable joints.")
        wham_valid |= joints_valid
        observed_joint_mask = np.isfinite(joints).all(axis=2)
        joints = fill_missing_motion(joints, observed_joint_mask, max_gap=24)
    else:
        fallback = lift_track_to_3d(
            track,
            data,
            body_height=args.body_height,
            focal_scale=args.focal_scale,
            min_keypoint_score=args.min_keypoint_score,
        )
        local_joints = fallback.joints - fallback.root[:, None, :]
        if trans_world is not None and np.any(np.isfinite(trans_world)):
            root_observed = np.isfinite(trans_world).all(axis=1)
            wham_valid |= root_observed
            filled_root = fill_missing_motion(trans_world.reshape(frame_count, 1, 3), root_observed[:, None], 24)
            root = filled_root.reshape(frame_count, 3)
            joints = root[:, None, :] + local_joints
            joints_source = "wham_root_with_kinematic_local_joints"
        else:
            joints = fallback.joints.copy()
            wham_valid[:] = True
            joints_source = "wham_import_missing_joints_using_fallback"

    if not np.any(wham_valid):
        raise ValueError(f"WHAM subject {subject.subject_id} has no frames on the project timeline.")

    visible_mask &= wham_valid[:, None]
    if trans_world is not None and np.any(np.isfinite(trans_world)) and subject.joints_world is None:
        root = joints[:, 19].copy()
    else:
        root = joints[:, 19].copy()
    yaw = estimate_root_yaw(joints)

    suffix = "_local_only" if args.wham_local_only else ""
    return Motion3D(
        role=role,
        stable_track_id=track.stable_track_id,
        joints=joints.astype(np.float32),
        confidence=confidence,
        visible_mask=visible_mask,
        root=root.astype(np.float32),
        yaw=yaw.astype(np.float32),
        foot_contact=np.zeros((frame_count, 2), dtype=bool),
        source=f"{joints_source}{suffix}",
    )


def build_motions_from_wham(
    bundle: WhamResultBundle,
    data: Pair2DData,
    args: argparse.Namespace,
) -> Dict[str, Motion3D]:
    track_map = parse_wham_track_map(args.wham_track_map)
    motions: Dict[str, Motion3D] = {}
    for role in ROLE_NAMES:
        default_subject_id = normalize_subject_id(data.tracks[role].stable_track_id)
        subject_id = track_map.get(role, default_subject_id)
        if subject_id not in bundle.subjects:
            available = ", ".join(sorted(bundle.subjects.keys()))
            raise ValueError(
                f"WHAM result does not contain subject {subject_id} for {role}. "
                f"Available subjects: {available}. Use --wham-track-map if WHAM ids differ."
            )
        motions[role] = motion_from_wham_subject(role, bundle.subjects[subject_id], data, args)
    return motions


def fill_missing_motion(values: np.ndarray, observed_mask: np.ndarray, max_gap: int) -> np.ndarray:
    filled = values.copy()
    frame_count = values.shape[0]
    trailing_shape = values.shape[1:]
    for index in np.ndindex(trailing_shape[:-1]):
        for axis in range(values.shape[-1]):
            series = values[(slice(None),) + index + (axis,)]
            if observed_mask.ndim == 2:
                valid = observed_mask[(slice(None),) + index]
            else:
                valid = np.isfinite(series)
            valid = valid & np.isfinite(series)
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices) == 0:
                filled[(slice(None),) + index + (axis,)] = 0.0
                continue
            interp = np.interp(np.arange(frame_count), valid_indices, series[valid_indices])
            # Long unobserved gaps are still filled, but later exports keep the mask so
            # downstream optimizers can treat them as weak/inpainted motion.
            filled[(slice(None),) + index + (axis,)] = interp
    return filled.astype(np.float32)


def apply_bidirectional_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    def forward_pass(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        previous = out[0].copy()
        for idx in range(1, len(out)):
            previous = alpha * out[idx] + (1.0 - alpha) * previous
            out[idx] = previous
        return out

    forward = forward_pass(values)
    backward = forward_pass(values[::-1])[::-1]
    return ((forward + backward) * 0.5).astype(np.float32)


def estimate_root_yaw(joints: np.ndarray) -> np.ndarray:
    shoulder_left = joints[:, 5]
    shoulder_right = joints[:, 6]
    shoulder_axis = shoulder_right - shoulder_left
    yaw = np.arctan2(shoulder_axis[:, 2], shoulder_axis[:, 0] + 1e-8) + math.pi * 0.5
    yaw = np.unwrap(yaw)
    return apply_bidirectional_ema(yaw.reshape(-1, 1), 0.35).reshape(-1)


def estimate_motion_height(motion: Motion3D) -> float:
    head_y = motion.joints[:, [0, 17], 1]
    foot_y = motion.joints[:, FOOT_JOINTS, 1]
    finite_head = head_y[np.isfinite(head_y)]
    finite_foot = foot_y[np.isfinite(foot_y)]
    if len(finite_head) == 0 or len(finite_foot) == 0:
        return float("nan")
    height = float(np.percentile(finite_head, 95) - np.percentile(finite_foot, 5))
    return height if 0.5 <= height <= 2.4 else float("nan")


def normalize_xz(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(2)
    if not np.isfinite(vector).all():
        return fallback.astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-5:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def transform_xz(points: np.ndarray, origin_xz: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    output = points.copy()
    xz = points[..., [0, 2]] - origin_xz.reshape((1,) * (points.ndim - 1) + (2,))
    output[..., 0] = xz[..., 0] * x_axis[0] + xz[..., 1] * x_axis[1]
    output[..., 2] = xz[..., 0] * z_axis[0] + xz[..., 1] * z_axis[1]
    return output


def transform_yaw_to_basis(yaw: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    direction = np.stack([np.sin(yaw), np.cos(yaw)], axis=1)
    transformed = np.stack(
        [
            direction[:, 0] * x_axis[0] + direction[:, 1] * x_axis[1],
            direction[:, 0] * z_axis[0] + direction[:, 1] * z_axis[1],
        ],
        axis=1,
    )
    return np.unwrap(np.arctan2(transformed[:, 0], transformed[:, 1])).astype(np.float32)


def compute_floor_offset(motions: Dict[str, Motion3D]) -> float:
    foot_heights = []
    for motion in motions.values():
        foot_y = motion.joints[:, FOOT_JOINTS, 1]
        finite = foot_y[np.isfinite(foot_y)]
        if len(finite):
            foot_heights.append(float(np.percentile(finite, 5)))
    return float(np.median(foot_heights)) if foot_heights else 0.0


def apply_floor_offset(motions: Dict[str, Motion3D], floor_offset: float) -> None:
    if not np.isfinite(floor_offset) or abs(floor_offset) < 1e-8:
        return
    for motion in motions.values():
        motion.joints[:, :, 1] -= floor_offset
        motion.root[:, 1] -= floor_offset


def align_world(motions: Dict[str, Motion3D]) -> dict:
    heights_before = {role: estimate_motion_height(motion) for role, motion in motions.items()}
    valid_heights = [height for height in heights_before.values() if np.isfinite(height)]
    target_height = float(np.median(valid_heights)) if valid_heights else float("nan")
    scale_factors: Dict[str, float] = {}

    for role, motion in motions.items():
        height = heights_before[role]
        scale = 1.0
        if np.isfinite(height) and np.isfinite(target_height) and height > 1e-5:
            scale = float(np.clip(target_height / height, 0.85, 1.15))
            motion.joints = motion.root[:, None, :] + (motion.joints - motion.root[:, None, :]) * scale
        scale_factors[role] = scale
        motion.root = motion.joints[:, 19].copy()

    motion_a = motions[ROLE_NAMES[0]]
    motion_b = motions[ROLE_NAMES[1]]
    all_roots = np.concatenate([motion.root for motion in motions.values()], axis=0)
    origin_xz = np.asarray(
        [float(np.nanmedian(all_roots[:, 0])), float(np.nanmedian(all_roots[:, 2]))],
        dtype=np.float32,
    )

    pair_delta = motion_b.root[:, [0, 2]] - motion_a.root[:, [0, 2]]
    pair_distance = np.linalg.norm(pair_delta, axis=1)
    valid_delta = np.isfinite(pair_delta).all(axis=1) & (pair_distance > 0.05)
    fallback_axis = np.asarray([1.0, 0.0], dtype=np.float32)
    if np.any(valid_delta):
        x_axis = normalize_xz(np.nanmedian(pair_delta[valid_delta], axis=0), fallback_axis)
    else:
        x_axis = fallback_axis
    z_axis = np.asarray([-x_axis[1], x_axis[0]], dtype=np.float32)

    for motion in motions.values():
        motion.joints = transform_xz(motion.joints, origin_xz, x_axis, z_axis)
        motion.root = transform_xz(motion.root, origin_xz, x_axis, z_axis)
        motion.yaw = transform_yaw_to_basis(motion.yaw, x_axis, z_axis)

    floor = compute_floor_offset(motions)
    apply_floor_offset(motions, floor)

    heights_after = {role: estimate_motion_height(motion) for role, motion in motions.items()}
    return {
        "mode": "pair_centric_world",
        "unit": "meter",
        "up_axis": "Y",
        "floor_y": 0.0,
        "source_floor_offset": floor,
        "origin_xz_before_alignment": [float(origin_xz[0]), float(origin_xz[1])],
        "x_axis_before_alignment": [float(x_axis[0]), float(x_axis[1])],
        "z_axis_before_alignment": [float(z_axis[0]), float(z_axis[1])],
        "axis_definition": "+X points from character_A root toward character_B root; +Z is the horizontal perpendicular.",
        "scale_factors": scale_factors,
        "estimated_height_before": {
            role: (float(value) if np.isfinite(value) else None) for role, value in heights_before.items()
        },
        "estimated_height_after": {
            role: (float(value) if np.isfinite(value) else None) for role, value in heights_after.items()
        },
        "target_pair_height": target_height if np.isfinite(target_height) else None,
    }


def enforce_bone_lengths(motion: Motion3D, iterations: int = 2, blend: float = 0.65) -> None:
    target_lengths = {}
    for parent, child in SKELETON_EDGES:
        lengths = np.linalg.norm(motion.joints[:, child] - motion.joints[:, parent], axis=1)
        lengths = lengths[np.isfinite(lengths) & (lengths > 0.01)]
        if len(lengths):
            target_lengths[(parent, child)] = float(np.median(lengths))

    for _ in range(iterations):
        for parent, child in SKELETON_EDGES:
            target = target_lengths.get((parent, child))
            if target is None:
                continue
            delta = motion.joints[:, child] - motion.joints[:, parent]
            length = np.linalg.norm(delta, axis=1)
            valid = np.isfinite(length) & (length > 1e-6)
            corrected = motion.joints[:, parent] + delta / np.maximum(length[:, None], 1e-6) * target
            motion.joints[valid, child] = (
                (1.0 - blend) * motion.joints[valid, child] + blend * corrected[valid]
            )
    motion.root = motion.joints[:, 19].copy()


def detect_and_lock_feet(
    motion: Motion3D,
    fps: float,
    contact_height: float,
    contact_speed: float,
) -> None:
    foot_contact = np.zeros((motion.joints.shape[0], 2), dtype=bool)
    foot_groups = ([15, 20, 22, 24], [16, 21, 23, 25])
    for foot_slot, foot_indices in enumerate(foot_groups):
        ankle_idx = ANKLE_JOINTS[foot_slot]
        positions = motion.joints[:, foot_indices]
        foot_center = np.nanmean(positions, axis=1)
        min_height = np.nanmin(positions[:, :, 1], axis=1)
        velocity = np.zeros((len(positions),), dtype=np.float32)
        velocity[1:] = np.linalg.norm(np.diff(foot_center[:, [0, 2]], axis=0), axis=1) * fps
        finite_height = min_height[np.isfinite(min_height)]
        adaptive_height = float(np.percentile(finite_height, 35)) + 0.015 if len(finite_height) else contact_height
        height_threshold = max(contact_height, adaptive_height)
        contact = (min_height <= height_threshold) & (velocity <= contact_speed)
        foot_contact[:, foot_slot] = smooth_boolean_segments(contact, min_length=3, bridge_gap=2)

        start = None
        for idx, value in enumerate(foot_contact[:, foot_slot].tolist() + [False]):
            if value and start is None:
                start = idx
            if (not value) and start is not None:
                end = idx
                segment = slice(start, end)
                min_y = np.nanmin(motion.joints[segment][:, foot_indices, 1], axis=1)
                motion.joints[segment, :, 1] -= np.minimum(min_y, 0.0)[:, None]
                for joint_idx in foot_indices:
                    anchor = np.median(motion.joints[segment, joint_idx, [0, 2]], axis=0)
                    motion.joints[segment, joint_idx, 0] = (
                        0.45 * motion.joints[segment, joint_idx, 0] + 0.55 * anchor[0]
                    )
                    motion.joints[segment, joint_idx, 2] = (
                        0.45 * motion.joints[segment, joint_idx, 2] + 0.55 * anchor[1]
                    )
                start = None
    motion.foot_contact = foot_contact
    motion.root = motion.joints[:, 19].copy()


def smooth_boolean_segments(mask: np.ndarray, min_length: int, bridge_gap: int) -> np.ndarray:
    output = mask.astype(bool).copy()
    idx = 0
    while idx < len(output):
        if output[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(output) and not output[idx]:
            idx += 1
        if start > 0 and idx < len(output) and idx - start <= bridge_gap:
            output[start:idx] = True

    idx = 0
    while idx < len(output):
        if not output[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(output) and output[idx]:
            idx += 1
        if idx - start < min_length:
            output[start:idx] = False
    return output


def prevent_root_interpenetration(motions: Dict[str, Motion3D], min_distance: float) -> None:
    motion_a = motions[ROLE_NAMES[0]]
    motion_b = motions[ROLE_NAMES[1]]
    for frame_idx in range(min(len(motion_a.root), len(motion_b.root))):
        delta = motion_b.root[frame_idx, [0, 2]] - motion_a.root[frame_idx, [0, 2]]
        distance = float(np.linalg.norm(delta))
        if distance >= min_distance:
            continue
        direction = delta / distance if distance > 1e-6 else np.asarray([1.0, 0.0], dtype=np.float32)
        push = 0.5 * (min_distance - distance) * direction
        for sign, motion in ((-1.0, motion_a), (1.0, motion_b)):
            motion.joints[frame_idx, :, 0] += sign * push[0]
            motion.joints[frame_idx, :, 2] += sign * push[1]
            motion.root[frame_idx, 0] += sign * push[0]
            motion.root[frame_idx, 2] += sign * push[1]


def build_occlusion_masks(data: Pair2DData) -> dict:
    frame_count = data.frame_count
    pair_occlusion = np.zeros((frame_count,), dtype=bool)
    for frame_idx in range(frame_count):
        box_a = data.tracks[ROLE_NAMES[0]].bbox_xyxy[frame_idx]
        box_b = data.tracks[ROLE_NAMES[1]].bbox_xyxy[frame_idx]
        pair_occlusion[frame_idx] = bbox_iou(box_a, box_b) > 0.4
    per_person = {
        role: {
            "visible_joint_mask": data.tracks[role].valid[:frame_count].astype(bool),
            "person_visible": data.tracks[role].visible[:frame_count].astype(bool),
        }
        for role in ROLE_NAMES
    }
    return {"pair_occlusion": pair_occlusion, "per_person": per_person}


def build_contact_events(
    motions: Dict[str, Motion3D],
    fps: float,
    contact_distance: float,
    min_frames: int,
) -> Tuple[List[dict], List[dict]]:
    motion_a = motions[ROLE_NAMES[0]]
    motion_b = motions[ROLE_NAMES[1]]
    events: List[dict] = []
    graph_frames: List[dict] = []
    frame_count = min(len(motion_a.joints), len(motion_b.joints))

    contact_pairs = []
    for a_joint in WRIST_JOINTS:
        for b_joint in WRIST_JOINTS:
            contact_pairs.append(("hand_hand", a_joint, b_joint, contact_distance))
    for a_joint in WRIST_JOINTS:
        for b_joint in TORSO_JOINTS:
            contact_pairs.append(("a_hand_b_body", a_joint, b_joint, contact_distance * 1.12))
    for a_joint in TORSO_JOINTS:
        for b_joint in WRIST_JOINTS:
            contact_pairs.append(("b_hand_a_body", a_joint, b_joint, contact_distance * 1.12))

    pair_distances: Dict[Tuple[str, int, int], np.ndarray] = {}
    for contact_type, a_joint, b_joint, threshold in contact_pairs:
        distances = np.linalg.norm(motion_a.joints[:, a_joint] - motion_b.joints[:, b_joint], axis=1)
        pair_distances[(contact_type, a_joint, b_joint)] = distances
        active = distances < threshold
        rel_speed = np.zeros_like(distances)
        rel_speed[1:] = np.abs(np.diff(distances)) * fps
        active &= rel_speed < 1.8
        active = smooth_boolean_segments(active, min_length=min_frames, bridge_gap=2)
        events.extend(mask_to_events(active, contact_type, a_joint, b_joint, distances, fps))

    for frame_idx in range(frame_count):
        root_delta = motion_b.root[frame_idx] - motion_a.root[frame_idx]
        frame_contacts = []
        for key, distances in pair_distances.items():
            contact_type, a_joint, b_joint = key
            distance = float(distances[frame_idx])
            if distance < contact_distance * 1.25:
                frame_contacts.append(
                    {
                        "type": contact_type,
                        "a_joint": HALPE26_NAMES[a_joint],
                        "b_joint": HALPE26_NAMES[b_joint],
                        "distance": round(distance, 4),
                    }
                )
        graph_frames.append(
            {
                "frame_index": frame_idx,
                "relative_root": [round(float(value), 5) for value in root_delta.tolist()],
                "root_distance": round(float(np.linalg.norm(root_delta[[0, 2]])), 5),
                "facing_angle": round(float(wrap_angle(motion_b.yaw[frame_idx] - motion_a.yaw[frame_idx])), 5),
                "contact_candidates": frame_contacts[:12],
            }
        )

    for role, motion in motions.items():
        for foot_slot, ankle_idx in enumerate(ANKLE_JOINTS):
            events.extend(
                mask_to_events(
                    motion.foot_contact[:, foot_slot],
                    "foot_ground",
                    ankle_idx,
                    ankle_idx,
                    motion.joints[:, ankle_idx, 1],
                    fps,
                    role=role,
                )
            )
    return events, graph_frames


def build_pair_frame(motions: Dict[str, Motion3D]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion_a = motions[ROLE_NAMES[0]]
    motion_b = motions[ROLE_NAMES[1]]
    frame_count = min(len(motion_a.root), len(motion_b.root))
    roots_a = motion_a.root[:frame_count]
    roots_b = motion_b.root[:frame_count]
    origin = (roots_a + roots_b) * 0.5
    origin[:, 1] = 0.0

    x_axis = np.zeros((frame_count, 2), dtype=np.float32)
    fallback = np.asarray([1.0, 0.0], dtype=np.float32)
    for frame_idx in range(frame_count):
        delta = roots_b[frame_idx, [0, 2]] - roots_a[frame_idx, [0, 2]]
        x_axis[frame_idx] = normalize_xz(delta, fallback)
        fallback = x_axis[frame_idx]
    z_axis = np.stack([-x_axis[:, 1], x_axis[:, 0]], axis=1).astype(np.float32)
    return origin.astype(np.float32), x_axis, z_axis


def transform_points_to_pair_frame(
    points: np.ndarray,
    origin: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
) -> np.ndarray:
    output = points.copy().astype(np.float32)
    if points.ndim == 2:
        xz = points[:, [0, 2]] - origin[:, [0, 2]]
        output[:, 0] = xz[:, 0] * x_axis[:, 0] + xz[:, 1] * x_axis[:, 1]
        output[:, 1] = points[:, 1] - origin[:, 1]
        output[:, 2] = xz[:, 0] * z_axis[:, 0] + xz[:, 1] * z_axis[:, 1]
        return output
    if points.ndim == 3:
        xz = points[..., [0, 2]] - origin[:, None, [0, 2]]
        output[..., 0] = xz[..., 0] * x_axis[:, None, 0] + xz[..., 1] * x_axis[:, None, 1]
        output[..., 1] = points[..., 1] - origin[:, None, 1]
        output[..., 2] = xz[..., 0] * z_axis[:, None, 0] + xz[..., 1] * z_axis[:, None, 1]
        return output
    raise ValueError(f"Expected points to have shape [T, 3] or [T, J, 3], got {points.shape}")


def transform_yaw_to_pair_frame(yaw: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    direction = np.stack([np.sin(yaw), np.cos(yaw)], axis=1)
    transformed = np.stack(
        [
            direction[:, 0] * x_axis[:, 0] + direction[:, 1] * x_axis[:, 1],
            direction[:, 0] * z_axis[:, 0] + direction[:, 1] * z_axis[:, 1],
        ],
        axis=1,
    )
    return np.unwrap(np.arctan2(transformed[:, 0], transformed[:, 1])).astype(np.float32)


def temporal_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        velocity[1:] = np.diff(values, axis=0) * fps
    return velocity


def role_frame_observed(motion: Motion3D) -> np.ndarray:
    torso = motion.visible_mask[:, TORSO_JOINTS]
    return (np.mean(motion.visible_mask, axis=1) >= 0.18) | np.any(torso, axis=1)


def min_joint_distance(
    joints_a: np.ndarray,
    joints_b: np.ndarray,
    indices_a: Sequence[int],
    indices_b: Sequence[int],
) -> np.ndarray:
    distances = np.linalg.norm(joints_a[:, indices_a, None, :] - joints_b[:, None, indices_b, :], axis=-1)
    return np.nanmin(distances.reshape(distances.shape[0], -1), axis=1).astype(np.float32)


def min_joint_distance_observed(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    indices_a: Sequence[int],
    indices_b: Sequence[int],
) -> np.ndarray:
    pair_mask = mask_a[:, indices_a, None] & mask_b[:, None, indices_b]
    return np.any(pair_mask.reshape(pair_mask.shape[0], -1), axis=1)


def append_feature(
    values_parts: List[np.ndarray],
    mask_parts: List[np.ndarray],
    schema: List[dict],
    name: str,
    values: np.ndarray,
    reliability: Optional[np.ndarray],
) -> None:
    raw_values = np.asarray(values, dtype=np.float32)
    if raw_values.ndim == 1:
        raw_values = raw_values.reshape(-1, 1)
    frame_count = raw_values.shape[0]
    finite_mask = np.isfinite(raw_values).reshape(frame_count, -1)
    flat_values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0).reshape(frame_count, -1)

    if reliability is None:
        flat_mask = finite_mask
    else:
        rel = np.asarray(reliability, dtype=bool)
        if rel.shape == raw_values.shape:
            flat_mask = rel.reshape(frame_count, -1)
        elif rel.shape == raw_values.shape[:-1]:
            flat_mask = np.repeat(rel[..., None], raw_values.shape[-1], axis=-1).reshape(frame_count, -1)
        elif rel.shape == (frame_count,):
            flat_mask = np.repeat(rel[:, None], flat_values.shape[1], axis=1)
        else:
            flat_mask = rel.reshape(frame_count, -1)
            if flat_mask.shape[1] != flat_values.shape[1]:
                raise ValueError(f"Reliability mask for {name} does not match values: {rel.shape} vs {raw_values.shape}")
        flat_mask &= finite_mask

    start = sum(part.shape[1] for part in values_parts)
    end = start + flat_values.shape[1]
    schema.append({"name": name, "start": start, "end": end, "shape": list(raw_values.shape[1:])})
    values_parts.append(flat_values.astype(np.float32))
    mask_parts.append(flat_mask.astype(bool))


def build_two_in_one_representation(
    motions: Dict[str, Motion3D],
    occlusion: dict,
    graph_frames: Sequence[dict],
    fps: float,
) -> Dict[str, Any]:
    frame_count = min(motion.joints.shape[0] for motion in motions.values())
    origin, x_axis, z_axis = build_pair_frame(motions)
    origin = origin[:frame_count]
    x_axis = x_axis[:frame_count]
    z_axis = z_axis[:frame_count]
    pair_occlusion = occlusion["pair_occlusion"][:frame_count].astype(bool)

    role_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    role_observed: Dict[str, np.ndarray] = {}
    for role in ROLE_NAMES:
        motion = motions[role]
        joints_pair = transform_points_to_pair_frame(motion.joints[:frame_count], origin, x_axis, z_axis)
        root_pair = transform_points_to_pair_frame(motion.root[:frame_count], origin, x_axis, z_axis)
        yaw_pair = transform_yaw_to_pair_frame(motion.yaw[:frame_count], x_axis, z_axis)
        role_arrays[role] = {
            "joints_pair": joints_pair,
            "root_pair": root_pair,
            "root_velocity_pair": temporal_velocity(root_pair, fps),
            "joint_velocity_pair": temporal_velocity(joints_pair, fps),
            "yaw_pair": yaw_pair,
            "yaw_sin_cos": np.stack([np.sin(yaw_pair), np.cos(yaw_pair)], axis=1).astype(np.float32),
        }
        role_observed[role] = role_frame_observed(motion)[:frame_count]

    root_a = role_arrays[ROLE_NAMES[0]]["root_pair"]
    root_b = role_arrays[ROLE_NAMES[1]]["root_pair"]
    relative_root = (root_b - root_a).astype(np.float32)
    relative_velocity = temporal_velocity(relative_root, fps)
    root_distance = np.linalg.norm(relative_root[:, [0, 2]], axis=1).astype(np.float32)
    root_distance_delta = np.zeros_like(root_distance)
    if frame_count > 1:
        root_distance_delta[1:] = np.diff(root_distance) * fps
    approach_speed = (-root_distance_delta).astype(np.float32)
    yaw_a = role_arrays[ROLE_NAMES[0]]["yaw_pair"]
    yaw_b = role_arrays[ROLE_NAMES[1]]["yaw_pair"]
    facing_angle = np.asarray([wrap_angle(float(b - a)) for a, b in zip(yaw_a, yaw_b)], dtype=np.float32)
    facing_sin_cos = np.stack([np.sin(facing_angle), np.cos(facing_angle)], axis=1).astype(np.float32)

    motion_a = motions[ROLE_NAMES[0]]
    motion_b = motions[ROLE_NAMES[1]]
    hand_hand_distance = min_joint_distance(motion_a.joints[:frame_count], motion_b.joints[:frame_count], WRIST_JOINTS, WRIST_JOINTS)
    a_hand_b_body_distance = min_joint_distance(motion_a.joints[:frame_count], motion_b.joints[:frame_count], WRIST_JOINTS, TORSO_JOINTS)
    b_hand_a_body_distance = min_joint_distance(motion_b.joints[:frame_count], motion_a.joints[:frame_count], WRIST_JOINTS, TORSO_JOINTS)
    hand_hand_observed = min_joint_distance_observed(motion_a.visible_mask[:frame_count], motion_b.visible_mask[:frame_count], WRIST_JOINTS, WRIST_JOINTS)
    a_hand_b_body_observed = min_joint_distance_observed(motion_a.visible_mask[:frame_count], motion_b.visible_mask[:frame_count], WRIST_JOINTS, TORSO_JOINTS)
    b_hand_a_body_observed = min_joint_distance_observed(motion_b.visible_mask[:frame_count], motion_a.visible_mask[:frame_count], WRIST_JOINTS, TORSO_JOINTS)
    both_roots_observed = role_observed[ROLE_NAMES[0]] & role_observed[ROLE_NAMES[1]]
    both_roots_with_prev = both_roots_observed.copy()
    both_roots_with_prev[1:] &= both_roots_observed[:-1]
    both_roots_with_prev[0] = False

    values_parts: List[np.ndarray] = []
    mask_parts: List[np.ndarray] = []
    feature_schema: List[dict] = []
    for role in ROLE_NAMES:
        motion = motions[role]
        arrays = role_arrays[role]
        observed = role_observed[role]
        observed_with_prev = observed.copy()
        observed_with_prev[1:] &= observed[:-1]
        observed_with_prev[0] = False
        append_feature(values_parts, mask_parts, feature_schema, f"{role}.root_pair", arrays["root_pair"], observed)
        append_feature(values_parts, mask_parts, feature_schema, f"{role}.root_velocity_pair", arrays["root_velocity_pair"], observed_with_prev)
        append_feature(values_parts, mask_parts, feature_schema, f"{role}.yaw_sin_cos_pair", arrays["yaw_sin_cos"], observed)
        append_feature(values_parts, mask_parts, feature_schema, f"{role}.joints_pair", arrays["joints_pair"], motion.visible_mask[:frame_count])
        joint_velocity_mask = motion.visible_mask[:frame_count].copy()
        joint_velocity_mask[1:] &= motion.visible_mask[: frame_count - 1]
        joint_velocity_mask[0] = False
        append_feature(values_parts, mask_parts, feature_schema, f"{role}.joint_velocity_pair", arrays["joint_velocity_pair"], joint_velocity_mask)
        append_feature(values_parts, mask_parts, feature_schema, f"{role}.visible_joint_mask", motion.visible_mask[:frame_count].astype(np.float32), np.ones_like(motion.visible_mask[:frame_count], dtype=bool))
        append_feature(values_parts, mask_parts, feature_schema, f"{role}.confidence", motion.confidence[:frame_count], np.ones_like(motion.confidence[:frame_count], dtype=bool))
        append_feature(values_parts, mask_parts, feature_schema, f"{role}.foot_contact", motion.foot_contact[:frame_count].astype(np.float32), np.ones_like(motion.foot_contact[:frame_count], dtype=bool))

    append_feature(values_parts, mask_parts, feature_schema, "pair.relative_root_pair", relative_root, both_roots_observed)
    append_feature(values_parts, mask_parts, feature_schema, "pair.relative_root_velocity_pair", relative_velocity, both_roots_with_prev)
    append_feature(values_parts, mask_parts, feature_schema, "pair.root_distance_xz", root_distance, both_roots_observed)
    append_feature(values_parts, mask_parts, feature_schema, "pair.facing_angle_sin_cos", facing_sin_cos, both_roots_observed)
    append_feature(values_parts, mask_parts, feature_schema, "pair.approach_speed", approach_speed, both_roots_with_prev)
    append_feature(values_parts, mask_parts, feature_schema, "pair.min_hand_hand_distance", hand_hand_distance, hand_hand_observed)
    append_feature(values_parts, mask_parts, feature_schema, "pair.min_a_hand_b_body_distance", a_hand_b_body_distance, a_hand_b_body_observed)
    append_feature(values_parts, mask_parts, feature_schema, "pair.min_b_hand_a_body_distance", b_hand_a_body_distance, b_hand_a_body_observed)
    append_feature(values_parts, mask_parts, feature_schema, "pair.occlusion_mask", pair_occlusion.astype(np.float32), np.ones((frame_count,), dtype=bool))

    x_pair = np.concatenate(values_parts, axis=1).astype(np.float32)
    x_pair_mask = np.concatenate(mask_parts, axis=1).astype(bool)
    contact_lookup = {int(frame.get("frame_index", idx)): frame.get("contact_candidates", []) for idx, frame in enumerate(graph_frames)}

    frames = []
    for frame_idx in range(frame_count):
        role_payload = {}
        for role in ROLE_NAMES:
            arrays = role_arrays[role]
            motion = motions[role]
            role_payload[role] = {
                "root_pair": [round(float(value), 6) for value in arrays["root_pair"][frame_idx].tolist()],
                "root_world": [round(float(value), 6) for value in motion.root[frame_idx].tolist()],
                "yaw_pair": round(float(arrays["yaw_pair"][frame_idx]), 6),
                "visible_joint_ratio": round(float(np.mean(motion.visible_mask[frame_idx])), 5),
                "foot_contact": [bool(value) for value in motion.foot_contact[frame_idx].tolist()],
            }
        frames.append(
            {
                "frame_index": frame_idx,
                "time": round(frame_idx / fps, 6),
                "pair_frame": {
                    "origin_world": [round(float(value), 6) for value in origin[frame_idx].tolist()],
                    "x_axis_world_xz": [round(float(value), 6) for value in x_axis[frame_idx].tolist()],
                    "z_axis_world_xz": [round(float(value), 6) for value in z_axis[frame_idx].tolist()],
                    "definition": "local +X points from character_A root to character_B root; local +Z is the horizontal perpendicular",
                },
                "roles": role_payload,
                "relation": {
                    "relative_root_pair": [round(float(value), 6) for value in relative_root[frame_idx].tolist()],
                    "root_distance_xz": round(float(root_distance[frame_idx]), 6),
                    "facing_angle": round(float(facing_angle[frame_idx]), 6),
                    "approach_speed": round(float(approach_speed[frame_idx]), 6),
                    "min_hand_hand_distance": round(float(hand_hand_distance[frame_idx]), 6),
                    "min_a_hand_b_body_distance": round(float(a_hand_b_body_distance[frame_idx]), 6),
                    "min_b_hand_a_body_distance": round(float(b_hand_a_body_distance[frame_idx]), 6),
                    "pair_occlusion": bool(pair_occlusion[frame_idx]),
                    "contact_candidates": contact_lookup.get(frame_idx, [])[:12],
                },
            }
        )

    schema = {
        "representation": "two_in_one_pair",
        "version": 2,
        "x_pair_shape": list(x_pair.shape),
        "mask_shape": list(x_pair_mask.shape),
        "feature_axis": feature_schema,
        "joint_names": HALPE26_NAMES,
        "roles": list(ROLE_NAMES),
        "coordinate_frames": {
            "world": "meter, Y-up, floor Y=0 after world alignment",
            "pair": "per-frame dynamic pair frame centered between roots on the floor plane",
        },
        "mask_semantics": "x_pair_mask marks observed/reliable coordinates; false values are filled or derived from weak observations.",
    }
    npz_arrays: Dict[str, Any] = {
        "x_pair": x_pair,
        "x_pair_mask": x_pair_mask,
        "schema_json": np.asarray(json.dumps(schema, ensure_ascii=False)),
        "pair_origin_world": origin,
        "pair_x_axis_world_xz": x_axis,
        "pair_z_axis_world_xz": z_axis,
        "relative_root_pair": relative_root,
        "relative_root_velocity_pair": relative_velocity,
        "root_distance_xz": root_distance,
        "facing_angle": facing_angle,
        "approach_speed": approach_speed,
        "min_hand_hand_distance": hand_hand_distance,
        "min_a_hand_b_body_distance": a_hand_b_body_distance,
        "min_b_hand_a_body_distance": b_hand_a_body_distance,
        "pair_occlusion": pair_occlusion,
    }
    for role in ROLE_NAMES:
        arrays = role_arrays[role]
        npz_arrays[f"{role}_joints_pair"] = arrays["joints_pair"]
        npz_arrays[f"{role}_root_pair"] = arrays["root_pair"]
        npz_arrays[f"{role}_yaw_pair"] = arrays["yaw_pair"]
        npz_arrays[f"{role}_observed_frame_mask"] = role_observed[role]

    return {"schema": schema, "frames": frames, "npz_arrays": npz_arrays}


def mask_to_events(
    mask: np.ndarray,
    contact_type: str,
    a_joint: int,
    b_joint: int,
    distances: np.ndarray,
    fps: float,
    role: Optional[str] = None,
) -> List[dict]:
    events = []
    start = None
    for idx, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = idx
        if (not value) and start is not None:
            end = idx - 1
            payload = {
                "type": contact_type,
                "start_frame": int(start),
                "end_frame": int(end),
                "start_time": round(float(start) / fps, 5),
                "end_time": round(float(end) / fps, 5),
                "duration_frames": int(end - start + 1),
                "mean_distance": round(float(np.mean(distances[start : end + 1])), 5),
                "a_joint": HALPE26_NAMES[a_joint],
                "b_joint": HALPE26_NAMES[b_joint],
            }
            if role is not None:
                payload["role"] = role
            events.append(payload)
            start = None
    return events


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def refine_motions(
    motions: Dict[str, Motion3D],
    data: Pair2DData,
    args: argparse.Namespace,
) -> dict:
    world_alignment = align_world(motions)
    for motion in motions.values():
        motion.joints = apply_bidirectional_ema(motion.joints, args.smooth_alpha)
        motion.root = motion.joints[:, 19].copy()
        motion.yaw = apply_bidirectional_ema(motion.yaw.reshape(-1, 1), 0.35).reshape(-1)
        enforce_bone_lengths(motion, iterations=args.bone_enforce_iterations)
        detect_and_lock_feet(
            motion,
            fps=data.fps,
            contact_height=args.ground_contact_height,
            contact_speed=args.ground_contact_speed,
        )
    prevent_root_interpenetration(motions, args.root_min_distance)
    for motion in motions.values():
        enforce_bone_lengths(motion, iterations=1, blend=0.35)
        motion.root = motion.joints[:, 19].copy()
    post_floor = compute_floor_offset(motions)
    apply_floor_offset(motions, post_floor)
    world_alignment["post_refine_floor_offset"] = post_floor
    world_alignment["estimated_height_final"] = {
        role: (float(value) if np.isfinite(value) else None)
        for role, value in {role: estimate_motion_height(motion) for role, motion in motions.items()}.items()
    }
    world_alignment["final_floor_y"] = 0.0
    return world_alignment


def compute_quality_report(
    motions: Dict[str, Motion3D],
    data: Pair2DData,
    events: Sequence[dict],
    occlusion: dict,
    initializer_label: str,
    world_alignment: Optional[dict] = None,
) -> dict:
    role_reports = {}
    for role, motion in motions.items():
        velocity = np.linalg.norm(np.diff(motion.joints, axis=0), axis=2) * data.fps
        acceleration = np.linalg.norm(np.diff(motion.joints, n=2, axis=0), axis=2) * data.fps * data.fps
        bone_errors = []
        for parent, child in SKELETON_EDGES:
            lengths = np.linalg.norm(motion.joints[:, child] - motion.joints[:, parent], axis=1)
            if len(lengths):
                bone_errors.append(float(np.std(lengths) / max(np.mean(lengths), 1e-6)))
        foot_y = motion.joints[:, FOOT_JOINTS, 1]
        penetration = np.minimum(foot_y, 0.0)
        contact_speed = []
        for foot_slot, ankle_idx in enumerate(ANKLE_JOINTS):
            active = motion.foot_contact[:, foot_slot]
            if np.any(active):
                speeds = np.zeros((motion.joints.shape[0],), dtype=np.float32)
                speeds[1:] = np.linalg.norm(np.diff(motion.joints[:, ankle_idx, [0, 2]], axis=0), axis=1) * data.fps
                contact_speed.extend(speeds[active].tolist())
        role_reports[role] = {
            "frames": int(motion.joints.shape[0]),
            "visible_joint_ratio": float(np.mean(motion.visible_mask)),
            "mean_velocity_mps": float(np.mean(velocity)) if velocity.size else 0.0,
            "p95_velocity_mps": float(np.percentile(velocity, 95)) if velocity.size else 0.0,
            "mean_acceleration": float(np.mean(acceleration)) if acceleration.size else 0.0,
            "p95_acceleration": float(np.percentile(acceleration, 95)) if acceleration.size else 0.0,
            "mean_bone_length_cv": float(np.mean(bone_errors)) if bone_errors else 0.0,
            "max_ground_penetration_m": float(abs(np.min(penetration))) if penetration.size else 0.0,
            "mean_contact_foot_speed_mps": float(np.mean(contact_speed)) if contact_speed else 0.0,
            "foot_contact_frames": int(np.count_nonzero(motion.foot_contact)),
        }

    root_delta = motions[ROLE_NAMES[1]].root - motions[ROLE_NAMES[0]].root
    pair_distance = np.linalg.norm(root_delta[:, [0, 2]], axis=1)
    return {
        "summary": {
            "frame_count": data.frame_count,
            "fps": data.fps,
            "method": f"{initializer_label} + Two-in-One pair refinement",
            "world_alignment_mode": (world_alignment or {}).get("mode"),
            "pair_occlusion_frames": int(np.count_nonzero(occlusion["pair_occlusion"])),
            "interaction_event_count": len(events),
            "mean_pair_root_distance_m": float(np.mean(pair_distance)),
            "min_pair_root_distance_m": float(np.min(pair_distance)),
        },
        "roles": role_reports,
        "notes": [
            "Visible masks and confidence are preserved so future Masked InterVAE/InterLDM modules can use the same exports.",
        ],
    }


def motion_to_serializable(motion: Motion3D) -> dict:
    frames = []
    for frame_idx in range(motion.joints.shape[0]):
        frames.append(
            {
                "frame_index": frame_idx,
                "root_translation": [round(float(value), 6) for value in motion.root[frame_idx].tolist()],
                "root_yaw": round(float(motion.yaw[frame_idx]), 6),
                "joints_3d": [
                    [round(float(coord), 6) for coord in point]
                    for point in motion.joints[frame_idx].tolist()
                ],
                "confidence": [round(float(value), 5) for value in motion.confidence[frame_idx].tolist()],
                "visible_mask": [bool(value) for value in motion.visible_mask[frame_idx].tolist()],
                "foot_contact": [bool(value) for value in motion.foot_contact[frame_idx].tolist()],
            }
        )
    return {
        "role": motion.role,
        "stable_track_id": motion.stable_track_id,
        "source": motion.source,
        "joint_names": HALPE26_NAMES,
        "skeleton_edges": [[HALPE26_NAMES[a], HALPE26_NAMES[b]] for a, b in SKELETON_EDGES],
        "frames": frames,
    }


def save_outputs(
    output_root: Path,
    data: Pair2DData,
    motions: Dict[str, Motion3D],
    occlusion: dict,
    events: Sequence[dict],
    graph_frames: Sequence[dict],
    quality: dict,
    preview: bool,
    video_path: Path,
    initializer_metadata: Optional[dict] = None,
    world_alignment: Optional[dict] = None,
) -> None:
    ensure_dir(output_root)
    pair_representation = build_two_in_one_representation(motions, occlusion, graph_frames, data.fps)
    metadata = {
        "source_video": str(video_path),
        "source_2d": data.metadata.get("format", "wham_2d_observations"),
        "fps": data.fps,
        "frame_count": data.frame_count,
        "frame_width": data.width,
        "frame_height": data.height,
        "coordinate_system": "meters, Y-up, floor at Y=0, X/Z horizontal",
        "method_steps": [
            "WHAM-compatible initialization",
            "world alignment",
            "Two-in-One pair representation",
            "occlusion mask generation",
            "masked interpolation and temporal smoothing",
            "contact and physical consistency refinement",
            "asset-oriented export",
        ],
        "initializer": initializer_metadata or {},
        "world_alignment": world_alignment or {},
    }
    motion_payload = {
        "metadata": metadata,
        "tracks": [motion_to_serializable(motions[role]) for role in ROLE_NAMES],
    }
    (output_root / "motion3d_sequences.json").write_text(
        json.dumps(motion_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pair_payload = {
        "metadata": metadata,
        "schema": pair_representation["schema"],
        "pair_occlusion": [bool(value) for value in occlusion["pair_occlusion"].tolist()],
        "frames": pair_representation["frames"],
    }
    (output_root / "two_in_one_pair.json").write_text(
        json.dumps(pair_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(output_root / "two_in_one_pair.npz", **pair_representation["npz_arrays"])

    (output_root / "interaction_events.json").write_text(
        json.dumps({"metadata": metadata, "events": list(events)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "world_alignment.json").write_text(
        json.dumps({"metadata": metadata, "world_alignment": world_alignment or {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    np.savez_compressed(
        output_root / "motion3d_sequences.npz",
        **{f"{role}_joints": motions[role].joints for role in ROLE_NAMES},
        **{f"{role}_root": motions[role].root for role in ROLE_NAMES},
        **{f"{role}_yaw": motions[role].yaw for role in ROLE_NAMES},
        x_pair=pair_representation["npz_arrays"]["x_pair"],
        x_pair_mask=pair_representation["npz_arrays"]["x_pair_mask"],
        two_in_one_schema_json=pair_representation["npz_arrays"]["schema_json"],
        pair_occlusion=occlusion["pair_occlusion"],
    )
    save_joint_csv(output_root / "joint_positions.csv", motions, data.fps)
    save_asset_metadata(output_root / "asset_metadata.json", metadata, motions, events)
    if preview:
        save_3d_preview(output_root / "preview_3d.mp4", motions, data.fps)


def save_joint_csv(path: Path, motions: Dict[str, Motion3D], fps: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frame", "time", "role", "joint_index", "joint_name", "x", "y", "z", "confidence", "visible"])
        for role, motion in motions.items():
            for frame_idx in range(motion.joints.shape[0]):
                for joint_idx, joint_name in enumerate(HALPE26_NAMES):
                    point = motion.joints[frame_idx, joint_idx]
                    writer.writerow(
                        [
                            frame_idx,
                            frame_idx / fps,
                            role,
                            joint_idx,
                            joint_name,
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                            float(motion.confidence[frame_idx, joint_idx]),
                            bool(motion.visible_mask[frame_idx, joint_idx]),
                        ]
                    )


def save_asset_metadata(path: Path, metadata: dict, motions: Dict[str, Motion3D], events: Sequence[dict]) -> None:
    payload = {
        "metadata": metadata,
        "engine_import": {
            "unit": "meter",
            "up_axis": "Y",
            "root_motion": {
                role: {
                    "translation_source": "root_translation",
                    "yaw_source": "root_yaw",
                }
                for role in ROLE_NAMES
            },
            "ik_targets": {
                role: {
                    "left_hand": "left_wrist",
                    "right_hand": "right_wrist",
                    "left_foot": "left_ankle",
                    "right_foot": "right_ankle",
                }
                for role in ROLE_NAMES
            },
        },
        "constraints": [
            {
                "type": "attach" if event["type"] != "foot_ground" else "foot_lock",
                "event_type": event["type"],
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "a_joint": event.get("a_joint"),
                "b_joint": event.get("b_joint"),
                "stiffness": 0.65,
            }
            for event in events
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def project_oblique(point: np.ndarray, scale: float, center: Tuple[int, int]) -> Tuple[int, int]:
    x, y, z = point
    px = center[0] + scale * (x - 0.42 * z)
    py = center[1] - scale * (y + 0.20 * z)
    return int(round(px)), int(round(py))


def save_3d_preview(path: Path, motions: Dict[str, Motion3D], fps: float) -> None:
    frame_count = min(motion.joints.shape[0] for motion in motions.values())
    width, height = 1280, 720
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_points = np.concatenate([motion.joints.reshape(-1, 3) for motion in motions.values()], axis=0)
    finite = all_points[np.isfinite(all_points).all(axis=1)]
    extent = np.percentile(np.linalg.norm(finite[:, [0, 2]], axis=1), 95) if len(finite) else 2.0
    scale = min(210.0, max(95.0, 260.0 / max(extent, 1.0)))

    for frame_idx in range(frame_count):
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, f"frame {frame_idx}", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
        cv2.line(canvas, (40, 610), (840, 610), (210, 210, 210), 2)

        for role in ROLE_NAMES:
            motion = motions[role]
            color = ROLE_COLORS[role]
            joints = motion.joints[frame_idx]
            for parent, child in SKELETON_EDGES:
                p1 = project_oblique(joints[parent], scale, (430, 550))
                p2 = project_oblique(joints[child], scale, (430, 550))
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
            for idx in [19, 18, 5, 6, 9, 10, 15, 16]:
                cv2.circle(canvas, project_oblique(joints[idx], scale, (430, 550)), 4, color, -1, cv2.LINE_AA)
            root = project_oblique(motion.root[frame_idx], scale, (430, 550))
            cv2.putText(canvas, role, (root[0] + 6, root[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Top-down panel.
        cv2.rectangle(canvas, (890, 70), (1245, 650), (230, 230, 230), 1)
        cv2.putText(canvas, "top-down X/Z", (910, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2)
        top_center = np.asarray([1065, 375], dtype=np.float32)
        top_scale = scale * 0.75
        for role in ROLE_NAMES:
            motion = motions[role]
            color = ROLE_COLORS[role]
            root = motion.root[frame_idx]
            px = int(round(top_center[0] + root[0] * top_scale))
            py = int(round(top_center[1] + root[2] * top_scale))
            cv2.circle(canvas, (px, py), 8, color, -1, cv2.LINE_AA)
            yaw = motion.yaw[frame_idx]
            end = (int(round(px + math.sin(yaw) * 34)), int(round(py + math.cos(yaw) * 34)))
            cv2.arrowedLine(canvas, (px, py), end, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, role[-1], (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        writer.write(canvas)
    writer.release()


def reconstruct_from_2d(observations_path: Path, output_root: Path, video_path: Path, args: argparse.Namespace) -> None:
    data = load_2d_observations(observations_path, max_frames=args.max_frames)
    initializer_metadata: Dict[str, Any] = {"requested_backend": args.reconstruction_backend}
    motions: Optional[Dict[str, Motion3D]] = None

    should_try_wham = args.reconstruction_backend == "wham" or (
        args.reconstruction_backend == "auto" and (args.wham_root or args.wham_results_path)
    )
    if should_try_wham:
        try:
            wham_bundle = load_or_run_wham(observations_path, output_root, video_path, args)
            motions = build_motions_from_wham(wham_bundle, data, args)
            initializer_metadata.update(
                {
                    "backend": "wham",
                    "source_format": wham_bundle.metadata.get("source_format"),
                    "source_path": wham_bundle.metadata.get("source_path") or wham_bundle.metadata.get("result_path"),
                    "subjects": sorted(wham_bundle.subjects.keys()),
                    "local_only": bool(args.wham_local_only),
                }
            )
        except Exception as exc:
            if args.reconstruction_backend == "wham":
                raise SystemExit(f"WHAM backend failed: {exc}") from exc
            print(f"WHAM backend unavailable, using deterministic fallback: {exc}")

    if motions is None:
        motions = {
            role: lift_track_to_3d(
                data.tracks[role],
                data,
                body_height=args.body_height,
                focal_scale=args.focal_scale,
                min_keypoint_score=args.min_keypoint_score,
            )
            for role in ROLE_NAMES
        }
        initializer_metadata.update(
            {
                "backend": "fallback",
                "source_format": "project_2d_observations",
                "note": "Deterministic monocular kinematic lift used because real WHAM was not selected or unavailable.",
            }
        )

    occlusion = build_occlusion_masks(data)
    world_alignment = refine_motions(motions, data, args)
    events, graph_frames = build_contact_events(
        motions,
        fps=data.fps,
        contact_distance=args.contact_distance,
        min_frames=args.contact_min_frames,
    )
    initializer_label = "WHAM SMPL/world initialization" if initializer_metadata.get("backend") == "wham" else "WHAM-compatible fallback initialization"
    quality = compute_quality_report(motions, data, events, occlusion, initializer_label, world_alignment)
    save_outputs(
        output_root=output_root,
        data=data,
        motions=motions,
        occlusion=occlusion,
        events=events,
        graph_frames=graph_frames,
        quality=quality,
        preview=not args.no_preview,
        video_path=video_path,
        initializer_metadata=initializer_metadata,
        world_alignment=world_alignment,
    )


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root).resolve())
    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    if args.two_d_root:
        observations_path = Path(args.two_d_root).resolve() / "wham_2d_observations.json"
        if not observations_path.exists():
            raise SystemExit(f"2D observations not found: {observations_path}")
    else:
        observations_path = run_2d_pipeline(args, output_root)

    three_d_root = ensure_dir(output_root / "motion3d")
    reconstruct_from_2d(observations_path, three_d_root, video_path, args)
    print(f"Finished 3D reconstruction. Output saved to: {three_d_root}")


if __name__ == "__main__":
    main()
