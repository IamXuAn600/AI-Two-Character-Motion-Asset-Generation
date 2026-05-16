#!/usr/bin/env python3
"""
Two-character tracking + RTMPose pipeline for motion asset generation.

This script keeps the original `video_detection.py` untouched and builds a new,
more production-oriented flow:

1. YOLO person detection + BoT-SORT tracking with ReID enabled
2. Role locking on top of tracker IDs to keep stable `character_A/B`
3. RTMPose top-down inference on each character crop
4. Lightweight temporal post-processing
5. JSON / CSV export plus optional debug video and crops

Example:
    python code/two_character_rtmpose_pipeline.py \
        --video-path video/1.mp4 \
        --output-root outputs/two_character_run \
        --display
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.utils import (
    ENHANCED_KEYPOINT_NAMES as LEGACY_ENHANCED_KEYPOINT_NAMES,
    KEYPOINT_NAMES as LEGACY_KEYPOINT_NAMES,
    build_enhanced_keypoints,
    compute_body_center,
)

try:
    import torch
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

MMPoseInferencer = None

try:
    from rtmlib import RTMPose as RTMLibPose
except Exception:
    RTMLibPose = None


DEFAULT_ROLE_NAMES = ("character_A", "character_B")

KEYPOINT_NAMES: Dict[int, List[str]] = {
    17: [
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
    ],
    26: [
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
    ],
}

ROLE_COLORS = {
    "character_A": (50, 205, 50),
    "character_B": (30, 144, 255),
}

SKELETONS = {
    17: [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
        (13, 15), (12, 14), (14, 16),
    ],
    26: [
        (17, 0), (0, 1), (0, 2), (1, 3), (2, 4), (17, 18),
        (18, 5), (18, 6), (5, 6), (5, 7), (7, 9), (6, 8),
        (8, 10), (18, 19), (19, 11), (19, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16), (15, 24),
        (15, 20), (15, 22), (16, 25), (16, 21), (16, 23),
    ],
}

RTMLIB_POSE_MODELS: Dict[str, Dict[str, Dict[str, object]]] = {
    "body17": {
        "lightweight": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
            "pose_input_size": (192, 256),
        },
        "balanced": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
            "pose_input_size": (192, 256),
        },
        "performance": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
            "pose_input_size": (288, 384),
        },
    },
    "body26": {
        "lightweight": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
            "pose_input_size": (192, 256),
        },
        "balanced": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
            "pose_input_size": (192, 256),
        },
        "performance": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
            "pose_input_size": (288, 384),
        },
    },
}


@dataclass
class Detection:
    tracker_id: int
    bbox: np.ndarray
    confidence: float

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


@dataclass
class RoleState:
    name: str
    tracker_id: Optional[int] = None
    last_bbox: Optional[np.ndarray] = None
    last_center: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    last_seen_frame: int = -1
    missing_frames: int = 0
    initialized: bool = False
    initial_center: Optional[np.ndarray] = None
    appearance_feature: Optional[np.ndarray] = None
    appearance_updates: int = 0
    identity_feature: Optional[np.ndarray] = None
    identity_updates: int = 0
    anchor_appearance_feature: Optional[np.ndarray] = None
    anchor_appearance_updates: int = 0


@dataclass
class InteractionAnchor:
    feature: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    frame_index: int = -1


@dataclass
class PoseResult:
    keypoints: np.ndarray
    scores: np.ndarray


@dataclass
class FrameRoleRecord:
    frame_index: int
    role: str
    visible: bool
    tracker_id: Optional[int]
    tracker_bbox: Optional[List[float]]
    pose_crop_bbox: Optional[List[int]]
    source_tracker_id: Optional[int] = None
    raw_detector_tracker_id: Optional[int] = None
    raw_keypoints: Optional[List[List[float]]] = None
    raw_scores: Optional[List[float]] = None
    smoothed_keypoints: Optional[List[List[float]]] = None
    source_track_id_changed: bool = False
    raw_detector_track_id_changed: bool = False
    is_recovered: bool = False
    role_id: int = -1
    stable_track_id: int = -1
    bbox_score: Optional[float] = None
    frame_time_sec: Optional[float] = None
    raw_valid_mask: Optional[List[bool]] = None
    smoothed_scores: Optional[List[float]] = None
    smoothed_valid_mask: Optional[List[bool]] = None
    keypoint_source: Optional[List[str]] = None


@dataclass
class HeldPoseState:
    keypoints: Optional[List[List[float]]] = None
    scores: Optional[List[float]] = None
    bbox: Optional[List[float]] = None
    tracker_id: Optional[int] = None
    center: Optional[np.ndarray] = None


@dataclass
class DetectionCandidate:
    detection: Detection
    pose_crop_bbox: Tuple[int, int, int, int]
    pose_result: Optional["PoseResult"]
    pose_mean_score: float
    confident_keypoint_count: int
    appearance_feature: Optional[np.ndarray] = None
    is_recovered: bool = False


@dataclass
class DetectionTrackHistory:
    centers: List[np.ndarray] = field(default_factory=list)
    bbox_heights: List[float] = field(default_factory=list)
    last_frame_index: int = -1


@dataclass
class TrackProbationState:
    consecutive_frames: int = 0
    last_seen_frame: int = -1
    approved: bool = False


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    default_yolo = project_root / "yolo11n.pt"
    if not default_yolo.exists():
        default_yolo = "yolo11n.pt"

    parser = argparse.ArgumentParser(
        description="Two-character RTMPose pipeline for motion asset generation."
    )
    parser.add_argument("--video-path", "--video", dest="video_path", default=str(project_root / "input" / "video_2.mp4"))
    parser.add_argument("--yolo-weights", "--model", dest="yolo_weights", default=str(default_yolo))
    parser.add_argument("--output-root", "--output", dest="output_root", default=str(project_root / "outputs"))
    parser.add_argument("--tracking-preset", choices=("default", "occlusion"), default="occlusion")
    parser.add_argument("--pose-backend", choices=("rtmlib", "mmpose"), default="rtmlib")
    parser.add_argument("--pose-alias", default="body26")
    parser.add_argument("--pose-mode", choices=("lightweight", "balanced", "performance"), default="balanced")
    parser.add_argument("--pose-config", default=None)
    parser.add_argument("--pose-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--keep-temp-pose-inputs", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    parser.add_argument("--bbox-padding", type=float, default=0.2)
    parser.add_argument("--min-keypoint-score", type=float, default=0.2)
    parser.add_argument("--temporal-smoothing-mode", choices=("none", "ema", "bidirectional", "adaptive"), default="bidirectional")
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--bidirectional-smoothing-alpha", type=float, default=0.45)
    parser.add_argument("--adaptive-smoothing-window", type=int, default=2)
    parser.add_argument("--adaptive-smoothing-min-strength", type=float, default=0.06)
    parser.add_argument("--adaptive-smoothing-max-strength", type=float, default=0.28)
    parser.add_argument("--adaptive-smoothing-velocity-scale", type=float, default=0.035)
    parser.add_argument("--interpolate-gap", type=int, default=6)
    parser.add_argument("--max-missed-frames", type=int, default=45)
    parser.add_argument("--track-buffer", type=int, default=60)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--appearance-thresh", type=float, default=0.35)
    parser.add_argument("--proximity-thresh", type=float, default=0.5)
    parser.add_argument("--disable-detection-filter", action="store_true")
    parser.add_argument("--filter-min-bbox-height-ratio", type=float, default=0.12)
    parser.add_argument("--filter-min-bbox-area-ratio", type=float, default=0.01)
    parser.add_argument("--filter-relative-min-bbox-height-ratio", type=float, default=0.38)
    parser.add_argument("--filter-relative-min-bbox-area-ratio", type=float, default=0.18)
    parser.add_argument("--filter-min-bbox-aspect-ratio", type=float, default=0.3)
    parser.add_argument("--filter-max-bbox-aspect-ratio", type=float, default=1.2)
    parser.add_argument("--filter-min-pose-mean-score", type=float, default=0.45)
    parser.add_argument("--filter-pose-keypoint-score", type=float, default=0.35)
    parser.add_argument("--filter-min-pose-keypoints", type=int, default=6)
    parser.add_argument("--filter-static-history", type=int, default=6)
    parser.add_argument("--filter-static-motion-threshold", type=float, default=12.0)
    parser.add_argument("--filter-static-height-jitter-ratio", type=float, default=0.08)
    parser.add_argument("--appearance-alpha", type=float, default=0.25)
    parser.add_argument("--appearance-weight", type=float, default=2.2)
    parser.add_argument("--appearance-min-pose-score", type=float, default=0.55)
    parser.add_argument("--identity-alpha", type=float, default=0.03)
    parser.add_argument("--identity-weight", type=float, default=1.6)
    parser.add_argument("--identity-min-pose-score", type=float, default=0.68)
    parser.add_argument("--initial-role-lock-frames", type=int, default=144)
    parser.add_argument("--initial-role-lock-weight", type=float, default=4.5)
    parser.add_argument("--anchor-appearance-alpha", type=float, default=0.08)
    parser.add_argument("--anchor-appearance-weight", type=float, default=2.8)
    parser.add_argument("--anchor-min-pose-score", type=float, default=0.72)
    parser.add_argument("--interaction-iou-threshold", type=float, default=0.08)
    parser.add_argument("--interaction-center-distance-ratio", type=float, default=0.9)
    parser.add_argument("--interaction-cooldown-frames", type=int, default=18)
    parser.add_argument("--interaction-side-prior-scale", type=float, default=0.18)
    parser.add_argument("--interaction-anchor-motion-weight", type=float, default=2.6)
    parser.add_argument("--interaction-anchor-distance-weight", type=float, default=2.4)
    parser.add_argument("--interaction-tracker-bonus-scale", type=float, default=0.28)
    parser.add_argument("--interaction-anchor-velocity-scale", type=float, default=0.9)
    parser.add_argument("--tracker-role-memory-frames", type=int, default=42)
    parser.add_argument("--tracker-role-min-history", type=int, default=3)
    parser.add_argument("--tracker-role-bonus", type=float, default=0.45)
    parser.add_argument("--tracker-role-penalty", type=float, default=2.2)
    parser.add_argument("--interaction-tracker-role-penalty-scale", type=float, default=1.35)
    parser.add_argument("--recovery-max-gap", type=int, default=12)
    parser.add_argument("--recovery-extra-padding", type=float, default=0.35)
    parser.add_argument("--recovery-velocity-scale", type=float, default=0.75)
    parser.add_argument("--recovery-min-pose-mean-score", type=float, default=0.2)
    parser.add_argument("--recovery-min-keypoints", type=int, default=4)
    parser.add_argument("--recovery-min-appearance-similarity", type=float, default=0.2)
    parser.add_argument("--recovery-appearance-margin", type=float, default=0.03)
    parser.add_argument("--recovery-max-center-distance-ratio", type=float, default=0.65)
    parser.add_argument("--new-track-probation-frames", type=int, default=3)
    parser.add_argument("--probation-extra-frames", type=int, default=2)
    parser.add_argument("--probation-low-pose-threshold", type=float, default=0.62)
    parser.add_argument("--probation-min-appearance-similarity", type=float, default=0.34)
    parser.add_argument("--role-switch-min-pose-score", type=float, default=0.5)
    parser.add_argument("--role-switch-min-keypoints", type=int, default=8)
    parser.add_argument("--role-switch-min-detection-score", type=float, default=0.3)
    parser.add_argument("--role-switch-max-center-jump-ratio", type=float, default=0.26)
    parser.add_argument("--role-switch-long-gap-frames", type=int, default=12)
    parser.add_argument("--disable-offline-reid", action="store_true")
    parser.add_argument("--offline-reid-anchor-frames", type=int, default=120)
    parser.add_argument("--offline-reid-min-bbox-score", type=float, default=0.6)
    parser.add_argument("--offline-reid-swap-margin", type=float, default=0.12)
    parser.add_argument("--offline-reid-keep-margin", type=float, default=0.08)
    parser.add_argument("--offline-reid-min-swap-segment", type=int, default=5)
    parser.add_argument("--offline-reid-bridge-gap", type=int, default=2)
    parser.add_argument("--offline-reid-single-margin", type=float, default=0.09)
    parser.add_argument("--offline-reid-min-single-segment", type=int, default=5)
    parser.add_argument(
        "--occlusion-hold",
        choices=("off", "partial", "full"),
        default="off",
        help=(
            "How aggressively to synthesize missing 2D joints. "
            "'off' keeps occluded joints masked, 'partial' only fills low-confidence joints "
            "inside otherwise observed poses, and 'full' restores the previous legacy hold behavior."
        ),
    )
    parser.add_argument(
        "--interpolated-score-scale",
        type=float,
        default=0.45,
        help="Confidence multiplier for short-gap interpolated keypoints; kept below min-keypoint-score by default.",
    )
    parser.add_argument("--partial-pose-hold-shift-px", type=float, default=10.0)
    parser.add_argument("--partial-pose-min-visible-keypoints", type=int, default=3)
    parser.add_argument("--no-3d", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-blend", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--blender-exe", default=r"D:\Blender\blender.exe", help=argparse.SUPPRESS)
    parser.add_argument("--blend-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--motion-output-root", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    apply_tracking_preset(args, sys.argv[1:])
    return args


def apply_tracking_preset(args: argparse.Namespace, argv: Sequence[str]) -> None:
    if args.tracking_preset == "default":
        return

    provided_flags = {item.split("=", 1)[0] for item in argv if item.startswith("--")}

    def set_if_default(flag: str, attr: str, value: object) -> None:
        if flag not in provided_flags:
            setattr(args, attr, value)

    if args.tracking_preset == "occlusion":
        set_if_default("--yolo-weights", "yolo_weights", "yolo11m.pt")
        set_if_default("--pose-mode", "pose_mode", "performance")
        set_if_default("--temporal-smoothing-mode", "temporal_smoothing_mode", "bidirectional")
        set_if_default("--bidirectional-smoothing-alpha", "bidirectional_smoothing_alpha", 0.45)
        set_if_default("--adaptive-smoothing-window", "adaptive_smoothing_window", 2)
        set_if_default("--adaptive-smoothing-min-strength", "adaptive_smoothing_min_strength", 0.06)
        set_if_default("--adaptive-smoothing-max-strength", "adaptive_smoothing_max_strength", 0.28)
        set_if_default("--adaptive-smoothing-velocity-scale", "adaptive_smoothing_velocity_scale", 0.035)
        set_if_default("--interpolate-gap", "interpolate_gap", 18)
        set_if_default("--ema-alpha", "ema_alpha", 0.5)
        set_if_default("--yolo-conf", "yolo_conf", 0.18)
        set_if_default("--yolo-iou", "yolo_iou", 0.62)
        set_if_default("--bbox-padding", "bbox_padding", 0.28)
        set_if_default("--max-missed-frames", "max_missed_frames", 120)
        set_if_default("--track-buffer", "track_buffer", 180)
        set_if_default("--match-thresh", "match_thresh", 0.72)
        set_if_default("--appearance-thresh", "appearance_thresh", 0.25)
        set_if_default("--proximity-thresh", "proximity_thresh", 0.35)
        set_if_default("--filter-min-bbox-height-ratio", "filter_min_bbox_height_ratio", 0.07)
        set_if_default("--filter-min-bbox-area-ratio", "filter_min_bbox_area_ratio", 0.004)
        set_if_default("--filter-relative-min-bbox-height-ratio", "filter_relative_min_bbox_height_ratio", 0.18)
        set_if_default("--filter-relative-min-bbox-area-ratio", "filter_relative_min_bbox_area_ratio", 0.06)
        set_if_default("--filter-min-pose-mean-score", "filter_min_pose_mean_score", 0.28)
        set_if_default("--filter-pose-keypoint-score", "filter_pose_keypoint_score", 0.25)
        set_if_default("--filter-min-pose-keypoints", "filter_min_pose_keypoints", 4)
        set_if_default("--filter-static-history", "filter_static_history", 10)
        set_if_default("--filter-static-motion-threshold", "filter_static_motion_threshold", 6.0)
        set_if_default("--appearance-weight", "appearance_weight", 2.8)
        set_if_default("--appearance-min-pose-score", "appearance_min_pose_score", 0.42)
        set_if_default("--identity-weight", "identity_weight", 2.2)
        set_if_default("--identity-min-pose-score", "identity_min_pose_score", 0.58)
        set_if_default("--initial-role-lock-weight", "initial_role_lock_weight", 6.0)
        set_if_default("--anchor-appearance-weight", "anchor_appearance_weight", 3.4)
        set_if_default("--anchor-min-pose-score", "anchor_min_pose_score", 0.55)
        set_if_default("--interaction-cooldown-frames", "interaction_cooldown_frames", 36)
        set_if_default("--interaction-tracker-bonus-scale", "interaction_tracker_bonus_scale", 0.18)
        set_if_default("--tracker-role-memory-frames", "tracker_role_memory_frames", 120)
        set_if_default("--tracker-role-min-history", "tracker_role_min_history", 2)
        set_if_default("--tracker-role-bonus", "tracker_role_bonus", 0.8)
        set_if_default("--tracker-role-penalty", "tracker_role_penalty", 3.0)
        set_if_default("--interaction-tracker-role-penalty-scale", "interaction_tracker_role_penalty_scale", 1.7)
        set_if_default("--recovery-max-gap", "recovery_max_gap", 48)
        set_if_default("--recovery-extra-padding", "recovery_extra_padding", 0.55)
        set_if_default("--recovery-velocity-scale", "recovery_velocity_scale", 0.5)
        set_if_default("--recovery-min-pose-mean-score", "recovery_min_pose_mean_score", 0.12)
        set_if_default("--recovery-min-keypoints", "recovery_min_keypoints", 3)
        set_if_default("--recovery-min-appearance-similarity", "recovery_min_appearance_similarity", 0.12)
        set_if_default("--recovery-appearance-margin", "recovery_appearance_margin", 0.0)
        set_if_default("--recovery-max-center-distance-ratio", "recovery_max_center_distance_ratio", 0.95)
        set_if_default("--new-track-probation-frames", "new_track_probation_frames", 2)
        set_if_default("--probation-extra-frames", "probation_extra_frames", 1)
        set_if_default("--probation-low-pose-threshold", "probation_low_pose_threshold", 0.45)
        set_if_default("--probation-min-appearance-similarity", "probation_min_appearance_similarity", 0.18)
        set_if_default("--role-switch-min-pose-score", "role_switch_min_pose_score", 0.56)
        set_if_default("--role-switch-min-keypoints", "role_switch_min_keypoints", 8)
        set_if_default("--role-switch-min-detection-score", "role_switch_min_detection_score", 0.34)
        set_if_default("--role-switch-max-center-jump-ratio", "role_switch_max_center_jump_ratio", 0.24)
        set_if_default("--role-switch-long-gap-frames", "role_switch_long_gap_frames", 12)
        set_if_default("--offline-reid-anchor-frames", "offline_reid_anchor_frames", 120)
        set_if_default("--offline-reid-min-bbox-score", "offline_reid_min_bbox_score", 0.6)
        set_if_default("--offline-reid-swap-margin", "offline_reid_swap_margin", 0.12)
        set_if_default("--offline-reid-keep-margin", "offline_reid_keep_margin", 0.08)
        set_if_default("--offline-reid-min-swap-segment", "offline_reid_min_swap_segment", 5)
        set_if_default("--offline-reid-bridge-gap", "offline_reid_bridge_gap", 2)
        set_if_default("--offline-reid-single-margin", "offline_reid_single_margin", 0.09)
        set_if_default("--offline-reid-min-single-segment", "offline_reid_min_single_segment", 5)
        set_if_default("--occlusion-hold", "occlusion_hold", "off")
        set_if_default("--interpolated-score-scale", "interpolated_score_scale", 0.45)
        set_if_default("--partial-pose-hold-shift-px", "partial_pose_hold_shift_px", 7.0)
        set_if_default("--partial-pose-min-visible-keypoints", "partial_pose_min_visible_keypoints", 3)


def select_device(device_arg: str, pose_backend: str) -> str:
    if device_arg != "auto":
        return device_arg
    if pose_backend == "rtmlib" and ort is not None:
        providers = ort.get_available_providers()
        if "CoreMLExecutionProvider" in providers or "MPSExecutionProvider" in providers:
            return "mps"
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    if (
        pose_backend == "mmpose"
        and torch is not None
        and hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_tracker_yaml(args: argparse.Namespace, output_root: Path) -> Path:
    tracker_path = output_root / "botsort_reid.yaml"
    tracker_path.write_text(
        "\n".join(
            [
                "tracker_type: botsort",
                "track_high_thresh: 0.25",
                "track_low_thresh: 0.1",
                "new_track_thresh: 0.25",
                f"track_buffer: {args.track_buffer}",
                f"match_thresh: {args.match_thresh}",
                "fuse_score: True",
                "gmc_method: sparseOptFlow",
                f"proximity_thresh: {args.proximity_thresh}",
                f"appearance_thresh: {args.appearance_thresh}",
                "with_reid: True",
                "model: auto",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tracker_path


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - inter_area
    )
    return inter_area / union_area if union_area > 0 else 0.0


def expand_bbox(
    bbox: np.ndarray,
    frame_shape: Sequence[int],
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    nx1 = max(0, int(round(x1 - pad_x)))
    ny1 = max(0, int(round(y1 - pad_y)))
    nx2 = min(width, int(round(x2 + pad_x)))
    ny2 = min(height, int(round(y2 + pad_y)))
    return nx1, ny1, nx2, ny2


def clip_bbox_to_frame(
    bbox: np.ndarray,
    frame_shape: Sequence[int],
) -> np.ndarray:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    x1 = min(max(0.0, x1), max(0.0, float(width - 1)))
    y1 = min(max(0.0, y1), max(0.0, float(height - 1)))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-8:
        return feature.astype(np.float32, copy=True)
    return (feature / norm).astype(np.float32, copy=False)


def compute_appearance_feature(
    crop_bgr: np.ndarray,
    pose_result: Optional[PoseResult],
    pose_score_threshold: float,
) -> Optional[np.ndarray]:
    if crop_bgr.size == 0:
        return None

    crop_h, crop_w = crop_bgr.shape[:2]
    sample = crop_bgr

    if pose_result is not None and len(pose_result.keypoints) >= 13:
        torso_indices = [5, 6, 11, 12]
        valid_points = []
        for idx in torso_indices:
            if idx >= len(pose_result.keypoints):
                continue
            if idx < len(pose_result.scores) and pose_result.scores[idx] < pose_score_threshold:
                continue
            x, y = pose_result.keypoints[idx]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            valid_points.append((float(x), float(y)))

        if len(valid_points) >= 2:
            xs = [point[0] for point in valid_points]
            ys = [point[1] for point in valid_points]
            torso_x1 = max(0, int(round(min(xs) - max(8.0, 0.18 * crop_w))))
            torso_y1 = max(0, int(round(min(ys) - max(8.0, 0.12 * crop_h))))
            torso_x2 = min(crop_w, int(round(max(xs) + max(8.0, 0.18 * crop_w))))
            torso_y2 = min(crop_h, int(round(max(ys) + max(8.0, 0.2 * crop_h))))
            if torso_x2 > torso_x1 and torso_y2 > torso_y1:
                sample = crop_bgr[torso_y1:torso_y2, torso_x1:torso_x2]

    if sample.size == 0:
        return None

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 12, 6], [0, 180, 0, 256, 0, 256])
    if hist is None:
        return None
    return normalize_feature(hist.reshape(-1))


def appearance_similarity(feature_a: Optional[np.ndarray], feature_b: Optional[np.ndarray]) -> float:
    if feature_a is None or feature_b is None:
        return 0.0
    return float(np.dot(feature_a, feature_b))


def predict_recovery_bbox(
    state: RoleState,
    frame_shape: Sequence[int],
    gap_frames: int,
    velocity_scale: float,
) -> Optional[np.ndarray]:
    if state.last_bbox is None or state.last_center is None:
        return None

    bbox = state.last_bbox.astype(np.float32).copy()
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    shift = np.zeros((2,), dtype=np.float32)

    if state.velocity is not None:
        shift = state.velocity.astype(np.float32) * float(min(gap_frames, 3)) * float(velocity_scale)
        shift_norm = float(np.linalg.norm(shift))
        max_shift = max(width, height) * (0.35 + 0.1 * float(min(gap_frames, 3)))
        if shift_norm > max_shift and shift_norm > 1e-6:
            shift *= max_shift / shift_norm

    predicted_center = state.last_center.astype(np.float32) + shift
    predicted_bbox = np.array(
        [
            predicted_center[0] - 0.5 * width,
            predicted_center[1] - 0.5 * height,
            predicted_center[0] + 0.5 * width,
            predicted_center[1] + 0.5 * height,
        ],
        dtype=np.float32,
    )
    return clip_bbox_to_frame(predicted_bbox, frame_shape)


def role_side_prior(role_name: str, detection: Detection, frame_width: int) -> float:
    center_x = float(detection.center[0]) / max(1.0, float(frame_width))
    if role_name == "character_A":
        return 1.0 - center_x
    return center_x


class RoleLockManager:
    def __init__(
        self,
        role_names: Sequence[str],
        max_missed_frames: int,
        appearance_alpha: float,
        appearance_weight: float,
        appearance_min_pose_score: float,
        identity_alpha: float,
        identity_weight: float,
        identity_min_pose_score: float,
        initial_role_lock_frames: int,
        initial_role_lock_weight: float,
        anchor_appearance_alpha: float,
        anchor_appearance_weight: float,
        anchor_min_pose_score: float,
        interaction_iou_threshold: float,
        interaction_center_distance_ratio: float,
        interaction_cooldown_frames: int,
        interaction_side_prior_scale: float,
        interaction_anchor_motion_weight: float,
        interaction_anchor_distance_weight: float,
        interaction_tracker_bonus_scale: float,
        interaction_anchor_velocity_scale: float,
        tracker_role_memory_frames: int,
        tracker_role_min_history: int,
        tracker_role_bonus: float,
        tracker_role_penalty: float,
        interaction_tracker_role_penalty_scale: float,
        role_switch_min_pose_score: float,
        role_switch_min_keypoints: int,
        role_switch_min_detection_score: float,
        role_switch_max_center_jump_ratio: float,
        role_switch_long_gap_frames: int,
    ) -> None:
        self.role_names = list(role_names)
        self.states = {name: RoleState(name=name) for name in role_names}
        self.max_missed_frames = max_missed_frames
        self.appearance_alpha = appearance_alpha
        self.appearance_weight = appearance_weight
        self.appearance_min_pose_score = appearance_min_pose_score
        self.identity_alpha = identity_alpha
        self.identity_weight = identity_weight
        self.identity_min_pose_score = identity_min_pose_score
        self.initial_role_lock_frames = initial_role_lock_frames
        self.initial_role_lock_weight = initial_role_lock_weight
        self.anchor_appearance_alpha = anchor_appearance_alpha
        self.anchor_appearance_weight = anchor_appearance_weight
        self.anchor_min_pose_score = anchor_min_pose_score
        self.interaction_iou_threshold = interaction_iou_threshold
        self.interaction_center_distance_ratio = interaction_center_distance_ratio
        self.interaction_cooldown_frames = interaction_cooldown_frames
        self.interaction_side_prior_scale = interaction_side_prior_scale
        self.interaction_anchor_motion_weight = interaction_anchor_motion_weight
        self.interaction_anchor_distance_weight = interaction_anchor_distance_weight
        self.interaction_tracker_bonus_scale = interaction_tracker_bonus_scale
        self.interaction_anchor_velocity_scale = interaction_anchor_velocity_scale
        self.tracker_role_memory_frames = tracker_role_memory_frames
        self.tracker_role_min_history = tracker_role_min_history
        self.tracker_role_bonus = tracker_role_bonus
        self.tracker_role_penalty = tracker_role_penalty
        self.interaction_tracker_role_penalty_scale = interaction_tracker_role_penalty_scale
        self.role_switch_min_pose_score = role_switch_min_pose_score
        self.role_switch_min_keypoints = role_switch_min_keypoints
        self.role_switch_min_detection_score = role_switch_min_detection_score
        self.role_switch_max_center_jump_ratio = role_switch_max_center_jump_ratio
        self.role_switch_long_gap_frames = role_switch_long_gap_frames
        self.interaction_cooldown = 0
        self.interaction_anchors = {
            name: InteractionAnchor() for name in role_names
        }
        self.tracker_role_history: Dict[int, List[Tuple[int, str]]] = {}
        self.switch_gate_stats: Dict[str, int] = {}

    def _clear_interaction_anchors(self) -> None:
        self.interaction_anchors = {
            name: InteractionAnchor() for name in self.role_names
        }

    def _capture_interaction_anchors(self, frame_index: int) -> None:
        for role_name, state in self.states.items():
            feature = state.anchor_appearance_feature
            if feature is None:
                feature = state.appearance_feature

            self.interaction_anchors[role_name] = InteractionAnchor(
                feature=None if feature is None else feature.copy(),
                bbox=None if state.last_bbox is None else state.last_bbox.copy(),
                center=None if state.last_center is None else state.last_center.copy(),
                velocity=None if state.velocity is None else state.velocity.copy(),
                frame_index=frame_index,
            )

    def _predict_interaction_anchor_bbox(
        self,
        role_name: str,
        frame_shape: Sequence[int],
        frame_index: int,
    ) -> Optional[np.ndarray]:
        anchor = self.interaction_anchors.get(role_name)
        if anchor is None or anchor.bbox is None:
            return None

        predicted_bbox = anchor.bbox.astype(np.float32).copy()
        if anchor.velocity is not None and anchor.frame_index >= 0:
            elapsed = max(0, frame_index - anchor.frame_index)
            if elapsed > 0:
                shift = (
                    anchor.velocity.astype(np.float32)
                    * float(min(elapsed, 6))
                    * float(self.interaction_anchor_velocity_scale)
                )
                predicted_bbox[[0, 2]] += shift[0]
                predicted_bbox[[1, 3]] += shift[1]
        return clip_bbox_to_frame(predicted_bbox, frame_shape)

    def _prune_tracker_role_history(self, frame_index: int) -> None:
        cutoff = frame_index - self.tracker_role_memory_frames
        stale_track_ids = []
        for track_id, observations in self.tracker_role_history.items():
            kept = [(obs_frame, obs_role) for obs_frame, obs_role in observations if obs_frame >= cutoff]
            if kept:
                self.tracker_role_history[track_id] = kept
            else:
                stale_track_ids.append(track_id)
        for track_id in stale_track_ids:
            self.tracker_role_history.pop(track_id, None)

    def _record_tracker_role(
        self,
        tracker_id: int,
        role_name: str,
        frame_index: int,
    ) -> None:
        if tracker_id < 0:
            return
        observations = self.tracker_role_history.setdefault(tracker_id, [])
        observations.append((frame_index, role_name))
        cutoff = frame_index - self.tracker_role_memory_frames
        self.tracker_role_history[tracker_id] = [
            (obs_frame, obs_role)
            for obs_frame, obs_role in observations
            if obs_frame >= cutoff
        ]

    def _is_close_interaction(
        self,
        detection_a: Detection,
        detection_b: Detection,
    ) -> bool:
        iou = bbox_iou(detection_a.bbox, detection_b.bbox)
        if iou >= self.interaction_iou_threshold:
            return True

        width_a = float(detection_a.bbox[2] - detection_a.bbox[0])
        height_a = float(detection_a.bbox[3] - detection_a.bbox[1])
        width_b = float(detection_b.bbox[2] - detection_b.bbox[0])
        height_b = float(detection_b.bbox[3] - detection_b.bbox[1])
        avg_diag = 0.5 * (math.hypot(width_a, height_a) + math.hypot(width_b, height_b))
        if avg_diag <= 1e-6:
            return False

        center_distance = float(np.linalg.norm(detection_a.center - detection_b.center))
        return center_distance / avg_diag <= self.interaction_center_distance_ratio

    def _assignment_pair_is_close(
        self,
        assignments: Dict[str, Optional[DetectionCandidate]],
    ) -> bool:
        if len(self.role_names) != 2:
            return False
        first = assignments.get(self.role_names[0])
        second = assignments.get(self.role_names[1])
        if first is None or second is None:
            return False
        return self._is_close_interaction(first.detection, second.detection)

    def _switch_gate_reason(
        self,
        role_name: str,
        candidate: DetectionCandidate,
    ) -> Optional[str]:
        state = self.states[role_name]
        detection = candidate.detection
        if candidate.is_recovered:
            return None
        if not state.initialized or state.tracker_id is None:
            return None
        if detection.tracker_id == state.tracker_id:
            return None
        if state.missing_frames > self.role_switch_long_gap_frames:
            return None
        if candidate.pose_result is None:
            return None

        weak_pose = (
            candidate.pose_mean_score < self.role_switch_min_pose_score
            or candidate.confident_keypoint_count < self.role_switch_min_keypoints
        )
        weak_detection = detection.confidence < self.role_switch_min_detection_score

        large_jump = False
        if state.last_center is not None and state.last_bbox is not None:
            predicted_center = state.last_center.astype(np.float32).copy()
            if state.velocity is not None:
                predicted_center += (
                    state.velocity.astype(np.float32)
                    * float(min(max(1, state.missing_frames + 1), 3))
                )
            bbox_width = max(1.0, float(state.last_bbox[2] - state.last_bbox[0]))
            bbox_height = max(1.0, float(state.last_bbox[3] - state.last_bbox[1]))
            bbox_diag = math.hypot(bbox_width, bbox_height)
            jump_ratio = float(np.linalg.norm(detection.center - predicted_center)) / max(bbox_diag, 1.0)
            large_jump = jump_ratio > self.role_switch_max_center_jump_ratio

        target_similarity = max(
            appearance_similarity(state.appearance_feature, candidate.appearance_feature),
            appearance_similarity(state.identity_feature, candidate.appearance_feature),
            appearance_similarity(state.anchor_appearance_feature, candidate.appearance_feature),
        )
        other_similarity = max(
            (
                max(
                    appearance_similarity(other_state.appearance_feature, candidate.appearance_feature),
                    appearance_similarity(other_state.identity_feature, candidate.appearance_feature),
                    appearance_similarity(other_state.anchor_appearance_feature, candidate.appearance_feature),
                )
                for other_role, other_state in self.states.items()
                if other_role != role_name
            ),
            default=0.0,
        )
        appearance_conflict = other_similarity > target_similarity + 0.08 and target_similarity < 0.55

        if weak_pose and weak_detection:
            return "weak_pose_and_detection"
        if weak_pose and large_jump:
            return "weak_pose_large_jump"
        if weak_detection and large_jump:
            return "weak_detection_large_jump"
        if appearance_conflict and (weak_pose or large_jump):
            return "appearance_conflict"
        return None

    def _suppress_unstable_track_switches(
        self,
        assignments: Dict[str, Optional[DetectionCandidate]],
    ) -> None:
        for role_name, candidate in list(assignments.items()):
            if candidate is None:
                continue
            reason = self._switch_gate_reason(role_name, candidate)
            if reason is None:
                continue
            self.switch_gate_stats[reason] = self.switch_gate_stats.get(reason, 0) + 1
            self.switch_gate_stats[f"{role_name}:{reason}"] = (
                self.switch_gate_stats.get(f"{role_name}:{reason}", 0) + 1
            )
            assignments[role_name] = None

    def assign(
        self,
        candidates: List[DetectionCandidate],
        frame_index: int,
        frame_shape: Sequence[int],
    ) -> Dict[str, Optional[DetectionCandidate]]:
        assignments = {name: None for name in self.role_names}
        if not candidates:
            self._update_states(assignments, frame_index)
            return assignments

        if not all(self.states[name].initialized for name in self.role_names) and len(candidates) >= 2:
            ordered = sorted(candidates, key=lambda item: float(item.detection.center[0]))
            assignments[self.role_names[0]] = ordered[0]
            assignments[self.role_names[1]] = ordered[-1]
            self._update_states(assignments, frame_index)
            return assignments

        frame_h, frame_w = frame_shape[:2]
        candidate_ids: List[Optional[int]] = [None] + list(range(len(candidates)))
        best_score = -float("inf")
        best_indices: Optional[Tuple[Optional[int], ...]] = None
        interaction_mode = self.interaction_cooldown > 0
        best_pair_is_close = False
        self._prune_tracker_role_history(frame_index)

        for index_combo in product(candidate_ids, repeat=len(self.role_names)):
            used = [idx for idx in index_combo if idx is not None]
            if len(set(used)) != len(used):
                continue

            combo_interaction_mode = interaction_mode
            pair_is_close = False
            score = 0.0
            if len(self.role_names) == 2:
                left = index_combo[0]
                right = index_combo[1]
                if left is not None and right is not None:
                    left_candidate = candidates[left]
                    right_candidate = candidates[right]
                    pair_is_close = self._is_close_interaction(
                        left_candidate.detection,
                        right_candidate.detection,
                    )
                    combo_interaction_mode = combo_interaction_mode or pair_is_close
                    left_x = left_candidate.detection.center[0]
                    right_x = right_candidate.detection.center[0]
                    state_a = self.states[self.role_names[0]]
                    state_b = self.states[self.role_names[1]]
                    if (
                        state_a.last_center is not None
                        and state_b.last_center is not None
                        and state_a.last_center[0] < state_b.last_center[0]
                        and left_x > right_x
                    ):
                        score -= 0.3

            for role_name, det_idx in zip(self.role_names, index_combo):
                candidate = candidates[det_idx] if det_idx is not None else None
                score += self._score_role_assignment(
                    role_name,
                    candidate,
                    frame_w,
                    frame_h,
                    frame_index,
                    interaction_mode=combo_interaction_mode,
                )

            if score > best_score:
                best_score = score
                best_indices = index_combo
                best_pair_is_close = pair_is_close

        if best_indices is not None:
            for role_name, det_idx in zip(self.role_names, best_indices):
                assignments[role_name] = candidates[det_idx] if det_idx is not None else None

        self._suppress_unstable_track_switches(assignments)
        best_pair_is_close = self._assignment_pair_is_close(assignments)

        if best_pair_is_close:
            if self.interaction_cooldown == 0:
                self._capture_interaction_anchors(frame_index)
            self.interaction_cooldown = self.interaction_cooldown_frames
        elif self.interaction_cooldown > 0:
            self.interaction_cooldown -= 1
            if self.interaction_cooldown == 0:
                self._clear_interaction_anchors()

        self._update_states(assignments, frame_index)
        return assignments

    def _score_role_assignment(
        self,
        role_name: str,
        candidate: Optional[DetectionCandidate],
        frame_width: int,
        frame_height: int,
        frame_index: int,
        interaction_mode: bool,
    ) -> float:
        state = self.states[role_name]

        if candidate is None:
            return -1.5 - 0.05 * state.missing_frames
        detection = candidate.detection

        score = 0.2 * detection.confidence
        side_prior_scale = self.interaction_side_prior_scale if interaction_mode else 1.0
        score += 0.7 * side_prior_scale * role_side_prior(role_name, detection, frame_width)

        if not state.initialized:
            if state.appearance_feature is not None and candidate.appearance_feature is not None:
                score += self.appearance_weight * appearance_similarity(
                    state.appearance_feature,
                    candidate.appearance_feature,
                )
            return score

        if state.tracker_id is not None and detection.tracker_id == state.tracker_id:
            tracker_bonus = 3.5
            if interaction_mode:
                tracker_bonus *= self.interaction_tracker_bonus_scale
            score += tracker_bonus

        if state.last_bbox is not None:
            last_bbox_iou_weight = 1.15 if interaction_mode else 2.0
            score += last_bbox_iou_weight * bbox_iou(state.last_bbox, detection.bbox)

            last_center = state.last_center if state.last_center is not None else detection.center
            frame_diag = math.hypot(frame_width, frame_height)
            distance = float(np.linalg.norm(detection.center - last_center)) / max(frame_diag, 1.0)
            center_distance_weight = 1.35 if interaction_mode else 2.5
            score -= center_distance_weight * distance

            last_area = max(state.last_bbox[2] - state.last_bbox[0], 1.0) * max(
                state.last_bbox[3] - state.last_bbox[1], 1.0
            )
            area_ratio = max(detection.area, 1.0) / max(last_area, 1.0)
            score -= 0.15 * abs(math.log(area_ratio))

        if (
            frame_index < self.initial_role_lock_frames
            and state.initial_center is not None
        ):
            initial_distance = abs(float(detection.center[0]) - float(state.initial_center[0])) / max(
                1.0,
                float(frame_width),
            )
            score -= self.initial_role_lock_weight * initial_distance

        if state.appearance_feature is not None and candidate.appearance_feature is not None:
            similarity = appearance_similarity(state.appearance_feature, candidate.appearance_feature)
            score += self.appearance_weight * similarity
            if state.appearance_updates >= 3 and similarity < 0.45:
                score -= 1.2

        if state.identity_feature is not None and candidate.appearance_feature is not None:
            identity_similarity = appearance_similarity(state.identity_feature, candidate.appearance_feature)
            score += self.identity_weight * identity_similarity
            if state.identity_updates >= 3 and identity_similarity < 0.35:
                score -= 0.8 if interaction_mode else 0.4

        if state.anchor_appearance_feature is not None and candidate.appearance_feature is not None:
            anchor_similarity = appearance_similarity(
                state.anchor_appearance_feature,
                candidate.appearance_feature,
            )
            anchor_weight = self.anchor_appearance_weight * (1.2 if interaction_mode else 0.75)
            score += anchor_weight * anchor_similarity
            if state.anchor_appearance_updates >= 3 and anchor_similarity < 0.42:
                score -= 1.4 if interaction_mode else 0.7

        if interaction_mode:
            interaction_anchor = self.interaction_anchors.get(role_name)
            if (
                interaction_anchor is not None
                and interaction_anchor.feature is not None
                and candidate.appearance_feature is not None
            ):
                interaction_similarity = appearance_similarity(
                    interaction_anchor.feature,
                    candidate.appearance_feature,
                )
                score += 3.2 * interaction_similarity
                if interaction_similarity < 0.4:
                    score -= 1.6

            predicted_anchor_bbox = self._predict_interaction_anchor_bbox(
                role_name,
                (frame_height, frame_width),
                frame_index,
            )
            if predicted_anchor_bbox is not None:
                score += self.interaction_anchor_motion_weight * bbox_iou(
                    predicted_anchor_bbox,
                    detection.bbox,
                )
                predicted_center = np.array(
                    [
                        0.5 * (predicted_anchor_bbox[0] + predicted_anchor_bbox[2]),
                        0.5 * (predicted_anchor_bbox[1] + predicted_anchor_bbox[3]),
                    ],
                    dtype=np.float32,
                )
                frame_diag = math.hypot(frame_width, frame_height)
                anchor_distance = float(np.linalg.norm(detection.center - predicted_center)) / max(
                    frame_diag,
                    1.0,
                )
                score -= self.interaction_anchor_distance_weight * anchor_distance

        tracker_history = self.tracker_role_history.get(detection.tracker_id)
        if tracker_history and len(tracker_history) >= self.tracker_role_min_history:
            same_role_votes = sum(1 for _, owner_role in tracker_history if owner_role == role_name)
            other_role_votes = len(tracker_history) - same_role_votes
            if same_role_votes > 0:
                score += self.tracker_role_bonus * (
                    float(same_role_votes) / float(len(tracker_history))
                )
            if other_role_votes > 0:
                penalty_scale = (
                    self.interaction_tracker_role_penalty_scale if interaction_mode else 1.0
                )
                score -= (
                    self.tracker_role_penalty
                    * penalty_scale
                    * (float(other_role_votes) / float(len(tracker_history)))
                )
                if same_role_votes == 0:
                    score -= 0.8 * penalty_scale

        if state.missing_frames > self.max_missed_frames:
            score -= 0.6

        return score

    def _update_states(
        self,
        assignments: Dict[str, Optional[DetectionCandidate]],
        frame_index: int,
    ) -> None:
        for role_name, candidate in assignments.items():
            state = self.states[role_name]
            if candidate is None:
                state.missing_frames += 1
                if state.missing_frames > self.max_missed_frames:
                    state.tracker_id = None
                continue

            self._apply_candidate_to_state(state, candidate, frame_index)

    def apply_recovered_candidate(
        self,
        role_name: str,
        candidate: DetectionCandidate,
        frame_index: int,
    ) -> None:
        self._apply_candidate_to_state(self.states[role_name], candidate, frame_index)

    def _apply_candidate_to_state(
        self,
        state: RoleState,
        candidate: DetectionCandidate,
        frame_index: int,
    ) -> None:
        detection = candidate.detection
        previous_center = None if state.last_center is None else state.last_center.copy()
        state.tracker_id = detection.tracker_id
        state.last_bbox = detection.bbox.copy()
        state.last_center = detection.center.copy()
        if state.initial_center is None:
            state.initial_center = state.last_center.copy()
        if previous_center is not None:
            state.velocity = state.last_center - previous_center
        state.last_seen_frame = frame_index
        state.missing_frames = 0
        state.initialized = True
        if (
            candidate.appearance_feature is not None
            and candidate.pose_mean_score >= self.appearance_min_pose_score
        ):
            if state.appearance_feature is None:
                state.appearance_feature = candidate.appearance_feature.copy()
            else:
                blended = (
                    (1.0 - self.appearance_alpha) * state.appearance_feature
                    + self.appearance_alpha * candidate.appearance_feature
                )
                state.appearance_feature = normalize_feature(blended)
            state.appearance_updates += 1

        if (
            self.interaction_cooldown == 0
            and not candidate.is_recovered
            and candidate.appearance_feature is not None
            and candidate.pose_mean_score >= self.identity_min_pose_score
        ):
            if state.identity_feature is None:
                state.identity_feature = candidate.appearance_feature.copy()
            else:
                identity_similarity = appearance_similarity(
                    state.identity_feature,
                    candidate.appearance_feature,
                )
                if state.identity_updates < 5 or identity_similarity >= 0.5:
                    identity_blended = (
                        (1.0 - self.identity_alpha) * state.identity_feature
                        + self.identity_alpha * candidate.appearance_feature
                    )
                    state.identity_feature = normalize_feature(identity_blended)
            state.identity_updates += 1

        if (
            self.interaction_cooldown == 0
            and candidate.appearance_feature is not None
            and candidate.pose_mean_score >= self.anchor_min_pose_score
        ):
            if state.anchor_appearance_feature is None:
                state.anchor_appearance_feature = candidate.appearance_feature.copy()
            else:
                anchor_blended = (
                    (1.0 - self.anchor_appearance_alpha) * state.anchor_appearance_feature
                    + self.anchor_appearance_alpha * candidate.appearance_feature
                )
                state.anchor_appearance_feature = normalize_feature(anchor_blended)
            state.anchor_appearance_updates += 1

        if not candidate.is_recovered:
            self._record_tracker_role(detection.tracker_id, state.name, frame_index)


def resolve_rtmlib_pose_spec(
    pose_alias: str,
    pose_mode: str,
    pose_checkpoint: Optional[str],
) -> Tuple[str, Tuple[int, int]]:
    if pose_alias not in RTMLIB_POSE_MODELS:
        raise ValueError(
            f"Unsupported rtmlib pose alias: {pose_alias}. "
            f"Choose from {sorted(RTMLIB_POSE_MODELS)}."
        )
    spec = RTMLIB_POSE_MODELS[pose_alias][pose_mode]
    pose_source = pose_checkpoint or str(spec["pose"])
    pose_input_size = tuple(spec["pose_input_size"])
    return pose_source, pose_input_size


class RTMPoseRunner:
    def __init__(
        self,
        pose_backend: str,
        pose_alias: str,
        pose_mode: str,
        pose_config: Optional[str],
        pose_checkpoint: Optional[str],
        device: str,
        temp_dir: Path,
        keep_temp_pose_inputs: bool,
    ) -> None:
        self.pose_backend = pose_backend
        self.temp_dir = temp_dir
        self.keep_temp_pose_inputs = keep_temp_pose_inputs
        self._allow_ndarray_input = True

        if pose_backend == "mmpose":
            global MMPoseInferencer
            if MMPoseInferencer is None:
                try:
                    from mmpose.apis import MMPoseInferencer as ImportedMMPoseInferencer
                except Exception as exc:
                    raise SystemExit(
                        "MMPose backend requested, but MMPose is not usable in this environment. "
                        "Use --pose-backend rtmlib or install full MMPose + mmcv."
                    ) from exc
                MMPoseInferencer = ImportedMMPoseInferencer
            kwargs = {
                "det_model": "whole_image",
                "device": device,
            }
            if pose_config:
                kwargs["pose2d"] = pose_config
                if pose_checkpoint:
                    kwargs["pose2d_weights"] = pose_checkpoint
            else:
                kwargs["pose2d"] = pose_alias

            self.inferencer = MMPoseInferencer(**kwargs)
            self.pose_source = pose_checkpoint or pose_config or pose_alias
            self.pose_input_size = None
        elif pose_backend == "rtmlib":
            if RTMLibPose is None:
                raise SystemExit(
                    "rtmlib backend requested, but rtmlib is not installed. "
                    "Install `rtmlib` and `onnxruntime` first."
                )
            pose_source, pose_input_size = resolve_rtmlib_pose_spec(
                pose_alias=pose_alias,
                pose_mode=pose_mode,
                pose_checkpoint=pose_checkpoint,
            )
            self.inferencer = RTMLibPose(
                pose_source,
                model_input_size=pose_input_size,
                to_openpose=False,
                backend="onnxruntime",
                device=device,
            )
            self.pose_source = pose_source
            self.pose_input_size = pose_input_size
        else:
            raise ValueError(f"Unsupported pose backend: {pose_backend}")

    def predict(self, crop_bgr: np.ndarray, frame_index: int, role_name: str) -> Optional[PoseResult]:
        if self.pose_backend == "rtmlib":
            keypoints, scores = self.inferencer(crop_bgr)
            keypoints_arr = np.asarray(keypoints, dtype=np.float32)
            scores_arr = np.asarray(scores, dtype=np.float32)

            if keypoints_arr.size == 0:
                return None
            if keypoints_arr.ndim == 3:
                keypoints_arr = keypoints_arr[0]
            if scores_arr.ndim == 2:
                scores_arr = scores_arr[0]
            if len(keypoints_arr) == 0:
                return None

            return PoseResult(keypoints=keypoints_arr, scores=scores_arr.reshape(-1))

        result = None

        if self._allow_ndarray_input:
            try:
                generator = self.inferencer(
                    crop_bgr,
                    show=False,
                    return_vis=False,
                    return_datasamples=False,
                )
                result = next(generator)
            except Exception:
                self._allow_ndarray_input = False

        if result is None:
            temp_path = self.temp_dir / f"{role_name}_{frame_index:06d}.jpg"
            cv2.imwrite(str(temp_path), crop_bgr)
            generator = self.inferencer(
                str(temp_path),
                show=False,
                return_vis=False,
                return_datasamples=False,
            )
            result = next(generator)
            if not self.keep_temp_pose_inputs and temp_path.exists():
                temp_path.unlink()

        return parse_pose_result(result)


def parse_pose_result(result: dict) -> Optional[PoseResult]:
    predictions = result.get("predictions", [])
    if not predictions:
        return None

    instances = predictions[0] if isinstance(predictions[0], list) else predictions
    if not instances:
        return None

    def instance_score(instance: dict) -> float:
        scores = instance.get("keypoint_scores") or instance.get("keypoints_visible")
        if scores is None:
            return 0.0
        scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        return float(scores_arr.mean()) if scores_arr.size else 0.0

    best_instance = max(instances, key=instance_score)
    keypoints = np.asarray(best_instance.get("keypoints"), dtype=np.float32)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]

    scores = best_instance.get("keypoint_scores")
    if scores is None:
        scores = best_instance.get("keypoints_visible")
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1) if scores is not None else None

    if keypoints.size == 0:
        return None

    if scores_arr is None or scores_arr.size != len(keypoints):
        scores_arr = np.ones((len(keypoints),), dtype=np.float32)

    return PoseResult(keypoints=keypoints, scores=scores_arr)


def update_detection_track_history(
    histories: Dict[int, DetectionTrackHistory],
    detection: Detection,
    frame_index: int,
    max_length: int,
) -> DetectionTrackHistory:
    history = histories.setdefault(detection.tracker_id, DetectionTrackHistory())
    if history.last_frame_index >= 0 and frame_index - history.last_frame_index > max_length:
        history.centers.clear()
        history.bbox_heights.clear()

    history.centers.append(detection.center.copy())
    history.bbox_heights.append(float(detection.bbox[3] - detection.bbox[1]))
    if len(history.centers) > max_length:
        history.centers = history.centers[-max_length:]
    if len(history.bbox_heights) > max_length:
        history.bbox_heights = history.bbox_heights[-max_length:]
    history.last_frame_index = frame_index
    return history


def is_static_background_detection(
    detection: Detection,
    history: DetectionTrackHistory,
    args: argparse.Namespace,
    is_small_candidate: bool,
) -> bool:
    if not is_small_candidate:
        return False
    if len(history.centers) < args.filter_static_history:
        return False

    centers = np.asarray(history.centers, dtype=np.float32)
    heights = np.asarray(history.bbox_heights, dtype=np.float32)
    if len(centers) < 2:
        return False

    mean_motion = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).mean())
    height_jitter_ratio = float(heights.std() / max(heights.mean(), 1.0)) if len(heights) else 0.0
    return (
        mean_motion <= args.filter_static_motion_threshold
        and height_jitter_ratio <= args.filter_static_height_jitter_ratio
    )


def increment_filter_stats(filter_stats: Dict[str, int], reason: str) -> None:
    filter_stats[reason] = filter_stats.get(reason, 0) + 1


def prepare_detection_candidates(
    detections: List[Detection],
    frame: np.ndarray,
    frame_index: int,
    pose_runner: RTMPoseRunner,
    args: argparse.Namespace,
    track_histories: Dict[int, DetectionTrackHistory],
    filter_stats: Dict[str, int],
) -> List[DetectionCandidate]:
    frame_height, frame_width = frame.shape[:2]
    frame_area = max(1.0, float(frame_height * frame_width))
    frame_max_bbox_height = max(
        (float(det.bbox[3] - det.bbox[1]) for det in detections),
        default=1.0,
    )
    frame_max_bbox_area = max((float(det.area) for det in detections), default=1.0)
    candidates: List[DetectionCandidate] = []

    for detection in detections:
        increment_filter_stats(filter_stats, "input_detections")
        history = update_detection_track_history(
            histories=track_histories,
            detection=detection,
            frame_index=frame_index,
            max_length=max(args.filter_static_history, 2),
        )

        bbox_width = float(detection.bbox[2] - detection.bbox[0])
        bbox_height = float(detection.bbox[3] - detection.bbox[1])
        aspect_ratio = bbox_width / max(bbox_height, 1.0)
        height_ratio = bbox_height / max(1.0, float(frame_height))
        area_ratio = detection.area / frame_area
        relative_height_ratio = bbox_height / max(frame_max_bbox_height, 1.0)
        relative_area_ratio = detection.area / max(frame_max_bbox_area, 1.0)
        is_tiny_absolute = (
            height_ratio < args.filter_min_bbox_height_ratio
            or area_ratio < args.filter_min_bbox_area_ratio
        )
        is_small_relative = (
            relative_height_ratio < args.filter_relative_min_bbox_height_ratio
            or relative_area_ratio < args.filter_relative_min_bbox_area_ratio
        )
        is_small_candidate = is_tiny_absolute or is_small_relative

        rejected = False
        if (
            aspect_ratio < args.filter_min_bbox_aspect_ratio
            or aspect_ratio > args.filter_max_bbox_aspect_ratio
        ):
            increment_filter_stats(filter_stats, "reject_bad_aspect")
            rejected = True
        if is_static_background_detection(detection, history, args, is_small_candidate):
            increment_filter_stats(filter_stats, "reject_static_background")
            rejected = True
        if rejected:
            continue

        crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
            detection.bbox,
            frame.shape,
            args.bbox_padding,
        )
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            increment_filter_stats(filter_stats, "reject_empty_crop")
            continue

        pose_result = pose_runner.predict(crop, frame_index, f"candidate_{detection.tracker_id}")
        if pose_result is None:
            increment_filter_stats(filter_stats, "reject_no_pose")
            continue

        pose_mean_score = float(np.mean(pose_result.scores)) if pose_result.scores.size else 0.0
        confident_keypoint_count = int(
            np.count_nonzero(pose_result.scores >= args.filter_pose_keypoint_score)
        )
        appearance_feature = compute_appearance_feature(
            crop,
            pose_result,
            pose_score_threshold=args.filter_pose_keypoint_score,
        )

        if pose_mean_score < args.filter_min_pose_mean_score and is_small_candidate:
            increment_filter_stats(filter_stats, "reject_low_pose_score")
            continue
        if confident_keypoint_count < args.filter_min_pose_keypoints and is_small_candidate:
            increment_filter_stats(filter_stats, "reject_sparse_pose")
            continue

        pose_result.keypoints[:, 0] += crop_x1
        pose_result.keypoints[:, 1] += crop_y1
        candidates.append(
            DetectionCandidate(
                detection=detection,
                pose_crop_bbox=(crop_x1, crop_y1, crop_x2, crop_y2),
                pose_result=pose_result,
                pose_mean_score=pose_mean_score,
                confident_keypoint_count=confident_keypoint_count,
                appearance_feature=appearance_feature,
            )
        )
        increment_filter_stats(filter_stats, "kept_candidates")

    return candidates


def extract_detections(result) -> List[Detection]:
    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones((len(boxes),))
    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
    if ids is None:
        return []

    detections = []
    for box, score, track_id in zip(boxes, scores, ids):
        x1, y1, x2, y2 = box.astype(np.float32)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                tracker_id=int(track_id),
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                confidence=float(score),
            )
        )
    return detections


def update_track_probation_states(
    candidates: Sequence[DetectionCandidate],
    probation_states: Dict[int, TrackProbationState],
    frame_index: int,
) -> None:
    for candidate in candidates:
        track_id = candidate.detection.tracker_id
        state = probation_states.setdefault(track_id, TrackProbationState())
        if state.last_seen_frame == frame_index - 1:
            state.consecutive_frames += 1
        else:
            state.consecutive_frames = 1
        state.last_seen_frame = frame_index


def filter_candidates_by_probation(
    candidates: Sequence[DetectionCandidate],
    probation_states: Dict[int, TrackProbationState],
    role_states: Dict[str, RoleState],
    frame_index: int,
    args: argparse.Namespace,
    probation_stats: Dict[str, int],
) -> List[DetectionCandidate]:
    probation_frames = args.new_track_probation_frames
    if probation_frames <= 1:
        return list(candidates)

    if not all(state.initialized for state in role_states.values()):
        increment_filter_stats(probation_stats, "initialization_bypass")
        return list(candidates)

    active_track_ids = {
        state.tracker_id
        for state in role_states.values()
        if state.tracker_id is not None and state.missing_frames <= 1
    }
    filtered: List[DetectionCandidate] = []

    for candidate in candidates:
        track_id = candidate.detection.tracker_id
        if candidate.is_recovered or track_id in active_track_ids:
            filtered.append(candidate)
            continue

        state = probation_states.setdefault(track_id, TrackProbationState())
        if state.approved:
            filtered.append(candidate)
            continue

        required_frames = probation_frames
        weak_pose = (
            candidate.pose_mean_score < args.probation_low_pose_threshold
            or candidate.confident_keypoint_count < args.filter_min_pose_keypoints + 2
        )
        if weak_pose:
            required_frames += args.probation_extra_frames
            increment_filter_stats(probation_stats, "extended_low_pose")

        role_similarities = [
            appearance_similarity(role_state.appearance_feature, candidate.appearance_feature)
            for role_state in role_states.values()
            if role_state.appearance_feature is not None
        ]
        if role_similarities:
            best_similarity = max(role_similarities)
            if best_similarity < args.probation_min_appearance_similarity:
                required_frames += args.probation_extra_frames
                increment_filter_stats(probation_stats, "extended_low_similarity")

        if state.last_seen_frame == frame_index and state.consecutive_frames >= required_frames:
            state.approved = True
            increment_filter_stats(probation_stats, "approved")
            filtered.append(candidate)
            continue

        increment_filter_stats(probation_stats, "blocked")

    return filtered


def approve_assigned_track_candidates(
    assignments: Dict[str, Optional[DetectionCandidate]],
    probation_states: Dict[int, TrackProbationState],
) -> None:
    for candidate in assignments.values():
        if candidate is None:
            continue
        track_id = candidate.detection.tracker_id
        state = probation_states.setdefault(track_id, TrackProbationState())
        state.approved = True


def attempt_recovery_candidate(
    role_name: str,
    state_before: Optional[RoleState],
    current_states: Dict[str, RoleState],
    frame: np.ndarray,
    frame_index: int,
    pose_runner: RTMPoseRunner,
    args: argparse.Namespace,
    recovery_stats: Dict[str, int],
) -> Optional[DetectionCandidate]:
    increment_filter_stats(recovery_stats, "attempted")
    if state_before is None or not state_before.initialized:
        increment_filter_stats(recovery_stats, "skip_uninitialized")
        return None
    if state_before.last_bbox is None or state_before.last_center is None:
        increment_filter_stats(recovery_stats, "skip_no_history")
        return None

    gap_frames = int(state_before.missing_frames) + 1
    if gap_frames > args.recovery_max_gap:
        increment_filter_stats(recovery_stats, "skip_gap_too_large")
        return None

    predicted_bbox = predict_recovery_bbox(
        state=state_before,
        frame_shape=frame.shape,
        gap_frames=gap_frames,
        velocity_scale=args.recovery_velocity_scale,
    )
    if predicted_bbox is None:
        increment_filter_stats(recovery_stats, "skip_no_prediction")
        return None

    crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
        predicted_bbox,
        frame.shape,
        args.bbox_padding + args.recovery_extra_padding,
    )
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        increment_filter_stats(recovery_stats, "reject_empty_crop")
        return None

    pose_result = pose_runner.predict(crop, frame_index, f"{role_name}_recovery")
    if pose_result is None:
        increment_filter_stats(recovery_stats, "reject_no_pose")
        return None

    pose_mean_score = float(np.mean(pose_result.scores)) if pose_result.scores.size else 0.0
    confident_keypoint_count = int(
        np.count_nonzero(pose_result.scores >= args.filter_pose_keypoint_score)
    )
    if pose_mean_score < args.recovery_min_pose_mean_score:
        increment_filter_stats(recovery_stats, "reject_low_pose_score")
        return None
    if confident_keypoint_count < args.recovery_min_keypoints:
        increment_filter_stats(recovery_stats, "reject_sparse_pose")
        return None

    pose_result.keypoints[:, 0] += crop_x1
    pose_result.keypoints[:, 1] += crop_y1

    valid_mask = pose_result.scores >= args.filter_pose_keypoint_score
    valid_points = pose_result.keypoints[valid_mask]
    if len(valid_points) == 0:
        valid_points = pose_result.keypoints[np.isfinite(pose_result.keypoints).all(axis=1)]
    if len(valid_points) == 0:
        increment_filter_stats(recovery_stats, "reject_invalid_pose")
        return None

    pose_center = valid_points.mean(axis=0)
    bbox_center = np.array(
        [
            0.5 * (predicted_bbox[0] + predicted_bbox[2]),
            0.5 * (predicted_bbox[1] + predicted_bbox[3]),
        ],
        dtype=np.float32,
    )
    center_distance_ratio = float(np.linalg.norm(pose_center - bbox_center)) / max(
        math.hypot(
            float(predicted_bbox[2] - predicted_bbox[0]),
            float(predicted_bbox[3] - predicted_bbox[1]),
        ),
        1.0,
    )
    if center_distance_ratio > args.recovery_max_center_distance_ratio:
        increment_filter_stats(recovery_stats, "reject_off_target_pose")
        return None

    local_pose = PoseResult(
        keypoints=pose_result.keypoints.copy(),
        scores=pose_result.scores.copy(),
    )
    local_pose.keypoints[:, 0] -= crop_x1
    local_pose.keypoints[:, 1] -= crop_y1
    appearance_feature = compute_appearance_feature(
        crop,
        local_pose,
        pose_score_threshold=args.filter_pose_keypoint_score,
    )

    target_state = current_states.get(role_name)
    target_similarity = appearance_similarity(
        target_state.appearance_feature if target_state is not None else None,
        appearance_feature,
    )
    target_identity_similarity = appearance_similarity(
        target_state.identity_feature if target_state is not None else None,
        appearance_feature,
    )
    other_best_similarity = -1.0
    other_best_identity_similarity = -1.0
    for other_role_name, other_state in current_states.items():
        if other_role_name == role_name:
            continue
        other_best_similarity = max(
            other_best_similarity,
            appearance_similarity(other_state.appearance_feature, appearance_feature),
        )
        other_best_identity_similarity = max(
            other_best_identity_similarity,
            appearance_similarity(other_state.identity_feature, appearance_feature),
        )

    if (
        target_state is not None
        and (
            target_state.appearance_feature is not None
            or target_state.identity_feature is not None
        )
    ):
        best_target_similarity = max(target_similarity, target_identity_similarity)
        best_other_similarity = max(other_best_similarity, other_best_identity_similarity)
        if best_target_similarity < args.recovery_min_appearance_similarity:
            increment_filter_stats(recovery_stats, "reject_appearance")
            return None
        if best_other_similarity > best_target_similarity + args.recovery_appearance_margin:
            increment_filter_stats(recovery_stats, "reject_other_role_match")
            return None

    tracker_id = state_before.tracker_id if state_before.tracker_id is not None else -1
    if tracker_id >= 0:
        for other_role_name, other_state in current_states.items():
            if other_role_name == role_name:
                continue
            if (
                other_state.tracker_id == tracker_id
                and other_state.last_seen_frame == frame_index
                and other_state.missing_frames == 0
            ):
                increment_filter_stats(recovery_stats, "reject_duplicate_tracker")
                return None
    detection = Detection(
        tracker_id=int(tracker_id),
        bbox=predicted_bbox,
        confidence=max(0.15, 0.35 - 0.02 * float(gap_frames - 1)),
    )
    increment_filter_stats(recovery_stats, "recovered")
    return DetectionCandidate(
        detection=detection,
        pose_crop_bbox=(crop_x1, crop_y1, crop_x2, crop_y2),
        pose_result=pose_result,
        pose_mean_score=pose_mean_score,
        confident_keypoint_count=confident_keypoint_count,
        appearance_feature=appearance_feature,
        is_recovered=True,
    )


def make_record(
    frame_index: int,
    role: str,
    role_id: int,
    fps: float,
    state_before: Optional[RoleState],
    detection: Optional[Detection],
    pose_crop_bbox: Optional[Tuple[int, int, int, int]],
    pose_result: Optional[PoseResult],
    is_recovered: bool = False,
) -> FrameRoleRecord:
    track_id_changed = False
    if state_before is not None and detection is not None:
        track_id_changed = state_before.tracker_id is not None and state_before.tracker_id != detection.tracker_id

    record = FrameRoleRecord(
        frame_index=frame_index,
        role=role,
        visible=detection is not None and pose_result is not None,
        tracker_id=int(role_id),
        tracker_bbox=detection.bbox.round(2).tolist() if detection is not None else None,
        pose_crop_bbox=list(pose_crop_bbox) if pose_crop_bbox is not None else None,
        source_tracker_id=int(role_id),
        raw_detector_tracker_id=detection.tracker_id if detection is not None else None,
        source_track_id_changed=False,
        raw_detector_track_id_changed=track_id_changed,
        is_recovered=is_recovered,
        role_id=int(role_id),
        stable_track_id=int(role_id),
        bbox_score=round(float(detection.confidence), 5) if detection is not None else None,
        frame_time_sec=round(float(frame_index) / max(float(fps), 1e-6), 6),
    )
    if pose_result is not None:
        record.raw_keypoints = pose_result.keypoints.round(3).tolist()
        record.raw_scores = pose_result.scores.round(5).tolist()
        record.raw_valid_mask = [bool(np.isfinite(point).all()) for point in pose_result.keypoints]
    return record


def _valid_pose_center(
    keypoints: Sequence[Sequence[float]],
    scores: Sequence[float],
    min_score: float,
) -> Optional[np.ndarray]:
    points = []
    for point, score in zip(keypoints, scores):
        if float(score) < min_score:
            continue
        if len(point) < 2 or not np.isfinite(float(point[0])) or not np.isfinite(float(point[1])):
            continue
        points.append([float(point[0]), float(point[1])])
    if not points:
        return None
    return np.asarray(points, dtype=np.float32).mean(axis=0)


def _clamped_shift(shift: np.ndarray, max_shift_px: float) -> np.ndarray:
    length = float(np.linalg.norm(shift))
    if length > max_shift_px and length > 1e-6:
        return shift * (max_shift_px / length)
    return shift


def _soft_clamp_predicted_limbs(
    points: List[List[float]],
    previous_points: Sequence[Sequence[float]],
    predicted_mask: Sequence[bool],
    skeleton: Sequence[Tuple[int, int]],
) -> None:
    for parent_idx, child_idx in skeleton:
        if parent_idx >= len(points) or child_idx >= len(points):
            continue
        if parent_idx >= len(previous_points) or child_idx >= len(previous_points):
            continue
        if not predicted_mask[child_idx]:
            continue

        parent = np.asarray(points[parent_idx], dtype=np.float32)
        child = np.asarray(points[child_idx], dtype=np.float32)
        prev_parent = np.asarray(previous_points[parent_idx], dtype=np.float32)
        prev_child = np.asarray(previous_points[child_idx], dtype=np.float32)
        if not (
            np.isfinite(parent).all()
            and np.isfinite(child).all()
            and np.isfinite(prev_parent).all()
            and np.isfinite(prev_child).all()
        ):
            continue

        previous_length = float(np.linalg.norm(prev_child - prev_parent))
        current_length = float(np.linalg.norm(child - parent))
        if previous_length <= 1.0 or current_length <= previous_length * 1.18 + 6.0:
            continue

        previous_offset = prev_child - prev_parent
        if float(np.linalg.norm(previous_offset)) <= 1e-6:
            continue
        points[child_idx] = (parent + previous_offset).round(3).tolist()


def apply_occlusion_hold(
    record: FrameRoleRecord,
    hold_state: HeldPoseState,
    min_keypoint_score: float,
    partial_shift_px: float,
    partial_min_visible_keypoints: int,
) -> None:
    """Keep a role drawn when detector/pose recovery has no usable observation."""
    if record.raw_keypoints is not None:
        scores = list(record.raw_scores or [1.0] * len(record.raw_keypoints))
        if hold_state.keypoints is not None:
            current_center = _valid_pose_center(record.raw_keypoints, scores, min_keypoint_score)
            previous_center = hold_state.center
            visible_count = sum(1 for score in scores if float(score) >= min_keypoint_score)
            shift = np.zeros((2,), dtype=np.float32)
            if current_center is not None and previous_center is not None:
                shift = _clamped_shift(current_center - previous_center, partial_shift_px)

            if visible_count >= partial_min_visible_keypoints:
                completed = [list(point) for point in record.raw_keypoints]
                predicted_mask = [False] * len(completed)
                for idx, score in enumerate(scores):
                    if float(score) >= min_keypoint_score:
                        continue
                    if idx >= len(hold_state.keypoints):
                        continue
                    previous_point = hold_state.keypoints[idx]
                    completed[idx] = [
                        round(float(previous_point[0]) + float(shift[0]), 3),
                        round(float(previous_point[1]) + float(shift[1]), 3),
                    ]
                    predicted_mask[idx] = True
                _soft_clamp_predicted_limbs(
                    completed,
                    hold_state.keypoints,
                    predicted_mask,
                    SKELETONS.get(len(completed), []),
                )
                record.smoothed_keypoints = completed

        hold_state.keypoints = [list(point) for point in record.raw_keypoints]
        if record.smoothed_keypoints is not None:
            hold_state.keypoints = [list(point) for point in record.smoothed_keypoints]
        hold_state.scores = scores
        hold_state.bbox = list(record.tracker_bbox) if record.tracker_bbox is not None else None
        hold_state.tracker_id = record.tracker_id
        hold_state.center = _valid_pose_center(hold_state.keypoints, hold_state.scores, min_keypoint_score)
        return

    if hold_state.keypoints is None:
        return

    record.smoothed_keypoints = [list(point) for point in hold_state.keypoints]
    if hold_state.scores is not None:
        record.raw_scores = [round(max(0.05, float(score) * 0.98), 5) for score in hold_state.scores]
        hold_state.scores = list(record.raw_scores)
    if record.tracker_bbox is None and hold_state.bbox is not None:
        record.tracker_bbox = list(hold_state.bbox)
    if record.tracker_id is None:
        record.tracker_id = hold_state.tracker_id
    record.is_recovered = True


def update_hold_state_from_record(
    record: FrameRoleRecord,
    hold_state: HeldPoseState,
    min_keypoint_score: float,
) -> None:
    """Update last observed pose state without writing synthetic points into the record."""
    if record.raw_keypoints is None:
        return
    scores = list(record.raw_scores or [1.0] * len(record.raw_keypoints))
    hold_state.keypoints = [list(point) for point in record.raw_keypoints]
    if record.smoothed_keypoints is not None:
        hold_state.keypoints = [list(point) for point in record.smoothed_keypoints]
    hold_state.scores = scores
    hold_state.bbox = list(record.tracker_bbox) if record.tracker_bbox is not None else None
    hold_state.tracker_id = record.tracker_id
    hold_state.center = _valid_pose_center(hold_state.keypoints, hold_state.scores, min_keypoint_score)


def interpolate_short_gaps(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    max_gap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    smoothed = keypoints.copy()
    smoothed_valid = valid_mask.copy()
    num_frames, num_keypoints, _ = smoothed.shape

    for keypoint_idx in range(num_keypoints):
        frame_idx = 0
        while frame_idx < num_frames:
            if smoothed_valid[frame_idx, keypoint_idx]:
                frame_idx += 1
                continue

            gap_start = frame_idx
            while frame_idx < num_frames and not smoothed_valid[frame_idx, keypoint_idx]:
                frame_idx += 1
            gap_end = frame_idx
            gap_length = gap_end - gap_start

            left_idx = gap_start - 1
            right_idx = gap_end

            if (
                gap_length <= max_gap
                and left_idx >= 0
                and right_idx < num_frames
                and smoothed_valid[left_idx, keypoint_idx]
                and smoothed_valid[right_idx, keypoint_idx]
            ):
                left_point = smoothed[left_idx, keypoint_idx]
                right_point = smoothed[right_idx, keypoint_idx]
                for offset, target_idx in enumerate(range(gap_start, gap_end), start=1):
                    ratio = offset / (gap_length + 1)
                    smoothed[target_idx, keypoint_idx] = left_point * (1.0 - ratio) + right_point * ratio
                    smoothed_valid[target_idx, keypoint_idx] = True

    return smoothed, smoothed_valid


def apply_ema_smoothing(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    smoothed = keypoints.copy()
    num_frames, num_keypoints, _ = smoothed.shape

    for keypoint_idx in range(num_keypoints):
        previous: Optional[np.ndarray] = None
        for frame_idx in range(num_frames):
            if not valid_mask[frame_idx, keypoint_idx]:
                continue
            current = smoothed[frame_idx, keypoint_idx]
            if previous is None:
                previous = current.copy()
            else:
                previous = alpha * current + (1.0 - alpha) * previous
                smoothed[frame_idx, keypoint_idx] = previous

    return smoothed


def apply_bidirectional_ema_smoothing(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    forward = apply_ema_smoothing(keypoints.copy(), valid_mask, alpha)
    backward = apply_ema_smoothing(keypoints[::-1].copy(), valid_mask[::-1].copy(), alpha)[::-1]
    return ((forward + backward) * 0.5).astype(np.float32)


def _local_motion_speed(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    frame_idx: int,
    keypoint_idx: int,
) -> float:
    current = keypoints[frame_idx, keypoint_idx]
    previous_speed: Optional[float] = None
    next_speed: Optional[float] = None

    prev_idx = frame_idx - 1
    while prev_idx >= 0:
        if valid_mask[prev_idx, keypoint_idx]:
            previous_speed = float(np.linalg.norm(current - keypoints[prev_idx, keypoint_idx]))
            break
        prev_idx -= 1

    next_idx = frame_idx + 1
    while next_idx < keypoints.shape[0]:
        if valid_mask[next_idx, keypoint_idx]:
            next_speed = float(np.linalg.norm(keypoints[next_idx, keypoint_idx] - current))
            break
        next_idx += 1

    speeds = [value for value in (previous_speed, next_speed) if value is not None]
    return float(max(speeds)) if speeds else 0.0


def apply_adaptive_local_smoothing(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    window_radius: int,
    min_strength: float,
    max_strength: float,
    velocity_scale_px: float,
) -> np.ndarray:
    smoothed = keypoints.copy()
    if window_radius <= 0:
        return smoothed

    min_strength = float(np.clip(min_strength, 0.0, 0.95))
    max_strength = float(np.clip(max_strength, min_strength, 0.95))
    velocity_scale_px = max(1.0, float(velocity_scale_px))
    num_frames, num_keypoints, _ = keypoints.shape

    for keypoint_idx in range(num_keypoints):
        for frame_idx in range(num_frames):
            if not valid_mask[frame_idx, keypoint_idx]:
                continue

            weighted_points: List[np.ndarray] = []
            weights: List[float] = []
            for neighbor_idx in range(
                max(0, frame_idx - window_radius),
                min(num_frames, frame_idx + window_radius + 1),
            ):
                if not valid_mask[neighbor_idx, keypoint_idx]:
                    continue
                distance = abs(neighbor_idx - frame_idx)
                weight = float(window_radius + 1 - distance)
                weighted_points.append(keypoints[neighbor_idx, keypoint_idx])
                weights.append(weight)

            if len(weighted_points) <= 1:
                continue

            local_average = np.average(np.asarray(weighted_points, dtype=np.float32), axis=0, weights=np.asarray(weights))
            speed = _local_motion_speed(keypoints, valid_mask, frame_idx, keypoint_idx)
            motion_ratio = speed / (speed + velocity_scale_px)
            strength = max_strength * (1.0 - motion_ratio) + min_strength * motion_ratio
            smoothed[frame_idx, keypoint_idx] = (
                (1.0 - strength) * keypoints[frame_idx, keypoint_idx]
                + strength * local_average
            )

    return smoothed


def estimate_body_height_px(records: Sequence[FrameRoleRecord]) -> float:
    heights = []
    for record in records:
        if record.tracker_bbox is None or len(record.tracker_bbox) < 4:
            continue
        y1 = float(record.tracker_bbox[1])
        y2 = float(record.tracker_bbox[3])
        if np.isfinite(y1) and np.isfinite(y2) and y2 > y1:
            heights.append(y2 - y1)
    return float(np.median(heights)) if heights else 300.0


def postprocess_records(
    records_by_role: Dict[str, List[FrameRoleRecord]],
    min_keypoint_score: float,
    interpolate_gap: int,
    temporal_smoothing_mode: str,
    ema_alpha: float,
    bidirectional_smoothing_alpha: float,
    adaptive_smoothing_window: int,
    adaptive_smoothing_min_strength: float,
    adaptive_smoothing_max_strength: float,
    adaptive_smoothing_velocity_scale: float,
    occlusion_hold: str,
    interpolated_score_scale: float,
    partial_shift_px: float,
    partial_min_visible_keypoints: int,
) -> int:
    keypoint_count = 0

    for role, records in records_by_role.items():
        valid_records = [record for record in records if record.raw_keypoints is not None]
        if not valid_records:
            continue

        keypoint_count = len(valid_records[0].raw_keypoints)
        frame_count = len(records)
        raw = np.full((frame_count, keypoint_count, 2), np.nan, dtype=np.float32)
        valid = np.zeros((frame_count, keypoint_count), dtype=bool)
        raw_score_values = np.zeros((frame_count, keypoint_count), dtype=np.float32)

        for record in records:
            record.raw_valid_mask = [False] * keypoint_count
            if record.raw_keypoints is None or record.raw_scores is None:
                continue
            keypoints = np.asarray(record.raw_keypoints, dtype=np.float32)
            scores = np.asarray(record.raw_scores, dtype=np.float32)
            if len(keypoints) != keypoint_count:
                continue
            raw[record.frame_index] = keypoints
            score_count = min(len(scores), keypoint_count)
            raw_score_values[record.frame_index, :score_count] = scores[:score_count]
            valid[record.frame_index] = scores >= min_keypoint_score
            record.raw_valid_mask = [bool(value) for value in valid[record.frame_index].tolist()]

        interpolated, interpolated_valid = interpolate_short_gaps(raw, valid, interpolate_gap)
        if temporal_smoothing_mode == "ema":
            smoothed = apply_ema_smoothing(interpolated, interpolated_valid, ema_alpha)
        elif temporal_smoothing_mode == "bidirectional":
            smoothed = apply_bidirectional_ema_smoothing(
                interpolated,
                interpolated_valid,
                bidirectional_smoothing_alpha,
            )
        elif temporal_smoothing_mode == "adaptive":
            body_height = estimate_body_height_px(records)
            smoothed = apply_adaptive_local_smoothing(
                interpolated,
                interpolated_valid,
                window_radius=adaptive_smoothing_window,
                min_strength=adaptive_smoothing_min_strength,
                max_strength=adaptive_smoothing_max_strength,
                velocity_scale_px=max(1.0, body_height * adaptive_smoothing_velocity_scale),
            )
        else:
            smoothed = interpolated
        last_completed: Optional[np.ndarray] = None
        last_scores: Optional[List[float]] = None
        low_confidence_score = round(
            max(0.01, min_keypoint_score * float(np.clip(interpolated_score_scale, 0.01, 0.95))),
            5,
        )
        allow_partial_hold = occlusion_hold in {"partial", "full"}

        for record in records:
            if not interpolated_valid[record.frame_index].any():
                record.smoothed_scores = [0.0] * keypoint_count
                record.smoothed_valid_mask = [False] * keypoint_count
                record.keypoint_source = ["missing"] * keypoint_count
                if record.smoothed_keypoints is not None and all(
                    point[0] is not None and point[1] is not None for point in record.smoothed_keypoints
                ):
                    last_completed = np.asarray(record.smoothed_keypoints, dtype=np.float32)
                    last_scores = list(record.smoothed_scores or record.raw_scores or [low_confidence_score] * keypoint_count)
                continue

            role_points = []
            smoothed_scores: List[float] = []
            smoothed_valid_mask: List[bool] = []
            keypoint_source: List[str] = []
            raw_valid = valid[record.frame_index]
            for kp_idx in range(keypoint_count):
                if interpolated_valid[record.frame_index, kp_idx]:
                    role_points.append(smoothed[record.frame_index, kp_idx].round(3).tolist())
                    if raw_valid[kp_idx]:
                        score = float(raw_score_values[record.frame_index, kp_idx])
                        smoothed_scores.append(round(score, 5))
                        smoothed_valid_mask.append(True)
                        keypoint_source.append("raw")
                    else:
                        smoothed_scores.append(low_confidence_score)
                        smoothed_valid_mask.append(False)
                        keypoint_source.append("interpolated")
                else:
                    role_points.append([None, None])
                    smoothed_scores.append(0.0)
                    smoothed_valid_mask.append(False)
                    keypoint_source.append("missing")

            frame_valid = interpolated_valid[record.frame_index]
            visible_count = int(np.count_nonzero(frame_valid))
            if allow_partial_hold and last_completed is not None and visible_count >= partial_min_visible_keypoints:
                current_valid_points = smoothed[record.frame_index, frame_valid]
                previous_valid_points = last_completed[frame_valid]
                if len(current_valid_points) and len(previous_valid_points):
                    shift = _clamped_shift(
                        current_valid_points.mean(axis=0) - previous_valid_points.mean(axis=0),
                        partial_shift_px,
                    )
                else:
                    shift = np.zeros((2,), dtype=np.float32)

                predicted_mask = [False] * keypoint_count
                for kp_idx in range(keypoint_count):
                    if frame_valid[kp_idx] or kp_idx >= len(last_completed):
                        continue
                    predicted = last_completed[kp_idx] + shift
                    role_points[kp_idx] = predicted.round(3).tolist()
                    predicted_mask[kp_idx] = True
                    previous_score = (
                        float(last_scores[kp_idx])
                        if last_scores is not None and kp_idx < len(last_scores)
                        else low_confidence_score
                    )
                    smoothed_scores[kp_idx] = round(min(low_confidence_score, max(0.01, previous_score * 0.5)), 5)
                    smoothed_valid_mask[kp_idx] = False
                    keypoint_source[kp_idx] = "held"
                _soft_clamp_predicted_limbs(
                    role_points,
                    last_completed,
                    predicted_mask,
                    SKELETONS.get(keypoint_count, []),
                )

            record.smoothed_keypoints = role_points
            record.smoothed_scores = smoothed_scores
            record.smoothed_valid_mask = smoothed_valid_mask
            record.keypoint_source = keypoint_source
            if all(point[0] is not None and point[1] is not None for point in role_points):
                last_completed = np.asarray(role_points, dtype=np.float32)
            last_scores = list(smoothed_scores)

    return keypoint_count


def draw_overlay(
    frame: np.ndarray,
    records_for_frame: Iterable[FrameRoleRecord],
    min_keypoint_score: float = 0.2,
) -> np.ndarray:
    canvas = frame.copy()
    for record in records_for_frame:
        color = ROLE_COLORS.get(record.role, (255, 255, 255))

        if record.tracker_bbox is not None:
            x1, y1, x2, y2 = map(int, record.tracker_bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = record.role
            if record.tracker_id is not None:
                label += f" | ID {record.tracker_id}"
            if record.is_recovered:
                label += " | recovered"
            cv2.putText(
                canvas,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        points = record.smoothed_keypoints or record.raw_keypoints
        if points is None:
            continue
        point_scores = record.smoothed_scores or record.raw_scores or [1.0] * len(points)

        skeleton = SKELETONS.get(len(points), [])
        for start, end in skeleton:
            if start >= len(points) or end >= len(points):
                continue
            start_point = points[start]
            end_point = points[end]
            if (
                start_point[0] is None
                or start_point[1] is None
                or end_point[0] is None
                or end_point[1] is None
            ):
                continue
            if (
                start < len(point_scores)
                and end < len(point_scores)
                and (float(point_scores[start]) < min_keypoint_score or float(point_scores[end]) < min_keypoint_score)
            ):
                continue
            cv2.line(
                canvas,
                (int(round(start_point[0])), int(round(start_point[1]))),
                (int(round(end_point[0])), int(round(end_point[1]))),
                color,
                2,
                cv2.LINE_AA,
            )

        for point_index, point in enumerate(points):
            if point[0] is None or point[1] is None:
                continue
            score = float(point_scores[point_index]) if point_index < len(point_scores) else 1.0
            if score <= 0:
                continue
            px, py = int(round(point[0])), int(round(point[1]))
            radius = 3 if score >= min_keypoint_score else 2
            thickness = -1 if score >= min_keypoint_score else 1
            cv2.circle(canvas, (px, py), radius, color, thickness)

    return canvas


def _crop_from_bbox(frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
    if bbox is None or len(bbox) < 4:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def compute_reid_feature_from_bbox(frame: np.ndarray, bbox: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    crop = _crop_from_bbox(frame, bbox or [])
    if crop is None:
        return None

    feature_parts: List[np.ndarray] = []
    for start_ratio, end_ratio in ((0.0, 1.0), (0.0, 0.45), (0.45, 1.0), (0.55, 1.0)):
        start_y = int(round(crop.shape[0] * start_ratio))
        end_y = max(start_y + 1, int(round(crop.shape[0] * end_ratio)))
        part = crop[start_y:end_y]
        if part.size == 0:
            continue

        hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [10, 8, 4],
            [0, 180, 0, 256, 0, 256],
        )
        if hist is not None:
            hist_values = hist.reshape(-1).astype(np.float32)
            hist_norm = float(np.linalg.norm(hist_values))
            if hist_norm > 1e-8:
                hist_values /= hist_norm
            feature_parts.append(hist_values * 0.5)

        hue, saturation, value = cv2.split(hsv)
        skin_like = (
            ((hue < 25) | (hue > 165))
            & (saturation > 35)
            & (saturation < 190)
            & (value > 55)
        )
        white_like = (saturation < 55) & (value > 145)
        dark_like = value < 65
        feature_parts.append(
            np.array(
                [
                    float(np.mean(skin_like)),
                    float(np.mean(white_like)),
                    float(np.mean(dark_like)),
                ],
                dtype=np.float32,
            )
            * 2.0
        )

    if not feature_parts:
        return None
    return normalize_feature(np.concatenate(feature_parts).astype(np.float32))


def _swap_record_observation(left: FrameRoleRecord, right: FrameRoleRecord) -> None:
    observation_fields = [
        "visible",
        "tracker_bbox",
        "pose_crop_bbox",
        "raw_detector_tracker_id",
        "raw_keypoints",
        "raw_scores",
        "smoothed_keypoints",
        "is_recovered",
        "bbox_score",
        "raw_valid_mask",
        "smoothed_scores",
        "smoothed_valid_mask",
        "keypoint_source",
    ]
    for field_name in observation_fields:
        left_value = getattr(left, field_name)
        setattr(left, field_name, getattr(right, field_name))
        setattr(right, field_name, left_value)


def _recompute_raw_detector_switch_flags(records_by_role: Dict[str, List[FrameRoleRecord]]) -> None:
    for records in records_by_role.values():
        previous_id: Optional[int] = None
        for record in records:
            current_id = record.raw_detector_tracker_id
            record.source_track_id_changed = False
            record.raw_detector_track_id_changed = (
                previous_id is not None
                and current_id is not None
                and current_id != previous_id
            )
            if current_id is not None:
                previous_id = current_id


def _smooth_swap_flags(
    swap_scores: Sequence[Optional[float]],
    swap_margin: float,
    keep_margin: float,
    min_segment_length: int,
    bridge_gap: int,
) -> Tuple[List[bool], List[dict]]:
    flags = [score is not None and score > swap_margin for score in swap_scores]

    if bridge_gap > 0:
        idx = 0
        while idx < len(flags):
            if flags[idx]:
                idx += 1
                continue
            gap_start = idx
            while idx < len(flags) and not flags[idx]:
                idx += 1
            gap_end = idx
            if (
                gap_start > 0
                and gap_end < len(flags)
                and gap_end - gap_start <= bridge_gap
                and all(
                    swap_scores[gidx] is None or swap_scores[gidx] > -keep_margin
                    for gidx in range(gap_start, gap_end)
                )
            ):
                for gidx in range(gap_start, gap_end):
                    flags[gidx] = True

    segments: List[dict] = []
    idx = 0
    while idx < len(flags):
        if not flags[idx]:
            idx += 1
            continue
        start = idx
        values: List[float] = []
        while idx < len(flags) and flags[idx]:
            if swap_scores[idx] is not None:
                values.append(float(swap_scores[idx]))
            idx += 1
        end = idx - 1
        length = end - start + 1
        if length < min_segment_length:
            for clear_idx in range(start, end + 1):
                flags[clear_idx] = False
            continue
        segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "length": length,
                "mean_swap_advantage": round(float(np.mean(values)) if values else 0.0, 5),
                "max_swap_advantage": round(float(np.max(values)) if values else 0.0, 5),
            }
        )
    return flags, segments


def apply_offline_reid_repair(
    video_path: Path,
    records_by_role: Dict[str, List[FrameRoleRecord]],
    anchor_frames: int,
    min_bbox_score: float,
    swap_margin: float,
    keep_margin: float,
    min_segment_length: int,
    bridge_gap: int,
    single_margin: float,
    min_single_segment_length: int,
) -> dict:
    if len(DEFAULT_ROLE_NAMES) != 2:
        return {"enabled": False, "reason": "only_two_role_reid_is_supported"}
    role_a, role_b = DEFAULT_ROLE_NAMES
    records_a = records_by_role.get(role_a, [])
    records_b = records_by_role.get(role_b, [])
    frame_count = min(len(records_a), len(records_b))
    if frame_count == 0:
        return {"enabled": False, "reason": "no_records"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"enabled": False, "reason": f"failed_to_open_video:{video_path}"}

    prototypes = {role_a: [], role_b: []}
    swap_scores: List[Optional[float]] = [None] * frame_count
    frame_features: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = [
        (None, None) for _ in range(frame_count)
    ]

    frame_idx = 0
    while frame_idx < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        record_a = records_a[frame_idx]
        record_b = records_b[frame_idx]
        feature_a = compute_reid_feature_from_bbox(frame, record_a.tracker_bbox)
        feature_b = compute_reid_feature_from_bbox(frame, record_b.tracker_bbox)
        frame_features[frame_idx] = (feature_a, feature_b)

        if frame_idx < anchor_frames:
            for role_name, record, feature in (
                (role_a, record_a, feature_a),
                (role_b, record_b, feature_b),
            ):
                if feature is None:
                    continue
                if not record.visible or record.is_recovered:
                    continue
                if record.bbox_score is None or float(record.bbox_score) < min_bbox_score:
                    continue
                prototypes[role_name].append(feature)
        frame_idx += 1

    cap.release()

    if not prototypes[role_a] or not prototypes[role_b]:
        return {"enabled": False, "reason": "insufficient_anchor_features"}

    proto_a = normalize_feature(np.mean(np.stack(prototypes[role_a]), axis=0).astype(np.float32))
    proto_b = normalize_feature(np.mean(np.stack(prototypes[role_b]), axis=0).astype(np.float32))

    for idx, (feature_a, feature_b) in enumerate(frame_features):
        if feature_a is None or feature_b is None:
            continue
        current_score = appearance_similarity(proto_a, feature_a) + appearance_similarity(proto_b, feature_b)
        swapped_score = appearance_similarity(proto_b, feature_a) + appearance_similarity(proto_a, feature_b)
        swap_scores[idx] = swapped_score - current_score

    swap_flags, swap_segments = _smooth_swap_flags(
        swap_scores=swap_scores,
        swap_margin=swap_margin,
        keep_margin=keep_margin,
        min_segment_length=min_segment_length,
        bridge_gap=bridge_gap,
    )

    for idx, should_swap in enumerate(swap_flags):
        if should_swap:
            _swap_record_observation(records_a[idx], records_b[idx])

    single_move_scores = {
        f"{role_a}_to_{role_b}": [None] * frame_count,
        f"{role_b}_to_{role_a}": [None] * frame_count,
    }
    for idx, (feature_a, feature_b) in enumerate(frame_features):
        record_a = records_a[idx]
        record_b = records_b[idx]
        a_observed = record_a.tracker_bbox is not None and record_a.visible and not record_a.is_recovered
        b_observed = record_b.tracker_bbox is not None and record_b.visible and not record_b.is_recovered
        if a_observed and not b_observed and feature_a is not None:
            single_move_scores[f"{role_a}_to_{role_b}"][idx] = (
                appearance_similarity(proto_b, feature_a) - appearance_similarity(proto_a, feature_a)
            )
        if b_observed and not a_observed and feature_b is not None:
            single_move_scores[f"{role_b}_to_{role_a}"][idx] = (
                appearance_similarity(proto_a, feature_b) - appearance_similarity(proto_b, feature_b)
            )

    single_move_segments: Dict[str, List[dict]] = {}
    single_moved_frames = 0
    for direction, scores in single_move_scores.items():
        move_flags, move_segments = _smooth_swap_flags(
            swap_scores=scores,
            swap_margin=single_margin,
            keep_margin=0.0,
            min_segment_length=min_single_segment_length,
            bridge_gap=bridge_gap,
        )
        single_move_segments[direction] = move_segments
        source_role, target_role = direction.split("_to_", 1)
        source_records = records_by_role[source_role]
        target_records = records_by_role[target_role]
        for idx, should_move in enumerate(move_flags):
            if not should_move:
                continue
            _swap_record_observation(source_records[idx], target_records[idx])
            single_moved_frames += 1

    _recompute_raw_detector_switch_flags(records_by_role)
    swapped_frames = int(sum(1 for value in swap_flags if value))
    return {
        "enabled": True,
        "anchor_frames": anchor_frames,
        "prototype_frames": {
            role_a: len(prototypes[role_a]),
            role_b: len(prototypes[role_b]),
        },
        "swap_margin": swap_margin,
        "keep_margin": keep_margin,
        "min_swap_segment": min_segment_length,
        "bridge_gap": bridge_gap,
        "swapped_frames": swapped_frames,
        "swap_segments": swap_segments,
        "single_margin": single_margin,
        "min_single_segment": min_single_segment_length,
        "single_moved_frames": single_moved_frames,
        "single_move_segments": single_move_segments,
    }


def save_preview_video(
    video_path: Path,
    output_root: Path,
    records_by_role: Dict[str, List[FrameRoleRecord]],
    fps: float,
    frame_width: int,
    frame_height: int,
    min_keypoint_score: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for preview: {video_path}")
    output_video_path = output_root / "preview.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    frame_index = 0
    frame_count = min((len(records) for records in records_by_role.values()), default=0)
    while frame_index < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        records_for_frame = [
            records[frame_index]
            for records in records_by_role.values()
            if frame_index < len(records)
        ]
        writer.write(draw_overlay(frame, records_for_frame, min_keypoint_score=min_keypoint_score))
        frame_index += 1
    cap.release()
    writer.release()


def save_records_json(
    output_root: Path,
    metadata: dict,
    records_by_role: Dict[str, List[FrameRoleRecord]],
) -> None:
    serializable_records = {
        role: [record.__dict__ for record in records]
        for role, records in records_by_role.items()
    }
    payload = {
        "metadata": metadata,
        "roles": serializable_records,
    }
    with (output_root / "pose_sequences.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def save_records_csv(
    output_root: Path,
    records_by_role: Dict[str, List[FrameRoleRecord]],
) -> None:
    csv_path = output_root / "pose_sequences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "frame",
                "time",
                "role",
                "role_id",
                "stable_track_id",
                "visible",
                "recovered",
                "tracker_id",
                "track_id_changed",
                "source_tracker_id",
                "source_track_id_changed",
                "raw_detector_tracker_id",
                "raw_detector_track_id_changed",
                "bbox_score",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "keypoint_index",
                "raw_x",
                "raw_y",
                "raw_score",
                "raw_valid",
                "smoothed_x",
                "smoothed_y",
                "smoothed_score",
                "smoothed_valid",
                "keypoint_source",
            ]
        )

        for role, records in records_by_role.items():
            for record in records:
                raw_points = record.raw_keypoints or []
                raw_scores = record.raw_scores or []
                smoothed_points = record.smoothed_keypoints or []
                smoothed_scores = record.smoothed_scores or []
                raw_valid_mask = record.raw_valid_mask or []
                smoothed_valid_mask = record.smoothed_valid_mask or []
                keypoint_source = record.keypoint_source or []
                count = max(
                    len(raw_points),
                    len(smoothed_points),
                    len(raw_scores),
                    len(smoothed_scores),
                    len(keypoint_source),
                    1,
                )

                bbox = record.tracker_bbox or [None, None, None, None]
                for keypoint_index in range(count):
                    raw_point = raw_points[keypoint_index] if keypoint_index < len(raw_points) else [None, None]
                    raw_score = raw_scores[keypoint_index] if keypoint_index < len(raw_scores) else None
                    smoothed_point = (
                        smoothed_points[keypoint_index]
                        if keypoint_index < len(smoothed_points)
                        else [None, None]
                    )
                    writer.writerow(
                        [
                            record.frame_index,
                            record.frame_time_sec,
                            role,
                            record.role_id,
                            record.stable_track_id,
                            record.visible,
                            record.is_recovered,
                            record.tracker_id,
                            False,
                            record.source_tracker_id,
                            record.source_track_id_changed,
                            record.raw_detector_tracker_id,
                            record.raw_detector_track_id_changed,
                            record.bbox_score,
                            bbox[0],
                            bbox[1],
                            bbox[2],
                            bbox[3],
                            keypoint_index,
                            raw_point[0],
                            raw_point[1],
                            raw_score,
                            raw_valid_mask[keypoint_index] if keypoint_index < len(raw_valid_mask) else False,
                            smoothed_point[0],
                            smoothed_point[1],
                            smoothed_scores[keypoint_index] if keypoint_index < len(smoothed_scores) else None,
                            smoothed_valid_mask[keypoint_index] if keypoint_index < len(smoothed_valid_mask) else False,
                            keypoint_source[keypoint_index] if keypoint_index < len(keypoint_source) else "missing",
                        ]
                    )


def resolve_keypoint_names(count: int) -> List[str]:
    names = KEYPOINT_NAMES.get(count)
    if names is not None:
        return names
    return [f"kp_{idx}" for idx in range(count)]


def _point_with_score(points: np.ndarray, scores: np.ndarray, index: int) -> np.ndarray:
    if index >= len(points) or index >= len(scores):
        return np.array([0.0, 0.0, 0.0], dtype=float)
    x, y = points[index]
    if not np.isfinite(x) or not np.isfinite(y):
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return np.array([float(x), float(y), float(scores[index])], dtype=float)


def _legacy_coco17_from_record(record: FrameRoleRecord) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    points_source = record.smoothed_keypoints or record.raw_keypoints
    if points_source is None:
        return None, None

    points = np.asarray(
        [
            [np.nan if point[0] is None else float(point[0]), np.nan if point[1] is None else float(point[1])]
            for point in points_source
        ],
        dtype=float,
    )
    scores = np.asarray(record.smoothed_scores or record.raw_scores or [0.0] * len(points), dtype=float).reshape(-1)
    if len(points) < 17:
        return None, None

    legacy = np.zeros((17, 3), dtype=float)
    for index in range(17):
        legacy[index] = _point_with_score(points, scores, index)
    return legacy, points


def _override_enhanced_from_body26(enhanced: np.ndarray, body_points: np.ndarray, scores: Sequence[float]) -> None:
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)

    def set_if_valid(target: int, source: int) -> None:
        if source >= len(body_points) or source >= len(scores_arr):
            return
        x, y = body_points[source]
        if not np.isfinite(x) or not np.isfinite(y):
            return
        enhanced[target] = np.array([float(x), float(y), float(scores_arr[source])], dtype=float)

    # Halpe26: 17=head, 18=neck, 19=hip, 20/21=big toes, 22/23=small toes, 24/25=heels.
    set_if_valid(24, 17)
    set_if_valid(17, 18)
    set_if_valid(18, 19)
    set_if_valid(26, 24)
    set_if_valid(27, 25)
    set_if_valid(28, 20)
    set_if_valid(29, 21)
    set_if_valid(30, 22)
    set_if_valid(31, 23)

    for target, sources in ((22, (20, 22)), (23, (21, 23))):
        valid = []
        valid_scores = []
        for source in sources:
            if source < len(body_points) and source < len(scores_arr):
                x, y = body_points[source]
                if np.isfinite(x) and np.isfinite(y):
                    valid.append([float(x), float(y)])
                    valid_scores.append(float(scores_arr[source]))
        if valid:
            xy = np.asarray(valid, dtype=float).mean(axis=0)
            enhanced[target] = np.array([xy[0], xy[1], min(valid_scores)], dtype=float)


def save_legacy_project_outputs(
    output_root: Path,
    metadata: dict,
    records_by_role: Dict[str, List[FrameRoleRecord]],
) -> None:
    fps = float(metadata.get("fps") or 30.0)
    keypoint_records = []
    track_rows = []
    trajectory_rows = []

    for person_id, role in enumerate(DEFAULT_ROLE_NAMES, start=1):
        for record in records_by_role.get(role, []):
            legacy_kpts, body_points = _legacy_coco17_from_record(record)
            if legacy_kpts is None or body_points is None:
                continue

            enhanced = build_enhanced_keypoints(legacy_kpts)
            if len(body_points) >= 26 and record.raw_scores is not None:
                _override_enhanced_from_body26(enhanced, body_points, record.raw_scores)

            bbox = record.tracker_bbox or [0.0, 0.0, 0.0, 0.0]
            bbox_arr = np.asarray(bbox, dtype=float)
            score_values = record.smoothed_scores or record.raw_scores or []
            score = float(np.nanmean(score_values)) if score_values else 0.0
            center_x, center_y, center_score = compute_body_center(legacy_kpts, bbox_arr)
            frame = int(record.frame_index)

            track_rows.append(
                {
                    "frame": frame,
                    "time": frame / fps,
                    "person_id": person_id,
                    "raw_person_id": record.raw_detector_tracker_id if record.raw_detector_tracker_id is not None else person_id,
                    "x1": float(bbox_arr[0]),
                    "y1": float(bbox_arr[1]),
                    "x2": float(bbox_arr[2]),
                    "y2": float(bbox_arr[3]),
                    "score": score,
                }
            )
            trajectory_rows.append(
                {
                    "frame": frame,
                    "time": frame / fps,
                    "person_id": person_id,
                    "x": center_x,
                    "y": center_y,
                    "score": center_score if center_score > 0 else score,
                }
            )
            keypoint_records.append(
                {
                    "frame": frame,
                    "time": frame / fps,
                    "person_id": person_id,
                    "bbox": [float(value) for value in bbox_arr.tolist()],
                    "score": score,
                    "keypoint_names": LEGACY_KEYPOINT_NAMES,
                    "keypoints": [[float(x), float(y), float(c)] for x, y, c in legacy_kpts.tolist()],
                    "enhanced_keypoint_names": LEGACY_ENHANCED_KEYPOINT_NAMES,
                    "enhanced_keypoints": [[float(x), float(y), float(c)] for x, y, c in enhanced.tolist()],
                    "role": role,
                    "source": "rtmpose",
                    "is_recovered": bool(record.is_recovered),
                    "stable_track_id": int(record.stable_track_id),
                    "source_tracker_id": record.source_tracker_id,
                    "raw_detector_tracker_id": record.raw_detector_tracker_id,
                    "raw_detector_track_id_changed": bool(record.raw_detector_track_id_changed),
                    "bbox_score": record.bbox_score,
                    "keypoint_source": list(record.keypoint_source or []),
                }
            )

    keypoint_records.sort(key=lambda item: (int(item["frame"]), int(item["person_id"])))
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "keypoints.json").open("w", encoding="utf-8") as fp:
        json.dump(keypoint_records, fp, ensure_ascii=False, indent=2)

    for csv_name, rows, headers in (
        (
            "tracks.csv",
            track_rows,
            ["frame", "time", "person_id", "raw_person_id", "x1", "y1", "x2", "y2", "score"],
        ),
        (
            "trajectory.csv",
            trajectory_rows,
            ["frame", "time", "person_id", "x", "y", "score"],
        ),
    ):
        with (output_root / csv_name).open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


OPENPOSE_BODY25_NAMES = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "mid_hip",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]

OPENPOSE_BODY25_TO_HALPE26 = [
    0,
    18,
    6,
    8,
    10,
    5,
    7,
    9,
    19,
    12,
    14,
    16,
    11,
    13,
    15,
    2,
    1,
    4,
    3,
    20,
    22,
    24,
    21,
    23,
    25,
]


def _finite_point(point: Sequence[float]) -> bool:
    return (
        len(point) >= 2
        and point[0] is not None
        and point[1] is not None
        and np.isfinite(float(point[0]))
        and np.isfinite(float(point[1]))
    )


def _record_points_scores_sources(record: FrameRoleRecord) -> Tuple[List[List[Optional[float]]], List[float], List[str], List[bool]]:
    points = record.smoothed_keypoints or record.raw_keypoints or []
    scores = record.smoothed_scores or record.raw_scores or [0.0] * len(points)
    sources = record.keypoint_source or ["raw" if record.raw_keypoints is not None else "missing"] * len(points)
    valid_mask = record.smoothed_valid_mask or record.raw_valid_mask or [False] * len(points)
    count = max(len(points), len(scores), len(sources), len(valid_mask))

    padded_points: List[List[Optional[float]]] = []
    padded_scores: List[float] = []
    padded_sources: List[str] = []
    padded_valid: List[bool] = []
    for idx in range(count):
        point = points[idx] if idx < len(points) else [None, None]
        if _finite_point(point):
            padded_points.append([round(float(point[0]), 3), round(float(point[1]), 3)])
        else:
            padded_points.append([None, None])
        padded_scores.append(round(float(scores[idx]), 5) if idx < len(scores) and scores[idx] is not None else 0.0)
        padded_sources.append(str(sources[idx]) if idx < len(sources) else "missing")
        padded_valid.append(bool(valid_mask[idx]) if idx < len(valid_mask) else False)
    return padded_points, padded_scores, padded_sources, padded_valid


def _points_to_xyc(
    points: Sequence[Sequence[Optional[float]]],
    scores: Sequence[float],
    indices: Sequence[int],
) -> List[List[float]]:
    output: List[List[float]] = []
    for source_idx in indices:
        if source_idx < len(points) and _finite_point(points[source_idx]):
            score = float(scores[source_idx]) if source_idx < len(scores) else 0.0
            output.append([float(points[source_idx][0]), float(points[source_idx][1]), max(0.0, score)])
        else:
            output.append([0.0, 0.0, 0.0])
    return output


def _bbox_xywh(bbox: Optional[Sequence[float]]) -> Optional[List[float]]:
    if bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    return [round(x1, 3), round(y1, 3), round(max(0.0, x2 - x1), 3), round(max(0.0, y2 - y1), 3)]


def _bbox_xyxy(bbox: Optional[Sequence[float]]) -> Optional[List[float]]:
    if bbox is None or len(bbox) < 4:
        return None
    values = [float(value) for value in bbox[:4]]
    if not all(np.isfinite(value) for value in values):
        return None
    return [round(value, 3) for value in values]


def _visible_segments(records: Sequence[FrameRoleRecord], visible_attr: str = "visible") -> List[dict]:
    gaps: List[dict] = []
    start: Optional[int] = None
    for idx, record in enumerate(records):
        visible = bool(getattr(record, visible_attr))
        if not visible and start is None:
            start = int(record.frame_index)
        if (visible or idx == len(records) - 1) and start is not None:
            end = int(records[idx - 1].frame_index if visible else record.frame_index)
            gaps.append({"start_frame": start, "end_frame": end, "length": end - start + 1})
            start = None
    return gaps


def build_quality_report(
    metadata: dict,
    records_by_role: Dict[str, List[FrameRoleRecord]],
) -> dict:
    width = float(metadata.get("frame_width") or metadata.get("width") or 1.0)
    height = float(metadata.get("frame_height") or metadata.get("height") or 1.0)
    diag = max(1.0, math.hypot(width, height))
    keypoint_count = int(metadata.get("keypoint_count") or 0)
    role_reports: Dict[str, dict] = {}

    for role, records in records_by_role.items():
        frames = len(records)
        visible_frames = sum(1 for record in records if record.visible)
        recovered_frames = sum(1 for record in records if record.is_recovered)
        source_id_switches = sum(1 for record in records if record.source_track_id_changed)
        raw_detector_id_switches = sum(1 for record in records if record.raw_detector_track_id_changed)
        tracker_ids = sorted({int(record.tracker_id) for record in records if record.tracker_id is not None})
        source_tracker_ids = sorted(
            {int(record.source_tracker_id) for record in records if record.source_tracker_id is not None}
        )
        raw_detector_tracker_ids = sorted(
            {int(record.raw_detector_tracker_id) for record in records if record.raw_detector_tracker_id is not None}
        )
        tracker_id_switches = 0
        previous_tracker_id: Optional[int] = None
        for record in records:
            if record.tracker_id is None:
                continue
            if previous_tracker_id is not None and record.tracker_id != previous_tracker_id:
                tracker_id_switches += 1
            previous_tracker_id = record.tracker_id
        raw_scores = [
            float(score)
            for record in records
            for score in (record.raw_scores or [])
            if score is not None and np.isfinite(float(score))
        ]
        smoothed_scores = [
            float(score)
            for record in records
            for score in (record.smoothed_scores or [])
            if score is not None and np.isfinite(float(score))
        ]
        valid_keypoints = sum(sum(1 for value in (record.smoothed_valid_mask or []) if value) for record in records)
        interpolated_keypoints = sum(
            sum(1 for value in (record.keypoint_source or []) if value == "interpolated") for record in records
        )
        held_keypoints = sum(sum(1 for value in (record.keypoint_source or []) if value == "held") for record in records)
        missing_keypoints = sum(sum(1 for value in (record.keypoint_source or []) if value == "missing") for record in records)
        centers: List[Tuple[int, np.ndarray]] = []
        for record in records:
            bbox = _bbox_xyxy(record.tracker_bbox)
            if bbox is None:
                continue
            centers.append((int(record.frame_index), np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])))
        jumps: List[float] = []
        previous: Optional[Tuple[int, np.ndarray]] = None
        for frame, center in centers:
            if previous is not None and frame == previous[0] + 1:
                jumps.append(float(np.linalg.norm(center - previous[1])) / diag)
            previous = (frame, center)

        role_reports[role] = {
            "role_id": records[0].role_id if records else -1,
            "stable_track_id": records[0].stable_track_id if records else -1,
            "frames": frames,
            "visible_frames": visible_frames,
            "visible_ratio": visible_frames / frames if frames else 0.0,
            "recovered_frames": recovered_frames,
            "tracker_ids": tracker_ids,
            "track_id_switches": tracker_id_switches,
            "source_tracker_ids": source_tracker_ids,
            "source_track_id_switches": source_id_switches,
            "raw_detector_tracker_ids": raw_detector_tracker_ids,
            "raw_detector_track_id_switches": raw_detector_id_switches,
            "mean_raw_confidence": float(np.mean(raw_scores)) if raw_scores else 0.0,
            "mean_smoothed_confidence": float(np.mean(smoothed_scores)) if smoothed_scores else 0.0,
            "valid_keypoint_ratio": valid_keypoints / max(1, frames * keypoint_count),
            "interpolated_keypoints": interpolated_keypoints,
            "held_keypoints": held_keypoints,
            "missing_keypoints": missing_keypoints,
            "occlusion_segments": _visible_segments(records),
            "max_occlusion_length": max((segment["length"] for segment in _visible_segments(records)), default=0),
            "mean_bbox_center_jump_norm": float(np.mean(jumps)) if jumps else 0.0,
            "p95_bbox_center_jump_norm": float(np.percentile(jumps, 95)) if jumps else 0.0,
        }

    return {
        "summary": {
            "fps": metadata.get("fps"),
            "frame_count": metadata.get("frame_count") or metadata.get("processed_frames"),
            "frame_width": metadata.get("frame_width"),
            "frame_height": metadata.get("frame_height"),
            "keypoint_count": metadata.get("keypoint_count"),
            "occlusion_policy": metadata.get("settings", {}).get("occlusion_hold"),
            "interpolated_score_scale": metadata.get("settings", {}).get("interpolated_score_scale"),
        },
        "checklist": {
            "video_preprocessing": "fps/resolution/frame_index recorded in metadata and per-frame records",
            "detection": "two stable role slots are exported every frame; source detector bbox score is preserved",
            "tracking": "exported tracker_id/source_tracker_id/stable_track_id are fixed to 0/1; raw detector IDs are preserved as raw_detector_tracker_id",
            "bbox_postprocess": "tracker bbox and pose crop bbox are both exported; WHAM xyxy/xywh are generated",
            "pose_2d": "raw/smoothed keypoints, confidence, masks, and per-keypoint source are exported",
            "missing_handling": "missing/occluded joints are masked; interpolated joints are low confidence",
            "temporal_smoothing": "bidirectional EMA reduces jitter without causal lag; raw observations remain available",
            "formats": "COCO17, OpenPose BODY25, and WHAM-style 2D JSON are generated",
            "quality_report": "coverage, ID switches, confidence, occlusion segments, and bbox jitter are reported",
            "visualization": "preview.mp4 overlays observed skeleton; low-confidence joints are hollow and not connected",
        },
        "roles": role_reports,
    }


def save_engineering_2d_outputs(
    output_root: Path,
    metadata: dict,
    records_by_role: Dict[str, List[FrameRoleRecord]],
) -> None:
    keypoint_names = resolve_keypoint_names(int(metadata.get("keypoint_count") or 0))
    coco_indices = list(range(min(17, len(keypoint_names))))
    wham_tracks = []
    coco_annotations = []
    openpose_frames: Dict[int, dict] = {}

    for role, records in records_by_role.items():
        wham_frames = []
        for record in records:
            points, scores, sources, valid_mask = _record_points_scores_sources(record)
            halpe26 = _points_to_xyc(points, scores, list(range(min(len(points), 26))))
            coco17 = _points_to_xyc(points, scores, coco_indices)
            openpose25 = _points_to_xyc(points, scores, OPENPOSE_BODY25_TO_HALPE26)
            bbox_xyxy = _bbox_xyxy(record.tracker_bbox)
            frame_payload = {
                "frame_index": int(record.frame_index),
                "time": record.frame_time_sec,
                "role": role,
                "role_id": int(record.role_id),
                "stable_track_id": int(record.stable_track_id),
                "tracker_id": int(record.tracker_id) if record.tracker_id is not None else None,
                "source_tracker_id": record.source_tracker_id,
                "source_track_id_changed": bool(record.source_track_id_changed),
                "raw_detector_tracker_id": record.raw_detector_tracker_id,
                "raw_detector_track_id_changed": bool(record.raw_detector_track_id_changed),
                "visible": bool(record.visible),
                "is_recovered": bool(record.is_recovered),
                "bbox_xyxy": bbox_xyxy,
                "bbox_xywh": _bbox_xywh(record.tracker_bbox),
                "bbox_score": record.bbox_score,
                "pose_crop_bbox": record.pose_crop_bbox,
                "keypoints_halpe26": halpe26,
                "keypoints_coco17": coco17,
                "valid_keypoint_mask": [bool(value) for value in valid_mask],
                "keypoint_source": sources,
            }
            wham_frames.append(frame_payload)
            coco_annotations.append(
                {
                    "frame_index": int(record.frame_index),
                    "time": record.frame_time_sec,
                    "image_id": int(record.frame_index),
                    "category_id": 1,
                    "role": role,
                    "stable_track_id": int(record.stable_track_id),
                    "tracker_id": int(record.tracker_id) if record.tracker_id is not None else None,
                    "source_tracker_id": record.source_tracker_id,
                    "source_track_id_changed": bool(record.source_track_id_changed),
                    "raw_detector_tracker_id": record.raw_detector_tracker_id,
                    "raw_detector_track_id_changed": bool(record.raw_detector_track_id_changed),
                    "bbox": _bbox_xywh(record.tracker_bbox) or [0.0, 0.0, 0.0, 0.0],
                    "bbox_score": record.bbox_score or 0.0,
                    "keypoints": coco17,
                    "num_keypoints": sum(1 for point in coco17 if point[2] > 0.0),
                    "valid_keypoint_mask": [bool(value) for value in valid_mask[: len(coco17)]],
                    "keypoint_source": sources[: len(coco17)],
                }
            )
            openpose_frame = openpose_frames.setdefault(
                int(record.frame_index),
                {"frame_index": int(record.frame_index), "people": []},
            )
            openpose_frame["people"].append(
                {
                    "person_id": [int(record.stable_track_id)],
                    "role": role,
                    "tracker_id": int(record.tracker_id) if record.tracker_id is not None else None,
                    "source_tracker_id": record.source_tracker_id,
                    "source_track_id_changed": bool(record.source_track_id_changed),
                    "raw_detector_tracker_id": record.raw_detector_tracker_id,
                    "raw_detector_track_id_changed": bool(record.raw_detector_track_id_changed),
                    "pose_keypoints_2d": [value for point in openpose25 for value in point],
                    "keypoint_names": OPENPOSE_BODY25_NAMES,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_score": record.bbox_score,
                }
            )
        wham_tracks.append(
            {
                "role": role,
                "role_id": records[0].role_id if records else -1,
                "stable_track_id": records[0].stable_track_id if records else -1,
                "keypoint_format": "halpe26_with_coco17_subset",
                "frames": wham_frames,
            }
        )

    wham_payload = {
        "metadata": {
            **metadata,
            "format": "engineering_2d_for_wham",
            "stable_track_ids": {"character_A": 0, "character_B": 1},
            "tracker_id_policy": "tracker_id and source_tracker_id are stable role IDs used by WHAM; raw_detector_tracker_id keeps the original detector/BoT-SORT ID for diagnostics only",
            "confidence_policy": "raw observations keep model confidence; interpolated/held points are below min-keypoint-score",
        },
        "tracks": wham_tracks,
    }
    with (output_root / "wham_2d_observations.json").open("w", encoding="utf-8") as fp:
        json.dump(wham_payload, fp, ensure_ascii=False, indent=2)

    coco_payload = {
        "metadata": metadata,
        "keypoint_names": keypoint_names[: len(coco_indices)],
        "annotations": coco_annotations,
    }
    with (output_root / "coco_keypoints.json").open("w", encoding="utf-8") as fp:
        json.dump(coco_payload, fp, ensure_ascii=False, indent=2)

    openpose_payload = {
        "metadata": metadata,
        "keypoint_names": OPENPOSE_BODY25_NAMES,
        "frames": [openpose_frames[index] for index in sorted(openpose_frames)],
    }
    with (output_root / "openpose_keypoints.json").open("w", encoding="utf-8") as fp:
        json.dump(openpose_payload, fp, ensure_ascii=False, indent=2)

    quality_report = build_quality_report(metadata, records_by_role)
    with (output_root / "quality_report.json").open("w", encoding="utf-8") as fp:
        json.dump(quality_report, fp, ensure_ascii=False, indent=2)


def run_improved_3d_and_blend(args: argparse.Namespace, output_root: Path) -> None:
    if args.no_3d:
        return

    project_root = Path(__file__).resolve().parent.parent
    motion_root = Path(args.motion_output_root).resolve() if args.motion_output_root else output_root / "motion3d"
    motion_root.mkdir(parents=True, exist_ok=True)

    reconstruction_script = project_root / "scripts" / "two_character_3d_reconstruction.py"
    motion_cmd = [
        sys.executable,
        str(reconstruction_script),
        "--sequence-json",
        str(output_root / "pose_sequences.json"),
        "--metadata-json",
        str(output_root / "metadata.json"),
        "--output-root",
        str(motion_root),
        "--use-smoothed",
        "--reconstruction-mode",
        "improved",
        "--visibility-mode",
        "occlusion_aware",
        "--foot-ground-threshold",
        "0.18",
        "--foot-contact-speed",
        "0.65",
        "--foot-lock-min-frames",
        "2",
        "--foot-lock-blend",
        "0.92",
        "--collision-radius",
        "0.42",
        "--skip-wham-export",
    ]
    subprocess.run(motion_cmd, check=True)

    if args.no_blend:
        return

    blender_exe = Path(args.blender_exe)
    if not blender_exe.exists():
        print(f"Blender executable not found, skipped .blend generation: {blender_exe}")
        return

    blend_output = Path(args.blend_output).resolve() if args.blend_output else output_root / "two_person_mocap.blend"
    blend_script = project_root / "blender" / "motion3d_to_blend.py"
    blend_cmd = [
        str(blender_exe),
        "--background",
        "--python",
        str(blend_script),
        "--",
        str(motion_root / "motion3d_sequences.json"),
        str(blend_output),
    ]
    subprocess.run(blend_cmd, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output_root = ensure_dir(Path(args.output_root).resolve())
    temp_pose_dir = ensure_dir(output_root / "_temp_pose_inputs")
    crops_dir = ensure_dir(output_root / "crops") if args.save_crops else None

    video_path = Path(args.video_path).resolve()
    yolo_weights = Path(args.yolo_weights).resolve()
    yolo_model_source = str(yolo_weights) if yolo_weights.exists() else args.yolo_weights
    device = select_device(args.device, args.pose_backend)
    tracker_yaml = build_tracker_yaml(args, output_root)

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    model = YOLO(yolo_model_source)
    pose_runner = RTMPoseRunner(
        pose_backend=args.pose_backend,
        pose_alias=args.pose_alias,
        pose_mode=args.pose_mode,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint,
        device=device,
        temp_dir=temp_pose_dir,
        keep_temp_pose_inputs=args.keep_temp_pose_inputs,
    )
    role_manager = RoleLockManager(
        DEFAULT_ROLE_NAMES,
        args.max_missed_frames,
        appearance_alpha=args.appearance_alpha,
        appearance_weight=args.appearance_weight,
        appearance_min_pose_score=args.appearance_min_pose_score,
        identity_alpha=args.identity_alpha,
        identity_weight=args.identity_weight,
        identity_min_pose_score=args.identity_min_pose_score,
        initial_role_lock_frames=args.initial_role_lock_frames,
        initial_role_lock_weight=args.initial_role_lock_weight,
        anchor_appearance_alpha=args.anchor_appearance_alpha,
        anchor_appearance_weight=args.anchor_appearance_weight,
        anchor_min_pose_score=args.anchor_min_pose_score,
        interaction_iou_threshold=args.interaction_iou_threshold,
        interaction_center_distance_ratio=args.interaction_center_distance_ratio,
        interaction_cooldown_frames=args.interaction_cooldown_frames,
        interaction_side_prior_scale=args.interaction_side_prior_scale,
        interaction_anchor_motion_weight=args.interaction_anchor_motion_weight,
        interaction_anchor_distance_weight=args.interaction_anchor_distance_weight,
        interaction_tracker_bonus_scale=args.interaction_tracker_bonus_scale,
        interaction_anchor_velocity_scale=args.interaction_anchor_velocity_scale,
        tracker_role_memory_frames=args.tracker_role_memory_frames,
        tracker_role_min_history=args.tracker_role_min_history,
        tracker_role_bonus=args.tracker_role_bonus,
        tracker_role_penalty=args.tracker_role_penalty,
        interaction_tracker_role_penalty_scale=args.interaction_tracker_role_penalty_scale,
        role_switch_min_pose_score=args.role_switch_min_pose_score,
        role_switch_min_keypoints=args.role_switch_min_keypoints,
        role_switch_min_detection_score=args.role_switch_min_detection_score,
        role_switch_max_center_jump_ratio=args.role_switch_max_center_jump_ratio,
        role_switch_long_gap_frames=args.role_switch_long_gap_frames,
    )
    detection_filter_enabled = not args.disable_detection_filter

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 24.0

    writer = None
    if args.save_video and not args.no_preview:
        output_video_path = output_root / "preview.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (frame_width, frame_height),
        )

    records_by_role: Dict[str, List[FrameRoleRecord]] = {
        role: [] for role in DEFAULT_ROLE_NAMES
    }
    held_pose_states: Dict[str, HeldPoseState] = {
        role: HeldPoseState() for role in DEFAULT_ROLE_NAMES
    }
    detection_track_histories: Dict[int, DetectionTrackHistory] = {}
    detection_filter_stats: Dict[str, int] = {}
    track_probation_states: Dict[int, TrackProbationState] = {}

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_index >= args.max_frames:
            break

        state_snapshot = {
            role: RoleState(
                name=state.name,
                tracker_id=state.tracker_id,
                last_bbox=None if state.last_bbox is None else state.last_bbox.copy(),
                last_center=None if state.last_center is None else state.last_center.copy(),
                velocity=None if state.velocity is None else state.velocity.copy(),
                last_seen_frame=state.last_seen_frame,
                missing_frames=state.missing_frames,
                initialized=state.initialized,
                initial_center=None if state.initial_center is None else state.initial_center.copy(),
                appearance_feature=None if state.appearance_feature is None else state.appearance_feature.copy(),
                appearance_updates=state.appearance_updates,
                identity_feature=None if state.identity_feature is None else state.identity_feature.copy(),
                identity_updates=state.identity_updates,
                anchor_appearance_feature=None
                if state.anchor_appearance_feature is None
                else state.anchor_appearance_feature.copy(),
                anchor_appearance_updates=state.anchor_appearance_updates,
            )
            for role, state in role_manager.states.items()
        }

        track_results = model.track(
            frame,
            persist=True,
            tracker=str(tracker_yaml),
            classes=[0],
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            verbose=False,
        )
        result = track_results[0]
        detections = extract_detections(result)
        if detection_filter_enabled:
            candidates = prepare_detection_candidates(
                detections=detections,
                frame=frame,
                frame_index=frame_index,
                pose_runner=pose_runner,
                args=args,
                track_histories=detection_track_histories,
                filter_stats=detection_filter_stats,
            )
        else:
            candidates = [
                DetectionCandidate(
                    detection=detection,
                    pose_crop_bbox=(0, 0, 0, 0),
                    pose_result=None,
                    pose_mean_score=0.0,
                    confident_keypoint_count=0,
                    appearance_feature=None,
                )
                for detection in detections
            ]

        update_track_probation_states(candidates, track_probation_states, frame_index)
        probation_stats: Dict[str, int] = {}
        probation_candidates = filter_candidates_by_probation(
            candidates=candidates,
            probation_states=track_probation_states,
            role_states=role_manager.states,
            frame_index=frame_index,
            args=args,
            probation_stats=probation_stats,
        )

        assignments = role_manager.assign(probation_candidates, frame_index, frame.shape)
        recovery_stats: Dict[str, int] = {}
        for role_name in DEFAULT_ROLE_NAMES:
            if assignments[role_name] is not None:
                continue
            recovered_candidate = attempt_recovery_candidate(
                role_name=role_name,
                state_before=state_snapshot.get(role_name),
                current_states=role_manager.states,
                frame=frame,
                frame_index=frame_index,
                pose_runner=pose_runner,
                args=args,
                recovery_stats=recovery_stats,
            )
            if recovered_candidate is None:
                continue
            assignments[role_name] = recovered_candidate
            role_manager.apply_recovered_candidate(role_name, recovered_candidate, frame_index)

        approve_assigned_track_candidates(assignments, track_probation_states)
        frame_records: List[FrameRoleRecord] = []

        for role_id, role_name in enumerate(DEFAULT_ROLE_NAMES):
            assigned_candidate = assignments[role_name]
            detection = assigned_candidate.detection if assigned_candidate is not None else None
            pose_crop_bbox = None
            pose_result = None

            if detection is not None:
                if detection_filter_enabled:
                    if assigned_candidate is not None:
                        pose_crop_bbox = assigned_candidate.pose_crop_bbox
                        pose_result = assigned_candidate.pose_result
                else:
                    crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
                        detection.bbox,
                        frame.shape,
                        args.bbox_padding,
                    )
                    pose_crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    if crop.size > 0:
                        pose_result = pose_runner.predict(crop, frame_index, role_name)
                        if pose_result is not None:
                            pose_result.keypoints[:, 0] += crop_x1
                            pose_result.keypoints[:, 1] += crop_y1

                if crops_dir is not None and pose_crop_bbox is not None:
                    crop_x1, crop_y1, crop_x2, crop_y2 = pose_crop_bbox
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop.size > 0:
                        role_dir = ensure_dir(crops_dir / role_name)
                        cv2.imwrite(str(role_dir / f"{frame_index:06d}.jpg"), crop)

            record = make_record(
                frame_index=frame_index,
                role=role_name,
                role_id=role_id,
                fps=fps,
                state_before=state_snapshot.get(role_name),
                detection=detection,
                pose_crop_bbox=pose_crop_bbox,
                pose_result=pose_result,
                is_recovered=assigned_candidate.is_recovered if assigned_candidate is not None else False,
            )
            if args.occlusion_hold == "full":
                apply_occlusion_hold(
                    record,
                    held_pose_states[role_name],
                    min_keypoint_score=args.min_keypoint_score,
                    partial_shift_px=args.partial_pose_hold_shift_px,
                    partial_min_visible_keypoints=args.partial_pose_min_visible_keypoints,
                )
            else:
                update_hold_state_from_record(
                    record,
                    held_pose_states[role_name],
                    min_keypoint_score=args.min_keypoint_score,
                )
            records_by_role[role_name].append(record)
            frame_records.append(record)

        stop_requested = False
        if writer is not None or args.display:
            vis_frame = draw_overlay(frame, frame_records, min_keypoint_score=args.min_keypoint_score)
            if writer is not None:
                writer.write(vis_frame)
            if args.display:
                cv2.imshow("two_character_rtmpose_pipeline", vis_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    stop_requested = True

        frame_index += 1
        for key, value in probation_stats.items():
            detection_filter_stats[f"probation_{key}"] = detection_filter_stats.get(f"probation_{key}", 0) + value
        for key, value in recovery_stats.items():
            detection_filter_stats[f"recovery_{key}"] = detection_filter_stats.get(f"recovery_{key}", 0) + value
        if stop_requested:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    keypoint_count = postprocess_records(
        records_by_role=records_by_role,
        min_keypoint_score=args.min_keypoint_score,
        interpolate_gap=args.interpolate_gap,
        temporal_smoothing_mode=args.temporal_smoothing_mode,
        ema_alpha=args.ema_alpha,
        bidirectional_smoothing_alpha=args.bidirectional_smoothing_alpha,
        adaptive_smoothing_window=args.adaptive_smoothing_window,
        adaptive_smoothing_min_strength=args.adaptive_smoothing_min_strength,
        adaptive_smoothing_max_strength=args.adaptive_smoothing_max_strength,
        adaptive_smoothing_velocity_scale=args.adaptive_smoothing_velocity_scale,
        occlusion_hold=args.occlusion_hold,
        interpolated_score_scale=args.interpolated_score_scale,
        partial_shift_px=args.partial_pose_hold_shift_px,
        partial_min_visible_keypoints=args.partial_pose_min_visible_keypoints,
    )
    offline_reid_stats = {"enabled": False, "reason": "disabled"}
    if not args.disable_offline_reid:
        offline_reid_stats = apply_offline_reid_repair(
            video_path=video_path,
            records_by_role=records_by_role,
            anchor_frames=args.offline_reid_anchor_frames,
            min_bbox_score=args.offline_reid_min_bbox_score,
            swap_margin=args.offline_reid_swap_margin,
            keep_margin=args.offline_reid_keep_margin,
            min_segment_length=args.offline_reid_min_swap_segment,
            bridge_gap=args.offline_reid_bridge_gap,
            single_margin=args.offline_reid_single_margin,
            min_single_segment_length=args.offline_reid_min_single_segment,
        )

    metadata = {
        "video_path": str(video_path),
        "source_video": str(video_path),
        "project_root": str(project_root),
        "yolo_weights": yolo_model_source,
        "pose_backend": args.pose_backend,
        "pose_alias": args.pose_alias,
        "pose_mode": args.pose_mode,
        "pose_config": args.pose_config,
        "pose_checkpoint": args.pose_checkpoint,
        "pose_source": pose_runner.pose_source,
        "pose_input_size": pose_runner.pose_input_size,
        "device": device,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "width": frame_width,
        "height": frame_height,
        "fps": fps,
        "processed_frames": len(records_by_role[DEFAULT_ROLE_NAMES[0]]),
        "frame_count": len(records_by_role[DEFAULT_ROLE_NAMES[0]]),
        "tracked_people": len(DEFAULT_ROLE_NAMES),
        "ignored_detections": 0,
        "role_names": list(DEFAULT_ROLE_NAMES),
        "keypoint_count": keypoint_count,
        "keypoint_names": resolve_keypoint_names(keypoint_count),
        "tracker_yaml": str(tracker_yaml),
        "settings": {
            "bbox_padding": args.bbox_padding,
            "tracking_preset": args.tracking_preset,
            "yolo_conf": args.yolo_conf,
            "yolo_iou": args.yolo_iou,
            "min_keypoint_score": args.min_keypoint_score,
            "ema_alpha": args.ema_alpha,
            "interpolate_gap": args.interpolate_gap,
            "max_missed_frames": args.max_missed_frames,
            "detection_filter_enabled": detection_filter_enabled,
            "filter_min_bbox_height_ratio": args.filter_min_bbox_height_ratio,
            "filter_min_bbox_area_ratio": args.filter_min_bbox_area_ratio,
            "filter_relative_min_bbox_height_ratio": args.filter_relative_min_bbox_height_ratio,
            "filter_relative_min_bbox_area_ratio": args.filter_relative_min_bbox_area_ratio,
            "filter_min_bbox_aspect_ratio": args.filter_min_bbox_aspect_ratio,
            "filter_max_bbox_aspect_ratio": args.filter_max_bbox_aspect_ratio,
            "filter_min_pose_mean_score": args.filter_min_pose_mean_score,
            "filter_pose_keypoint_score": args.filter_pose_keypoint_score,
            "filter_min_pose_keypoints": args.filter_min_pose_keypoints,
            "filter_static_history": args.filter_static_history,
            "filter_static_motion_threshold": args.filter_static_motion_threshold,
            "filter_static_height_jitter_ratio": args.filter_static_height_jitter_ratio,
            "temporal_smoothing_mode": args.temporal_smoothing_mode,
            "bidirectional_smoothing_alpha": args.bidirectional_smoothing_alpha,
            "adaptive_smoothing_window": args.adaptive_smoothing_window,
            "adaptive_smoothing_min_strength": args.adaptive_smoothing_min_strength,
            "adaptive_smoothing_max_strength": args.adaptive_smoothing_max_strength,
            "adaptive_smoothing_velocity_scale": args.adaptive_smoothing_velocity_scale,
            "appearance_alpha": args.appearance_alpha,
            "appearance_weight": args.appearance_weight,
            "appearance_min_pose_score": args.appearance_min_pose_score,
            "identity_alpha": args.identity_alpha,
            "identity_weight": args.identity_weight,
            "identity_min_pose_score": args.identity_min_pose_score,
            "initial_role_lock_frames": args.initial_role_lock_frames,
            "initial_role_lock_weight": args.initial_role_lock_weight,
            "anchor_appearance_alpha": args.anchor_appearance_alpha,
            "anchor_appearance_weight": args.anchor_appearance_weight,
            "anchor_min_pose_score": args.anchor_min_pose_score,
            "interaction_iou_threshold": args.interaction_iou_threshold,
            "interaction_center_distance_ratio": args.interaction_center_distance_ratio,
            "interaction_cooldown_frames": args.interaction_cooldown_frames,
            "interaction_side_prior_scale": args.interaction_side_prior_scale,
            "interaction_anchor_motion_weight": args.interaction_anchor_motion_weight,
            "interaction_anchor_distance_weight": args.interaction_anchor_distance_weight,
            "interaction_tracker_bonus_scale": args.interaction_tracker_bonus_scale,
            "interaction_anchor_velocity_scale": args.interaction_anchor_velocity_scale,
            "tracker_role_memory_frames": args.tracker_role_memory_frames,
            "tracker_role_min_history": args.tracker_role_min_history,
            "tracker_role_bonus": args.tracker_role_bonus,
            "tracker_role_penalty": args.tracker_role_penalty,
            "interaction_tracker_role_penalty_scale": args.interaction_tracker_role_penalty_scale,
            "recovery_max_gap": args.recovery_max_gap,
            "recovery_extra_padding": args.recovery_extra_padding,
            "recovery_velocity_scale": args.recovery_velocity_scale,
            "recovery_min_pose_mean_score": args.recovery_min_pose_mean_score,
            "recovery_min_keypoints": args.recovery_min_keypoints,
            "recovery_min_appearance_similarity": args.recovery_min_appearance_similarity,
            "recovery_appearance_margin": args.recovery_appearance_margin,
            "recovery_max_center_distance_ratio": args.recovery_max_center_distance_ratio,
            "new_track_probation_frames": args.new_track_probation_frames,
            "probation_extra_frames": args.probation_extra_frames,
            "probation_low_pose_threshold": args.probation_low_pose_threshold,
            "probation_min_appearance_similarity": args.probation_min_appearance_similarity,
            "role_switch_min_pose_score": args.role_switch_min_pose_score,
            "role_switch_min_keypoints": args.role_switch_min_keypoints,
            "role_switch_min_detection_score": args.role_switch_min_detection_score,
            "role_switch_max_center_jump_ratio": args.role_switch_max_center_jump_ratio,
            "role_switch_long_gap_frames": args.role_switch_long_gap_frames,
            "offline_reid_enabled": not args.disable_offline_reid,
            "offline_reid_anchor_frames": args.offline_reid_anchor_frames,
            "offline_reid_min_bbox_score": args.offline_reid_min_bbox_score,
            "offline_reid_swap_margin": args.offline_reid_swap_margin,
            "offline_reid_keep_margin": args.offline_reid_keep_margin,
            "offline_reid_min_swap_segment": args.offline_reid_min_swap_segment,
            "offline_reid_bridge_gap": args.offline_reid_bridge_gap,
            "offline_reid_single_margin": args.offline_reid_single_margin,
            "offline_reid_min_single_segment": args.offline_reid_min_single_segment,
            "occlusion_hold": args.occlusion_hold,
            "occlusion_hold_missing_poses": args.occlusion_hold == "full",
            "interpolated_score_scale": args.interpolated_score_scale,
            "partial_pose_hold_shift_px": args.partial_pose_hold_shift_px,
            "partial_pose_min_visible_keypoints": args.partial_pose_min_visible_keypoints,
        },
        "detection_filter_stats": dict(sorted(detection_filter_stats.items())),
        "role_switch_gate_stats": dict(sorted(role_manager.switch_gate_stats.items())),
        "offline_reid_stats": offline_reid_stats,
    }

    with (output_root / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    save_records_json(output_root, metadata, records_by_role)
    save_records_csv(output_root, records_by_role)
    save_legacy_project_outputs(output_root, metadata, records_by_role)
    save_engineering_2d_outputs(output_root, metadata, records_by_role)
    if args.save_video and not args.no_preview:
        save_preview_video(
            video_path=video_path,
            output_root=output_root,
            records_by_role=records_by_role,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            min_keypoint_score=args.min_keypoint_score,
        )
    run_improved_3d_and_blend(args, output_root)

    if not args.keep_temp_pose_inputs and temp_pose_dir.exists():
        try:
            temp_pose_dir.rmdir()
        except OSError:
            pass

    print(f"Finished. Output saved to: {output_root}")


if __name__ == "__main__":
    main()
