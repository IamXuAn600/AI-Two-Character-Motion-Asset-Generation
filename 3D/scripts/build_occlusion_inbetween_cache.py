"""Build a z-swapped WHAM cache with smooth in-between motion for missing spans.

This replaces the older occlusion-freeze workflow. It does not create
Occluded labels, hold meshes, or visibility freezes. Instead it records each
visible->missing->visible span and fills the missing frames by interpolating
from the last valid pose before the gap to the first valid pose after it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROLE_LABELS = {
    "character_A": "Person 1",
    "character_B": "Person 2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply z-swap and occlusion in-betweening to a stable WHAM cache.")
    parser.add_argument("--cache", required=True, help="Stable-role WHAM cache from prepare_stable_wham_blend_cache.py.")
    parser.add_argument("--observations", required=True, help="wham_2d_observations.json with visible/missing flags.")
    parser.add_argument("--wham-output", required=True, help="WHAM wham_output.pkl for endpoint pose records.")
    parser.add_argument("--video", required=True, help="Source video for start/end frame snapshots.")
    parser.add_argument("--output-cache", required=True, help="Output .npz cache with z-swap and in-betweened gaps.")
    parser.add_argument("--events-json", required=True, help="Output JSON with occlusion start/end records.")
    parser.add_argument("--image-dir", required=True, help="Output folder for event start/end images.")
    parser.add_argument("--method", choices=("linear", "ease_in_out", "cubic", "sinusoidal"), default="ease_in_out")
    parser.add_argument("--min-gap-frames", type=int, default=1, help="Minimum missing-frame span to in-between.")
    parser.add_argument(
        "--occlusion-prediction-min-missing-frames",
        type=int,
        default=1,
        help="Consecutive undetected frames needed to flag a missing span as predicted occlusion/red-marker worthy.",
    )
    parser.add_argument("--swap-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mirror-x", action="store_true", help="Mirror both characters across a vertical source-X plane.")
    parser.add_argument(
        "--mirror-center-x",
        type=float,
        default=None,
        help="Source X coordinate used as mirror center. Defaults to the mean root X of all roles.",
    )
    parser.add_argument("--min-depth-separation", type=float, default=0.24)
    parser.add_argument(
        "--occlusion-depth-fade-frames",
        type=int,
        default=30,
        help="Small z-depth correction fade before/after an occlusion span to avoid depth pops.",
    )
    parser.add_argument(
        "--occlusion-depth-context-frames",
        type=int,
        default=120,
        help="Maximum close-contact context before/after an occlusion span where the occluded role stays behind.",
    )
    parser.add_argument("--penetration-root-distance", type=float, default=0.55)
    parser.add_argument("--close-distance-ratio", type=float, default=0.75)
    parser.add_argument("--max-z-delta-per-frame", type=float, default=0.012)
    parser.add_argument("--z-range-limit", type=float, default=2.50)
    parser.add_argument("--z-smoothing-alpha", type=float, default=0.12)
    parser.add_argument(
        "--occlusion-front-share",
        type=float,
        default=0.35,
        help="Share of missing z separation assigned to the visible/front person moving slightly forward.",
    )
    parser.add_argument("--hand-penetration-min-depth", type=float, default=0.34)
    parser.add_argument("--hand-keypoint-confidence", type=float, default=0.15)
    parser.add_argument(
        "--mesh-clearance-margin",
        type=float,
        default=0.10,
        help="Extra source-z clearance between front/back mesh envelopes to reduce limb/body penetration.",
    )
    parser.add_argument("--mesh-clearance-fade-frames", type=int, default=18)
    parser.add_argument("--prediction-velocity-window", type=int, default=8)
    parser.add_argument("--partial-observation-weight", type=float, default=0.45)
    parser.add_argument(
        "--lock-depth-during-z-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When character mesh z-depth intervals overlap, prevent front-person z from moving backward and back-person z from moving forward.",
    )
    parser.add_argument("--z-overlap-margin", type=float, default=0.0)
    parser.add_argument("--ground-y", type=float, default=0.0, help="Source-space floor height used to keep feet grounded.")
    parser.add_argument(
        "--max-final-depth-step",
        type=float,
        default=0.22,
        help="Deprecated compatibility option; use --max-z-delta-per-frame for the stabilizer.",
    )
    return parser.parse_args()


def interpolation_weight(t: float, method: str) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if method == "linear":
        return t
    if method == "ease_in_out":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    if method == "cubic":
        return t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    if method == "sinusoidal":
        return math.sin((t - 0.5) * math.pi) * 0.5 + 0.5
    raise ValueError(f"Unknown interpolation method: {method}")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def role_frame_maps(observations: dict[str, Any]) -> dict[str, dict[str, Any]]:
    maps = {}
    for track in observations.get("tracks", []):
        role = str(track.get("role"))
        maps[role] = {
            "role": role,
            "role_id": int(track.get("role_id", track.get("stable_track_id", len(maps)))),
            "stable_track_id": int(track.get("stable_track_id", track.get("role_id", len(maps)))),
            "frames": {int(frame["frame_index"]): frame for frame in track.get("frames", [])},
        }
    return maps


def frame_is_detected(frame: dict[str, Any] | None) -> bool:
    return bool(frame and frame.get("visible", False))


def frame_bbox(frame: dict[str, Any] | None) -> np.ndarray | None:
    if frame is None or not frame.get("bbox_xyxy"):
        return None
    return np.asarray(frame["bbox_xyxy"], dtype=np.float32)


def bbox_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return -1.0
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / (aa + bb - inter + 1e-6))


def center_ratio(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 2.0
    ca = np.asarray([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=np.float32)
    cb = np.asarray([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32)
    scale = max(float(a[3] - a[1]), float(b[3] - b[1]), 1.0)
    return float(np.linalg.norm(ca - cb) / scale)


def roles_close(role_frames: dict[str, dict[str, Any]], role_a: str, role_b: str, frame_idx: int, close_distance_ratio: float) -> bool:
    box_a = frame_bbox(role_frames[role_a]["frames"].get(frame_idx))
    box_b = frame_bbox(role_frames[role_b]["frames"].get(frame_idx))
    return center_ratio(box_a, box_b) <= close_distance_ratio or bbox_iou(box_a, box_b) >= 0.08


def detect_missing_events(observations: dict[str, Any], min_gap_frames: int, prediction_min_missing_frames: int) -> list[dict[str, Any]]:
    maps = role_frame_maps(observations)
    fps = float(observations.get("metadata", {}).get("fps") or 30.0)
    events: list[dict[str, Any]] = []
    event_id = 1
    min_gap_frames = max(1, int(min_gap_frames))
    prediction_min_missing_frames = max(1, int(prediction_min_missing_frames))
    event_threshold = min(min_gap_frames, prediction_min_missing_frames)

    for role, payload in maps.items():
        frames = payload["frames"]
        frame_ids = sorted(frames)
        if not frame_ids:
            continue
        in_gap = False
        start_frame: int | None = None
        anchor_frame: int | None = None
        last_visible: int | None = None
        previous_visible = frame_is_detected(frames.get(frame_ids[0]))
        if previous_visible:
            last_visible = frame_ids[0]

        for frame_idx in frame_ids[1:]:
            visible = frame_is_detected(frames.get(frame_idx))
            if previous_visible and not visible:
                in_gap = True
                start_frame = frame_idx
                anchor_frame = last_visible
            elif in_gap and visible:
                end_frame = frame_idx
                if anchor_frame is not None and start_frame is not None:
                    missing_count = end_frame - start_frame
                    if missing_count >= event_threshold:
                        prediction_trigger_frame = int(start_frame + min(prediction_min_missing_frames, missing_count) - 1)
                        events.append(
                            {
                                "event_id": event_id,
                                "role": role,
                                "person_label": ROLE_LABELS.get(role, f"Person {payload['role_id'] + 1}"),
                                "role_id": int(payload["role_id"]),
                                "stable_track_id": int(payload["stable_track_id"]),
                                "occlusion_start_frame": int(start_frame),
                                "start_anchor_frame": int(anchor_frame),
                                "occlusion_end_frame": int(end_frame),
                                "missing_frame_count": int(missing_count),
                                "duration_seconds": float(missing_count / fps),
                                "occlusion_prediction": bool(missing_count >= prediction_min_missing_frames),
                                "prediction_trigger_frame": prediction_trigger_frame,
                                "prediction_min_missing_frames": int(prediction_min_missing_frames),
                                "red_head_marker": True,
                            }
                        )
                        event_id += 1
                in_gap = False
                start_frame = None
                anchor_frame = None
            if visible:
                last_visible = frame_idx
            previous_visible = visible
        if in_gap and anchor_frame is not None and start_frame is not None:
            end_frame = frame_ids[-1]
            missing_count = end_frame - start_frame + 1
            if missing_count >= event_threshold:
                prediction_trigger_frame = int(start_frame + min(prediction_min_missing_frames, missing_count) - 1)
                events.append(
                    {
                        "event_id": event_id,
                        "role": role,
                        "person_label": ROLE_LABELS.get(role, f"Person {payload['role_id'] + 1}"),
                        "role_id": int(payload["role_id"]),
                        "stable_track_id": int(payload["stable_track_id"]),
                        "occlusion_start_frame": int(start_frame),
                        "start_anchor_frame": int(anchor_frame),
                        "occlusion_end_frame": int(end_frame),
                        "missing_frame_count": int(missing_count),
                        "duration_seconds": float(missing_count / fps),
                        "occlusion_prediction": True,
                        "prediction_trigger_frame": prediction_trigger_frame,
                        "prediction_min_missing_frames": int(prediction_min_missing_frames),
                        "terminal_gap": True,
                        "red_head_marker": True,
                    }
                )
                event_id += 1
    events = sorted(events, key=lambda item: (item["occlusion_start_frame"], item["role"]))
    for idx, event in enumerate(events, start=1):
        event["event_id"] = idx
    return events


def copy_cache_arrays(cache_path: Path) -> dict[str, np.ndarray]:
    cache = np.load(cache_path, allow_pickle=True)
    return {key: cache[key].copy() for key in cache.files}


def apply_depth_swap(arrays: dict[str, np.ndarray], role_a: str = "character_A", role_b: str = "character_B") -> None:
    roots_a_key = f"roots__{role_a}"
    roots_b_key = f"roots__{role_b}"
    verts_a_key = f"verts__{role_a}"
    verts_b_key = f"verts__{role_b}"
    if roots_a_key not in arrays or roots_b_key not in arrays:
        return
    roots_a = arrays[roots_a_key].copy()
    roots_b = arrays[roots_b_key].copy()
    verts_a = arrays[verts_a_key].copy()
    verts_b = arrays[verts_b_key].copy()
    za = roots_a[:, 2].copy()
    zb = roots_b[:, 2].copy()
    delta_a = (zb - za).astype(np.float32)
    delta_b = (za - zb).astype(np.float32)
    roots_a[:, 2] += delta_a
    roots_b[:, 2] += delta_b
    verts_a[:, :, 2] += delta_a[:, None]
    verts_b[:, :, 2] += delta_b[:, None]
    arrays[roots_a_key] = roots_a.astype(np.float32)
    arrays[roots_b_key] = roots_b.astype(np.float32)
    arrays[verts_a_key] = verts_a.astype(np.float32)
    arrays[verts_b_key] = verts_b.astype(np.float32)


def apply_mirror_x(arrays: dict[str, np.ndarray], center_x: float | None = None) -> float:
    subject_ids = [str(item) for item in arrays["subject_ids"].tolist()]
    if center_x is None:
        root_samples = []
        for role in subject_ids:
            key = f"roots__{role}"
            if key in arrays:
                root_samples.append(np.asarray(arrays[key][:, 0], dtype=np.float32))
        if root_samples:
            values = np.concatenate(root_samples)
            values = values[np.isfinite(values)]
            center_x = float(np.mean(values)) if len(values) else 0.0
        else:
            center_x = 0.0

    for role in subject_ids:
        roots_key = f"roots__{role}"
        verts_key = f"verts__{role}"
        if roots_key in arrays:
            roots = arrays[roots_key].copy()
            roots[:, 0] = float(2.0 * center_x) - roots[:, 0]
            arrays[roots_key] = roots.astype(np.float32)
        if verts_key in arrays:
            vertices = arrays[verts_key].copy()
            vertices[:, :, 0] = float(2.0 * center_x) - vertices[:, :, 0]
            arrays[verts_key] = vertices.astype(np.float32)

    if "faces" in arrays:
        faces = np.asarray(arrays["faces"], dtype=np.int32)
        if faces.ndim == 2 and faces.shape[1] == 3:
            arrays["faces"] = faces[:, [0, 2, 1]].copy()
    return float(center_x)


def observation_center(frame: dict[str, Any] | None, confidence_threshold: float = 0.05) -> np.ndarray | None:
    if not frame:
        return None
    bbox = frame_bbox(frame)
    if bbox is not None:
        return np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
    keypoints = frame.get("keypoints_coco17") or frame.get("keypoints_halpe26")
    if keypoints is None:
        return None
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return None
    valid = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & (points[:, 2] >= confidence_threshold)
    if int(np.count_nonzero(valid)) < 3:
        return None
    return np.mean(points[valid, :2], axis=0).astype(np.float32)


def estimate_root_velocity(roots: np.ndarray, frame_idx: int, window: int, direction: str) -> np.ndarray:
    frame_idx = int(np.clip(frame_idx, 0, len(roots) - 1))
    window = max(1, int(window))
    if direction == "before":
        other = max(0, frame_idx - window)
        if other == frame_idx:
            return np.zeros(3, dtype=np.float32)
        return ((roots[frame_idx] - roots[other]) / float(frame_idx - other)).astype(np.float32)
    other = min(len(roots) - 1, frame_idx + window)
    if other == frame_idx:
        return np.zeros(3, dtype=np.float32)
    return ((roots[other] - roots[frame_idx]) / float(other - frame_idx)).astype(np.float32)


def hermite_root(root_a: np.ndarray, root_b: np.ndarray, velocity_a: np.ndarray, velocity_b: np.ndarray, t: float, span: int) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00 * root_a
        + h10 * velocity_a * float(max(1, span))
        + h01 * root_b
        + h11 * velocity_b * float(max(1, span))
    ).astype(np.float32)


def partial_observation_progress(
    maps: dict[str, dict[str, Any]],
    role: str,
    anchor_frame: int,
    end_frame: int,
    frame_idx: int,
    fallback_t: float,
    partial_weight: float,
) -> tuple[float, bool]:
    frames = maps.get(role, {}).get("frames", {})
    anchor_center = observation_center(frames.get(anchor_frame))
    end_center = observation_center(frames.get(end_frame))
    current_center = observation_center(frames.get(frame_idx))
    if anchor_center is None or end_center is None or current_center is None:
        return float(np.clip(fallback_t, 0.0, 1.0)), False
    path = end_center - anchor_center
    denom = float(np.dot(path, path))
    if denom <= 1e-4:
        return float(np.clip(fallback_t, 0.0, 1.0)), False
    observed_t = float(np.dot(current_center - anchor_center, path) / denom)
    observed_t = float(np.clip(observed_t, 0.0, 1.0))
    partial_weight = float(np.clip(partial_weight, 0.0, 0.9))
    return float((1.0 - partial_weight) * fallback_t + partial_weight * observed_t), True


def interpolate_missing_motion(
    arrays: dict[str, np.ndarray],
    observations: dict[str, Any],
    events: list[dict[str, Any]],
    method: str,
    velocity_window: int,
    partial_observation_weight: float,
) -> None:
    maps = role_frame_maps(observations)
    for event in events:
        role = event["role"]
        start_anchor = int(event["start_anchor_frame"])
        start = int(event["occlusion_start_frame"])
        end = int(event["occlusion_end_frame"])
        if end <= start_anchor:
            continue
        roots = arrays[f"roots__{role}"]
        verts = arrays[f"verts__{role}"]
        root_a = roots[start_anchor].copy()
        root_b = roots[end].copy()
        verts_a = verts[start_anchor].copy()
        verts_b = verts[end].copy()
        velocity_a = estimate_root_velocity(roots, start_anchor, velocity_window, "before")
        velocity_b = estimate_root_velocity(roots, end, velocity_window, "after")
        span = max(1, end - start_anchor)
        used_partial_frames = 0
        max_motion_extra = 0.0
        previous_t = 0.0
        for frame_idx in range(start, end):
            raw_t = (frame_idx - start_anchor) / float(span)
            predicted_t, used_partial = partial_observation_progress(
                maps,
                role,
                start_anchor,
                end,
                frame_idx,
                raw_t,
                partial_observation_weight,
            )
            predicted_t = float(np.clip(max(previous_t, predicted_t), 0.0, 1.0))
            previous_t = predicted_t
            w = interpolation_weight(predicted_t, method)
            root_linear = (root_a * (1.0 - w) + root_b * w).astype(np.float32)
            root_predicted = hermite_root(root_a, root_b, velocity_a, velocity_b, predicted_t, span)
            extra = root_predicted - root_linear
            extra_cap = max(0.08, float(np.linalg.norm(root_b - root_a)) * 0.55 + 0.01 * float(span))
            extra_norm = float(np.linalg.norm(extra))
            if extra_norm > extra_cap:
                extra *= np.float32(extra_cap / max(extra_norm, 1e-6))
            root_final = (root_linear + extra).astype(np.float32)
            verts_linear = (verts_a * (1.0 - w) + verts_b * w).astype(np.float32)
            roots[frame_idx] = root_final
            verts[frame_idx] = verts_linear + (root_final - root_linear)[None, :]
            used_partial_frames += int(used_partial)
            max_motion_extra = max(max_motion_extra, float(np.linalg.norm(root_final - root_linear)))
        event["motion_prediction"] = {
            "method": "velocity_hermite_with_partial_2d_progress",
            "velocity_window": int(max(1, velocity_window)),
            "partial_observation_weight": float(np.clip(partial_observation_weight, 0.0, 0.9)),
            "partial_observation_frames": int(used_partial_frames),
            "max_root_extra_motion": float(max_motion_extra),
        }


def apply_depth_offsets(arrays: dict[str, np.ndarray], role_a: str, role_b: str, frame_idx: int, target_diff: float) -> None:
    roots_a = arrays[f"roots__{role_a}"]
    roots_b = arrays[f"roots__{role_b}"]
    verts_a = arrays[f"verts__{role_a}"]
    verts_b = arrays[f"verts__{role_b}"]
    current_diff = float(roots_a[frame_idx, 2] - roots_b[frame_idx, 2])
    adjustment = float((target_diff - current_diff) * 0.5)
    roots_a[frame_idx, 2] += adjustment
    roots_b[frame_idx, 2] -= adjustment
    verts_a[frame_idx, :, 2] += adjustment
    verts_b[frame_idx, :, 2] -= adjustment


class DepthStabilizer:
    """Smooth, capped z-depth repair for two-character WHAM caches."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        observations: dict[str, Any],
        events: list[dict[str, Any]],
        min_z_separation: float,
        max_z_delta_per_frame: float,
        z_range_limit: float,
        alpha: float,
        context_frames: int,
        occlusion_front_share: float,
        penetration_root_distance: float,
        close_distance_ratio: float,
        hand_penetration_min_depth: float,
        hand_keypoint_confidence: float,
        mesh_clearance_margin: float,
        lock_depth_during_z_overlap: bool,
        z_overlap_margin: float,
    ) -> None:
        self.arrays = arrays
        self.events = events
        self.roles = [str(item) for item in arrays["subject_ids"].tolist()]
        self.min_z_separation = max(0.0, float(min_z_separation))
        self.max_z_delta_per_frame = max(0.0, float(max_z_delta_per_frame))
        self.z_range_limit = max(0.0, float(z_range_limit))
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.context_frames = max(0, int(context_frames))
        self.occlusion_front_share = float(np.clip(occlusion_front_share, 0.0, 0.8))
        self.penetration_root_distance = max(0.0, float(penetration_root_distance))
        self.close_distance_ratio = float(close_distance_ratio)
        self.hand_penetration_min_depth = max(self.min_z_separation, float(hand_penetration_min_depth))
        self.hand_keypoint_confidence = max(0.0, float(hand_keypoint_confidence))
        self.mesh_clearance_margin = max(0.0, float(mesh_clearance_margin))
        self.lock_depth_during_z_overlap = bool(lock_depth_during_z_overlap)
        self.z_overlap_margin = max(0.0, float(z_overlap_margin))
        self.maps = role_frame_maps(observations)
        self.frame_count = self._frame_count()
        self.z_extent_offsets = self._precompute_z_extent_offsets()
        self.baseline_z = self._estimate_baselines()
        self.occlusion_constraints = self._build_occlusion_constraints()
        self.stats: dict[str, Any] = {
            "baseline_z": {role: float(value) for role, value in self.baseline_z.items()},
            "min_z_separation": float(self.min_z_separation),
            "max_z_delta_per_frame": float(self.max_z_delta_per_frame),
            "z_range_limit": float(self.z_range_limit),
            "alpha": float(self.alpha),
            "occlusion_front_share": float(self.occlusion_front_share),
            "hand_penetration_min_depth": float(self.hand_penetration_min_depth),
            "mesh_clearance_margin": float(self.mesh_clearance_margin),
            "lock_depth_during_z_overlap": bool(self.lock_depth_during_z_overlap),
            "z_overlap_margin": float(self.z_overlap_margin),
            "range_clamp_count": 0,
            "temporal_cap_count": 0,
            "separation_correction_count": 0,
            "penetration_correction_count": 0,
            "hand_penetration_correction_count": 0,
            "z_overlap_direction_lock_count": 0,
            "z_overlap_motion_freeze_count": 0,
            "semantic_occlusion_order_frame_count": 0,
            "dynamic_mesh_clearance_max": 0.0,
            "dynamic_mesh_clearance_count": 0,
        }

    def _frame_count(self) -> int:
        lengths = [len(self.arrays[f"roots__{role}"]) for role in self.roles if f"roots__{role}" in self.arrays]
        return min(lengths) if lengths else 0

    def _precompute_z_extent_offsets(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        extents: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role in self.roles:
            roots_key = f"roots__{role}"
            verts_key = f"verts__{role}"
            if roots_key not in self.arrays or verts_key not in self.arrays:
                continue
            roots_z = np.asarray(self.arrays[roots_key][: self.frame_count, 2], dtype=np.float32)
            verts_z = np.asarray(self.arrays[verts_key][: self.frame_count, :, 2], dtype=np.float32)
            extents[role] = (
                (np.nanmin(verts_z, axis=1) - roots_z).astype(np.float32),
                (np.nanmax(verts_z, axis=1) - roots_z).astype(np.float32),
            )
        return extents

    def _estimate_baselines(self) -> dict[str, float]:
        baselines: dict[str, float] = {}
        for role in self.roles:
            roots_key = f"roots__{role}"
            if roots_key not in self.arrays:
                continue
            roots = self.arrays[roots_key]
            other_roles = [candidate for candidate in self.roles if candidate != role and candidate in self.maps]
            normal_samples = []
            visible_samples = []
            for frame_idx in range(min(len(roots), self.frame_count)):
                z_value = float(roots[frame_idx, 2])
                if not np.isfinite(z_value):
                    continue
                frame = self.maps.get(role, {}).get("frames", {}).get(frame_idx)
                if frame_is_detected(frame) and not bool(frame.get("is_recovered", False)):
                    visible_samples.append(z_value)
                    if not any(roles_close(self.maps, role, other, frame_idx, self.close_distance_ratio) for other in other_roles):
                        normal_samples.append(z_value)
            samples = normal_samples if len(normal_samples) >= 5 else visible_samples
            if not samples:
                values = np.asarray(roots[: self.frame_count, 2], dtype=np.float32)
                values = values[np.isfinite(values)]
                samples = values.astype(float).tolist()
            baselines[role] = float(np.median(np.asarray(samples, dtype=np.float32))) if samples else 0.0
        return baselines

    def _clip_to_range(self, role: str, value: float) -> float:
        clipped = float(np.clip(value, self.baseline_z[role] - self.z_range_limit, self.baseline_z[role] + self.z_range_limit))
        if abs(clipped - value) > 1e-6:
            self.stats["range_clamp_count"] += 1
        return clipped

    def _bounded_from_previous(self, role: str, desired: float, previous: float | None) -> float:
        desired = self._clip_to_range(role, desired)
        if previous is None or self.max_z_delta_per_frame <= 0.0:
            return desired
        delta = desired - float(previous)
        if abs(delta) > self.max_z_delta_per_frame:
            self.stats["temporal_cap_count"] += 1
            desired = float(previous) + math.copysign(self.max_z_delta_per_frame, delta)
        return self._clip_to_range(role, desired)

    def _smoothed_value(self, role: str, raw_value: float, previous: float | None) -> float:
        raw_value = self._clip_to_range(role, raw_value)
        if previous is None:
            return raw_value
        desired = self.alpha * raw_value + (1.0 - self.alpha) * float(previous)
        return self._bounded_from_previous(role, desired, previous)

    def _close_context(self, role: str, other: str, frame_idx: int) -> bool:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return False
        if roles_close(self.maps, role, other, frame_idx, self.close_distance_ratio):
            return True
        role_root = self.arrays[f"roots__{role}"][frame_idx]
        other_root = self.arrays[f"roots__{other}"][frame_idx]
        return float(abs(role_root[0] - other_root[0])) <= self.penetration_root_distance

    def _build_occlusion_constraints(self) -> dict[int, list[tuple[str, str]]]:
        constraints: dict[int, list[tuple[str, str]]] = {}
        if len(self.roles) != 2:
            return constraints
        for event in self.events:
            role = str(event["role"])
            if role not in self.roles:
                continue
            other_roles = [candidate for candidate in self.roles if candidate != role]
            if not other_roles:
                continue
            other = other_roles[0]
            frame_count = min(len(self.arrays[f"roots__{role}"]), len(self.arrays[f"roots__{other}"]))
            start = max(0, min(int(event["occlusion_start_frame"]), frame_count - 1))
            end_visible = max(0, min(int(event["occlusion_end_frame"]), frame_count - 1))
            if end_visible < start:
                continue
            full_start = max(0, start - self.context_frames)
            full_end = end_visible
            for frame_idx in range(end_visible + 1, min(frame_count, end_visible + self.context_frames + 1)):
                if not self._close_context(role, other, frame_idx):
                    break
                full_end = frame_idx
            for frame_idx in range(full_start, full_end + 1):
                constraints.setdefault(frame_idx, []).append((role, other))
            event["depth_order_rule"] = {
                "occluded_role": role,
                "visible_role": other,
                "required": "occluded_z >= visible_z + min_z_separation",
                "min_z_separation": float(self.min_z_separation),
                "mesh_clearance_margin": float(self.mesh_clearance_margin),
                "full_start_frame": int(full_start),
                "full_end_frame": int(full_end),
                "correction": "smooth_capped_depth_stabilizer",
                "max_z_delta_per_frame": float(self.max_z_delta_per_frame),
                "z_range_limit": float(self.z_range_limit),
                "alpha": float(self.alpha),
                "front_share": float(self.occlusion_front_share),
                "z_overlap_direction_lock": bool(self.lock_depth_during_z_overlap),
            }
        return constraints

    def _z_mesh_overlap(self, z_values: dict[str, float], frame_idx: int) -> bool:
        if len(self.roles) != 2:
            return False
        role_a, role_b = self.roles[:2]
        if role_a not in self.z_extent_offsets or role_b not in self.z_extent_offsets:
            return abs(float(z_values[role_a] - z_values[role_b])) < self.min_z_separation
        a_min_offsets, a_max_offsets = self.z_extent_offsets[role_a]
        b_min_offsets, b_max_offsets = self.z_extent_offsets[role_b]
        a_min = float(z_values[role_a] + a_min_offsets[frame_idx])
        a_max = float(z_values[role_a] + a_max_offsets[frame_idx])
        b_min = float(z_values[role_b] + b_min_offsets[frame_idx])
        b_max = float(z_values[role_b] + b_max_offsets[frame_idx])
        return bool(min(a_max, b_max) - max(a_min, b_min) > -self.z_overlap_margin)

    def _mesh_clearance_separation(self, back_role: str, front_role: str, frame_idx: int, minimum: float | None = None) -> float:
        target = max(self.min_z_separation, float(minimum) if minimum is not None else 0.0)
        if back_role not in self.z_extent_offsets or front_role not in self.z_extent_offsets:
            return target
        back_min_offsets, _ = self.z_extent_offsets[back_role]
        _, front_max_offsets = self.z_extent_offsets[front_role]
        mesh_target = float(front_max_offsets[frame_idx] - back_min_offsets[frame_idx] + self.mesh_clearance_margin)
        if mesh_target > target:
            self.stats["dynamic_mesh_clearance_count"] += 1
            self.stats["dynamic_mesh_clearance_max"] = max(float(self.stats["dynamic_mesh_clearance_max"]), mesh_target)
            target = mesh_target
        return target

    def _apply_z_overlap_direction_lock(
        self,
        z_values: dict[str, float],
        previous_z: dict[str, float | None],
        back_role: str,
        front_role: str,
        frame_idx: int,
        semantic_order: bool,
    ) -> None:
        if not self.lock_depth_during_z_overlap or not self._z_mesh_overlap(z_values, frame_idx):
            return
        role_a, role_b = self.roles[:2]
        if not semantic_order and previous_z[role_a] is not None and previous_z[role_b] is not None:
            if float(previous_z[role_a]) >= float(previous_z[role_b]):
                back_role, front_role = role_a, role_b
            else:
                back_role, front_role = role_b, role_a
        changed = False
        previous_front = previous_z.get(front_role)
        previous_back = previous_z.get(back_role)
        if previous_front is not None and z_values[front_role] > float(previous_front):
            z_values[front_role] = float(previous_front)
            changed = True
        if previous_back is not None and z_values[back_role] < float(previous_back):
            z_values[back_role] = float(previous_back)
            changed = True
        if changed:
            self.stats["z_overlap_direction_lock_count"] += 1

    def _freeze_motion_during_z_overlap(
        self,
        z_values: dict[str, float],
        previous_z: dict[str, float | None],
        frame_idx: int,
    ) -> bool:
        if not self.lock_depth_during_z_overlap or not self._z_mesh_overlap(z_values, frame_idx):
            return False
        changed = False
        for role in self.roles:
            previous = previous_z.get(role)
            if previous is None:
                continue
            if abs(float(z_values[role]) - float(previous)) <= 1e-6:
                continue
            z_values[role] = float(previous)
            changed = True
        if changed:
            self.stats["z_overlap_motion_freeze_count"] += 1
        return changed

    def _frame_keypoints(self, role: str, frame_idx: int) -> np.ndarray | None:
        frame = self.maps.get(role, {}).get("frames", {}).get(frame_idx)
        if not frame:
            return None
        keypoints = frame.get("keypoints_coco17") or frame.get("keypoints_halpe26")
        if keypoints is None:
            return None
        points = np.asarray(keypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) < 11:
            return None
        return points

    def _arm_points_inside_other_bbox(self, role: str, other: str, frame_idx: int) -> int:
        points = self._frame_keypoints(role, frame_idx)
        other_box = frame_bbox(self.maps.get(other, {}).get("frames", {}).get(frame_idx))
        if points is None or other_box is None:
            return 0
        x1, y1, x2, y2 = [float(value) for value in other_box]
        count = 0
        for idx in (7, 8, 9, 10):
            if idx >= len(points):
                continue
            x, y, conf = [float(value) for value in points[idx, :3]]
            if conf < self.hand_keypoint_confidence or not np.isfinite([x, y, conf]).all():
                continue
            if x1 <= x <= x2 and y1 <= y <= y2:
                count += 1
        return count

    def _hand_penetration_target(self, z_values: dict[str, float], frame_idx: int, back_role: str, front_role: str) -> float:
        if len(self.roles) != 2:
            return 0.0
        role_a, role_b = self.roles[:2]
        close_2d = roles_close(self.maps, role_a, role_b, frame_idx, self.close_distance_ratio)
        box_iou = bbox_iou(frame_bbox(self.maps[role_a]["frames"].get(frame_idx)), frame_bbox(self.maps[role_b]["frames"].get(frame_idx)))
        if not close_2d and box_iou < 0.04:
            return 0.0
        intrusion_count = self._arm_points_inside_other_bbox(role_a, role_b, frame_idx) + self._arm_points_inside_other_bbox(role_b, role_a, frame_idx)
        if intrusion_count <= 0:
            return 0.0
        target = self._mesh_clearance_separation(back_role, front_role, frame_idx, self.hand_penetration_min_depth)
        diff = float(z_values[back_role] - z_values[front_role])
        if diff >= target:
            return 0.0
        self.stats["hand_penetration_correction_count"] += 1
        return float(target)

    def _penetration_target(
        self,
        z_values: dict[str, float],
        frame_idx: int,
        depth_order: float,
        back_role: str,
        front_role: str,
    ) -> float:
        if len(self.roles) != 2:
            return 0.0
        role_a, role_b = self.roles[:2]
        close_2d = roles_close(self.maps, role_a, role_b, frame_idx, self.close_distance_ratio)
        roots_a = self.arrays[f"roots__{role_a}"][frame_idx]
        roots_b = self.arrays[f"roots__{role_b}"][frame_idx]
        root_plane_distance = float(np.linalg.norm(np.asarray([roots_a[0], z_values[role_a]]) - np.asarray([roots_b[0], z_values[role_b]])))
        diff = float(z_values[role_a] - z_values[role_b])
        current_order = 1.0 if diff >= 0.0 else -1.0
        target = self._mesh_clearance_separation(back_role, front_role, frame_idx)
        ordered_diff = float(z_values[back_role] - z_values[front_role])
        order_confused = abs(diff) > self.min_z_separation * 0.5 and current_order != depth_order
        too_close = ordered_diff < target
        contact = close_2d or root_plane_distance <= self.penetration_root_distance
        return target if contact and (too_close or order_confused) else 0.0

    def _separate_pair(
        self,
        z_values: dict[str, float],
        previous_z: dict[str, float | None],
        back_role: str,
        front_role: str,
        target_separation: float,
        front_share: float,
    ) -> bool:
        missing = float(target_separation) - float(z_values[back_role] - z_values[front_role])
        if missing <= 0.0:
            return False
        front_share = float(np.clip(front_share, 0.0, 0.8))
        z_values[back_role] = self._bounded_from_previous(back_role, z_values[back_role] + missing * (1.0 - front_share), previous_z[back_role])
        z_values[front_role] = self._bounded_from_previous(front_role, z_values[front_role] - missing * front_share, previous_z[front_role])
        self.stats["separation_correction_count"] += 1
        return True

    def stabilize(self) -> dict[str, Any]:
        if self.frame_count <= 0 or len(self.roles) < 2:
            return self.stats
        role_a, role_b = self.roles[:2]
        previous_z: dict[str, float | None] = {role: None for role in self.roles}
        stabilized = {role: np.zeros(self.frame_count, dtype=np.float32) for role in self.roles}
        initial_diff = float(self.arrays[f"roots__{role_a}"][0, 2] - self.arrays[f"roots__{role_b}"][0, 2])
        depth_order = 1.0 if initial_diff >= 0.0 else -1.0

        for frame_idx in range(self.frame_count):
            z_values = {
                role: self._smoothed_value(role, float(self.arrays[f"roots__{role}"][frame_idx, 2]), previous_z[role])
                for role in self.roles
            }
            self._freeze_motion_during_z_overlap(z_values, previous_z, frame_idx)
            has_semantic_occlusion_order = False
            locked_back_role = role_a if depth_order >= 0.0 else role_b
            locked_front_role = role_b if locked_back_role == role_a else role_a
            constraints = self.occlusion_constraints.get(frame_idx, [])
            if constraints:
                has_semantic_occlusion_order = True
                self.stats["semantic_occlusion_order_frame_count"] += 1
                for back_role, front_role in constraints:
                    locked_back_role = back_role
                    locked_front_role = front_role
                    target = self._mesh_clearance_separation(back_role, front_role, frame_idx)
                    self._separate_pair(z_values, previous_z, back_role, front_role, target, self.occlusion_front_share)
                    depth_order = 1.0 if back_role == role_a else -1.0
            else:
                back_role = role_a if depth_order >= 0.0 else role_b
                front_role = role_b if back_role == role_a else role_a
                target = self._penetration_target(z_values, frame_idx, depth_order, back_role, front_role)
                if target > 0.0:
                    locked_back_role = back_role
                    locked_front_role = front_role
                    if self._separate_pair(z_values, previous_z, back_role, front_role, target, 0.5):
                        self.stats["penetration_correction_count"] += 1

            hand_target = self._hand_penetration_target(z_values, frame_idx, locked_back_role, locked_front_role)
            if hand_target > 0.0:
                if self._separate_pair(z_values, previous_z, locked_back_role, locked_front_role, hand_target, 0.5):
                    self.stats["penetration_correction_count"] += 1

            self._apply_z_overlap_direction_lock(
                z_values,
                previous_z,
                locked_back_role,
                locked_front_role,
                frame_idx,
                semantic_order=has_semantic_occlusion_order,
            )

            diff = float(z_values[role_a] - z_values[role_b])
            if not constraints and abs(diff) >= self.min_z_separation * 0.5:
                depth_order = 1.0 if diff >= 0.0 else -1.0

            for role, value in z_values.items():
                stabilized[role][frame_idx] = float(value)
                previous_z[role] = float(value)

        for role, z_values in stabilized.items():
            roots = self.arrays[f"roots__{role}"]
            vertices = self.arrays[f"verts__{role}"]
            original_z = roots[: self.frame_count, 2].copy()
            offset = (z_values - original_z).astype(np.float32)
            roots[: self.frame_count, 2] = z_values.astype(np.float32)
            vertices[: self.frame_count, :, 2] += offset[:, None]
            self.stats.setdefault("final_max_z_step", {})[role] = float(np.max(np.abs(np.diff(z_values)))) if len(z_values) > 1 else 0.0
            self.stats.setdefault("final_z_range", {})[role] = float(np.max(z_values) - np.min(z_values)) if len(z_values) else 0.0
        return self.stats


def enforce_depth_continuity(
    arrays: dict[str, np.ndarray],
    observations: dict[str, Any],
    min_depth_separation: float,
    penetration_root_distance: float,
    close_distance_ratio: float,
) -> None:
    subject_ids = [str(item) for item in arrays["subject_ids"].tolist()]
    if "character_A" not in subject_ids or "character_B" not in subject_ids:
        return
    role_a, role_b = "character_A", "character_B"
    maps = role_frame_maps(observations)
    roots_a = arrays[f"roots__{role_a}"]
    roots_b = arrays[f"roots__{role_b}"]
    depth_order = 1.0 if float(roots_a[0, 2] - roots_b[0, 2]) >= 0 else -1.0
    for frame_idx in range(len(roots_a)):
        diff = float(roots_a[frame_idx, 2] - roots_b[frame_idx, 2])
        root_plane_distance = float(np.linalg.norm(roots_a[frame_idx, [0, 2]] - roots_b[frame_idx, [0, 2]]))
        close_or_intersecting = roles_close(maps, role_a, role_b, frame_idx, close_distance_ratio) or root_plane_distance <= penetration_root_distance
        if close_or_intersecting:
            current_order = 1.0 if diff >= 0 else -1.0
            swapped_order = abs(diff) > min_depth_separation * 0.5 and current_order != depth_order
            too_close = abs(diff) < min_depth_separation
            if swapped_order or too_close:
                target_magnitude = max(abs(diff), min_depth_separation)
                apply_depth_offsets(arrays, role_a, role_b, frame_idx, depth_order * target_magnitude)
                diff = float(roots_a[frame_idx, 2] - roots_b[frame_idx, 2])
        if abs(diff) >= min_depth_separation * 0.5:
            depth_order = 1.0 if diff >= 0 else -1.0


def enforce_occlusion_depth_order(
    arrays: dict[str, np.ndarray],
    observations: dict[str, Any],
    events: list[dict[str, Any]],
    min_depth_separation: float,
    fade_frames: int,
    context_frames: int,
    penetration_root_distance: float,
    close_distance_ratio: float,
) -> None:
    """Keep the missing/occluded role behind the visible role without ID swaps.

    WHAM/source z-depth uses larger values as farther from camera in this
    project. During a missing span, the role marked by the event is the
    occluded person, so it must satisfy:

        occluded_z >= visible_z + min_depth_separation

    The correction is a pure depth translation on that same role's root and
    vertices. It never exchanges role ids, assigned WHAM fragments, or meshes.
    """

    subject_ids = [str(item) for item in arrays["subject_ids"].tolist()]
    if len(subject_ids) != 2:
        return
    fade_frames = max(0, int(fade_frames))
    context_frames = max(0, int(context_frames))
    maps = role_frame_maps(observations)
    corrections = {
        role: np.zeros(len(arrays[f"roots__{role}"]), dtype=np.float32)
        for role in subject_ids
        if f"roots__{role}" in arrays and f"verts__{role}" in arrays
    }

    def close_context(role: str, other: str, frame_idx: int) -> bool:
        if frame_idx < 0 or frame_idx >= len(arrays[f"roots__{role}"]):
            return False
        if roles_close(maps, role, other, frame_idx, close_distance_ratio):
            return True
        role_root = arrays[f"roots__{role}"][frame_idx]
        other_root = arrays[f"roots__{other}"][frame_idx]
        return float(abs(role_root[0] - other_root[0])) <= float(penetration_root_distance)

    def eased_ramp(value: float, weight: float) -> float:
        weight = float(np.clip(weight, 0.0, 1.0))
        return float(value) * (0.5 - 0.5 * math.cos(math.pi * weight))

    for event in events:
        role = str(event["role"])
        if role not in corrections:
            continue
        other_roles = [candidate for candidate in subject_ids if candidate != role]
        if not other_roles:
            continue
        other = other_roles[0]
        if f"roots__{other}" not in arrays:
            continue
        role_roots = arrays[f"roots__{role}"]
        other_roots = arrays[f"roots__{other}"]
        frame_count = min(len(role_roots), len(other_roots))
        start = max(0, min(int(event["occlusion_start_frame"]), frame_count - 1))
        end_visible = max(0, min(int(event["occlusion_end_frame"]), frame_count - 1))
        if end_visible < start:
            continue

        full_start = start
        for frame_idx in range(start - 1, max(-1, start - context_frames - 1), -1):
            if not close_context(role, other, frame_idx):
                break
            full_start = frame_idx

        full_end = end_visible
        for frame_idx in range(end_visible + 1, min(frame_count, end_visible + context_frames + 1)):
            if not close_context(role, other, frame_idx):
                break
            full_end = frame_idx

        required = np.maximum(
            0.0,
            other_roots[full_start : full_end + 1, 2] + float(min_depth_separation) - role_roots[full_start : full_end + 1, 2],
        ).astype(np.float32)
        plateau_correction = float(np.max(required)) if len(required) else 0.0
        if plateau_correction > 0.0:
            corrections[role][full_start : full_end + 1] = np.maximum(corrections[role][full_start : full_end + 1], plateau_correction)

            for idx in range(1, fade_frames + 1):
                pre_frame = full_start - idx
                if pre_frame >= 0:
                    weight = float(fade_frames - idx + 1) / float(fade_frames + 1)
                    corrections[role][pre_frame] = max(corrections[role][pre_frame], eased_ramp(plateau_correction, weight))
                post_frame = full_end + idx
                if post_frame < frame_count:
                    weight = float(fade_frames - idx + 1) / float(fade_frames + 1)
                    corrections[role][post_frame] = max(corrections[role][post_frame], eased_ramp(plateau_correction, weight))

        event["depth_order_rule"] = {
            "occluded_role": role,
            "visible_role": other,
            "required": "occluded_z >= visible_z + min_depth_separation",
            "min_depth_separation": float(min_depth_separation),
            "full_start_frame": int(full_start),
            "full_end_frame": int(full_end),
            "fade_frames": int(fade_frames),
            "context_frames": int(context_frames),
            "max_depth_correction": plateau_correction,
        }

    for role, correction in corrections.items():
        if not np.any(correction > 0.0):
            continue
        arrays[f"roots__{role}"][:, 2] += correction
        arrays[f"verts__{role}"][:, :, 2] += correction[:, None]


def limit_final_depth_steps(arrays: dict[str, np.ndarray], max_step: float) -> dict[str, float]:
    max_step = float(max_step)
    if max_step <= 0.0:
        return {}
    stats: dict[str, float] = {}
    for role in [str(item) for item in arrays["subject_ids"].tolist()]:
        roots_key = f"roots__{role}"
        verts_key = f"verts__{role}"
        if roots_key not in arrays or verts_key not in arrays:
            continue
        roots = arrays[roots_key]
        vertices = arrays[verts_key]
        before = float(np.max(np.abs(np.diff(roots[:, 2])))) if len(roots) > 1 else 0.0
        for frame_idx in range(1, len(roots)):
            previous = float(roots[frame_idx - 1, 2])
            current = float(roots[frame_idx, 2])
            delta = current - previous
            if abs(delta) <= max_step:
                continue
            target = previous + math.copysign(max_step, delta)
            offset = float(target - current)
            roots[frame_idx:, 2] += offset
            vertices[frame_idx:, :, 2] += offset
        after = float(np.max(np.abs(np.diff(roots[:, 2])))) if len(roots) > 1 else 0.0
        stats[role] = after
        if before > max_step:
            print(f"Depth step limited for {role}: {before:.3f} -> {after:.3f}")
    return stats


def enforce_feet_on_ground(arrays: dict[str, np.ndarray], ground_y: float) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    subject_ids = [str(item) for item in arrays["subject_ids"].tolist()]
    for role in subject_ids:
        roots_key = f"roots__{role}"
        verts_key = f"verts__{role}"
        if roots_key not in arrays or verts_key not in arrays:
            continue
        roots = arrays[roots_key]
        vertices = arrays[verts_key]
        if len(roots) == 0 or len(vertices) == 0:
            continue
        frame_count = min(len(roots), len(vertices))
        min_y = np.nanmin(vertices[:frame_count, :, 1], axis=1).astype(np.float32)
        offsets = (float(ground_y) - min_y).astype(np.float32)
        vertices[:frame_count, :, 1] += offsets[:, None]
        roots[:frame_count, 1] += offsets
        stats[role] = {
            "max_abs_offset": float(np.max(np.abs(offsets))) if len(offsets) else 0.0,
            "mean_abs_offset": float(np.mean(np.abs(offsets))) if len(offsets) else 0.0,
            "max_abs_ground_error_after": float(
                np.max(np.abs(np.nanmin(vertices[:frame_count, :, 1], axis=1) - float(ground_y)))
            )
            if frame_count
            else 0.0,
        }
    return stats


def enforce_occlusion_mesh_clearance(
    arrays: dict[str, np.ndarray],
    events: list[dict[str, Any]],
    mesh_clearance_margin: float,
    front_share: float,
    fade_frames: int,
) -> dict[str, Any]:
    subject_ids = [str(item) for item in arrays["subject_ids"].tolist()]
    if len(subject_ids) != 2:
        return {}
    margin = max(0.0, float(mesh_clearance_margin))
    fade_frames = max(0, int(fade_frames))
    front_share = float(np.clip(front_share, 0.0, 0.8))
    stats: dict[str, Any] = {
        "target_mesh_gap": margin,
        "fade_frames": fade_frames,
        "corrected_event_count": 0,
        "max_added_root_separation": 0.0,
        "events": [],
    }

    def fade_weight(frame_idx: int, start: int, end: int, window_start: int, window_end: int) -> float:
        if start <= frame_idx <= end:
            return 1.0
        if fade_frames <= 0 and (window_start >= start or window_end <= end):
            return 0.0
        if frame_idx < start:
            denom = max(1, start - window_start)
            t = float(frame_idx - window_start) / float(denom)
        else:
            denom = max(1, window_end - end)
            t = float(window_end - frame_idx) / float(denom)
        t = float(np.clip(t, 0.0, 1.0))
        return 0.5 - 0.5 * math.cos(math.pi * t)

    for event in events:
        back_role = str(event["role"])
        front_roles = [role for role in subject_ids if role != back_role]
        if not front_roles:
            continue
        front_role = front_roles[0]
        required_keys = [f"roots__{back_role}", f"verts__{back_role}", f"roots__{front_role}", f"verts__{front_role}"]
        if any(key not in arrays for key in required_keys):
            continue
        back_roots = arrays[f"roots__{back_role}"]
        front_roots = arrays[f"roots__{front_role}"]
        back_vertices = arrays[f"verts__{back_role}"]
        front_vertices = arrays[f"verts__{front_role}"]
        frame_count = min(len(back_roots), len(front_roots), len(back_vertices), len(front_vertices))
        if frame_count <= 0:
            continue
        start = max(0, min(int(event["occlusion_start_frame"]), frame_count - 1))
        end = max(0, min(int(event["occlusion_end_frame"]), frame_count - 1))
        if end < start:
            continue
        gaps = np.nanmin(back_vertices[start : end + 1, :, 2], axis=1) - np.nanmax(front_vertices[start : end + 1, :, 2], axis=1)
        min_gap = float(np.min(gaps)) if len(gaps) else margin
        added = max(0.0, margin - min_gap)
        if added <= 1e-6:
            continue
        depth_rule = event.get("depth_order_rule", {})
        window_start = max(0, min(start, int(depth_rule.get("full_start_frame", start - fade_frames))))
        window_end = min(frame_count - 1, max(end, int(depth_rule.get("full_end_frame", end + fade_frames))))
        back_delta = added * (1.0 - front_share)
        front_delta = -added * front_share
        for frame_idx in range(window_start, window_end + 1):
            weight = fade_weight(frame_idx, start, end, window_start, window_end)
            if weight <= 0.0:
                continue
            back_offset = np.float32(back_delta * weight)
            front_offset = np.float32(front_delta * weight)
            back_roots[frame_idx, 2] += back_offset
            back_vertices[frame_idx, :, 2] += back_offset
            front_roots[frame_idx, 2] += front_offset
            front_vertices[frame_idx, :, 2] += front_offset
        corrected_gap = float(
            np.min(np.nanmin(back_vertices[start : end + 1, :, 2], axis=1) - np.nanmax(front_vertices[start : end + 1, :, 2], axis=1))
        )
        event["mesh_clearance_rule"] = {
            "back_role": back_role,
            "front_role": front_role,
            "target_mesh_gap": margin,
            "min_gap_before": min_gap,
            "min_gap_after": corrected_gap,
            "added_root_separation": added,
            "fade_frames": fade_frames,
            "window_start_frame": int(window_start),
            "window_end_frame": int(window_end),
        }
        stats["corrected_event_count"] += 1
        stats["max_added_root_separation"] = max(float(stats["max_added_root_separation"]), float(added))
        stats["events"].append(event["mesh_clearance_rule"])
    return stats


def z_intervals_overlap(
    arrays: dict[str, np.ndarray],
    role_a: str,
    role_b: str,
    frame_idx: int,
    margin: float,
) -> bool:
    verts_a = arrays[f"verts__{role_a}"][frame_idx, :, 2]
    verts_b = arrays[f"verts__{role_b}"][frame_idx, :, 2]
    a_min = float(np.nanmin(verts_a))
    a_max = float(np.nanmax(verts_a))
    b_min = float(np.nanmin(verts_b))
    b_max = float(np.nanmax(verts_b))
    return bool(min(a_max, b_max) - max(a_min, b_min) > -float(margin))


def freeze_z_motion_while_overlapped(
    arrays: dict[str, np.ndarray],
    events: list[dict[str, Any]],
    overlap_margin: float,
    max_z_delta_per_frame: float,
    mesh_clearance_margin: float,
) -> dict[str, Any]:
    subject_ids = [str(item) for item in arrays["subject_ids"].tolist()]
    if len(subject_ids) != 2:
        return {}
    role_a, role_b = subject_ids[:2]
    required = [f"roots__{role_a}", f"verts__{role_a}", f"roots__{role_b}", f"verts__{role_b}"]
    if any(key not in arrays for key in required):
        return {}
    frame_count = min(len(arrays[f"roots__{role_a}"]), len(arrays[f"roots__{role_b}"]))
    max_step = max(0.0, float(max_z_delta_per_frame))
    stats: dict[str, Any] = {
        "enabled": True,
        "overlap_margin": float(overlap_margin),
        "max_z_delta_per_frame": float(max_step),
        "frozen_frame_count": 0,
        "direction_locked_frame_count": 0,
        "skipped_occlusion_frame_count": 0,
        "capped_step_count": 0,
        "clearance_protected_frame_count": 0,
        "backfilled_step_count": 0,
        "semantic_transition_frames": 90,
        "max_abs_step_after": {role_a: 0.0, role_b: 0.0},
    }
    occlusion_frames: dict[int, list[dict[str, Any]]] = {}
    semantic_order_frames: dict[int, tuple[str, str]] = {}
    for event in events:
        start = int(event.get("occlusion_start_frame", -1))
        end = int(event.get("occlusion_end_frame", -1))
        if end < start:
            continue
        for frame_idx in range(max(0, start), min(frame_count - 1, end) + 1):
            occlusion_frames.setdefault(frame_idx, []).append(event)
        back_role = str(event.get("role", ""))
        front_candidates = [role for role in (role_a, role_b) if role != back_role]
        if not back_role or not front_candidates:
            continue
        transition_frames = int(stats["semantic_transition_frames"])
        rule_start = max(0, start - transition_frames)
        rule_end = min(frame_count - 1, end + transition_frames)
        if rule_end < rule_start:
            continue
        front_role = front_candidates[0]
        for frame_idx in range(rule_start, rule_end + 1):
            semantic_order_frames[frame_idx] = (back_role, front_role)

    def event_clearance_ok(event: dict[str, Any], frame_idx: int) -> bool:
        back_role = str(event.get("role", ""))
        front_candidates = [role for role in (role_a, role_b) if role != back_role]
        if not back_role or not front_candidates:
            return True
        front_role = front_candidates[0]
        back_roots = arrays[f"roots__{back_role}"]
        front_roots = arrays[f"roots__{front_role}"]
        back_vertices = arrays[f"verts__{back_role}"]
        front_vertices = arrays[f"verts__{front_role}"]
        if float(back_roots[frame_idx, 2] - front_roots[frame_idx, 2]) < -1e-5:
            return False
        gap = float(np.nanmin(back_vertices[frame_idx, :, 2]) - np.nanmax(front_vertices[frame_idx, :, 2]))
        return bool(gap >= float(mesh_clearance_margin) - 1e-4)

    def cap_frame_step(frame_idx: int, protect_events: list[dict[str, Any]] | None = None) -> None:
        if max_step <= 0.0:
            return
        offsets: dict[str, np.float32] = {}
        for role in (role_a, role_b):
            roots = arrays[f"roots__{role}"]
            vertices = arrays[f"verts__{role}"]
            delta = float(roots[frame_idx, 2] - roots[frame_idx - 1, 2])
            if abs(delta) <= max_step:
                continue
            target_z = float(roots[frame_idx - 1, 2]) + math.copysign(max_step, delta)
            offset = np.float32(target_z - float(roots[frame_idx, 2]))
            roots[frame_idx, 2] += offset
            vertices[frame_idx, :, 2] += offset
            offsets[role] = offset
        if protect_events and offsets:
            if not all(event_clearance_ok(event, frame_idx) for event in protect_events):
                for role, offset in offsets.items():
                    roots = arrays[f"roots__{role}"]
                    vertices = arrays[f"verts__{role}"]
                    roots[frame_idx, 2] -= offset
                    vertices[frame_idx, :, 2] -= offset
                stats["clearance_protected_frame_count"] += 1
                return
        stats["capped_step_count"] += len(offsets)

    for frame_idx in range(1, frame_count):
        if frame_idx in occlusion_frames:
            stats["skipped_occlusion_frame_count"] += 1
            cap_frame_step(frame_idx, occlusion_frames[frame_idx])
            continue
        candidate_overlap = z_intervals_overlap(arrays, role_a, role_b, frame_idx, overlap_margin)
        if candidate_overlap:
            semantic_order = semantic_order_frames.get(frame_idx)
            if semantic_order is None:
                changed = False
                for role in (role_a, role_b):
                    roots = arrays[f"roots__{role}"]
                    vertices = arrays[f"verts__{role}"]
                    target_z = float(roots[frame_idx - 1, 2])
                    offset = np.float32(target_z - float(roots[frame_idx, 2]))
                    if abs(float(offset)) <= 1e-7:
                        continue
                    roots[frame_idx, 2] += offset
                    vertices[frame_idx, :, 2] += offset
                    changed = True
                if changed:
                    stats["frozen_frame_count"] += 1
                continue
            else:
                ordered_roles = ((semantic_order[0], 1), (semantic_order[1], -1))
            changed = False
            for role, allowed_sign in ordered_roles:
                if allowed_sign == 0:
                    target_z = float(arrays[f"roots__{role}"][frame_idx - 1, 2])
                else:
                    roots = arrays[f"roots__{role}"]
                    previous = float(roots[frame_idx - 1, 2])
                    current = float(roots[frame_idx, 2])
                    delta = current - previous
                    if delta == 0.0 or math.copysign(1.0, delta) == float(allowed_sign):
                        continue
                    target_z = previous
                roots = arrays[f"roots__{role}"]
                vertices = arrays[f"verts__{role}"]
                offset = np.float32(target_z - float(roots[frame_idx, 2]))
                if abs(float(offset)) <= 1e-7:
                    continue
                roots[frame_idx, 2] += offset
                vertices[frame_idx, :, 2] += offset
                changed = True
            if changed:
                stats["direction_locked_frame_count"] += 1
            cap_frame_step(frame_idx)
            continue
        cap_frame_step(frame_idx)

    if max_step > 0.0:
        for frame_idx in range(frame_count - 1, 0, -1):
            for role in (role_a, role_b):
                roots = arrays[f"roots__{role}"]
                vertices = arrays[f"verts__{role}"]
                delta = float(roots[frame_idx, 2] - roots[frame_idx - 1, 2])
                if abs(delta) <= max_step:
                    continue
                target_previous_z = float(roots[frame_idx, 2]) - math.copysign(max_step, delta)
                offset = np.float32(target_previous_z - float(roots[frame_idx - 1, 2]))
                if abs(float(offset)) <= 1e-7:
                    continue
                roots[frame_idx - 1, 2] += offset
                vertices[frame_idx - 1, :, 2] += offset
                stats["backfilled_step_count"] += 1
    for role in (role_a, role_b):
        roots = arrays[f"roots__{role}"]
        stats["max_abs_step_after"][role] = float(np.max(np.abs(np.diff(roots[:frame_count, 2])))) if frame_count > 1 else 0.0
    return stats


def wham_index(raw_results: dict[str, Any]) -> dict[str, dict[int, int]]:
    output = {}
    for subject_id, payload in raw_results.items():
        frame_ids = np.asarray(payload.get("frame_ids", []), dtype=np.int64).reshape(-1)
        output[str(subject_id)] = {int(frame_id): idx for idx, frame_id in enumerate(frame_ids.tolist())}
    return output


def as_jsonable(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return []
    return arr.astype(float).tolist()


def frame_info(
    role: str,
    frame_idx: int,
    arrays: dict[str, np.ndarray],
    maps: dict[str, dict[str, Any]],
    raw_results: dict[str, Any],
    raw_frame_index: dict[str, dict[int, int]],
) -> dict[str, Any]:
    frame = maps[role]["frames"].get(frame_idx, {})
    assigned_key = f"assigned_subjects__{role}"
    assigned_subject = ""
    if assigned_key in arrays and 0 <= frame_idx < len(arrays[assigned_key]):
        assigned_subject = str(arrays[assigned_key][frame_idx])

    pose = None
    trans = None
    trans_world = None
    if assigned_subject and assigned_subject in raw_results:
        local_idx = raw_frame_index.get(assigned_subject, {}).get(frame_idx)
        if local_idx is not None:
            payload = raw_results[assigned_subject]
            if "pose" in payload:
                pose = np.asarray(payload["pose"], dtype=np.float32)[local_idx]
            if "trans" in payload:
                trans = np.asarray(payload["trans"], dtype=np.float32)[local_idx]
            if "trans_world" in payload:
                trans_world = np.asarray(payload["trans_world"], dtype=np.float32)[local_idx]

    roots = arrays[f"roots__{role}"]
    return {
        "frame_id": int(frame_idx),
        "person_id": ROLE_LABELS.get(role, role),
        "role": role,
        "track_id": int(maps[role]["stable_track_id"]),
        "visible": bool(frame.get("visible", False)),
        "is_recovered": bool(frame.get("is_recovered", False)),
        "bbox": frame.get("bbox_xyxy"),
        "keypoints_coco17": frame.get("keypoints_coco17"),
        "keypoints_halpe26": frame.get("keypoints_halpe26"),
        "assigned_subject_id": assigned_subject,
        "pose": as_jsonable(pose),
        "source_trans": as_jsonable(trans),
        "source_trans_world": as_jsonable(trans_world),
        "global_position": as_jsonable(roots[frame_idx]),
        "z_depth": float(roots[frame_idx, 2]),
    }


def interpolate_array(start: np.ndarray | None, end: np.ndarray | None, t: float, method: str, confidence_threshold: float = 0.05) -> list | None:
    if start is None or end is None:
        return None
    a = np.asarray(start, dtype=np.float32)
    b = np.asarray(end, dtype=np.float32)
    if a.shape != b.shape:
        return None
    aa = a.copy()
    bb = b.copy()
    if aa.ndim == 2 and aa.shape[1] >= 3:
        weak_a = (~np.isfinite(aa[:, 0])) | (~np.isfinite(aa[:, 1])) | (aa[:, 2] < confidence_threshold)
        weak_b = (~np.isfinite(bb[:, 0])) | (~np.isfinite(bb[:, 1])) | (bb[:, 2] < confidence_threshold)
        aa[weak_a, :2] = bb[weak_a, :2]
        bb[weak_b, :2] = aa[weak_b, :2]
        aa[:, 2] = np.nan_to_num(aa[:, 2], nan=0.0)
        bb[:, 2] = np.nan_to_num(bb[:, 2], nan=0.0)
    w = interpolation_weight(t, method)
    return np.nan_to_num(aa * (1.0 - w) + bb * w, nan=0.0).astype(float).tolist()


def add_inbetween_records(events: list[dict[str, Any]], arrays: dict[str, np.ndarray], maps: dict[str, dict[str, Any]], method: str) -> None:
    for event in events:
        role = event["role"]
        anchor_frame = int(event["start_anchor_frame"])
        start_frame = int(event["occlusion_start_frame"])
        end_frame = int(event["occlusion_end_frame"])
        anchor = maps[role]["frames"].get(anchor_frame, {})
        end = maps[role]["frames"].get(end_frame, {})
        records = []
        for frame_idx in range(start_frame, end_frame):
            t = (frame_idx - anchor_frame) / float(end_frame - anchor_frame)
            record = {
                "frame_id": int(frame_idx),
                "weight": float(interpolation_weight(t, method)),
                "bbox": interpolate_array(anchor.get("bbox_xyxy"), end.get("bbox_xyxy"), t, method),
                "keypoints_coco17": interpolate_array(anchor.get("keypoints_coco17"), end.get("keypoints_coco17"), t, method),
                "keypoints_halpe26": interpolate_array(anchor.get("keypoints_halpe26"), end.get("keypoints_halpe26"), t, method),
                "global_position": as_jsonable(arrays[f"roots__{role}"][frame_idx]),
                "z_depth": float(arrays[f"roots__{role}"][frame_idx, 2]),
            }
            records.append(record)
        event["inbetween_frames"] = records


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame {frame_idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_snapshot(frame_rgb: np.ndarray, event: dict[str, Any], maps: dict[str, dict[str, Any]], snapshot: str, output_path: Path) -> None:
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    width, _height = image.size
    font_big = load_font(max(30, width // 70))
    font = load_font(max(22, width // 100))
    role = event["role"]
    frame_idx = int(event["occlusion_start_frame"] if snapshot == "start" else event["occlusion_end_frame"])
    bbox_frame = int(event["start_anchor_frame"] if snapshot == "start" else event["occlusion_end_frame"])
    bbox = maps[role]["frames"].get(bbox_frame, {}).get("bbox_xyxy")
    color = (255, 190, 60, 255)
    if bbox:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        line_width = max(5, width // 500)
        for offset in range(line_width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        draw.rectangle((x1, max(0, y1 - 48), min(width, x1 + 620), y1), fill=(0, 0, 0, 180))
        draw.text((x1 + 12, max(0, y1 - 42)), f"{event['person_label']} / Track ID {event['stable_track_id']}", font=font, fill=color)

    title = "In-between Start" if snapshot == "start" else "In-between End"
    lines = [
        title,
        f"Person: {event['person_label']} ({role}, Track ID {event['stable_track_id']})",
        f"Start frame: {event['occlusion_start_frame']}",
        f"End frame: {event['occlusion_end_frame']}",
        f"Missing frames: {event['missing_frame_count']}",
        f"Duration: {event['duration_seconds']:.3f} s",
        f"Current frame: {frame_idx}",
    ]
    pad = 24
    line_h = int(font.size * 1.35) if hasattr(font, "size") else 32
    box_w = min(width - 2 * pad, max(960, width // 3))
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle((pad, pad, pad + box_w, pad + box_h), fill=(0, 0, 0, 190), outline=color, width=4)
    draw.text((pad + 18, pad + 14), lines[0], font=font_big, fill=color)
    y = pad + 14 + line_h + 12
    for line in lines[1:]:
        draw.text((pad + 18, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def attach_event_details(
    events: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    observations: dict[str, Any],
    raw_results: dict[str, Any],
    video_path: Path,
    image_dir: Path,
) -> None:
    maps = role_frame_maps(observations)
    raw_frame_index = wham_index(raw_results)
    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    for event in events:
        event["start_frame_info"] = frame_info(event["role"], int(event["occlusion_start_frame"]), arrays, maps, raw_results, raw_frame_index)
        event["start_anchor_info"] = frame_info(event["role"], int(event["start_anchor_frame"]), arrays, maps, raw_results, raw_frame_index)
        event["end_frame_info"] = frame_info(event["role"], int(event["occlusion_end_frame"]), arrays, maps, raw_results, raw_frame_index)
        stem = f"event_{event['event_id']:03d}_{event['role']}_f{event['occlusion_start_frame']:04d}_to_f{event['occlusion_end_frame']:04d}"
        start_path = image_dir / f"{stem}_start.png"
        end_path = image_dir / f"{stem}_end.png"
        draw_snapshot(read_frame(cap, int(event["occlusion_start_frame"])), event, maps, "start", start_path)
        draw_snapshot(read_frame(cap, int(event["occlusion_end_frame"])), event, maps, "end", end_path)
        event["start_image"] = str(start_path)
        event["end_image"] = str(end_path)
    cap.release()


def update_bounds_and_metadata(arrays: dict[str, np.ndarray], metadata_update: dict[str, Any]) -> None:
    bounds_min = np.full(3, np.inf, dtype=np.float32)
    bounds_max = np.full(3, -np.inf, dtype=np.float32)
    for role in [str(item) for item in arrays["subject_ids"].tolist()]:
        vertices = arrays[f"verts__{role}"]
        bounds_min = np.minimum(bounds_min, np.nanmin(vertices.reshape(-1, 3), axis=0))
        bounds_max = np.maximum(bounds_max, np.nanmax(vertices.reshape(-1, 3), axis=0))
    arrays["bounds_min"] = bounds_min
    arrays["bounds_max"] = bounds_max

    metadata = {}
    if "metadata_json" in arrays:
        metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update(metadata_update)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    observations = json.loads(Path(args.observations).read_text(encoding="utf-8"))
    raw_results = {str(key): value for key, value in joblib.load(args.wham_output).items()}
    arrays = copy_cache_arrays(Path(args.cache))
    events = detect_missing_events(observations, args.min_gap_frames, args.occlusion_prediction_min_missing_frames)

    if args.swap_depth:
        apply_depth_swap(arrays)
    interpolate_missing_motion(
        arrays,
        observations,
        events,
        args.method,
        args.prediction_velocity_window,
        args.partial_observation_weight,
    )
    depth_stabilizer = DepthStabilizer(
        arrays,
        observations,
        events,
        args.min_depth_separation,
        args.max_z_delta_per_frame,
        args.z_range_limit,
        args.z_smoothing_alpha,
        args.occlusion_depth_context_frames,
        args.occlusion_front_share,
        args.penetration_root_distance,
        args.close_distance_ratio,
        args.hand_penetration_min_depth,
        args.hand_keypoint_confidence,
        args.mesh_clearance_margin,
        args.lock_depth_during_z_overlap,
        args.z_overlap_margin,
    )
    depth_stabilizer_stats = depth_stabilizer.stabilize()
    mesh_clearance_stats = enforce_occlusion_mesh_clearance(
        arrays,
        events,
        args.mesh_clearance_margin,
        args.occlusion_front_share,
        args.mesh_clearance_fade_frames,
    )
    z_overlap_freeze_stats = freeze_z_motion_while_overlapped(
        arrays,
        events,
        max(args.z_overlap_margin, args.mesh_clearance_margin),
        args.max_z_delta_per_frame,
        args.mesh_clearance_margin,
    )
    mirror_center_x = None
    if args.mirror_x:
        mirror_center_x = apply_mirror_x(arrays, args.mirror_center_x)
    grounding_stats = enforce_feet_on_ground(arrays, args.ground_y)

    maps = role_frame_maps(observations)
    add_inbetween_records(events, arrays, maps, args.method)
    image_dir = Path(args.image_dir).resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    for old_image in image_dir.glob("*.png"):
        old_image.unlink()
    attach_event_details(events, arrays, observations, raw_results, Path(args.video), image_dir)

    update_bounds_and_metadata(
        arrays,
        {
            "depth_swapped": bool(args.swap_depth),
            "fps": float(observations.get("metadata", {}).get("fps") or 30.0),
            "frame_count": int(observations.get("metadata", {}).get("frame_count", 0)),
            "depth_swap_axis": "source_z_depth",
            "occlusion_handling": "inbetween_interpolation",
            "occlusion_freeze_overlays": False,
            "inbetween_method": args.method,
            "motion_prediction": "velocity_hermite_with_partial_2d_progress",
            "prediction_velocity_window": int(args.prediction_velocity_window),
            "partial_observation_weight": float(args.partial_observation_weight),
            "inbetween_event_count": len(events),
            "occlusion_prediction_min_missing_frames": int(args.occlusion_prediction_min_missing_frames),
            "occlusion_depth_order": "occluded_z >= visible_z + min_depth_separation",
            "occlusion_depth_fade_frames": int(args.occlusion_depth_fade_frames),
            "occlusion_depth_context_frames": int(args.occlusion_depth_context_frames),
            "depth_stabilizer": depth_stabilizer_stats,
            "max_z_delta_per_frame": float(args.max_z_delta_per_frame),
            "z_range_limit": float(args.z_range_limit),
            "z_smoothing_alpha": float(args.z_smoothing_alpha),
            "occlusion_front_share": float(args.occlusion_front_share),
            "hand_penetration_min_depth": float(args.hand_penetration_min_depth),
            "mesh_clearance_margin": float(args.mesh_clearance_margin),
            "mesh_clearance_fade_frames": int(args.mesh_clearance_fade_frames),
            "mesh_clearance_postprocess": mesh_clearance_stats,
            "z_overlap_freeze_postprocess": z_overlap_freeze_stats,
            "lock_depth_during_z_overlap": bool(args.lock_depth_during_z_overlap),
            "z_overlap_margin": float(args.z_overlap_margin),
            "feet_grounded": True,
            "ground_y": float(args.ground_y),
            "grounding_stats": grounding_stats,
            "mirrored_x": bool(args.mirror_x),
            "mirror_center_x": mirror_center_x,
        },
    )

    output_cache = Path(args.output_cache).resolve()
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_cache, **arrays)

    payload = {
        "metadata": {
            "fps": float(observations.get("metadata", {}).get("fps") or 30.0),
            "frame_count": int(observations.get("metadata", {}).get("frame_count", 0)),
            "definition": "visible=False spans are in-betweened from the previous visible frame to the next visible frame; no freeze overlays are created",
            "method": args.method,
            "motion_prediction": "velocity_hermite_with_partial_2d_progress",
            "prediction_velocity_window": int(args.prediction_velocity_window),
            "partial_observation_weight": float(args.partial_observation_weight),
            "min_gap_frames": int(args.min_gap_frames),
            "occlusion_prediction_min_missing_frames": int(args.occlusion_prediction_min_missing_frames),
            "depth_swapped": bool(args.swap_depth),
            "occlusion_depth_order": "occluded_z >= visible_z + min_depth_separation",
            "occlusion_depth_fade_frames": int(args.occlusion_depth_fade_frames),
            "occlusion_depth_context_frames": int(args.occlusion_depth_context_frames),
            "depth_stabilizer": depth_stabilizer_stats,
            "max_z_delta_per_frame": float(args.max_z_delta_per_frame),
            "z_range_limit": float(args.z_range_limit),
            "z_smoothing_alpha": float(args.z_smoothing_alpha),
            "occlusion_front_share": float(args.occlusion_front_share),
            "hand_penetration_min_depth": float(args.hand_penetration_min_depth),
            "mesh_clearance_margin": float(args.mesh_clearance_margin),
            "mesh_clearance_fade_frames": int(args.mesh_clearance_fade_frames),
            "mesh_clearance_postprocess": mesh_clearance_stats,
            "z_overlap_freeze_postprocess": z_overlap_freeze_stats,
            "lock_depth_during_z_overlap": bool(args.lock_depth_during_z_overlap),
            "z_overlap_margin": float(args.z_overlap_margin),
            "feet_grounded": True,
            "ground_y": float(args.ground_y),
            "grounding_stats": grounding_stats,
            "mirrored_x": bool(args.mirror_x),
            "mirror_center_x": mirror_center_x,
            "source_cache": str(Path(args.cache).resolve()),
            "output_cache": str(output_cache),
        },
        "events": events,
    }
    events_json = Path(args.events_json).resolve()
    events_json.parent.mkdir(parents=True, exist_ok=True)
    events_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote in-between cache: {output_cache}")
    print(f"Wrote {len(events)} in-between events: {events_json}")
    print(f"Event snapshots: {image_dir}")


if __name__ == "__main__":
    main()
