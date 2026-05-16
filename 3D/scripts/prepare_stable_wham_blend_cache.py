"""Build a stable two-role WHAM mesh cache for Blender.

WHAM demo outputs can split or swap temporary subject ids during close contact.
This script stitches those WHAM subject fragments back onto the project's stable
2D roles by combining bbox agreement with 3D motion/depth continuity.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROLE_ORDER = ["character_A", "character_B"]


def patch_numpy_for_legacy_chumpy() -> None:
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stable two-role WHAM cache for Blender.")
    parser.add_argument("--wham-output", required=True)
    parser.add_argument("--tracking-results", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--smpl-model-dir", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--smooth-depth-window", type=int, default=31)
    parser.add_argument("--max-depth-step", type=float, default=0.85)
    parser.add_argument("--min-depth-separation", type=float, default=0.24)
    parser.add_argument("--penetration-root-distance", type=float, default=0.55)
    parser.add_argument("--close-distance-ratio", type=float, default=0.75)
    return parser.parse_args()


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


def wham_bbox_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    cx, cy, scale = [float(value) for value in bbox[:3]]
    half = 100.0 * scale
    return np.asarray([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def is_stable_detection(frame: dict[str, Any] | None) -> bool:
    return bool(frame and frame.get("visible", False) and not frame.get("is_recovered", False))


def frame_bbox(frame: dict[str, Any] | None) -> np.ndarray | None:
    if frame is None or not frame.get("bbox_xyxy"):
        return None
    return np.asarray(frame["bbox_xyxy"], dtype=np.float32)


def roles_close_in_2d(
    role_frames: dict[str, dict[int, dict[str, Any]]],
    role_a: str,
    role_b: str,
    frame_idx: int,
    close_distance_ratio: float,
) -> bool:
    box_a = frame_bbox(role_frames[role_a].get(frame_idx))
    box_b = frame_bbox(role_frames[role_b].get(frame_idx))
    return center_ratio(box_a, box_b) <= close_distance_ratio or bbox_iou(box_a, box_b) >= 0.08


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if window < 3 or len(values) < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values.copy()
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(values, window_length=window, polyorder=2 if window >= 5 else 1, mode="interp").astype(np.float32)
    except Exception:
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def build_subject_data(raw_results: dict[str, Any], tracking_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    subjects = {}
    for subject_id, payload in raw_results.items():
        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64).reshape(-1)
        vertices = np.asarray(payload["verts"], dtype=np.float32)
        roots = np.asarray(payload.get("trans", vertices.mean(axis=1)), dtype=np.float32)
        frame_count = min(len(frame_ids), len(vertices), len(roots))
        frame_ids = frame_ids[:frame_count]
        vertices = vertices[:frame_count]
        roots = roots[:frame_count]
        frame_to_local = {int(frame_id): idx for idx, frame_id in enumerate(frame_ids.tolist())}

        tracking = tracking_results.get(subject_id)
        bboxes = {}
        if tracking is not None:
            track_frames = np.asarray(tracking["frame_id"], dtype=np.int64).reshape(-1)
            track_boxes = np.asarray(tracking["bbox"], dtype=np.float32)
            for frame_id, bbox in zip(track_frames.tolist(), track_boxes):
                bboxes[int(frame_id)] = wham_bbox_to_xyxy(bbox)

        subjects[subject_id] = {
            "frame_ids": frame_ids,
            "vertices": vertices,
            "roots": roots,
            "frame_to_local": frame_to_local,
            "bboxes": bboxes,
        }
    return subjects


def active_candidates(subjects: dict[str, dict[str, Any]], frame_idx: int) -> list[dict[str, Any]]:
    output = []
    for subject_id, subject in subjects.items():
        local_idx = subject["frame_to_local"].get(frame_idx)
        if local_idx is None:
            continue
        output.append(
            {
                "subject_id": subject_id,
                "local_idx": local_idx,
                "bbox": subject["bboxes"].get(frame_idx),
                "root": subject["roots"][local_idx],
                "vertices": subject["vertices"][local_idx],
            }
        )
    return output


def role_maps(observations: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    maps = {}
    for track in observations.get("tracks", []):
        role = str(track.get("role"))
        maps[role] = {int(frame["frame_index"]): frame for frame in track.get("frames", [])}
    return maps


def candidate_score(
    role: str,
    candidate: dict[str, Any] | None,
    role_frame: dict[str, Any] | None,
    previous_root: np.ndarray | None,
    previous_subject: str | None,
    max_depth_step: float,
) -> float:
    stable = is_stable_detection(role_frame)
    role_box = frame_bbox(role_frame)
    if candidate is None:
        return -8.0 if stable else -1.0

    score = 0.0
    if role_box is not None and candidate["bbox"] is not None:
        iou = bbox_iou(candidate["bbox"], role_box)
        dist = center_ratio(candidate["bbox"], role_box)
        score += 8.0 * max(iou, 0.0) - 2.2 * dist
        if stable and iou < 0.03 and dist > 0.9:
            score -= 4.0
    elif stable:
        score -= 2.0

    if previous_root is not None:
        delta = np.asarray(candidate["root"], dtype=np.float32) - previous_root
        horizontal = float(np.linalg.norm(delta[[0, 2]]))
        depth = float(abs(delta[2]))
        score -= 0.75 * horizontal + 1.25 * depth
        if depth > max_depth_step:
            score -= 5.0 * (depth - max_depth_step)

    if previous_subject is not None and candidate["subject_id"] == previous_subject:
        score += 1.2

    if not stable:
        # During unreliable 2D periods, continuity should dominate bbox noise.
        score += 0.8 if candidate["subject_id"] == previous_subject else -0.4
    return score


def choose_assignment(
    roles: list[str],
    candidates: list[dict[str, Any]],
    frame_idx: int,
    role_frames: dict[str, dict[int, dict[str, Any]]],
    previous_roots: dict[str, np.ndarray | None],
    previous_subjects: dict[str, str | None],
    max_depth_step: float,
    min_depth_separation: float,
    penetration_root_distance: float,
    close_distance_ratio: float,
) -> dict[str, dict[str, Any] | None]:
    options = candidates + [None]
    best_score = -1e18
    best_assignment = {role: None for role in roles}
    for choices in product(options, repeat=len(roles)):
        subject_ids = [choice["subject_id"] for choice in choices if choice is not None]
        if len(subject_ids) != len(set(subject_ids)):
            continue
        score = 0.0
        assignment = {}
        for role, choice in zip(roles, choices):
            assignment[role] = choice
            score += candidate_score(
                role,
                choice,
                role_frames[role].get(frame_idx),
                previous_roots[role],
                previous_subjects[role],
                max_depth_step,
            )
        if len(roles) == 2 and all(assignment[role] is not None for role in roles):
            role_a, role_b = roles
            root_a = np.asarray(assignment[role_a]["root"], dtype=np.float32)
            root_b = np.asarray(assignment[role_b]["root"], dtype=np.float32)
            prev_a = previous_roots[role_a]
            prev_b = previous_roots[role_b]
            root_plane_distance = float(np.linalg.norm(root_a[[0, 2]] - root_b[[0, 2]]))
            close_2d = roles_close_in_2d(role_frames, role_a, role_b, frame_idx, close_distance_ratio)
            if close_2d or root_plane_distance <= penetration_root_distance:
                depth_diff = float(root_a[2] - root_b[2])
                if abs(depth_diff) < min_depth_separation:
                    score -= 8.0 * (min_depth_separation - abs(depth_diff))
                if prev_a is not None and prev_b is not None:
                    prev_diff = float(prev_a[2] - prev_b[2])
                    if abs(prev_diff) > min_depth_separation * 0.5 and depth_diff * prev_diff < 0:
                        score -= 12.0 + 4.0 * min(abs(prev_diff), abs(depth_diff))
        if score > best_score:
            best_score = score
            best_assignment = assignment

    if len(roles) == 2 and all(best_assignment[role] is not None for role in roles):
        role_a, role_b = roles
        prev_a = previous_roots[role_a]
        prev_b = previous_roots[role_b]
        if prev_a is not None and prev_b is not None:
            root_a = np.asarray(best_assignment[role_a]["root"], dtype=np.float32)
            root_b = np.asarray(best_assignment[role_b]["root"], dtype=np.float32)
            direct = float(np.linalg.norm(root_a[[0, 2]] - prev_a[[0, 2]]) + np.linalg.norm(root_b[[0, 2]] - prev_b[[0, 2]]))
            swapped = float(np.linalg.norm(root_b[[0, 2]] - prev_a[[0, 2]]) + np.linalg.norm(root_a[[0, 2]] - prev_b[[0, 2]]))
            if direct > swapped + 0.35:
                best_assignment = {role_a: best_assignment[role_b], role_b: best_assignment[role_a]}

    return best_assignment


def first_available(subjects: dict[str, dict[str, Any]], role: str, role_frames: dict[str, dict[int, dict[str, Any]]], frame_count: int) -> tuple[np.ndarray, np.ndarray, str | None]:
    best = None
    best_score = -1e18
    for frame_idx in range(frame_count):
        candidates = active_candidates(subjects, frame_idx)
        role_frame = role_frames[role].get(frame_idx)
        for candidate in candidates:
            score = candidate_score(role, candidate, role_frame, None, None, 0.85)
            if score > best_score:
                best_score = score
                best = candidate
        if best is not None and is_stable_detection(role_frame):
            break
    if best is None:
        any_subject = next(iter(subjects.values()))
        return any_subject["vertices"][0], any_subject["roots"][0], None
    return best["vertices"], best["root"], best["subject_id"]


def build_stable_role_sequences(
    subjects: dict[str, dict[str, Any]],
    observations: dict[str, Any],
    max_depth_step: float,
    min_depth_separation: float,
    penetration_root_distance: float,
    close_distance_ratio: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str | None]]]:
    frame_count = int(observations.get("metadata", {}).get("frame_count") or observations.get("metadata", {}).get("processed_frames") or 0)
    roles = [role for role in ROLE_ORDER if role in role_maps(observations)]
    maps = role_maps(observations)
    role_vertices = {}
    role_roots = {}
    role_subjects = {}
    previous_roots: dict[str, np.ndarray | None] = {}
    previous_subjects: dict[str, str | None] = {}

    for role in roles:
        vertex, root, subject_id = first_available(subjects, role, maps, frame_count)
        role_vertices[role] = np.repeat(vertex[None, ...], frame_count, axis=0).astype(np.float32)
        role_roots[role] = np.repeat(root[None, ...], frame_count, axis=0).astype(np.float32)
        role_subjects[role] = [subject_id for _ in range(frame_count)]
        previous_roots[role] = np.asarray(root, dtype=np.float32)
        previous_subjects[role] = subject_id

    for frame_idx in range(frame_count):
        candidates = active_candidates(subjects, frame_idx)
        assignment = choose_assignment(
            roles,
            candidates,
            frame_idx,
            maps,
            previous_roots,
            previous_subjects,
            max_depth_step,
            min_depth_separation,
            penetration_root_distance,
            close_distance_ratio,
        )
        for role in roles:
            choice = assignment.get(role)
            if choice is None:
                if frame_idx > 0:
                    role_vertices[role][frame_idx] = role_vertices[role][frame_idx - 1]
                    role_roots[role][frame_idx] = role_roots[role][frame_idx - 1]
                    role_subjects[role][frame_idx] = role_subjects[role][frame_idx - 1]
                continue
            role_vertices[role][frame_idx] = choice["vertices"]
            role_roots[role][frame_idx] = choice["root"]
            role_subjects[role][frame_idx] = choice["subject_id"]
            previous_roots[role] = np.asarray(choice["root"], dtype=np.float32)
            previous_subjects[role] = choice["subject_id"]

    resolve_depth_swaps_and_interpenetration(
        roles,
        role_vertices,
        role_roots,
        maps,
        min_depth_separation,
        penetration_root_distance,
        close_distance_ratio,
    )
    return role_vertices, role_roots, role_subjects


def apply_depth_offsets(
    role_vertices: dict[str, np.ndarray],
    role_roots: dict[str, np.ndarray],
    role_a: str,
    role_b: str,
    frame_idx: int,
    target_diff: float,
) -> None:
    current_diff = float(role_roots[role_a][frame_idx, 2] - role_roots[role_b][frame_idx, 2])
    adjustment = float((target_diff - current_diff) * 0.5)
    role_roots[role_a][frame_idx, 2] += adjustment
    role_vertices[role_a][frame_idx, :, 2] += adjustment
    role_roots[role_b][frame_idx, 2] -= adjustment
    role_vertices[role_b][frame_idx, :, 2] -= adjustment


def resolve_depth_swaps_and_interpenetration(
    roles: list[str],
    role_vertices: dict[str, np.ndarray],
    role_roots: dict[str, np.ndarray],
    role_frames: dict[str, dict[int, dict[str, Any]]],
    min_depth_separation: float,
    penetration_root_distance: float,
    close_distance_ratio: float,
) -> None:
    if len(roles) != 2:
        return
    role_a, role_b = roles
    frame_count = len(role_roots[role_a])
    initial_diff = float(role_roots[role_a][0, 2] - role_roots[role_b][0, 2])
    depth_order = 1.0 if initial_diff >= 0 else -1.0

    for frame_idx in range(frame_count):
        root_a = role_roots[role_a][frame_idx]
        root_b = role_roots[role_b][frame_idx]
        diff = float(root_a[2] - root_b[2])
        root_plane_distance = float(np.linalg.norm(root_a[[0, 2]] - root_b[[0, 2]]))
        close_2d = roles_close_in_2d(role_frames, role_a, role_b, frame_idx, close_distance_ratio)
        close_or_intersecting = close_2d or root_plane_distance <= penetration_root_distance

        if close_or_intersecting:
            current_order = 1.0 if diff >= 0 else -1.0
            swapped_depth_order = abs(diff) > min_depth_separation * 0.5 and current_order != depth_order
            too_close_in_depth = abs(diff) < min_depth_separation
            if swapped_depth_order or too_close_in_depth:
                target_magnitude = max(abs(diff), min_depth_separation)
                apply_depth_offsets(role_vertices, role_roots, role_a, role_b, frame_idx, depth_order * target_magnitude)
                diff = float(role_roots[role_a][frame_idx, 2] - role_roots[role_b][frame_idx, 2])

        if abs(diff) >= min_depth_separation * 0.5:
            depth_order = 1.0 if diff >= 0 else -1.0


def smooth_depth_offset(vertices: np.ndarray, roots: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window < 3 or len(roots) < 3:
        return vertices, roots
    smoothed_depth = smooth_1d(roots[:, 2], window)
    offset = (smoothed_depth - roots[:, 2]).astype(np.float32)
    vertices = vertices.copy()
    roots = roots.copy()
    vertices[..., 2] += offset[:, None]
    roots[:, 2] = smoothed_depth
    return vertices, roots


def main() -> None:
    args = parse_args()
    patch_numpy_for_legacy_chumpy()
    import smplx

    observations = json.loads(Path(args.observations).read_text(encoding="utf-8"))
    raw_results = {str(key): value for key, value in joblib.load(args.wham_output).items()}
    tracking_results = {str(key): value for key, value in joblib.load(args.tracking_results).items()}
    subjects = build_subject_data(raw_results, tracking_results)
    role_vertices, role_roots, role_subjects = build_stable_role_sequences(
        subjects,
        observations,
        args.max_depth_step,
        args.min_depth_separation,
        args.penetration_root_distance,
        args.close_distance_ratio,
    )

    model = smplx.SMPL(
        str(Path(args.smpl_model_dir).resolve()),
        gender="neutral",
        batch_size=1,
        create_betas=False,
        create_body_pose=False,
        create_global_orient=False,
        create_transl=False,
    )

    arrays: dict[str, np.ndarray] = {
        "faces": np.asarray(model.faces, dtype=np.int32),
        "subject_ids": np.asarray(list(role_vertices.keys())),
    }

    smoothed_vertices = {}
    smoothed_roots = {}
    for role in role_vertices:
        vertices, roots = smooth_depth_offset(role_vertices[role], role_roots[role], args.smooth_depth_window)
        smoothed_vertices[role] = vertices
        smoothed_roots[role] = roots

    resolve_depth_swaps_and_interpenetration(
        list(smoothed_vertices.keys()),
        smoothed_vertices,
        smoothed_roots,
        role_maps(observations),
        args.min_depth_separation,
        args.penetration_root_distance,
        args.close_distance_ratio,
    )

    floor_y = max(float(np.nanmax(vertices[..., 1])) for vertices in smoothed_vertices.values())
    bounds_min = np.full(3, np.inf, dtype=np.float32)
    bounds_max = np.full(3, -np.inf, dtype=np.float32)
    frame_ids = np.arange(int(observations["metadata"]["frame_count"]), dtype=np.int64)

    for role in smoothed_vertices:
        vertices = smoothed_vertices[role]
        roots = smoothed_roots[role]
        vertices = vertices.copy()
        roots = roots.copy()
        vertices[..., 1] = floor_y - vertices[..., 1]
        roots[..., 1] = floor_y - roots[..., 1]
        arrays[f"frame_ids__{role}"] = frame_ids
        arrays[f"verts__{role}"] = vertices.astype(np.float32)
        arrays[f"roots__{role}"] = roots.astype(np.float32)
        arrays[f"assigned_subjects__{role}"] = np.asarray(["" if value is None else str(value) for value in role_subjects[role]])
        bounds_min = np.minimum(bounds_min, np.nanmin(vertices.reshape(-1, 3), axis=0))
        bounds_max = np.maximum(bounds_max, np.nanmax(vertices.reshape(-1, 3), axis=0))

    assignment_summary = {
        role: {subject: int(np.count_nonzero(arrays[f"assigned_subjects__{role}"] == subject)) for subject in sorted(set(arrays[f"assigned_subjects__{role}"].tolist()))}
        for role in role_vertices
    }
    metadata = {
        "source": "stable-role WHAM wham_output.pkl",
        "wham_output": str(Path(args.wham_output).resolve()),
        "observations": str(Path(args.observations).resolve()),
        "tracking_results": str(Path(args.tracking_results).resolve()),
        "vertex_source": "wham-verts-stable-roles",
        "source_y_axis": "down",
        "floor_y_reference": floor_y,
        "smooth_depth_window": int(args.smooth_depth_window),
        "max_depth_step": float(args.max_depth_step),
        "min_depth_separation": float(args.min_depth_separation),
        "penetration_root_distance": float(args.penetration_root_distance),
        "close_distance_ratio": float(args.close_distance_ratio),
        "assignment_summary": assignment_summary,
        "bounds_min_after_floor": bounds_min.tolist(),
        "bounds_max_after_floor": bounds_max.tolist(),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    arrays["bounds_min"] = bounds_min
    arrays["bounds_max"] = bounds_max

    output_cache = Path(args.output_cache).resolve()
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_cache, **arrays)
    print(f"Wrote stable WHAM cache: {output_cache}")
    print(json.dumps(assignment_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
