#!/usr/bin/env python3
"""Run official WHAM using this project's fixed two-person 2D observations.

This bridge is intentionally isolated from the main 3D pipeline because WHAM has
its own Python/CUDA/SMPL dependency stack. The main pipeline calls this script
with the WHAM environment's Python executable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


VIS_THRESH = 0.30
MINIMUM_JOINTS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge project 2D observations into official WHAM.")
    parser.add_argument("--wham-root", required=True, help="Official WHAM checkout.")
    parser.add_argument("--video", required=True, help="Original full video path.")
    parser.add_argument("--observations", required=True, help="wham_2d_observations.json from the project 2D stage.")
    parser.add_argument("--output-root", required=True, help="Output root. A video-name subfolder is created inside.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-track-frames", type=int, default=30)
    parser.add_argument("--calib", default=None)
    parser.add_argument("--local-only", action="store_true", help="Skip WHAM global SLAM/world trajectory estimation.")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--run-smplify", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None, help="Override WHAM cfg.DEVICE, e.g. cuda or cuda:0.")
    return parser.parse_args()


def ensure_wham_on_path(wham_root: Path) -> None:
    sys.path.insert(0, str(wham_root))
    os.chdir(wham_root)


def video_info(video_path: Path) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return length, fps, width, height


def finite_bbox_xyxy(values: Sequence[Any]) -> Optional[np.ndarray]:
    if values is None or len(values) < 4:
        return None
    bbox = np.asarray(values[:4], dtype=np.float32)
    if not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def compute_wham_bbox(keypoints: np.ndarray, fallback_xyxy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    mask = np.isfinite(keypoints).all(axis=1) & (keypoints[:, 2] > VIS_THRESH)
    if int(np.count_nonzero(mask)) >= MINIMUM_JOINTS:
        pts = keypoints[mask]
        x1, y1 = pts[:, :2].min(axis=0)
        x2, y2 = pts[:, :2].max(axis=0)
        cx = float((x1 + x2) * 0.5)
        cy = float((y1 + y2) * 0.5)
        scale = float(max(x2 - x1, y2 - y1) * 1.20 / 200.0)
        return np.asarray([cx, cy, max(scale, 1e-4)], dtype=np.float32)
    if fallback_xyxy is not None:
        cx = float((fallback_xyxy[0] + fallback_xyxy[2]) * 0.5)
        cy = float((fallback_xyxy[1] + fallback_xyxy[3]) * 0.5)
        scale = float(max(fallback_xyxy[2] - fallback_xyxy[0], fallback_xyxy[3] - fallback_xyxy[1]) * 1.05 / 200.0)
        return np.asarray([cx, cy, max(scale, 1e-4)], dtype=np.float32)
    return None


def smooth_bbox_sequence(bboxes: np.ndarray, fps: float) -> np.ndarray:
    if len(bboxes) < 3:
        return bboxes
    try:
        import scipy.signal as signal

        kernel = int(int(fps / 2) / 2) * 2 + 1
        kernel = max(3, min(kernel, len(bboxes) if len(bboxes) % 2 == 1 else len(bboxes) - 1))
        if kernel < 3:
            return bboxes
        return np.asarray([signal.medfilt(param, kernel) for param in bboxes.T], dtype=np.float32).T
    except Exception:
        return bboxes


def convert_observations_to_wham_tracking(
    observations_path: Path,
    fps: float,
    max_frames: Optional[int],
    min_track_frames: int,
) -> Dict[int, Dict[str, Any]]:
    payload = json.loads(observations_path.read_text(encoding="utf-8"))
    tracking = defaultdict(lambda: defaultdict(list))

    for track in payload.get("tracks", []):
        subject_id = int(track.get("stable_track_id", track.get("role_id", len(tracking))))
        frames = track.get("frames", [])
        if max_frames is not None:
            frames = frames[:max_frames]
        for frame in frames:
            if not frame.get("visible", True):
                continue
            keypoints = np.asarray(frame.get("keypoints_coco17") or [], dtype=np.float32)
            if keypoints.shape != (17, 3):
                continue
            if int(np.count_nonzero(keypoints[:, 2] > VIS_THRESH)) < MINIMUM_JOINTS:
                continue
            bbox = compute_wham_bbox(keypoints, finite_bbox_xyxy(frame.get("bbox_xyxy")))
            if bbox is None:
                continue
            tracking[subject_id]["frame_id"].append(int(frame.get("frame_index", len(tracking[subject_id]["frame_id"]))))
            tracking[subject_id]["bbox"].append(bbox)
            tracking[subject_id]["keypoints"].append(keypoints)

    output: Dict[int, Dict[str, Any]] = {}
    for subject_id, values in tracking.items():
        if len(values["frame_id"]) < min_track_frames:
            continue
        output[subject_id] = {
            "frame_id": np.asarray(values["frame_id"], dtype=np.int64),
            "bbox": smooth_bbox_sequence(np.asarray(values["bbox"], dtype=np.float32), fps),
            "keypoints": np.asarray(values["keypoints"], dtype=np.float32),
            "features": [],
            "flipped_bbox": [],
            "flipped_keypoints": [],
            "flipped_features": [],
        }

    if not output:
        raise RuntimeError("No project tracks had enough valid 2D observations for WHAM.")
    return output


def identity_slam(length: int) -> np.ndarray:
    slam_results = np.zeros((length, 7), dtype=np.float32)
    slam_results[:, 3] = 1.0
    return slam_results


def run_shared_slam(video_path: Path, output_path: Path, width: int, height: int, calib: Optional[str], local_only: bool) -> np.ndarray:
    length, _, _, _ = video_info(video_path)
    if local_only:
        return identity_slam(length)
    try:
        from lib.models.preproc.slam import SLAMModel
    except Exception as exc:
        print(f"WHAM SLAM unavailable; using identity camera trajectory: {exc}")
        return identity_slam(length)

    cap = cv2.VideoCapture(str(video_path))
    slam = SLAMModel(str(video_path), str(output_path), width, height, calib)
    while cap.isOpened():
        ok, _ = cap.read()
        if not ok:
            break
        slam.track()
    cap.release()
    return slam.process()


def remove_if_force(paths: Sequence[Path], force: bool) -> None:
    if not force:
        return
    for path in paths:
        if path.exists():
            path.unlink()


def prepare_wham_preprocess(
    cfg: Any,
    video_path: Path,
    observations_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    import joblib
    from lib.models.preproc.extractor import FeatureExtractor

    tracking_path = output_path / "tracking_results.pth"
    slam_path = output_path / "slam_results.pth"
    remove_if_force([tracking_path, slam_path], args.force)
    if tracking_path.exists() and slam_path.exists():
        return

    length, fps, width, height = video_info(video_path)
    tracking_results = convert_observations_to_wham_tracking(
        observations_path=observations_path,
        fps=fps,
        max_frames=args.max_frames,
        min_track_frames=args.min_track_frames,
    )
    extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
    import torch

    with torch.inference_mode():
        tracking_results = extractor.run(str(video_path), tracking_results)
    if cfg.DEVICE.lower().startswith("cuda"):
        torch.cuda.empty_cache()
    slam_results = run_shared_slam(video_path, output_path, width, height, args.calib, args.local_only)
    if len(slam_results) != length:
        print(f"Warning: SLAM length {len(slam_results)} differs from video length {length}.")

    joblib.dump(tracking_results, tracking_path)
    joblib.dump(slam_results, slam_path)


def expand_betas(betas: np.ndarray, frame_count: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        return np.repeat(betas.reshape(1, -1), frame_count, axis=0)
    if len(betas) == frame_count:
        return betas
    if len(betas) == 1:
        return np.repeat(betas, frame_count, axis=0)
    return np.repeat(betas[:1], frame_count, axis=0)


def compute_smpl_joints_world(
    smpl_model: Any,
    pose_world: Optional[np.ndarray],
    trans_world: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    device: str,
) -> Optional[np.ndarray]:
    if pose_world is None or trans_world is None or betas is None:
        return None
    pose_world = np.asarray(pose_world, dtype=np.float32)
    trans_world = np.asarray(trans_world, dtype=np.float32)
    if pose_world.ndim != 2 or pose_world.shape[1] < 72 or trans_world.ndim != 2 or trans_world.shape[1] < 3:
        return None

    import torch

    frame_count = min(len(pose_world), len(trans_world))
    betas = expand_betas(betas, frame_count)
    chunks = []
    for start in range(0, frame_count, 256):
        end = min(frame_count, start + 256)
        global_orient = torch.as_tensor(pose_world[start:end, :3], dtype=torch.float32, device=device)
        body_pose = torch.as_tensor(pose_world[start:end, 3:72], dtype=torch.float32, device=device)
        betas_chunk = torch.as_tensor(betas[start:end, :10], dtype=torch.float32, device=device)
        transl = torch.as_tensor(trans_world[start:end, :3], dtype=torch.float32, device=device)
        with torch.no_grad():
            output = smpl_model.get_output(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas_chunk,
                transl=transl,
                pose2rot=True,
            )
        chunks.append(output.joints.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else None


def export_project_wham_npz(output_path: Path, smpl_model: Any, device: str, local_only: bool) -> Path:
    import joblib

    wham_output = output_path / "wham_output.pkl"
    if not wham_output.exists():
        raise RuntimeError(f"WHAM did not write expected output: {wham_output}")
    raw_results = joblib.load(wham_output)
    arrays: Dict[str, np.ndarray] = {}
    subject_ids = []

    for raw_subject_id, result in raw_results.items():
        subject_id = str(raw_subject_id)
        subject_ids.append(subject_id)
        prefix = f"subject_{subject_id}_"
        frame_ids = np.asarray(result.get("frame_ids"), dtype=np.int64).reshape(-1)
        pose_world = np.asarray(result.get("pose_world"), dtype=np.float32) if result.get("pose_world") is not None else None
        pose_body = pose_world[:, 3:72] if pose_world is not None and pose_world.shape[1] >= 72 else None
        trans_world = np.asarray(result.get("trans_world"), dtype=np.float32) if result.get("trans_world") is not None else None
        betas = np.asarray(result.get("betas"), dtype=np.float32) if result.get("betas") is not None else None
        joints_world = compute_smpl_joints_world(smpl_model, pose_world, trans_world, betas, device)

        arrays[f"{prefix}frame_ids"] = frame_ids
        if joints_world is not None:
            arrays[f"{prefix}joints_world"] = joints_world
        if trans_world is not None:
            arrays[f"{prefix}trans_world"] = trans_world
        if pose_world is not None:
            arrays[f"{prefix}pose_world"] = pose_world
        if pose_body is not None:
            arrays[f"{prefix}pose_body"] = pose_body
        if betas is not None:
            arrays[f"{prefix}betas"] = betas

    metadata = {
        "source_format": "project_npz",
        "wham_output_pkl": str(wham_output),
        "generated_by": "run_wham_with_observations.py",
        "uses_project_2d_tracks": True,
        "local_only": bool(local_only),
    }
    project_npz = output_path / "project_wham_output.npz"
    np.savez_compressed(
        project_npz,
        subject_ids=np.asarray(subject_ids),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        **arrays,
    )
    return project_npz


def main() -> None:
    args = parse_args()
    wham_root = Path(args.wham_root).resolve()
    video_path = Path(args.video).resolve()
    observations_path = Path(args.observations).resolve()
    output_root = Path(args.output_root).resolve()
    sequence = video_path.stem
    output_path = output_root / sequence
    output_path.mkdir(parents=True, exist_ok=True)

    project_npz = output_path / "project_wham_output.npz"
    if project_npz.exists() and not args.force:
        print(f"PROJECT_WHAM_OUTPUT={project_npz}")
        return

    ensure_wham_on_path(wham_root)

    import demo as wham_demo
    from configs.config import get_cfg_defaults
    from lib.models import build_body_model, build_network

    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/yamls/demo.yaml")
    if args.device:
        cfg.defrost()
        cfg.DEVICE = args.device
        cfg.freeze()

    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()

    prepare_wham_preprocess(cfg, video_path, observations_path, output_path, args)

    wham_output = output_path / "wham_output.pkl"
    remove_if_force([wham_output], args.force)
    wham_demo.args = SimpleNamespace(run_smplify=args.run_smplify)
    wham_demo.smpl = smpl
    if not wham_output.exists():
        wham_demo.run(
            cfg,
            str(video_path),
            str(output_path),
            network,
            args.calib,
            run_global=not args.local_only,
            save_pkl=True,
            visualize=args.visualize,
        )

    project_npz = export_project_wham_npz(output_path, getattr(network, "smpl", smpl), cfg.DEVICE, args.local_only)
    print(f"PROJECT_WHAM_OUTPUT={project_npz}")


if __name__ == "__main__":
    main()
