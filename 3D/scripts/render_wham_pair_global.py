#!/usr/bin/env python3
"""Render WHAM results with all active subjects in the global SLAM panel."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import imageio
import joblib
import numpy as np
import torch
from progress.bar import Bar
from pytorch3d.renderer import PointLights
from pytorch3d.renderer.cameras import look_at_rotation


PALETTE = np.asarray(
    [
        [0.18, 0.78, 0.28, 1.0],
        [0.96, 0.45, 0.18, 1.0],
        [0.22, 0.47, 0.95, 1.0],
        [0.88, 0.22, 0.72, 1.0],
        [0.95, 0.82, 0.24, 1.0],
        [0.18, 0.78, 0.82, 1.0],
        [0.85, 0.85, 0.85, 1.0],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a two/person global WHAM SLAM visualization.")
    parser.add_argument("--wham-root", required=True, help="Official WHAM checkout path.")
    parser.add_argument("--results-pkl", required=True, help="WHAM wham_output.pkl path.")
    parser.add_argument("--base-video", required=True, help="Existing WHAM output.mp4; left half is reused.")
    parser.add_argument("--output", required=True, help="Output mp4 path.")
    parser.add_argument("--subject-ids", default=None, help="Optional comma-separated WHAM subject ids to render.")
    parser.add_argument("--distance-scale", type=float, default=1.25, help="Global camera distance multiplier.")
    return parser.parse_args()


def setup_wham_imports(wham_root: Path):
    sys.path.insert(0, str(wham_root))
    os.chdir(str(wham_root))
    from configs.config import get_cfg_defaults
    from lib.models import build_body_model
    from lib.vis.renderer import Renderer

    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/yamls/demo.yaml")
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    return cfg, smpl, Renderer


def selected_subjects(results: Dict, subject_ids: str | None) -> List[str]:
    available = [str(key) for key in results.keys()]
    if not subject_ids:
        return sorted(available, key=lambda value: int(value) if value.isdigit() else value)
    requested = [item.strip() for item in subject_ids.split(",") if item.strip()]
    missing = [item for item in requested if item not in available]
    if missing:
        raise ValueError(f"Missing requested WHAM subject ids {missing}; available: {available}")
    return requested


def to_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).float().to(device)


def compute_global_vertices(results: Dict, subject_ids: Sequence[str], smpl, device: str) -> Dict[str, torch.Tensor]:
    verts_by_subject: Dict[str, torch.Tensor] = {}
    floor_min = None
    for subject_id in subject_ids:
        payload = results[subject_id]
        with torch.no_grad():
            output = smpl.get_output(
                body_pose=to_tensor(payload["pose_world"][:, 3:], device),
                global_orient=to_tensor(payload["pose_world"][:, :3], device),
                betas=to_tensor(payload["betas"], device),
                transl=to_tensor(payload["trans_world"], device),
            )
        verts = output.vertices.detach().cpu()
        subject_floor = float(verts[..., 1].min())
        floor_min = subject_floor if floor_min is None else min(floor_min, subject_floor)
        verts_by_subject[subject_id] = verts

    floor = float(floor_min or 0.0)
    for subject_id in subject_ids:
        verts_by_subject[subject_id][..., 1] -= floor
    return verts_by_subject


def build_frame_index(results: Dict, subject_ids: Sequence[str]) -> Dict[int, List[Tuple[str, int]]]:
    frame_index: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for subject_id in subject_ids:
        frame_ids = np.asarray(results[subject_id]["frame_ids"], dtype=np.int64)
        for local_idx, frame_id in enumerate(frame_ids.tolist()):
            frame_index[int(frame_id)].append((subject_id, local_idx))
    return frame_index


def compute_scene_geometry(
    verts_by_subject: Dict[str, torch.Tensor],
    frame_index: Dict[int, List[Tuple[str, int]]],
    frame_count: int,
) -> Tuple[np.ndarray, float, float, float]:
    all_centers = []
    frame_targets = np.zeros((frame_count, 3), dtype=np.float32)
    last_target = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    for frame_idx in range(frame_count):
        centers = []
        for subject_id, local_idx in frame_index.get(frame_idx, []):
            centers.append(verts_by_subject[subject_id][local_idx].mean(0).numpy())
        if centers:
            target = np.mean(np.stack(centers, axis=0), axis=0)
            last_target = target.astype(np.float32)
            all_centers.extend(centers)
        frame_targets[frame_idx] = last_target

    if not all_centers:
        return frame_targets, 4.0, 0.0, 0.0

    centers = np.stack(all_centers, axis=0)
    min_xz = centers[:, [0, 2]].min(axis=0)
    max_xz = centers[:, [0, 2]].max(axis=0)
    center_x, center_z = ((min_xz + max_xz) * 0.5).tolist()
    span = float(np.max(max_xz - min_xz))
    ground_scale = max(4.0, span * 1.7)
    return frame_targets, ground_scale, float(center_x), float(center_z)


def create_scene_camera(target: np.ndarray, device: str, distance: float, position=(-5.0, 5.0, 0.0)):
    target_t = torch.tensor(target, dtype=torch.float32, device=device).reshape(1, 3)
    position_t = torch.tensor(position, dtype=torch.float32, device=device).reshape(1, 3)
    direction = target_t - position_t
    direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6) * distance
    camera_pos = target_t - direction
    rotation = look_at_rotation(camera_pos, target_t).mT.to(device)
    translation = -(rotation @ camera_pos.unsqueeze(-1)).squeeze(-1)
    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


def subject_color(subject_id: str) -> torch.Tensor:
    try:
        index = int(subject_id)
    except ValueError:
        index = abs(hash(subject_id))
    return torch.from_numpy(PALETTE[index % len(PALETTE)])


def render_pair_global(args: argparse.Namespace) -> None:
    wham_root = Path(args.wham_root).resolve()
    cfg, smpl, Renderer = setup_wham_imports(wham_root)
    device = cfg.DEVICE
    results = {str(key): value for key, value in joblib.load(args.results_pkl).items()}
    subject_ids = selected_subjects(results, args.subject_ids)
    verts_by_subject = compute_global_vertices(results, subject_ids, smpl, device)

    cap = cv2.VideoCapture(args.base_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open base video: {args.base_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_width = total_width // 2
    focal_length = float((panel_width**2 + height**2) ** 0.5)

    renderer = Renderer(panel_width, height, focal_length, device, smpl.faces)
    frame_index = build_frame_index(results, subject_ids)
    targets, ground_scale, center_x, center_z = compute_scene_geometry(verts_by_subject, frame_index, frame_count)
    renderer.set_ground(ground_scale, center_x, center_z)
    faces = renderer.faces.clone().squeeze(0)
    camera_distance = max(5.0, ground_scale * args.distance_scale)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=fps, mode="I", format="FFMPEG", macro_block_size=1)
    bar = Bar("Rendering pair global SLAM", fill="#", max=frame_count)

    try:
        for frame_idx in range(frame_count):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = frame_bgr[..., ::-1].copy()
            left_panel = frame_rgb[:, :panel_width]

            active = frame_index.get(frame_idx, [])
            if active:
                verts = torch.stack(
                    [verts_by_subject[subject_id][local_idx] for subject_id, local_idx in active],
                    dim=0,
                ).to(device)
                colors = torch.stack([subject_color(subject_id) for subject_id, _ in active], dim=0).to(device)
                rotation, translation, lights = create_scene_camera(targets[frame_idx], device, camera_distance)
                cameras = renderer.create_camera(rotation, translation)
                global_panel = renderer.render_with_ground(verts, faces, colors, cameras, lights)
            else:
                global_panel = np.ones_like(left_panel, dtype=np.uint8) * 255

            label = "global SLAM: all active WHAM subjects"
            cv2.putText(global_panel, label, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
            ids_text = "subjects: " + ",".join(subject_id for subject_id, _ in active)
            cv2.putText(global_panel, ids_text, (28, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
            writer.append_data(np.concatenate((left_panel, global_panel), axis=1))
            bar.next()
    finally:
        writer.close()
        cap.release()


def main() -> None:
    render_pair_global(parse_args())


if __name__ == "__main__":
    main()
