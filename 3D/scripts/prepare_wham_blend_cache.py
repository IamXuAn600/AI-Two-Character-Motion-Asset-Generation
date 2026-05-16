"""Prepare WHAM/SMPL mesh data for Blender export.

This script is meant to be run with the project Python environment, not inside
Blender. It reads WHAM's joblib pickle and writes a temporary npz cache that
Blender can load without WHAM, torch, smplx, or joblib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch


DEFAULT_BATCH_SIZE = 128


def patch_numpy_for_legacy_chumpy() -> None:
    """SMPL pkl files often reference chumpy, which expects old numpy aliases."""

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
    parser = argparse.ArgumentParser(description="Prepare WHAM mesh cache for Blender.")
    parser.add_argument("--wham-output", required=True, help="Path to WHAM wham_output.pkl.")
    parser.add_argument("--smpl-model-dir", required=True, help="Folder containing SMPL_NEUTRAL.pkl.")
    parser.add_argument("--output-cache", required=True, help="Temporary .npz cache path for Blender.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--vertex-source",
        choices=("smpl-world", "wham-verts"),
        default="smpl-world",
        help=(
            "smpl-world recomputes vertices from pose_world/trans_world/betas. "
            "wham-verts uses WHAM's exported verts and trans directly, matching "
            "the visualization output more closely."
        ),
    )
    parser.add_argument(
        "--smooth-depth-window",
        type=int,
        default=0,
        help=(
            "Temporal smoothing window for the WHAM/source depth axis. "
            "Only changes the global per-frame depth offset, not the body pose."
        ),
    )
    parser.add_argument(
        "--subject-ids",
        default=None,
        help="Optional comma-separated WHAM subject ids. Defaults to all subjects in wham_output.pkl.",
    )
    return parser.parse_args()


def selected_subject_ids(raw_results: dict, requested: str | None) -> list[str]:
    available = [str(key) for key in raw_results.keys()]
    available = sorted(available, key=lambda value: int(value) if value.isdigit() else value)
    if not requested:
        return available
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    missing = [item for item in wanted if item not in available]
    if missing:
        raise ValueError(f"Missing requested WHAM subject ids {missing}; available: {available}")
    return wanted


def as_frame_betas(betas: np.ndarray, frame_count: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        return np.repeat(betas.reshape(1, -1), frame_count, axis=0)
    if len(betas) == frame_count:
        return betas
    if len(betas) == 1:
        return np.repeat(betas, frame_count, axis=0)
    return np.repeat(betas[:1], frame_count, axis=0)


def compute_subject_vertices(
    model,
    payload: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    pose_world = np.asarray(payload["pose_world"], dtype=np.float32)
    trans_world = np.asarray(payload["trans_world"], dtype=np.float32)
    frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64).reshape(-1)
    frame_count = min(len(frame_ids), len(pose_world), len(trans_world))
    pose_world = pose_world[:frame_count]
    trans_world = trans_world[:frame_count]
    frame_ids = frame_ids[:frame_count]
    betas = as_frame_betas(payload["betas"], frame_count)[:, :10]

    chunks = []
    with torch.inference_mode():
        for start in range(0, frame_count, batch_size):
            end = min(frame_count, start + batch_size)
            output = model(
                global_orient=torch.as_tensor(pose_world[start:end, :3], dtype=torch.float32, device=device),
                body_pose=torch.as_tensor(pose_world[start:end, 3:72], dtype=torch.float32, device=device),
                betas=torch.as_tensor(betas[start:end], dtype=torch.float32, device=device),
                transl=torch.as_tensor(trans_world[start:end, :3], dtype=torch.float32, device=device),
                pose2rot=True,
            )
            chunks.append(output.vertices.detach().cpu().numpy().astype(np.float32))
    return frame_ids, np.concatenate(chunks, axis=0)


def load_wham_vertices(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64).reshape(-1)
    vertices = np.asarray(payload["verts"], dtype=np.float32)
    frame_count = min(len(frame_ids), len(vertices))
    return frame_ids[:frame_count], vertices[:frame_count]


def subject_roots(payload: dict, vertices: np.ndarray, frame_count: int, vertex_source: str) -> np.ndarray:
    if vertex_source == "wham-verts" and payload.get("trans") is not None:
        roots = np.asarray(payload["trans"], dtype=np.float32)
    elif payload.get("trans_world") is not None:
        roots = np.asarray(payload["trans_world"], dtype=np.float32)
    else:
        roots = vertices.mean(axis=1)

    if len(roots) < frame_count:
        fallback = vertices.mean(axis=1)
        output = fallback.copy()
        output[: len(roots)] = roots
        return output
    return roots[:frame_count].astype(np.float32)


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    window = int(window)
    if window < 3 or len(values) < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values.copy()

    try:
        from scipy.signal import savgol_filter

        polyorder = 2 if window >= 5 else 1
        return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp").astype(np.float32)
    except Exception:
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)


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

    wham_output = Path(args.wham_output).resolve()
    smpl_model_dir = Path(args.smpl_model_dir).resolve()
    output_cache = Path(args.output_cache).resolve()
    output_cache.parent.mkdir(parents=True, exist_ok=True)

    raw_results = {str(key): value for key, value in joblib.load(wham_output).items()}
    subject_ids = selected_subject_ids(raw_results, args.subject_ids)

    device = torch.device(args.device)
    model = smplx.SMPL(
        str(smpl_model_dir),
        gender="neutral",
        batch_size=max(1, int(args.batch_size)),
        create_betas=False,
        create_body_pose=False,
        create_global_orient=False,
        create_transl=False,
    ).to(device)
    model.eval()

    arrays: dict[str, np.ndarray] = {
        "faces": np.asarray(model.faces, dtype=np.int32),
        "subject_ids": np.asarray(subject_ids),
    }
    metadata = {
        "source": "WHAM wham_output.pkl",
        "wham_output": str(wham_output),
        "smpl_model_dir": str(smpl_model_dir),
        "coordinate_system": "WHAM/SMPL meters, Y-up before Blender conversion",
        "vertex_source": args.vertex_source,
        "smooth_depth_window": int(args.smooth_depth_window),
        "subjects": subject_ids,
    }

    y_down = args.vertex_source == "wham-verts"
    floor_y = None
    bounds_min = np.full(3, np.inf, dtype=np.float32)
    bounds_max = np.full(3, -np.inf, dtype=np.float32)
    for subject_id in subject_ids:
        payload = raw_results[subject_id]
        if args.vertex_source == "wham-verts":
            frame_ids, vertices = load_wham_vertices(payload)
        else:
            frame_ids, vertices = compute_subject_vertices(
                model=model,
                payload=payload,
                batch_size=max(1, int(args.batch_size)),
                device=device,
            )
        roots = subject_roots(payload, vertices, len(frame_ids), args.vertex_source)
        vertices, roots = smooth_depth_offset(vertices, roots, args.smooth_depth_window)
        subject_floor = float(np.nanmax(vertices[..., 1]) if y_down else np.nanmin(vertices[..., 1]))
        if floor_y is None:
            floor_y = subject_floor
        elif y_down:
            floor_y = max(floor_y, subject_floor)
        else:
            floor_y = min(floor_y, subject_floor)
        arrays[f"frame_ids__{subject_id}"] = frame_ids
        arrays[f"verts__{subject_id}"] = vertices
        arrays[f"roots__{subject_id}"] = roots
        print(f"Prepared WHAM subject {subject_id}: {len(frame_ids)} frames, {vertices.shape[1]} vertices")

    floor_y = float(floor_y or 0.0)
    bounds_min[:] = np.inf
    bounds_max[:] = -np.inf
    for subject_id in subject_ids:
        if y_down:
            arrays[f"verts__{subject_id}"][..., 1] = floor_y - arrays[f"verts__{subject_id}"][..., 1]
            arrays[f"roots__{subject_id}"][..., 1] = floor_y - arrays[f"roots__{subject_id}"][..., 1]
        else:
            arrays[f"verts__{subject_id}"][..., 1] -= floor_y
            arrays[f"roots__{subject_id}"][..., 1] -= floor_y
        vertices = arrays[f"verts__{subject_id}"]
        bounds_min = np.minimum(bounds_min, np.nanmin(vertices.reshape(-1, 3), axis=0))
        bounds_max = np.maximum(bounds_max, np.nanmax(vertices.reshape(-1, 3), axis=0))
    metadata["floor_y_reference"] = floor_y
    metadata["source_y_axis"] = "down" if y_down else "up"
    metadata["bounds_min_after_floor"] = bounds_min.tolist()
    metadata["bounds_max_after_floor"] = bounds_max.tolist()
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    arrays["bounds_min"] = bounds_min
    arrays["bounds_max"] = bounds_max

    np.savez(output_cache, **arrays)
    print(f"Wrote WHAM Blender cache: {output_cache}")


if __name__ == "__main__":
    main()
