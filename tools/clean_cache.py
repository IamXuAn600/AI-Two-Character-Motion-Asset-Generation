#!/usr/bin/env python3
"""Conservative cache cleanup for the project.

By default this script only prints what it would remove. Pass --apply to delete.
It intentionally avoids model, checkpoint, video, WHAM, Blender, JSON, CSV, and
NPZ artifacts because those may be expensive or impossible to recreate.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "_temp_pose_inputs",
}

CACHE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
    ".bak",
}

PROTECTED_SUFFIXES = {
    ".pt",
    ".pth",
    ".pkl",
    ".npz",
    ".npy",
    ".blend",
    ".blend1",
    ".mp4",
    ".mov",
    ".avi",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely clean cache files.")
    parser.add_argument("--apply", action="store_true", help="Actually delete matched cache files/directories.")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    return parser.parse_args()


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def collect_targets(root: Path) -> list[Path]:
    targets: list[Path] = []
    for path in root.rglob("*"):
        if any(part in {".git"} for part in path.parts):
            continue
        if path.is_dir() and path.name in CACHE_DIR_NAMES:
            targets.append(path)
            continue
        if path.is_file() and path.suffix.lower() in CACHE_FILE_SUFFIXES:
            if path.suffix.lower() not in PROTECTED_SUFFIXES:
                targets.append(path)
    return sorted(set(targets), key=lambda item: (len(item.parts), str(item)))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    targets = [path for path in collect_targets(root) if is_inside(path, root)]
    if not targets:
        print("No cache targets found.")
        return

    action = "Removing" if args.apply else "Would remove"
    for path in targets:
        print(f"{action}: {path.relative_to(root)}")
        if args.apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete these cache targets.")


if __name__ == "__main__":
    main()
