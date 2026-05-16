#!/usr/bin/env python3
"""Run only WHAM's shared SLAM preprocessing for an existing video.

This is useful when tracking/features already exist and we only need to replace
an identity/local SLAM trajectory with a real DPVO global camera trajectory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WHAM SLAMModel and save slam_results.pth.")
    parser.add_argument("--wham-root", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True, help="Folder that will receive slam_results.pth.")
    parser.add_argument("--calib", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wham_root = Path(args.wham_root).resolve()
    video_path = Path(args.video).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(wham_root))
    os.chdir(str(wham_root))

    from lib.models.preproc.slam import SLAMModel

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    slam = SLAMModel(str(video_path), str(output), width, height, args.calib)

    processed = 0
    while cap.isOpened():
        ok, _ = cap.read()
        if not ok:
            break
        slam.track()
        processed += 1
        if processed % 100 == 0:
            print(f"SLAM frames: {processed}/{frame_count}", flush=True)

    cap.release()
    slam_results = slam.process()
    joblib.dump(slam_results, output / "slam_results.pth")
    print(f"Saved {output / 'slam_results.pth'} with {len(slam_results)} frames", flush=True)


if __name__ == "__main__":
    main()
