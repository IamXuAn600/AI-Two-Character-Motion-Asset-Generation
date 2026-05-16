# Two Character Motion Pipeline

This repository contains a two-character video motion pipeline:

```text
MP4 video
-> 2D detection, tracking, and RTMPose keypoints
-> WHAM-compatible observations
-> pair-centric 3D reconstruction
-> WHAM / Blender visualization assets
```

The current preserved working sample is `video_2`. Large inputs, model weights,
and generated outputs are intentionally ignored by git; keep them locally or in a
separate artifact store.

## Layout

```text
2D/                         2D tracking and keypoint pipeline
3D/                         3D reconstruction and WHAM/Blender scripts
videos/                     Local source videos, ignored by git
3D/results/                 Local generated outputs, ignored by git
```

## Quick Start

Install dependencies for the stage you want to run:

```powershell
cd 2D
pip install -r requirements.txt
python run_pipeline.py --video input/video_2.mp4 --output outputs/video_2
```

For the end-to-end 3D stage:

```powershell
cd 3D
pip install -r requirements.txt
python run_pipeline.py
```

The 3D default uses `../videos/video_2.mp4` and writes to
`results/two_in_one_alignment/video_2`.

## Local Artifacts

The repository keeps code and documentation git-friendly. These local files are
not meant to be committed:

- `videos/*.mp4`
- `2D/input/*.mp4`
- `2D/outputs/`
- generated 2D root exports such as `2D/pose_sequences.json`
- `3D/results/`
- model files such as `*.pt`, `*.pth`, `*.pkl`, `*.npz`, and Blender files

For the current local workspace, the `video_2` inputs and outputs have been left
in place, including WHAM outputs under `3D/results/wham_visualizations/video_2`.

See `docs/ARTIFACT_POLICY.md` for the keep/delete policy. To preview safe cache
cleanup:

```powershell
python tools/clean_cache.py
```

To apply it:

```powershell
python tools/clean_cache.py --apply
```
