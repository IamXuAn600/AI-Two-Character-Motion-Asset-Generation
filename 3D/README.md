# Two-in-One 3D Pipeline

This folder contains the engineering 3D stage for the two-character project.

Default test material:

```bash
python run_pipeline.py
```

The default command uses `../videos/video_2.mp4`, runs the current 2D pipeline
into `results/two_in_one_alignment/video_2/2d`, then reconstructs a shared
two-person 3D motion into `results/two_in_one_alignment/video_2/motion3d`.

Main outputs:

```text
motion3d/
|- motion3d_sequences.json
|- motion3d_sequences.npz
|- world_alignment.json
|- two_in_one_pair.json
|- two_in_one_pair.npz
|- interaction_events.json
|- asset_metadata.json
|- joint_positions.csv
|- quality_report.json
`- preview_3d.mp4
```

Implemented report stages:

- WHAM-compatible initialization. If real WHAM/SMPL outputs are not available,
  the pipeline uses deterministic monocular kinematic lifting so the full 3D
  asset path remains runnable.
- Shared pair-centric world alignment with meters, Y-up, floor at `Y=0`,
  height normalization, root centering, and a stable `A -> B` horizontal axis.
- Two-in-One pair representation with dynamic pair-frame roots, joint positions,
  velocities, relative root, facing angle, visible masks, pair occlusion, and
  contact candidates. The dense `x_pair` tensor and `x_pair_mask` are exported
  in `two_in_one_pair.npz`.
- Occlusion masks from 2D confidence/person visibility and bbox overlap.
- Masked interpolation plus temporal smoothing and bone-length stabilization.
- Foot-ground locking, root collision separation, and contact event extraction.
- Asset-oriented JSON/NPZ/CSV exports and a 3D preview video.

## Canonical Data Flow

```text
2D/run_pipeline.py
  -> wham_2d_observations.json
  -> scripts/run_wham_with_observations.py
  -> project_wham_output.npz
  -> scripts/two_in_one_pipeline.py
  -> world_alignment.json
  -> two_in_one_pair.json / two_in_one_pair.npz
```

The 2D stage owns stable project identities (`character_A`, `character_B`),
bbox/keypoint confidence, visible masks, and pair occlusion cues. The WHAM bridge
does not reassign identities: it converts those stable tracks into WHAM
`tracking_results.pth`, runs WHAM feature extraction plus optional global SLAM,
and exports a compact `project_wham_output.npz`. The final pipeline imports that
NPZ, aligns both bodies into one pair-centric world frame, and writes the
Two-in-One representation.

For quick debugging, use official WHAM pickle imports with `--wham-results-path`.
For production runs, prefer the bridge path so WHAM follows the project 2D
identity assignment instead of its own demo subject IDs.

Useful options:

```bash
python run_pipeline.py --video ../videos/video_2.mp4 --output results/two_in_one_alignment/video_2
python run_pipeline.py --two-d-root path/to/2d_output --skip-2d
python run_pipeline.py --force-2d
python run_pipeline.py --max-frames 120
```

## Real WHAM backend

The pipeline supports three initializers:

```bash
python run_pipeline.py --reconstruction-backend fallback
python run_pipeline.py --reconstruction-backend auto
python run_pipeline.py --reconstruction-backend wham --wham-root D:/path/to/WHAM --wham-python D:/path/to/wham_env/python.exe
```

`auto` is the default: it tries real WHAM only when `--wham-root` or
`--wham-results-path` is provided, then falls back to the deterministic
WHAM-compatible lift if WHAM is unavailable.

Recommended first WHAM smoke test:

```bash
python run_pipeline.py ^
  --video ../videos/video_2.mp4 ^
  --output results/two_in_one_alignment/video_2 ^
  --reconstruction-backend wham ^
  --wham-root D:/path/to/WHAM ^
  --wham-python D:/path/to/wham_env/python.exe ^
  --wham-local-only
```

Then run world/global mode by dropping `--wham-local-only` and optionally passing
camera calibration:

```bash
python run_pipeline.py --reconstruction-backend wham --wham-root D:/path/to/WHAM --wham-calib D:/path/to/calib.txt
```

If WHAM has already been run, import the cached bridge result:

```bash
python run_pipeline.py --reconstruction-backend wham --wham-results-path results/two_in_one_alignment/video_2/wham/video_2/project_wham_output.npz
```

The bridge script `scripts/run_wham_with_observations.py` converts this project's
stable 2D tracks (`bbox + COCO17 keypoints + frame_id`) into WHAM
`tracking_results.pth`, reuses WHAM feature extraction and shared SLAM, and
exports `project_wham_output.npz` with `joints_world`, `trans_world`,
`pose_world`, `pose_body`, and `betas` when SMPL files are available. Downstream
outputs keep the same schema: per-role `root_translation`, `root_yaw`,
`joints_3d`, confidence, visible masks, foot contacts, world alignment metadata,
and the dense Two-in-One `x_pair` representation.

## Current Checked Outputs

```text
results/two_in_one_alignment/video_2/
|- 2d/
`- motion3d/

results/wham_visualizations/video_2/
`- WHAM pkl/slam/tracking plus local and pair-global videos
```

The current preserved `motion3d/` folder uses the full Two-in-One schema.
`video_2` has 2127 frames.

## WHAM Blender Occlusion Repair

The current two-character Blender path uses the WHAM pickle plus the stable 2D
role observations, then repairs occlusion spans before exporting the `.blend`.
The repair keeps `character_A` / `character_B` identities stable and uses:

- missing-frame detection from `wham_2d_observations.json`, with red head dots
  triggered from the first predicted occlusion frame;
- velocity Hermite in-betweening, so long missing spans continue the recent
  motion instead of looking like slow linear interpolation;
- partial 2D observations from exposed keypoints, when available, to adjust the
  in-between progress during occlusion;
- smooth capped z-depth stabilization with per-frame z speed limits, per-track
  baselines, and semantic order: the occluded role stays behind the visible
  role;
- dynamic mesh clearance from each frame's body envelope, so raised hands or
  extended legs reserve extra z distance before they can penetrate the front
  body;
- source-space foot grounding at `Y=0`;
- small red head markers in Blender for predicted occlusion spans.

Example for `video_2` output:

```powershell
$modelSource = "C:\Users\pps12\Downloads\SMPL_python_v.1.1.0\smpl\models\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
$tempDir = "3D\results\tmp_smpl_model"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
New-Item -ItemType HardLink -Path "$tempDir\SMPL_NEUTRAL.pkl" -Target $modelSource

python 3D\scripts\prepare_stable_wham_blend_cache.py `
  --wham-output 3D\results\wham_visualizations\video_2\wham_output.pkl `
  --tracking-results 3D\results\wham_visualizations\video_2\tracking_results.pth `
  --observations 3D\results\two_in_one_alignment\video_2\2d\wham_2d_observations.json `
  --smpl-model-dir 3D\results\tmp_smpl_model `
  --output-cache 3D\results\wham_visualizations\video_2\wham_blend_cache_stable_regenerated.npz

python 3D\scripts\build_occlusion_inbetween_cache.py `
  --cache 3D\results\wham_visualizations\video_2\wham_blend_cache_stable_regenerated.npz `
  --observations 3D\results\two_in_one_alignment\video_2\2d\wham_2d_observations.json `
  --wham-output 3D\results\wham_visualizations\video_2\wham_output.pkl `
  --video videos\video_2.mp4 `
  --output-cache 3D\results\wham_visualizations\video_2\wham_blend_cache_z_swapped_inbetween.npz `
  --events-json 3D\results\wham_visualizations\video_2\occlusion_inbetween_events.json `
  --image-dir 3D\results\wham_visualizations\video_2\occlusion_inbetween

& "D:\Blender\blender.exe" --background `
  --python 3D\scripts\export_wham_cache_to_blend.py -- `
  --cache 3D\results\wham_visualizations\video_2\wham_blend_cache_z_swapped_inbetween.npz `
  --output 3D\results\wham_visualizations\video_2\wham_output_mesh.blend

& "D:\Blender\blender.exe" --background `
  --python 3D\scripts\add_occlusion_head_markers_to_blend.py -- `
  --input 3D\results\wham_visualizations\video_2\wham_output_mesh.blend `
  --cache 3D\results\wham_visualizations\video_2\wham_blend_cache_z_swapped_inbetween.npz `
  --events-json 3D\results\wham_visualizations\video_2\occlusion_inbetween_events.json `
  --output 3D\results\wham_visualizations\video_2\wham_output_mesh_z_swapped_inbetween_head_markers.blend
```

Add `--mirror-x` if a mirrored scene is desired. The main tuning defaults live in
`scripts/build_occlusion_inbetween_cache.py`: `max-z-delta-per-frame`,
`z-range-limit`, `occlusion-depth-context-frames`, `mesh-clearance-margin`,
`prediction-velocity-window`, and `partial-observation-weight`.
