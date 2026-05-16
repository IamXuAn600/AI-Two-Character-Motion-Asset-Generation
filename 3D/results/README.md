# 3D Results Index

Current result folders are grouped by video.

```text
wham_visualizations/
`- video_2/
   |- calib.txt
   |- tracking_results.pth
   |- slam_results.pth
   |- wham_output.pkl
   |- wham_local_output.mp4
   |- wham_pair_global_slam_output.mp4
   `- Blender/cache occlusion repair outputs
```

`video_2` uses the available official WHAM local output plus a two-subject
global-view render for subjects `0,1`; its `slam_results.pth` is the
local/identity trajectory from that WHAM run.

```text
two_in_one_alignment/
`- video_2/
   |- 2d/
   `- motion3d/
```

The current `motion3d/` folder contains the full Two-in-One schema:

- `world_alignment.json`
- `two_in_one_pair.json`
- `two_in_one_pair.npz`
- `motion3d_sequences.json`
- `motion3d_sequences.npz`
- `interaction_events.json`
- `quality_report.json`
- `asset_metadata.json`
- `joint_positions.csv`
- `preview_3d.mp4`

The current `video_2/motion3d/` is regenerated over the full 2127 frame
timeline.
