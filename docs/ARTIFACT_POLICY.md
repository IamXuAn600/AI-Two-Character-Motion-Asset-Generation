# Artifact Policy

The repository is organized so code and documentation are committed to GitHub,
while large or reproducible local artifacts stay outside normal git history.

## Keep Locally

Do not delete these unless you have a separate backup or can regenerate them:

- source videos in `videos/` and `2D/input/`;
- detector/checkpoint/model files such as `*.pt`, `*.pth`, and `*.pkl`;
- WHAM and SMPL outputs such as `wham_output.pkl`, `tracking_results.pth`,
  `slam_results.pth`, and `*.npz`;
- Blender scenes and media exports such as `*.blend`, `*.blend1`, and `*.mp4`;
- final project outputs under `3D/results/` and preserved 2D output exports.

These files are ignored by `.gitignore` so they remain local by default. If a
large artifact must be versioned, use Git LFS and add it intentionally.

## Safe To Clean

The conservative cache cleanup script only targets:

- `__pycache__/`;
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`;
- `2D/_temp_pose_inputs/` and similarly named temporary pose input folders;
- files ending in `.pyc`, `.pyo`, `.tmp`, `.log`, or `.bak`.

It never deletes model/checkpoint/result extensions such as `.pt`, `.pth`,
`.pkl`, `.npz`, `.blend`, `.mp4`, `.json`, or `.csv`.

Preview cleanup without deleting:

```powershell
python tools/clean_cache.py
```

Apply cleanup:

```powershell
python tools/clean_cache.py --apply
```
