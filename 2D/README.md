
# Video Mocap Pipeline: MP4 → 多人轨迹/关键点 → MATLAB/Blender

这个项目可以把普通 MP4 视频转换为多人轨迹和人体关键点数据。

当前版本支持：

```text
MP4 视频
→ YOLO detector 只做人框检测
→ BoT-SORT + 角色锁定跟踪 character_A / character_B
→ RTMPose body26 做 2D 人体关键点
→ 遮挡恢复 / 短缺失预测 / 角色身份保护
→ 改进 3D 重建 / foot lock / 碰撞穿模修正
→ 输出 keypoints.json / trajectory.csv / tracks.csv
→ 输出带骨架和ID的 preview.mp4
→ 输出 refined 3D stick mocap 的 two_person_mocap.blend
→ MATLAB 分析速度/轨迹
→ 用 Blender 脚本生成多人 3D stick mocap 场景 / FBX
```

## 1. 安装环境

建议 Python 3.10 或 3.11。

```bash
conda create -n video_mocap python=3.10 -y
conda activate video_mocap
pip install -r requirements.txt
```

本机 Windows 环境已验证可用的 conda 环境是 `yolo`：

```powershell
conda activate yolo
python run_pipeline.py --video ..\videos\video_2.mp4 --output outputs\video_2 --yolo-weights .\yolo11m.pt
```

该环境已安装 `ultralytics`、`opencv-python`、`opencv-contrib-python`、`onnxruntime`、
`rtmlib`、`lap`、`pandas`、`Pillow`、`torch` 等依赖，并设置了
`KMP_DUPLICATE_LIB_OK=TRUE` 以避开 Windows 上常见的 OpenMP runtime 冲突。
RTMPose performance 模型已缓存到：

```text
C:\Users\Youran Fang\.cache\rtmlib\hub\checkpoints\rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.onnx
```

如果看到 ONNXRuntime 提示 `CUDAExecutionProvider` 不可用，可以先忽略；当前会回退到
CPU provider。若想完全压掉这个提示，可显式加 `--device cpu`。

## 2. 放入视频

把你的 MP4 放到：

```text
input/video_2.mp4
```

## 3. 一键运行

```bash
python run_pipeline.py --video input/video_2.mp4 --output outputs/video_2
```

切换 RTMPose 精度：

```bash
python run_pipeline.py --video input/video_2.mp4 --output outputs/video_2 --pose-mode performance
```

更快的 RTMPose：

```bash
python run_pipeline.py --video input/video_2.mp4 --output outputs/video_2 --pose-mode lightweight
```

第一次运行会自动下载 detector-only `yolo11n.pt` 和 RTMPose ONNX 权重。旧的 YOLO-pose 关键点模型不再使用。

默认会使用遮挡增强预设 `--tracking-preset occlusion`，并自动生成改进 3D 和 `.blend`。如果只想跑 2D：

```bash
python run_pipeline.py --video input/video_2.mp4 --output outputs/video_2 --no-3d --no-blend
```

## 4. 输出文件

```text
outputs/
├── pose_sequences.json
├── pose_sequences.csv
├── quality_report.json
├── wham_2d_observations.json
├── coco_keypoints.json
├── openpose_keypoints.json
├── motion3d/
│   ├── motion3d_sequences.json
│   ├── contacts.json
│   └── metrics.json
├── keypoints.json
├── tracks.csv
├── trajectory.csv
├── preview.mp4
└── two_person_mocap.blend
```

- `pose_sequences.json`: RTMPose body26 原始/平滑输出，按 `character_A/B` 保存。
- `quality_report.json`: 2D 工程质量报告，包含覆盖率、ID 切换、置信度、遮挡段和 bbox 抖动。
- `wham_2d_observations.json`: 面向 WHAM/后续 3D 的 2D 观测包，稳定角色 ID 固定为 `0/1`，同时保留原始 tracker ID。
- `coco_keypoints.json`: COCO17 风格导出。
- `openpose_keypoints.json`: OpenPose BODY25 风格导出。
- `keypoints.json`: 兼容旧 3D 阶段的 17 个 COCO 关键点和 32 个增强关键点。
- `tracks.csv`: 每一帧每个人的检测框和 ID。
- `trajectory.csv`: 每个人的身体中心点轨迹。
- `preview.mp4`: 可视化结果。
- `motion3d/`: 改进 3D 重建结果，包含 foot lock、接触和碰撞修正指标。
- `two_person_mocap.blend`: 从 refined 3D motion 生成的 Blender 场景。

## 5. MATLAB 分析

在 MATLAB 里运行：

```matlab
run("matlab/analyse_trajectory.m")
```

## 6. Blender 多人动画

当前推荐方案不是导入 BVH，而是直接从 `keypoints.json` 在 Blender 里生成两个 3D stick mocap 角色。

生成 `.blend`：

```bash
D:\Blender\blender.exe --background --python blender\keypoints_to_fbx.py -- outputs\keypoints.json outputs\two_person_mocap.blend 2
```

可选生成 `.fbx`：

```bash
D:\Blender\blender.exe --background --python blender\keypoints_to_fbx.py -- outputs\keypoints.json outputs\two_person_mocap.fbx 2
```

详细复现说明见：

```text
docs/MOCAP_FBX_WORKFLOW.md
```

## 7. 重要说明

这个 pipeline 能做多人追踪和 2D 姿态估计，但不是商业级 3D mocap。  
如果你要高质量 FBX，需要继续加入：

```text
2D keypoints → 3D pose lifting → IK retargeting → FBX export
```

## 8. WHAM 前的 2D 工程约定

默认遮挡策略是：

```text
--occlusion-hold off
```

这表示遮挡帧不会被上一帧人体姿态强制补全为有效观测。短缺口插值仍会写入
`smoothed_keypoints`，但对应 `smoothed_scores` 会低于 `min_keypoint_score`，
`smoothed_valid_mask=false`，`keypoint_source=interpolated`。后续 WHAM/3D 阶段应把
这些点视为弱提示或缺失，而不是强观测。

如需复现实验性的旧行为，可以显式使用：

```bash
python run_pipeline.py --video input/video_2.mp4 --output outputs/video_2 --occlusion-hold full
```

但不建议将 `full` 模式用于正式 WHAM 输入。
