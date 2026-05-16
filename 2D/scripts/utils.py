
from __future__ import annotations

from typing import Dict, List, Tuple
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - fallback keeps OpenCV-only installs usable.
    Image = None
    ImageDraw = None
    ImageFont = None

SKELETON_EDGES: List[Tuple[int, int]] = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4)
]

ENHANCED_SKELETON_EDGES: List[Tuple[int, int]] = [
    (17, 18), (17, 19), (18, 11), (18, 12), (19, 0),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 24), (17, 25),
    (17, 5), (17, 6),
    (5, 7), (7, 9), (9, 20),
    (6, 8), (8, 10), (10, 21),
    (11, 13), (13, 15), (15, 22), (15, 26), (15, 28), (22, 28),
    (12, 14), (14, 16), (16, 23), (16, 27), (16, 29), (23, 29),
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

ENHANCED_KEYPOINT_NAMES = KEYPOINT_NAMES + [
    "neck", "mid_hip", "spine", "left_hand", "right_hand", "left_foot", "right_foot",
    "head_top", "chin", "left_heel", "right_heel", "left_toe", "right_toe",
    "left_outer_foot", "right_outer_foot",
]


CHINESE_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def safe_int_id(value, fallback: int) -> int:
    try:
        if value is None:
            return fallback
        return int(value)
    except Exception:
        return fallback


def compute_body_center(kpts: np.ndarray, bbox: np.ndarray) -> tuple[float, float, float]:
    if kpts is not None and kpts.shape[0] >= 13:
        left_hip = kpts[11]
        right_hip = kpts[12]
        if left_hip[2] > 0.2 and right_hip[2] > 0.2:
            x = float((left_hip[0] + right_hip[0]) / 2.0)
            y = float((left_hip[1] + right_hip[1]) / 2.0)
            score = float((left_hip[2] + right_hip[2]) / 2.0)
            return x, y, score

    x1, y1, x2, y2 = bbox
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0), 0.0


def weighted_midpoint(kpts: np.ndarray, a: int, b: int) -> np.ndarray:
    if a >= len(kpts) or b >= len(kpts):
        return np.array([0.0, 0.0, 0.0], dtype=float)

    pa = kpts[a]
    pb = kpts[b]
    conf = min(float(pa[2]), float(pb[2]))
    if conf <= 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    return np.array([(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, conf], dtype=float)


def extrapolate_endpoint(kpts: np.ndarray, joint_index: int, parent_index: int, scale: float = 0.25) -> np.ndarray:
    if joint_index >= len(kpts) or parent_index >= len(kpts):
        return np.array([0.0, 0.0, 0.0], dtype=float)

    joint = kpts[joint_index]
    parent = kpts[parent_index]
    conf = min(float(joint[2]), float(parent[2]))
    if conf <= 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    direction = joint[:2] - parent[:2]
    point = joint[:2] + direction * scale
    return np.array([point[0], point[1], conf], dtype=float)


def build_enhanced_keypoints(kpts: np.ndarray) -> np.ndarray:
    if kpts is None:
        return np.zeros((len(ENHANCED_KEYPOINT_NAMES), 3), dtype=float)

    enhanced = np.zeros((len(ENHANCED_KEYPOINT_NAMES), 3), dtype=float)
    count = min(len(kpts), len(KEYPOINT_NAMES))
    enhanced[:count] = kpts[:count]

    neck = weighted_midpoint(enhanced, 5, 6)
    mid_hip = weighted_midpoint(enhanced, 11, 12)
    if neck[2] > 0 and mid_hip[2] > 0:
        spine = np.array([(neck[0] + mid_hip[0]) / 2.0, (neck[1] + mid_hip[1]) / 2.0, min(neck[2], mid_hip[2])])
    else:
        spine = np.array([0.0, 0.0, 0.0], dtype=float)

    enhanced[17] = neck
    enhanced[18] = mid_hip
    enhanced[19] = spine
    enhanced[20] = extrapolate_endpoint(enhanced, 9, 7, scale=0.18)
    enhanced[21] = extrapolate_endpoint(enhanced, 10, 8, scale=0.18)
    enhanced[22] = extrapolate_endpoint(enhanced, 15, 13, scale=0.18)
    enhanced[23] = extrapolate_endpoint(enhanced, 16, 14, scale=0.18)
    if neck[2] > 0 and enhanced[0][2] > 0:
        head_vec = enhanced[0, :2] - neck[:2]
        head_len = float(np.linalg.norm(head_vec))
        torso_len = float(np.linalg.norm(neck[:2] - mid_hip[:2])) if mid_hip[2] > 0 else head_len * 2.0
        max_head_len = max(18.0, min(95.0, torso_len * 0.42))
        if head_len > 1e-5:
            head_vec = head_vec / head_len * min(max_head_len, max(12.0, head_len * 0.65))
            enhanced[24] = np.array([neck[0] + head_vec[0] * 1.25, neck[1] + head_vec[1] * 1.25, min(neck[2], enhanced[0][2])], dtype=float)
            enhanced[25] = np.array([neck[0] - head_vec[0] * 0.20, neck[1] - head_vec[1] * 0.20, min(neck[2], enhanced[0][2])], dtype=float)
    else:
        enhanced[24] = extrapolate_endpoint(enhanced, 0, 17, scale=0.30)
        enhanced[25] = extrapolate_endpoint(enhanced, 17, 0, scale=0.12)
    enhanced[26] = extrapolate_endpoint(enhanced, 15, 13, scale=-0.18)
    enhanced[27] = extrapolate_endpoint(enhanced, 16, 14, scale=-0.18)

    for target, ankle_idx, foot_idx, side in ((28, 15, 22, -1.0), (29, 16, 23, 1.0)):
        ankle = enhanced[ankle_idx]
        foot = enhanced[foot_idx]
        conf = min(float(ankle[2]), float(foot[2]))
        if conf <= 0:
            continue
        direction = foot[:2] - ankle[:2]
        length = float(np.linalg.norm(direction))
        if length <= 1e-5:
            continue
        direction = direction / length
        forward_len = max(10.0, min(34.0, length * 0.95))
        side_vec = np.array([side * forward_len * 0.22, 0.0], dtype=float)
        point = ankle[:2] + direction * forward_len + side_vec
        enhanced[target] = np.array([point[0], point[1], conf], dtype=float)
    return enhanced


class PoseSmoother:
    def __init__(
        self,
        alpha: float = 0.35,
        min_confidence: float = 0.2,
        hold_confidence_decay: float = 0.92,
        max_jump_px: float = 80.0,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.min_confidence = min_confidence
        self.hold_confidence_decay = hold_confidence_decay
        self.max_jump_px = max_jump_px
        self._previous: Dict[int, np.ndarray] = {}

    def smooth(self, person_id: int, kpts: np.ndarray) -> np.ndarray:
        current = np.array(kpts, dtype=float, copy=True)
        previous = self._previous.get(person_id)
        if previous is None or previous.shape != current.shape:
            self._previous[person_id] = current
            return current

        smoothed = previous.copy()
        for idx, point in enumerate(current):
            conf = float(point[2])
            if conf < self.min_confidence:
                smoothed[idx, 2] = max(0.0, previous[idx, 2] * self.hold_confidence_decay)
                continue

            prev_xy = previous[idx, :2]
            curr_xy = point[:2]
            distance = float(np.linalg.norm(curr_xy - prev_xy))
            if distance > self.max_jump_px:
                curr_xy = prev_xy + (curr_xy - prev_xy) * (self.max_jump_px / distance)

            confidence_alpha = self.alpha * np.clip(conf / 0.7, 0.35, 1.0)
            smoothed[idx, :2] = previous[idx, :2] * (1.0 - confidence_alpha) + curr_xy * confidence_alpha
            smoothed[idx, 2] = max(conf, previous[idx, 2] * self.hold_confidence_decay)

        self._previous[person_id] = smoothed
        return smoothed


def _contains_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


@lru_cache(maxsize=16)
def _load_text_font(size: int):
    if ImageFont is None:
        return None

    for font_path in CHINESE_FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw text on an OpenCV BGR frame.

    OpenCV's Hershey fonts cannot render Chinese, so non-ASCII text is drawn
    through Pillow with a real system font.
    """
    if not _contains_non_ascii(text) or Image is None:
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    font_size = max(12, int(30 * font_scale))
    font = _load_text_font(font_size)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    return frame


def draw_pose(frame: np.ndarray, kpts: np.ndarray, person_id: int, bbox: np.ndarray, score: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(int).tolist()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)
    draw_text(
        frame,
        f"ID {person_id} {score:.2f}",
        (x1, max(20, y1 - 8)),
        (80, 220, 80),
        0.6,
        2,
    )

    if kpts is None:
        return frame

    enhanced_kpts = build_enhanced_keypoints(kpts)

    for a, b in ENHANCED_SKELETON_EDGES:
        if a < len(enhanced_kpts) and b < len(enhanced_kpts):
            xa, ya, ca = enhanced_kpts[a]
            xb, yb, cb = enhanced_kpts[b]
            if ca > 0.25 and cb > 0.25:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 180, 80), 2)

    for idx, (x, y, c) in enumerate(enhanced_kpts):
        if c > 0.25:
            radius = 4 if idx >= len(KEYPOINT_NAMES) else 3
            color = (120, 230, 255) if idx >= len(KEYPOINT_NAMES) else (80, 160, 255)
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)

    return frame
