"""从彩色视频胸腔 ROI 提取起伏运动信号（非亮度）。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from respiration.dlc_io import bodypart_xy_likelihood, load_dlc_csv
from respiration.roi import chest_roi_params_series, crop_aligned_patch


def _smooth_patch(patch: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(patch, (3, 3), 0)


def motion_mad(prev: np.ndarray, curr: np.ndarray) -> float:
    """帧间平均绝对差，反映 ROI 内整体像素变化（起伏、毛发、阴影移动）。"""
    return float(np.mean(np.abs(curr - prev)))


def motion_heave(prev: np.ndarray, curr: np.ndarray) -> float:
    """身体轴对齐后，垂直于体轴方向的位移幅度（俯视起伏）。"""
    shift, _ = cv2.phaseCorrelate(
        prev.astype(np.float64),
        curr.astype(np.float64),
    )
    return float(abs(shift[1]))


def compute_frame_motion(
    prev: np.ndarray | None,
    curr: np.ndarray,
    *,
    metric: str,
) -> float:
    if prev is None:
        return float("nan")
    p = _smooth_patch(prev)
    c = _smooth_patch(curr)
    if metric == "mad":
        return motion_mad(p, c)
    if metric == "heave":
        return motion_heave(p, c)
    if metric == "combo":
        return motion_mad(p, c) + motion_heave(p, c)
    raise ValueError(f"未知 motion metric: {metric}")


def extract_chest_motion(
    video_path: Path,
    dlc_csv: Path,
    out_csv: Path,
    *,
    fps: float | None = None,
    chest_fraction: float,
    roi_width_scale: float,
    roi_length_scale: float,
    p_cutoff: float,
    motion_metric: str,
    patch_width: int,
    patch_height: int,
    max_frames: int | None = None,
) -> Path:
    """逐帧胸腔 ROI 运动强度 → csv（列 motion, valid, …）。"""
    df = load_dlc_csv(dlc_csv)
    lx, ly, pl = bodypart_xy_likelihood(df, "left_eye")
    rx, ry, pr = bodypart_xy_likelihood(df, "right_eye")
    tx, ty, pt = bodypart_xy_likelihood(df, "tail_base")

    cx, cy, hw, hh, ang, valid = chest_roi_params_series(
        lx, ly, rx, ry, tx, ty, pl, pr, pt,
        chest_fraction=chest_fraction,
        roi_width_scale=roi_width_scale,
        roi_length_scale=roi_length_scale,
        p_cutoff=p_cutoff,
    )

    if not video_path.is_file():
        raise FileNotFoundError(f"视频不存在: {video_path.resolve()}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path.resolve()}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    use_fps = fps if fps and fps > 0 else (vid_fps if vid_fps > 0 else 25.0)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(len(lx), n_video)
    if max_frames:
        n = min(n, max_frames)

    motion = np.full(n, np.nan)
    prev_patch: np.ndarray | None = None

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            n = i
            break
        if not valid[i]:
            prev_patch = None
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch = crop_aligned_patch(
            gray, cx[i], cy[i], hw[i], hh[i], ang[i],
            out_w=patch_width, out_h=patch_height,
        )
        if patch is None:
            prev_patch = None
            continue
        motion[i] = compute_frame_motion(prev_patch, patch, metric=motion_metric)
        prev_patch = patch

    cap.release()

    out = pd.DataFrame(
        {
            "motion": motion[:n],
            "valid": valid[:n],
            "cx": cx[:n],
            "cy": cy[:n],
            "half_w": hw[:n],
            "half_h": hh[:n],
            "angle_deg": ang[:n],
        },
        index=pd.RangeIndex(n, name="frame"),
    )
    out["time_s"] = out.index / use_fps
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv)
    meta_path = out_csv.with_suffix(".meta.txt")
    meta_path.write_text(
        f"video={video_path}\n"
        f"dlc_csv={dlc_csv}\n"
        f"fps={use_fps}\n"
        f"frames={n}\n"
        f"signal=motion\n"
        f"motion_metric={motion_metric}\n"
        f"patch_size={patch_width}x{patch_height}\n",
        encoding="utf-8",
    )
    return out_csv


# 兼容旧名
extract_chest_signal = extract_chest_motion
