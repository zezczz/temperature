"""由双眼与尾根估计胸腔旋转矩形 ROI。"""
from __future__ import annotations

import cv2
import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0])
    return v / n


def chest_roi_params(
    lx: float,
    ly: float,
    rx: float,
    ry: float,
    tx: float,
    ty: float,
    *,
    chest_fraction: float,
    roi_width_scale: float,
    roi_length_scale: float,
) -> tuple[float, float, float, float, float] | None:
    """返回 (cx, cy, half_w, half_h, angle_deg)。

    half_w 沿垂直于身体轴；half_h 沿身体轴（头→尾）。angle 为 OpenCV 旋转角（度）。
    """
    eye_mid = np.array([(lx + rx) * 0.5, (ly + ry) * 0.5], dtype=float)
    inter_eye = float(np.hypot(rx - lx, ry - ly))
    if inter_eye < 1e-3:
        return None

    tail = np.array([tx, ty], dtype=float)
    body = tail - eye_mid
    body_len = float(np.linalg.norm(body))
    if body_len < 1e-3:
        return None

    u = _unit(body)
    center = eye_mid + chest_fraction * body
    half_w = max(4.0, roi_width_scale * inter_eye * 0.5)
    half_h = max(4.0, roi_length_scale * body_len * 0.5)
    angle_deg = float(np.degrees(np.arctan2(u[1], u[0])))
    return float(center[0]), float(center[1]), half_w, half_h, angle_deg


def chest_roi_params_series(
    lx: np.ndarray,
    ly: np.ndarray,
    rx: np.ndarray,
    ry: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    pl: np.ndarray,
    pr: np.ndarray,
    pt: np.ndarray,
    *,
    chest_fraction: float,
    roi_width_scale: float,
    roi_length_scale: float,
    p_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """逐帧 ROI；无效帧对应 valid=False。"""
    n = len(lx)
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    hw = np.full(n, np.nan)
    hh = np.full(n, np.nan)
    ang = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    ok = (pl >= p_cutoff) & (pr >= p_cutoff) & (pt >= p_cutoff)
    for i in np.where(ok)[0]:
        params = chest_roi_params(
            lx[i], ly[i], rx[i], ry[i], tx[i], ty[i],
            chest_fraction=chest_fraction,
            roi_width_scale=roi_width_scale,
            roi_length_scale=roi_length_scale,
        )
        if params is None:
            continue
        cx[i], cy[i], hw[i], hh[i], ang[i] = params
        valid[i] = True
    return cx, cy, hw, hh, ang, valid


def crop_aligned_patch(
    gray: np.ndarray,
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
    angle_deg: float,
    *,
    out_w: int,
    out_h: int,
) -> np.ndarray | None:
    """旋转使身体轴沿水平方向，裁切胸腔区并缩放到固定尺寸（float32）。

    旋转后：x 沿头→尾，y 垂直于身体轴（俯视时起伏主要体现在 y 方向）。
    """
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    warped = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    x0 = int(round(cx - half_h))
    x1 = int(round(cx + half_h))
    y0 = int(round(cy - half_w))
    y1 = int(round(cy + half_w))
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = warped[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)
