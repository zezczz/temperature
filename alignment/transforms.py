"""加载/保存变换，将热成像 warp 到彩色坐标系。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from alignment.config import COLOR_SIZE, THERMAL_SIZE, TRANSFORM_PATH

Mode = Literal["affine", "homography"]


def _as_float32(matrix: list | np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32)


_PREROT_TO_CV = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply_prerot(img: np.ndarray, deg: int) -> np.ndarray:
    """对热图做 0/90/180/270 度预旋转（绕图像中心）。"""
    deg = int(deg) % 360
    code = _PREROT_TO_CV.get(deg)
    return img if code is None else cv2.rotate(img, code)


def prerot_point(x: float, y: float, w: int, h: int, deg: int) -> tuple[float, float]:
    """同 apply_prerot 但作用在点坐标上。w, h 为旋转前的图像宽高。"""
    deg = int(deg) % 360
    if deg == 0:
        return float(x), float(y)
    if deg == 90:  # CW
        return float(h - 1 - y), float(x)
    if deg == 180:
        return float(w - 1 - x), float(h - 1 - y)
    if deg == 270:  # CCW
        return float(y), float(w - 1 - x)
    raise ValueError(f"prerot must be one of 0/90/180/270, got {deg}")


def inv_prerot_point(xr: float, yr: float, w: int, h: int, deg: int) -> tuple[float, float]:
    """prerot_point 的逆：把旋转后图像的坐标 (xr, yr) 反算回原图 (x, y)。
    w, h 仍是旋转前的图像宽高。"""
    deg = int(deg) % 360
    if deg == 0:
        return float(xr), float(yr)
    if deg == 90:
        return float(yr), float(h - 1 - xr)
    if deg == 180:
        return float(w - 1 - xr), float(h - 1 - yr)
    if deg == 270:
        return float(w - 1 - yr), float(xr)
    raise ValueError(f"prerot must be one of 0/90/180/270, got {deg}")


def identity_transform(mode: Mode = "homography") -> dict:
    if mode == "affine":
        matrix = np.eye(2, 3, dtype=np.float32).tolist()
    else:
        matrix = np.eye(3, dtype=np.float32).tolist()
    return {
        "version": 1,
        "mode": mode,
        "src": "thermal",
        "dst": "color",
        "src_size": list(THERMAL_SIZE),
        "dst_size": list(COLOR_SIZE),
        "prerot": 0,
        "matrix": matrix,
    }


def load_transform(path: Path | None = None) -> dict:
    path = path or TRANSFORM_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到变换文件: {path}\n"
            "请先运行: python -m alignment.calibrate  或  python -m alignment.tune"
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data["matrix"] = _as_float32(data["matrix"])
    return data


def save_transform(data: dict, path: Path | None = None) -> Path:
    path = path or TRANSFORM_PATH
    out = dict(data)
    out["matrix"] = np.asarray(out["matrix"], dtype=np.float64).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


def warp_thermal(
    thermal_bgr: np.ndarray,
    transform: dict | None = None,
    dst_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """将热成像帧映射到彩色相机像素坐标。"""
    transform = transform or load_transform()
    w, h = dst_size or tuple(transform.get("dst_size", COLOR_SIZE))
    img = apply_prerot(thermal_bgr, transform.get("prerot", 0))
    mode: Mode = transform["mode"]
    M = _as_float32(transform["matrix"])
    if mode == "affine":
        return cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
    return cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def blend_overlay(color_bgr: np.ndarray, thermal_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(color_bgr, 1.0 - alpha, thermal_bgr, alpha, 0)


def affine_from_params(
    scale_x: float,
    scale_y: float,
    tx: float,
    ty: float,
    deg: float = 0.0,
) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    scale = np.array([[scale_x, 0], [0, scale_y]], dtype=np.float32)
    rs = rot @ scale
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = rs
    M[:, 2] = (tx, ty)
    return M


def homography_from_point_pairs(
    thermal_pts: np.ndarray,
    color_pts: np.ndarray,
) -> np.ndarray:
    if len(thermal_pts) < 4:
        raise ValueError("单应性标定至少需要 4 对对应点")
    H, _ = cv2.findHomography(
        np.asarray(thermal_pts, dtype=np.float32),
        np.asarray(color_pts, dtype=np.float32),
        cv2.RANSAC,
        3.0,
    )
    if H is None:
        raise RuntimeError("无法估计单应矩阵，请检查对应点")
    return H.astype(np.float32)


def map_thermal_point_to_color(
    x: float,
    y: float,
    transform: dict | None = None,
) -> tuple[float, float]:
    """将热成像像素坐标映射到彩色图像坐标。"""
    transform = transform or load_transform()
    src_w, src_h = tuple(transform.get("src_size", THERMAL_SIZE))
    xr, yr = prerot_point(x, y, src_w, src_h, transform.get("prerot", 0))
    pt = np.array([[[xr, yr]]], dtype=np.float32)
    M = _as_float32(transform["matrix"])
    if transform["mode"] == "affine":
        out = cv2.transform(pt, M)
    else:
        out = cv2.perspectiveTransform(pt, M)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def map_color_point_to_thermal(
    x: float,
    y: float,
    transform: dict | None = None,
) -> tuple[float, float]:
    """map_thermal_point_to_color 的反向：把彩色像素坐标映射回热成像坐标。"""
    transform = transform or load_transform()
    src_w, src_h = tuple(transform.get("src_size", THERMAL_SIZE))
    M = _as_float32(transform["matrix"]).astype(np.float64)
    if transform["mode"] == "affine":
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M
    else:
        H = M
    H_inv = np.linalg.inv(H).astype(np.float32)
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H_inv)
    xr, yr = float(out[0, 0, 0]), float(out[0, 0, 1])
    return inv_prerot_point(xr, yr, src_w, src_h, transform.get("prerot", 0))


def affine_from_point_pairs(
    thermal_pts: np.ndarray,
    color_pts: np.ndarray,
) -> np.ndarray:
    if len(thermal_pts) != 3:
        raise ValueError("仿射标定需要恰好 3 对对应点")
    return cv2.getAffineTransform(
        np.asarray(thermal_pts, dtype=np.float32),
        np.asarray(color_pts, dtype=np.float32),
    )
