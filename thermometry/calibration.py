"""伪彩亮度 (intensity, 0~255) 与实际温度 (°C) 之间的标定。

支持两类映射：

    linear     :  T = a * I + b           （最少 2 个 anchor）
    piecewise  :  分段线性插值（np.interp）   （任意 ≥2 个 anchor）

标定结果保存为 JSON，便于后续在 ``compute.apply_to_dataframe`` 中复用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np

Array = Union[float, np.ndarray]


@dataclass
class LinearCalibration:
    """T = a * I + b。"""

    a: float
    b: float

    def apply(self, intensity: Array) -> Array:
        arr = np.asarray(intensity, dtype=float)
        out = arr * self.a + self.b
        return out

    def describe(self) -> str:
        return f"linear: T = {self.a:.6f} * I + {self.b:.6f}"


@dataclass
class PiecewiseCalibration:
    """单调分段线性插值。

    端点之外的值会被 ``np.interp`` 截断到端点温度（保守做法，避免外推爆炸）。
    """

    intensities: list[float]
    temperatures: list[float]

    def __post_init__(self) -> None:
        if len(self.intensities) != len(self.temperatures):
            raise ValueError("intensities 与 temperatures 长度必须一致")
        order = np.argsort(np.asarray(self.intensities, dtype=float))
        self.intensities = list(np.asarray(self.intensities, dtype=float)[order])
        self.temperatures = list(np.asarray(self.temperatures, dtype=float)[order])

    def apply(self, intensity: Array) -> Array:
        arr = np.asarray(intensity, dtype=float)
        return np.interp(
            arr,
            self.intensities,
            self.temperatures,
            left=self.temperatures[0],
            right=self.temperatures[-1],
        )

    def describe(self) -> str:
        n = len(self.intensities)
        rng = f"I ∈ [{self.intensities[0]:.1f}, {self.intensities[-1]:.1f}]"
        return f"piecewise: {n} anchors, {rng}"


Calibration = Union[LinearCalibration, PiecewiseCalibration]


def fit_linear(anchors: Sequence[tuple[float, float]]) -> LinearCalibration:
    """最小二乘拟合 T = a * I + b。"""
    arr = np.asarray(anchors, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("anchors 形状应为 (N, 2)：[(intensity, temperature), ...]")
    if len(arr) < 2:
        raise ValueError("线性拟合至少需要 2 个 anchor")
    xs, ys = arr[:, 0], arr[:, 1]
    if np.allclose(xs.max() - xs.min(), 0):
        raise ValueError("anchor 的 intensity 全相同，无法拟合斜率")
    a, b = np.polyfit(xs, ys, 1)
    return LinearCalibration(float(a), float(b))


def fit_piecewise(anchors: Sequence[tuple[float, float]]) -> PiecewiseCalibration:
    """直接把所有 anchor 作为插值节点。"""
    arr = np.asarray(anchors, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("anchors 形状应为 (N, 2)：[(intensity, temperature), ...]")
    if len(arr) < 2:
        raise ValueError("分段插值至少需要 2 个 anchor")
    return PiecewiseCalibration(arr[:, 0].tolist(), arr[:, 1].tolist())


def fit(
    anchors: Sequence[tuple[float, float]],
    mode: str = "linear",
) -> Calibration:
    if mode == "linear":
        return fit_linear(anchors)
    if mode == "piecewise":
        return fit_piecewise(anchors)
    raise ValueError(f"未知模式: {mode}（可选 linear / piecewise）")


def load(path: Path) -> Calibration:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"未找到 calibration: {path}\n"
            "请先 `python -m thermometry init` 或 `python -m thermometry fit`"
        )
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    mode = data.get("mode", "linear")
    if mode == "linear":
        d = data["linear"]
        return LinearCalibration(float(d["a"]), float(d["b"]))
    if mode == "piecewise":
        d = data["piecewise"]
        return PiecewiseCalibration(list(d["intensities"]), list(d["temperatures"]))
    raise ValueError(f"未知 calibration mode: {mode}")


def save(
    calibration: Calibration,
    path: Path,
    *,
    extra: dict | None = None,
) -> Path:
    if isinstance(calibration, LinearCalibration):
        data: dict = {
            "version": 1,
            "mode": "linear",
            "linear": {"a": float(calibration.a), "b": float(calibration.b)},
        }
    elif isinstance(calibration, PiecewiseCalibration):
        data = {
            "version": 1,
            "mode": "piecewise",
            "piecewise": {
                "intensities": [float(x) for x in calibration.intensities],
                "temperatures": [float(x) for x in calibration.temperatures],
            },
        }
    else:
        raise TypeError(f"不支持的 calibration 类型: {type(calibration)!r}")
    if extra:
        data["meta"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_anchors(path: Path) -> tuple[list[tuple[float, float]], str | None]:
    """读 anchors.json，返回 [(intensity, temperature), ...] 与可选的 mode。"""
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("anchors", [])
    anchors: list[tuple[float, float]] = []
    for a in raw:
        anchors.append((float(a["intensity"]), float(a["temperature"])))
    return anchors, data.get("mode")
