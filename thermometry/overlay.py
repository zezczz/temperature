"""在热成像视频上叠加 bodypart 圆点 + 温度数字。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_PALETTE = [
    (0, 0, 255),     # red
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
]


def _bodyparts_in(df: pd.DataFrame, suffix: str) -> list[str]:
    return [c[: -len(suffix)] for c in df.columns if c.endswith(suffix)]


def render(
    intensity_csv: Path,
    temperature_csv: Path,
    thermal_video: Path,
    out_video: Path,
    *,
    p_cutoff: float = 0.5,
    dot_radius: int = 6,
    font_scale: float = 0.5,
    max_frames: int | None = None,
) -> Path:
    """在 ``thermal_video`` 上叠加 bodypart 圆点和温度数字，写到 ``out_video``。

    intensity_csv 提供 bodypart 在 thermal 上的 (x, y) 与 likelihood；
    temperature_csv 提供 <bp>_temperature 与可选的 body_temperature。
    """
    intensity_df = pd.read_csv(intensity_csv, index_col=0)
    temp_df = pd.read_csv(temperature_csv, index_col=0)

    cap = cv2.VideoCapture(str(thermal_video))
    if not cap.isOpened():
        raise FileNotFoundError(thermal_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_total, len(intensity_df), len(temp_df))
    if max_frames:
        n = min(n, max_frames)

    bodyparts = _bodyparts_in(intensity_df, "_intensity")

    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    try:
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break

            for k, bp in enumerate(bodyparts):
                hx = intensity_df.get(f"{bp}_hot_x")
                hy = intensity_df.get(f"{bp}_hot_y")
                xc = intensity_df.get(f"{bp}_x")
                yc = intensity_df.get(f"{bp}_y")
                lk = intensity_df.get(f"{bp}_likelihood")
                if hx is not None and hy is not None and not (
                    pd.isna(hx.iloc[i]) or pd.isna(hy.iloc[i])
                ):
                    x = float(hx.iloc[i])
                    y = float(hy.iloc[i])
                elif xc is not None and yc is not None:
                    x = float(xc.iloc[i])
                    y = float(yc.iloc[i])
                else:
                    continue
                if np.isnan(x) or np.isnan(y) or not (0 <= x < w and 0 <= y < h):
                    continue
                if lk is not None and float(lk.iloc[i]) < p_cutoff:
                    continue

                t_series = temp_df.get(f"{bp}_temperature")
                t_val = float(t_series.iloc[i]) if t_series is not None else float("nan")

                color = _PALETTE[k % len(_PALETTE)]
                xi, yi = int(round(x)), int(round(y))
                cv2.circle(frame, (xi, yi), dot_radius, color, -1)

                label = f"{bp}: {t_val:.2f}C" if not np.isnan(t_val) else bp
                cv2.putText(
                    frame,
                    label,
                    (xi + 8, yi - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            if "body_temperature" in temp_df.columns:
                bt = float(temp_df["body_temperature"].iloc[i])
                if not np.isnan(bt):
                    cv2.putText(
                        frame,
                        f"Tbody = {bt:.2f} C",
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            writer.write(frame)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{n}")
    finally:
        cap.release()
        writer.release()

    return out_video
