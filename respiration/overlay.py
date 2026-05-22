"""在彩色视频上叠加胸腔 ROI 与呼吸频率。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from respiration.plot import load_summary, resolve_result_paths


def _interp_bpm_at_times(inst_df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    if inst_df.empty or "time_s" not in inst_df.columns:
        return np.full(len(times), np.nan)
    t0 = inst_df["time_s"].to_numpy(dtype=float)
    bpm = inst_df["breaths_per_min"].to_numpy(dtype=float)
    return np.interp(times, t0, bpm, left=np.nan, right=np.nan)


def _draw_rotated_roi(
    frame: np.ndarray,
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
    angle_deg: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(half_w) and np.isfinite(half_h)):
        return
    rect = ((float(cx), float(cy)), (float(2 * half_h), float(2 * half_w)), float(angle_deg))
    box = cv2.boxPoints(rect)
    cv2.polylines(frame, [np.int32(box)], isClosed=True, color=color, thickness=thickness)


def render_overlay_video(
    video_path: Path,
    signal_csv: Path,
    out_video: Path,
    *,
    instant_rate_csv: Path | None = None,
    summary_txt: Path | None = None,
    global_bpm: float | None = None,
    max_frames: int | None = None,
    show_instant: bool = True,
) -> Path:
    """彩色 mp4 上画胸腔 ROI 框 + 全段/瞬时呼吸率文字。"""
    motion_df = pd.read_csv(signal_csv, index_col=0)
    inst_df = pd.DataFrame()
    if instant_rate_csv and Path(instant_rate_csv).is_file():
        inst_df = pd.read_csv(instant_rate_csv, index_col=0)

    summary = load_summary(summary_txt) if summary_txt and Path(summary_txt).is_file() else {}
    if global_bpm is None and "global_breaths_per_min" in summary:
        try:
            global_bpm = float(summary["global_breaths_per_min"])
        except ValueError:
            global_bpm = None

    meta = signal_csv.with_suffix(".meta.txt")
    fps = 25.0
    if meta.is_file():
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.startswith("fps="):
                fps = float(line.split("=", 1)[1])
                break

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_total, len(motion_df))
    if max_frames:
        n = min(n, max_frames)

    times = np.arange(n, dtype=float) / vid_fps
    bpm_series = _interp_bpm_at_times(inst_df, times) if show_instant else np.full(n, np.nan)

    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        vid_fps,
        (w, h),
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break

        row = motion_df.iloc[i]
        if bool(row.get("valid", False)):
            _draw_rotated_roi(
                frame,
                float(row["cx"]),
                float(row["cy"]),
                float(row["half_w"]),
                float(row["half_h"]),
                float(row["angle_deg"]),
                color=(0, 255, 0),
                thickness=2,
            )
        else:
            cv2.putText(
                frame, "ROI invalid", (12, 28), font, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
            )

        y = 28
        if global_bpm is not None and np.isfinite(global_bpm):
            cv2.putText(
                frame,
                f"Resp global: {global_bpm:.1f} bpm",
                (12, y),
                font,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 26
        if show_instant and np.isfinite(bpm_series[i]):
            cv2.putText(
                frame,
                f"Resp now: {bpm_series[i]:.1f} bpm",
                (12, y),
                font,
                0.65,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        if (i + 1) % 500 == 0:
            print(f"  叠加进度 {i + 1}/{n}")

    cap.release()
    writer.release()
    return out_video


def overlay_from_saved(
    *,
    stamp: str | None = None,
    prefix: Path | None = None,
    video: Path | None = None,
    out_video: Path | None = None,
    max_frames: int | None = None,
) -> Path:
    paths = resolve_result_paths(stamp=stamp, prefix=prefix)
    meta = paths["meta"]
    video_path = video
    if video_path is None and meta.is_file():
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.startswith("video="):
                video_path = Path(line.split("=", 1)[1].strip())
                if not video_path.is_absolute():
                    from respiration.config import PROJECT_ROOT
                    video_path = PROJECT_ROOT / video_path
                break
    if video_path is None or not Path(video_path).is_file():
        raise FileNotFoundError("未找到彩色视频，请用 --video 指定")

    if out_video is None:
        from respiration.config import OVERLAY_DIR
        out_video = OVERLAY_DIR / f"{paths['base'].name}_respiration_overlay.mp4"

    return render_overlay_video(
        Path(video_path),
        paths["signal"],
        out_video,
        instant_rate_csv=paths["instant"],
        summary_txt=paths["summary"],
        max_frames=max_frames,
    )
