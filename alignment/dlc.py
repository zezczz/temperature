"""
把彩色视频上跑出来的 DLC 关键点应用到对应的热成像。

三个子命令：
    convert : 坐标转换 color → thermal，输出与原 DLC csv 同结构的新 csv
    sample  : 在 thermal 视频上读取每个关键点位置的像素强度（伪彩亮度，
              作为温度的相对指标），输出宽表 csv
    draw    : 在 thermal 视频上画出反映射后的 DLC 关键点，导出叠加 mp4

用法示例:
    python -m alignment.dlc convert ^
        --csv data/top2026-05-17-14-44-35DLC_..._best-30.csv ^
        --out data/aligned/dlc/top2026-05-17-14-44-35_thermal_coords.csv

    python -m alignment.dlc sample ^
        --csv data/top2026-05-17-14-44-35DLC_..._best-30.csv ^
        --thermal "data/thermal/temp2026-05-17 14-44-35.mp4" ^
        --out data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv ^
        --radius 3

    python -m alignment.dlc draw ^
        --csv data/top2026-05-17-14-44-35DLC_..._best-30.csv ^
        --thermal "data/thermal/temp2026-05-17 14-44-35.mp4" ^
        --out data/aligned/dlc/top2026-05-17-14-44-35_thermal_labeled.mp4 ^
        --p-cutoff 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from alignment.transforms import load_transform, map_color_point_to_thermal


def _load_dlc_csv(path: Path) -> pd.DataFrame:
    """读 DLC 标准 csv（3 级表头：scorer / bodyparts / coords）。"""
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def _bodyparts(df: pd.DataFrame) -> list[tuple[str, str]]:
    """返回 [(bodypart, scorer), ...]，保持列顺序。"""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for scorer, bp, _ in df.columns:
        if bp not in seen:
            seen.add(bp)
            out.append((bp, scorer))
    return out


def _patch_mean(gray: np.ndarray, x: float, y: float, radius: int) -> float:
    h, w = gray.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return float("nan")
    if radius <= 0:
        return float(gray[yi, xi])
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    patch = gray[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else float("nan")


def _hot_patch(
    gray: np.ndarray,
    mean_map: np.ndarray | None,
    x: float,
    y: float,
    radius: int,
    search_radius: int,
    aggregator: str,
) -> tuple[float, float, float]:
    """在 (x, y) 附近 ±search_radius 的窗口里，找一个 (2*radius+1) 邻域均值最高
    （或 p95 / mean）的位置，返回 (intensity, hot_x, hot_y)。

    - search_radius <= 0  → 退化为标点中心 (2r+1)×(2r+1) 邻域均值（兼容旧行为）。
    - aggregator == "max" → 默认，取搜索窗内最高的 7×7 均值，并记录该位置。
    - aggregator == "p95" → 取搜索窗内 7×7 均值的 95 分位。
    - aggregator == "mean"→ 取搜索窗内全部 7×7 均值的平均（≈ 大块平均，最稳但偏低）。

    `mean_map` 应为 gray 的 (2r+1)×(2r+1) 邻域均值图（外部用 cv2.boxFilter 算好，
    一帧只算一次以摊销多个 bodypart 的成本）。
    """
    h, w = gray.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return float("nan"), float("nan"), float("nan")
    if search_radius <= 0 or mean_map is None:
        val = _patch_mean(gray, x, y, radius)
        return val, float(xi), float(yi)

    sx0, sx1 = max(0, xi - search_radius), min(w, xi + search_radius + 1)
    sy0, sy1 = max(0, yi - search_radius), min(h, yi + search_radius + 1)
    sub = mean_map[sy0:sy1, sx0:sx1]
    if sub.size == 0:
        return float("nan"), float("nan"), float("nan")
    if aggregator == "mean":
        return float(sub.mean()), float(xi), float(yi)
    if aggregator == "p95":
        return float(np.quantile(sub, 0.95)), float(xi), float(yi)
    idx = int(np.argmax(sub))
    yy, xx = divmod(idx, sub.shape[1])
    return float(sub[yy, xx]), float(sx0 + xx), float(sy0 + yy)


def cmd_convert(args: argparse.Namespace) -> None:
    df = _load_dlc_csv(args.csv)
    tform = load_transform()
    out = df.copy()
    for bp, scorer in _bodyparts(df):
        xs = df[(scorer, bp, "x")].to_numpy(dtype=float)
        ys = df[(scorer, bp, "y")].to_numpy(dtype=float)
        new_x = np.full_like(xs, np.nan)
        new_y = np.full_like(ys, np.nan)
        valid = ~(np.isnan(xs) | np.isnan(ys))
        for i in np.where(valid)[0]:
            xt, yt = map_color_point_to_thermal(float(xs[i]), float(ys[i]), tform)
            new_x[i] = xt
            new_y[i] = yt
        out[(scorer, bp, "x")] = new_x
        out[(scorer, bp, "y")] = new_y
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out)
    print(f"已写入 thermal 坐标 csv: {args.out}  (帧数 {len(out)})")


def cmd_sample(args: argparse.Namespace) -> None:
    df = _load_dlc_csv(args.csv)
    tform = load_transform()
    cap = cv2.VideoCapture(str(args.thermal))
    if not cap.isOpened():
        raise FileNotFoundError(args.thermal)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_video, len(df))
    if args.max_frames:
        n = min(n, args.max_frames)
    bp_list = _bodyparts(df)
    kernel = 2 * args.radius + 1
    rows: list[dict] = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.search_radius > 0:
            mean_map = cv2.boxFilter(
                gray, ddepth=cv2.CV_32F, ksize=(kernel, kernel), normalize=True
            )
        else:
            mean_map = None
        row: dict = {"frame": i}
        for bp, scorer in bp_list:
            xc = float(df[(scorer, bp, "x")].iloc[i])
            yc = float(df[(scorer, bp, "y")].iloc[i])
            lk = float(df[(scorer, bp, "likelihood")].iloc[i])
            if np.isnan(xc) or np.isnan(yc) or (args.p_cutoff and lk < args.p_cutoff):
                xt = yt = val = hx = hy = float("nan")
            else:
                xt, yt = map_color_point_to_thermal(xc, yc, tform)
                val, hx, hy = _hot_patch(
                    gray,
                    mean_map,
                    xt,
                    yt,
                    args.radius,
                    args.search_radius,
                    args.patch_aggregator,
                )
            row[f"{bp}_x"] = xt
            row[f"{bp}_y"] = yt
            row[f"{bp}_hot_x"] = hx
            row[f"{bp}_hot_y"] = hy
            row[f"{bp}_intensity"] = val
            row[f"{bp}_likelihood"] = lk
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    cap.release()
    out_df = pd.DataFrame(rows).set_index("frame")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out)
    print(f"已写入强度 csv: {args.out}  (帧数 {len(out_df)})")


_PALETTE = [
    (0, 0, 255),     # red
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
]


def cmd_draw(args: argparse.Namespace) -> None:
    df = _load_dlc_csv(args.csv)
    tform = load_transform()
    cap = cv2.VideoCapture(str(args.thermal))
    if not cap.isOpened():
        raise FileNotFoundError(args.thermal)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_video, len(df))
    if args.max_frames:
        n = min(n, args.max_frames)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bp_list = _bodyparts(df)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        for k, (bp, scorer) in enumerate(bp_list):
            xc = float(df[(scorer, bp, "x")].iloc[i])
            yc = float(df[(scorer, bp, "y")].iloc[i])
            lk = float(df[(scorer, bp, "likelihood")].iloc[i])
            if np.isnan(xc) or np.isnan(yc) or lk < args.p_cutoff:
                continue
            xt, yt = map_color_point_to_thermal(xc, yc, tform)
            if not (0 <= xt < w and 0 <= yt < h):
                continue
            color = _PALETTE[k % len(_PALETTE)]
            cv2.circle(frame, (int(round(xt)), int(round(yt))), args.dot_radius, color, -1)
            cv2.putText(
                frame,
                bp,
                (int(round(xt)) + 8, int(round(yt)) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        writer.write(frame)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    cap.release()
    writer.release()
    print(f"已写入叠加视频: {args.out}  ({n} 帧)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="把 DLC 在彩色画面上的关键点应用到热成像")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("convert", help="坐标转换 color→thermal，输出新 DLC csv")
    s.add_argument("--csv", required=True, type=Path, help="DLC 在彩色视频上的输出 csv")
    s.add_argument("--out", required=True, type=Path, help="目标 csv 路径")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("sample", help="按 DLC 关键点位置在 thermal 上取像素亮度")
    s.add_argument("--csv", required=True, type=Path)
    s.add_argument("--thermal", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--radius", type=int, default=3,
                   help="邻域窗口半径，最终 (2r+1)×(2r+1)；默认 3 → 7×7")
    s.add_argument("--search-radius", type=int, default=8,
                   help="在反映射点附近 ±N px 内搜索最佳位置；0 = 仅用中心点（旧行为）")
    s.add_argument("--patch-aggregator", choices=["max", "p95", "mean"], default="max",
                   help="搜索窗内的聚合方式：max（最热的 7×7 均值，推荐）/ p95 / mean")
    s.add_argument("--p-cutoff", type=float, default=0.0,
                   help="低于该置信度的点记 NaN")
    s.add_argument("--max-frames", type=int, default=None)
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("draw", help="在 thermal 视频上画 DLC 关键点叠加")
    s.add_argument("--csv", required=True, type=Path)
    s.add_argument("--thermal", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--p-cutoff", type=float, default=0.5)
    s.add_argument("--dot-radius", type=int, default=6)
    s.add_argument("--max-frames", type=int, default=None)
    s.set_defaults(func=cmd_draw)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
