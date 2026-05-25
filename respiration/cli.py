"""小鼠呼吸频率测量 CLI。

流程：彩色视频 + DLC csv → 胸腔 ROI 起伏运动 → 带通滤波 → FFT

示例:
    python -m respiration list
    python -m respiration run --stamp 2026-05-17-14-44-35 --motion-metric heave --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from respiration import config as cfg
from respiration.analyze import analyze_signal_csv, save_analysis
from respiration.extract import extract_chest_motion
from respiration.pairing import list_pairs, resolve_pair
from respiration.overlay import overlay_from_saved
from respiration.plot import plot_from_saved, plot_respiration


def _default_signal_path(video: Path) -> Path:
    return cfg.SIGNAL_DIR / f"{video.stem}_chest_motion.csv"


def _default_prefix(signal_csv: Path) -> Path:
    for suffix in ("_chest_motion.csv", "_chest_signal.csv"):
        if signal_csv.name.endswith(suffix):
            return cfg.RATE_DIR / signal_csv.name[: -len(suffix)]
    return cfg.RATE_DIR / signal_csv.stem


def cmd_list(_args: argparse.Namespace) -> None:
    pairs = list_pairs()
    if not pairs:
        print(f"未找到可配对条目。请确认 {cfg.COLOR_DIR} 有 top*.mp4，"
              f"{cfg.DLC_RESULTS_DIR} 有 top*DLC_*.csv")
        return
    print(f"共 {len(pairs)} 组（stamp / 视频 / DLC csv）:\n")
    for stamp, video, dlc in pairs:
        print(f"  {stamp}")
        print(f"    video: {video}")
        print(f"    dlc:   {dlc.name}\n")
    print("运行示例:")
    print(f"  python -m respiration run --stamp {pairs[0][0]} --motion-metric heave --plot")


def cmd_extract(args: argparse.Namespace) -> None:
    video, dlc_csv, _ = resolve_pair(
        stamp=args.stamp, video=args.video, dlc_csv=args.dlc_csv,
    )
    out = Path(args.out) if args.out else _default_signal_path(video)
    path = extract_chest_motion(
        video,
        dlc_csv,
        out,
        fps=args.fps,
        chest_fraction=args.chest_fraction,
        roi_width_scale=args.roi_width_scale,
        roi_length_scale=args.roi_length_scale,
        p_cutoff=args.p_cutoff,
        motion_metric=args.motion_metric,
        patch_width=args.patch_width,
        patch_height=args.patch_height,
        max_frames=args.max_frames,
    )
    print(f"胸腔运动序列已写入: {path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    signal_csv = Path(args.signal)
    fps = args.fps
    if fps is None or fps <= 0:
        meta = signal_csv.with_suffix(".meta.txt")
        if meta.is_file():
            for line in meta.read_text(encoding="utf-8").splitlines():
                if line.startswith("fps="):
                    fps = float(line.split("=", 1)[1])
                    break
    if fps is None or fps <= 0:
        fps = cfg.DEFAULT_FPS

    result = analyze_signal_csv(
        signal_csv,
        fps=fps,
        f_min=args.f_min,
        f_max=args.f_max,
        filter_order=args.filter_order,
        fft_window_sec=args.fft_window_sec,
        fft_hop_sec=args.fft_hop_sec,
    )
    prefix = Path(args.out_prefix) if args.out_prefix else _default_prefix(signal_csv)
    rate_path, filtered_path, summary_path = save_analysis(result, signal_csv, prefix)
    print(f"全段呼吸率: {result.global_bpm:.2f} 次/分 ({result.global_peak_hz:.3f} Hz)")
    if np.isfinite(result.median_inst_bpm):
        print(f"瞬时率中位数: {result.median_inst_bpm:.2f} 次/分")
    print(f"有效 ROI 帧占比: {result.valid_frac * 100:.1f}%")
    print(f"瞬时率: {rate_path}")
    print(f"滤波信号: {filtered_path}")
    print(f"摘要: {summary_path}")

    if args.plot:
        png = plot_from_saved(signal_csv=signal_csv, out_png=cfg.PLOT_DIR / f"{prefix.name}_respiration.png", show=args.show)
        print(f"图: {png}")


def cmd_plot(args: argparse.Namespace) -> None:
    png = plot_from_saved(
        stamp=args.stamp,
        prefix=args.prefix,
        signal_csv=Path(args.signal) if args.signal else None,
        out_png=Path(args.out) if args.out else None,
        show=args.show,
    )
    print(f"已保存图表: {png}")
    if args.video_overlay:
        mp4 = overlay_from_saved(
            stamp=args.stamp,
            prefix=args.prefix,
            video=Path(args.video) if args.video else None,
            out_video=Path(args.overlay_out) if args.overlay_out else None,
            max_frames=args.max_frames,
        )
        print(f"已保存叠加视频: {mp4}")


def cmd_run(args: argparse.Namespace) -> None:
    video, dlc_csv, stamp = resolve_pair(
        stamp=args.stamp, video=args.video, dlc_csv=args.dlc_csv,
    )
    print(f"stamp={stamp}")
    print(f"  video: {video}")
    print(f"  dlc:   {dlc_csv}")
    signal_csv = Path(args.out) if args.out else _default_signal_path(video)
    extract_chest_motion(
        video,
        dlc_csv,
        signal_csv,
        fps=args.fps,
        chest_fraction=args.chest_fraction,
        roi_width_scale=args.roi_width_scale,
        roi_length_scale=args.roi_length_scale,
        p_cutoff=args.p_cutoff,
        motion_metric=args.motion_metric,
        patch_width=args.patch_width,
        patch_height=args.patch_height,
        max_frames=args.max_frames,
    )
    args.signal = str(signal_csv)
    args.out_prefix = None
    cmd_analyze(args)
    if getattr(args, "video_overlay", False):
        mp4 = overlay_from_saved(stamp=stamp, video=video, max_frames=args.max_frames)
        print(f"叠加视频: {mp4}")


def _add_roi_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chest-fraction", type=float, default=cfg.DEFAULT_CHEST_FRACTION,
                   help="胸腔中心沿 眼中点→尾根 轴的位置 (0–1)")
    p.add_argument("--roi-width-scale", type=float, default=cfg.DEFAULT_ROI_WIDTH_SCALE)
    p.add_argument("--roi-length-scale", type=float, default=cfg.DEFAULT_ROI_LENGTH_SCALE)
    p.add_argument("--p-cutoff", type=float, default=cfg.DEFAULT_P_CUTOFF)


def _add_motion_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--motion-metric",
        choices=["mad", "heave", "combo"],
        default=cfg.DEFAULT_MOTION_METRIC,
        help="mad=帧间绝对差; heave=垂直体轴位移; combo=两者之和",
    )
    p.add_argument("--patch-width", type=int, default=cfg.MOTION_PATCH_WIDTH)
    p.add_argument("--patch-height", type=int, default=cfg.MOTION_PATCH_HEIGHT)


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fps", type=float, default=None, help="帧率；省略则从 .meta.txt 或视频读取")
    p.add_argument(
        "--f-min",
        type=float,
        default=cfg.DEFAULT_F_MIN_HZ,
        help=f"带通/FFT 下限 (Hz)，默认 {cfg.DEFAULT_F_MIN_HZ:.3f} ≈ {cfg.DEFAULT_BPM_MIN} 次/分",
    )
    p.add_argument(
        "--f-max",
        type=float,
        default=cfg.DEFAULT_F_MAX_HZ,
        help=f"带通/FFT 上限 (Hz)，默认 {cfg.DEFAULT_F_MAX_HZ:.3f} ≈ {cfg.DEFAULT_BPM_MAX} 次/分",
    )
    p.add_argument("--filter-order", type=int, default=cfg.DEFAULT_FILTER_ORDER)
    p.add_argument("--fft-window-sec", type=float, default=cfg.DEFAULT_FFT_WINDOW_SEC)
    p.add_argument("--fft-hop-sec", type=float, default=cfg.DEFAULT_FFT_HOP_SEC)
    p.add_argument("--out-prefix", type=Path, default=None)
    p.add_argument("--plot", action="store_true", help="分析后保存 PNG")
    p.add_argument("--show", action="store_true")


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--stamp",
        type=str,
        default=None,
        help="时间戳，如 2026-05-17-14-44-35（自动匹配 top*.mp4 与 top*DLC_*.csv）",
    )
    p.add_argument("--video", type=Path, default=None, help="彩色视频；与 --stamp 二选一")
    p.add_argument("--dlc-csv", type=Path, default=None, help="DLC 推理 csv；与 --stamp 二选一")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="小鼠胸腔 ROI 起伏运动 → FFT 呼吸频率")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="列出可配对的视频与 DLC csv")
    ls.set_defaults(func=cmd_list)

    ex = sub.add_parser("extract", help="从视频提取胸腔起伏运动序列")
    _add_input_args(ex)
    ex.add_argument("--out", type=Path, default=None)
    ex.add_argument("--fps", type=float, default=None)
    ex.add_argument("--max-frames", type=int, default=None)
    _add_roi_args(ex)
    _add_motion_args(ex)
    ex.set_defaults(func=cmd_extract)

    an = sub.add_parser("analyze", help="对已提取运动序列做滤波与 FFT")
    an.add_argument("--signal", type=Path, required=True)
    _add_analyze_args(an)
    an.set_defaults(func=cmd_analyze)

    rn = sub.add_parser("run", help="extract + analyze 一步完成")
    _add_input_args(rn)
    rn.add_argument("--out", type=Path, default=None)
    rn.add_argument("--max-frames", type=int, default=None)
    rn.add_argument("--video-overlay", action="store_true", help="分析后额外生成 ROI+呼吸率叠加 mp4")
    _add_roi_args(rn)
    _add_motion_args(rn)
    _add_analyze_args(rn)
    rn.set_defaults(func=cmd_run)

    pl = sub.add_parser("plot", help="从已保存结果生成 PNG（可选叠加视频）")
    pl.add_argument("--stamp", type=str, default=None)
    pl.add_argument("--prefix", type=Path, default=None)
    pl.add_argument("--signal", type=Path, default=None, help="*_chest_motion.csv 路径")
    pl.add_argument("--out", type=Path, default=None, help="PNG 输出路径")
    pl.add_argument("--video", type=Path, default=None, help="彩色视频（叠加时用）")
    pl.add_argument("--overlay-out", type=Path, default=None)
    pl.add_argument("--video-overlay", action="store_true", help="同时生成叠加 mp4")
    pl.add_argument("--max-frames", type=int, default=None, help="叠加视频最多处理帧数")
    pl.add_argument("--show", action="store_true", help="弹出 matplotlib 窗口")
    pl.set_defaults(func=cmd_plot)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
