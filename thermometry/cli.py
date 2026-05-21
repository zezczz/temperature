"""thermometry 命令行入口。

子命令一览：

    init       生成占位 calibration.json（线性 T = 0.05*I + 32，仅供联调）
    measure    在 intensity csv 中读出某段时间、某 bodypart 的 intensity 描述统计
    fit        读 anchors.json，拟合并写入 calibration.json
    apply      单个 *_thermal_intensity.csv -> 温度 csv
    batch      扫描 INTENSITY_DIR 下所有 *_thermal_intensity.csv 批量转
    summary    汇总每段视频的体温统计
    overlay    在 thermal 视频上叠加 bodypart 圆点 + 温度数字

例：
    python -m thermometry init
    python -m thermometry measure --csv data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv --bodypart tail_base
    python -m thermometry fit
    python -m thermometry apply --csv data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv
    python -m thermometry batch
    python -m thermometry summary
    python -m thermometry overlay ^
        --intensity   data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv ^
        --temperature data/aligned/dlc/top2026-05-17-14-44-35_temperature.csv ^
        --thermal     "data/thermal/temp2026-05-17 14-44-35.mp4"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from thermometry import calibration as cal
from thermometry import compute, overlay, plot, summary
from thermometry.config import (
    ANCHORS_PATH,
    CALIBRATION_PATH,
    DEFAULT_AGGREGATION,
    DEFAULT_P_CUTOFF,
    DEFAULT_QUANTILE,
    INTENSITY_DIR,
    OVERLAY_DIR,
    SUMMARY_DIR,
    TEMPERATURE_DIR,
)


def _add_apply_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--calibration", type=Path, default=None,
                    help=f"默认 {CALIBRATION_PATH}")
    sp.add_argument("--bodyparts", nargs="*", default=None,
                    help="参与体温估计的 bodypart 列表，省略表示全部")
    sp.add_argument("--aggregator", default=DEFAULT_AGGREGATION,
                    choices=["max", "mean", "median", "quantile"],
                    help="多 bodypart 聚合方式")
    sp.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE,
                    help="仅在 --aggregator quantile 时使用")
    sp.add_argument("--p-cutoff", type=float, default=DEFAULT_P_CUTOFF,
                    help="低于该 likelihood 的 bodypart 单帧记 NaN")
    sp.add_argument("--ffill", action="store_true",
                    help="body_temperature 全 NaN 帧用上一帧值前向填充")


def cmd_init(args: argparse.Namespace) -> None:
    if CALIBRATION_PATH.exists() and not args.force:
        print(f"已存在: {CALIBRATION_PATH}（--force 覆盖）")
        return
    model = cal.LinearCalibration(a=0.05, b=32.0)
    cal.save(
        model,
        CALIBRATION_PATH,
        extra={
            "note": (
                "占位标定 T = 0.05*I + 32。仅用于联调；"
                "请用 anchors.json + `python -m thermometry fit` 重新标定。"
            ),
        },
    )
    print(f"已写入: {CALIBRATION_PATH}")
    print(model.describe())


def cmd_measure(args: argparse.Namespace) -> None:
    stats = compute.measure_bodypart(
        args.csv,
        args.bodypart,
        start=args.start,
        end=args.end,
        p_cutoff=args.p_cutoff,
    )
    print(f"file     : {args.csv}")
    print(f"bodypart : {stats['bodypart']}")
    print(f"frames   : {stats['n_frames']}")
    if stats["n_frames"] == 0:
        print("（区间内没有有效帧）")
        return
    print(f"mean     : {stats['mean']:.3f}")
    print(f"median   : {stats['median']:.3f}")
    print(f"std      : {stats['std']:.3f}")
    print(f"min..max : {stats['min']:.3f} .. {stats['max']:.3f}")
    print(f"p10..p90 : {stats['p10']:.3f} .. {stats['p90']:.3f}")


def cmd_fit(args: argparse.Namespace) -> None:
    src = args.anchors or ANCHORS_PATH
    if not src.is_file():
        raise FileNotFoundError(
            f"未找到 anchors 文件: {src}\n"
            "请参考 thermometry/anchors.example.json 编写一份"
        )
    anchors, json_mode = cal.load_anchors(src)
    if not anchors:
        raise ValueError(f"{src} 中没有 anchor")
    mode = args.mode or json_mode or "linear"
    model = cal.fit(anchors, mode=mode)
    print(model.describe())
    if mode == "linear":
        # 给一些点上看看拟合残差
        preds = [model.apply(x) for x, _ in anchors]
        for (x, y), p in zip(anchors, preds):
            print(f"  I={x:8.3f}  T_true={y:6.3f}  T_pred={float(p):6.3f}  Δ={float(p) - y:+.3f}")
    out = args.out or CALIBRATION_PATH
    cal.save(
        model,
        out,
        extra={
            "anchors_file": str(src),
            "n_anchors": len(anchors),
            "fit_mode": mode,
        },
    )
    print(f"已写入: {out}")


def cmd_apply(args: argparse.Namespace) -> None:
    out = compute.process_csv(
        args.csv,
        args.out,
        args.calibration,
        bodyparts=args.bodyparts if args.bodyparts else None,
        aggregator=args.aggregator,
        quantile=args.quantile,
        p_cutoff=args.p_cutoff,
        ffill_body=args.ffill,
    )
    print(f"已写入: {out}")


def cmd_batch(args: argparse.Namespace) -> None:
    outs = compute.batch(
        intensity_dir=args.intensity_dir,
        out_dir=args.out_dir,
        calibration_path=args.calibration,
        bodyparts=args.bodyparts if args.bodyparts else None,
        aggregator=args.aggregator,
        quantile=args.quantile,
        p_cutoff=args.p_cutoff,
        ffill_body=args.ffill,
    )
    for p in outs:
        print(f"  {p}")
    print(f"共写入 {len(outs)} 个文件")


def cmd_summary(args: argparse.Namespace) -> None:
    res = summary.summarize_dir(args.dir or TEMPERATURE_DIR)
    out = args.out or (SUMMARY_DIR / "temperature_summary.csv")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"已写入: {out}")
    if not res.empty:
        # 只打印关键列以免太宽
        cols = ["file", "n_frames"]
        for c in ("body_temperature_mean", "body_temperature_median",
                  "body_temperature_max", "body_temperature_p90"):
            if c in res.columns:
                cols.append(c)
        with __import__("pandas").option_context("display.float_format", "{:.3f}".format):
            print(res[cols].to_string(index=False))


def cmd_plot(args: argparse.Namespace) -> None:
    out_dir = args.out_dir or (SUMMARY_DIR / "plots")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.csv.stem.replace("_temperature", "")

    p1 = plot.plot_body_temperature(
        args.csv,
        out_dir / f"{stem}_body.png",
        fps=args.fps,
        smooth=args.smooth,
    )
    print(f"已写入: {p1}")

    p2 = plot.plot_bodyparts(
        args.csv,
        out_dir / f"{stem}_bodyparts.png",
        bodyparts=args.bodyparts if args.bodyparts else None,
        fps=args.fps,
    )
    print(f"已写入: {p2}")


def cmd_overlay(args: argparse.Namespace) -> None:
    out = args.out or (OVERLAY_DIR / f"{args.thermal.stem}_temp_overlay.mp4")
    overlay.render(
        args.intensity,
        args.temperature,
        args.thermal,
        out,
        p_cutoff=args.p_cutoff,
        dot_radius=args.dot_radius,
        max_frames=args.max_frames,
    )
    print(f"已写入: {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m thermometry",
        description="伪彩 intensity → 实际体温 (°C) 的标定与计算",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="生成占位 calibration.json")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("measure", help="读 intensity csv 中某段 bodypart 的描述统计")
    s.add_argument("--csv", required=True, type=Path)
    s.add_argument("--bodypart", required=True)
    s.add_argument("--start", type=int, default=None)
    s.add_argument("--end", type=int, default=None)
    s.add_argument("--p-cutoff", type=float, default=DEFAULT_P_CUTOFF)
    s.set_defaults(func=cmd_measure)

    s = sub.add_parser("fit", help="从 anchors.json 拟合并写入 calibration.json")
    s.add_argument("--anchors", type=Path, default=None,
                   help=f"默认 {ANCHORS_PATH}")
    s.add_argument("--mode", choices=["linear", "piecewise"], default=None)
    s.add_argument("--out", type=Path, default=None,
                   help=f"默认 {CALIBRATION_PATH}")
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("apply", help="单个 *_thermal_intensity.csv -> 温度 csv")
    s.add_argument("--csv", required=True, type=Path)
    s.add_argument("--out", type=Path, default=None)
    _add_apply_args(s)
    s.set_defaults(func=cmd_apply)

    s = sub.add_parser("batch", help="批量处理 INTENSITY_DIR 下所有 *_thermal_intensity.csv")
    s.add_argument("--intensity-dir", type=Path, default=None,
                   help=f"默认 {INTENSITY_DIR}")
    s.add_argument("--out-dir", type=Path, default=None,
                   help=f"默认 {TEMPERATURE_DIR}")
    _add_apply_args(s)
    s.set_defaults(func=cmd_batch)

    s = sub.add_parser("summary", help="汇总每段视频的体温统计")
    s.add_argument("--dir", type=Path, default=None,
                   help=f"默认 {TEMPERATURE_DIR}")
    s.add_argument("--out", type=Path, default=None,
                   help=f"默认 {SUMMARY_DIR / 'temperature_summary.csv'}")
    s.set_defaults(func=cmd_summary)

    s = sub.add_parser("plot", help="读 *_temperature.csv 画体温折线图")
    s.add_argument("--csv", required=True, type=Path,
                   help="*_temperature.csv（apply 的输出）")
    s.add_argument("--out-dir", type=Path, default=None,
                   help=f"默认 {SUMMARY_DIR / 'plots'}")
    s.add_argument("--fps", type=float, default=None,
                   help="若提供则 x 轴换算成秒，否则用帧号")
    s.add_argument("--smooth", type=int, default=15,
                   help="body_temperature 的滚动平均窗口（帧），1 = 不平滑")
    s.add_argument("--bodyparts", nargs="*", default=None,
                   help="bodyparts 子图里要画哪些点，省略表示全部")
    s.set_defaults(func=cmd_plot)

    s = sub.add_parser("overlay", help="在 thermal 视频上叠加 bodypart 温度")
    s.add_argument("--intensity", required=True, type=Path)
    s.add_argument("--temperature", required=True, type=Path)
    s.add_argument("--thermal", required=True, type=Path)
    s.add_argument("--out", type=Path, default=None,
                   help=f"默认 {OVERLAY_DIR}/<stem>_temp_overlay.mp4")
    s.add_argument("--p-cutoff", type=float, default=DEFAULT_P_CUTOFF)
    s.add_argument("--dot-radius", type=int, default=6)
    s.add_argument("--max-frames", type=int, default=None)
    s.set_defaults(func=cmd_overlay)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
