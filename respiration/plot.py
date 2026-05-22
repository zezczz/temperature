"""呼吸结果可视化：静态图与汇总读盘。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from respiration.analyze import RespirationResult, analyze_signal_csv
from respiration.config import (
    DEFAULT_BPM_MAX,
    DEFAULT_BPM_MIN,
    DEFAULT_F_MAX_HZ,
    DEFAULT_F_MIN_HZ,
    PLOT_DIR,
)


def _read_meta_fps(meta_path: Path) -> float | None:
    if not meta_path.is_file():
        return None
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("fps="):
            return float(line.split("=", 1)[1])
    return None


def _stem_to_prefix(stem: str) -> str:
    for suffix in ("_chest_motion", "_chest_signal"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_result_paths(
    *,
    stamp: str | None = None,
    prefix: Path | None = None,
    signal_csv: Path | None = None,
) -> dict[str, Path]:
    """根据 stamp / prefix / signal_csv 定位一组结果文件。"""
    if signal_csv is not None:
        p = Path(signal_csv)
        base = p.parent / _stem_to_prefix(p.stem)
    elif prefix is not None:
        base = Path(prefix)
        if base.suffix:
            base = base.parent / _stem_to_prefix(base.stem)
    elif stamp:
        st = stamp.strip()
        if st.startswith("top"):
            st = st[3:]
        from respiration.config import RATE_DIR

        base = RATE_DIR / f"top{st}"
    else:
        raise ValueError("需要 --stamp、--prefix 或 --signal")

    paths = {
        "base": base,
        "signal": base.with_name(base.name + "_chest_motion.csv"),
        "filtered": base.with_name(base.name + "_filtered.csv"),
        "instant": base.with_name(base.name + "_instant_rate.csv"),
        "summary": base.with_name(base.name + "_summary.txt"),
        "meta": base.with_name(base.name + "_chest_motion.meta.txt"),
    }
    if not paths["signal"].is_file():
        legacy = base.with_name(base.name + "_chest_signal.csv")
        if legacy.is_file():
            paths["signal"] = legacy
    if not paths["signal"].is_file():
        raise FileNotFoundError(f"未找到运动 csv: {paths['signal']}")
    return paths


def load_summary(summary_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if summary_path.is_file():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def plot_respiration(
    result: RespirationResult,
    signal_csv: Path,
    out_png: Path,
    *,
    summary: dict[str, str] | None = None,
    show: bool = False,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    _candidates = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "Arial Unicode MS"]
    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _name in _candidates:
        if _name in _installed:
            plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_csv(signal_csv, index_col=0)
    if "motion" in df.columns:
        raw = df["motion"].to_numpy(dtype=float)
        raw_label = "胸腔运动 (motion)"
    else:
        raw = df["raw"].to_numpy(dtype=float)
        raw_label = "信号 (legacy raw)"
    t = result.time_s
    title_stem = signal_csv.stem.replace("_chest_motion", "").replace("_chest_signal", "")

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 0.9], width_ratios=[2.2, 1.0], hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, raw, lw=0.6, color="#aaaaaa", alpha=0.9, label=raw_label)
    if len(result.filtered):
        ax0.plot(t[: len(result.filtered)], result.filtered, lw=1.1, color="#1f77b4", label="带通滤波")
    valid = df["valid"].to_numpy(dtype=bool) if "valid" in df.columns else np.ones(len(raw), dtype=bool)
    if valid.sum() < len(valid):
        ax0.fill_between(
            t, raw.min(), raw.max(), where=~valid, color="#ffcccc", alpha=0.25, label="无效帧 (DLC)",
        )
    ax0.set_ylabel("运动强度 (a.u.)")
    ax0.set_xlabel("时间 (s)")
    ax0.set_title(f"胸腔起伏信号 — {title_stem}")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[1, :])
    if len(result.inst_time_s) and len(result.inst_bpm):
        ax1.plot(
            result.inst_time_s,
            result.inst_bpm,
            lw=1.4,
            color="#d62728",
            marker="o",
            ms=3,
            label="滑动 FFT 瞬时率",
        )
    if np.isfinite(result.global_bpm):
        ax1.axhline(
            result.global_bpm,
            color="#2ca02c",
            ls="--",
            lw=1.2,
            label=f"全段主峰 ≈ {result.global_bpm:.1f} 次/分",
        )
    ax1.set_ylabel("呼吸频率 (次/分)")
    ax1.set_xlabel("时间 (s)")
    ax1.set_title("呼吸频率随时间变化")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[2, 0])
    if len(result.freqs_hz):
        band = (result.freqs_hz >= 0) & (result.freqs_hz <= 8)
        ax2.plot(result.freqs_hz[band], result.spectrum[band], lw=1.0, color="#9467bd")
        if np.isfinite(result.global_peak_hz):
            ax2.axvline(
                result.global_peak_hz,
                color="#2ca02c",
                ls="--",
                lw=1.0,
                label=f"{result.global_peak_hz:.2f} Hz",
            )
        ax2.axvspan(
            DEFAULT_F_MIN_HZ,
            DEFAULT_F_MAX_HZ,
            color="#e8f4e8",
            alpha=0.5,
            label=f"搜索频段 {DEFAULT_BPM_MIN}–{DEFAULT_BPM_MAX} 次/分",
        )
    ax2.set_xlabel("频率 (Hz)")
    ax2.set_ylabel("|FFT|")
    ax2.set_title("全段频谱 (滤波后)")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axis("off")
    lines = [
        f"片段: {title_stem}",
        f"全段呼吸率: {result.global_bpm:.1f} 次/分" if np.isfinite(result.global_bpm) else "全段呼吸率: —",
        f"主峰频率: {result.global_peak_hz:.3f} Hz" if np.isfinite(result.global_peak_hz) else "主峰频率: —",
        f"帧率: {result.fps:.1f} fps",
        f"时长: {t[-1]:.1f} s" if len(t) else "",
    ]
    if summary:
        if "motion_metric" in summary or "signal" in summary:
            pass
        for key in ("motion_metric", "video", "dlc_csv"):
            if key in summary:
                val = summary[key]
                if len(val) > 48:
                    val = "…" + val[-45:]
                lines.append(f"{key}: {val}")
    meta_path = signal_csv.with_suffix(".meta.txt")
    if meta_path.is_file():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("motion_metric="):
                lines.append(f"运动指标: {line.split('=', 1)[1]}")
    ax3.text(
        0.02, 0.98, "\n".join(lines),
        va="top", ha="left", fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    fig.suptitle("呼吸频率分析", fontsize=13, y=0.995)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def plot_from_saved(
    *,
    stamp: str | None = None,
    prefix: Path | None = None,
    signal_csv: Path | None = None,
    out_png: Path | None = None,
    fps: float | None = None,
    f_min: float = DEFAULT_F_MIN_HZ,
    f_max: float = DEFAULT_F_MAX_HZ,
    show: bool = False,
) -> Path:
    """从已保存的 csv 重新绘图（无需重新读视频）。"""
    paths = resolve_result_paths(stamp=stamp, prefix=prefix, signal_csv=signal_csv)
    use_fps = fps or _read_meta_fps(paths["meta"]) or 25.0

    result = analyze_signal_csv(
        paths["signal"],
        fps=use_fps,
        f_min=f_min,
        f_max=f_max,
        filter_order=4,
        fft_window_sec=6.0,
        fft_hop_sec=0.5,
    )

    if out_png is None:
        out_png = PLOT_DIR / f"{paths['base'].name}_respiration.png"

    summary = load_summary(paths["summary"])
    return plot_respiration(
        result, paths["signal"], out_png, summary=summary, show=show,
    )
