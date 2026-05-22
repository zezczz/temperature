"""读 *_temperature.csv 画体温曲线。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def _frame_to_seconds(df: pd.DataFrame, fps: float | None) -> tuple[np.ndarray, str]:
    if fps and fps > 0:
        return df.index.to_numpy() / fps, "time (s)"
    return df.index.to_numpy(), "frame"


def plot_body_temperature(
    temperature_csv: Path,
    out_png: Path,
    *,
    fps: float | None = None,
    smooth: int = 1,
    show: bool = False,
) -> Path:
    """画 body_temperature 曲线。smooth>1 时叠加 rolling mean。"""
    import matplotlib.pyplot as plt

    df = _load(temperature_csv)
    if "body_temperature" not in df.columns:
        raise KeyError("csv 中没有 body_temperature 列")

    x, xlabel = _frame_to_seconds(df, fps)
    y = df["body_temperature"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, y, lw=0.8, color="#888888", label="body_temperature (raw)")
    if smooth and smooth > 1:
        ys = df["body_temperature"].rolling(smooth, min_periods=1, center=True).mean()
        ax.plot(x, ys.to_numpy(), lw=1.6, color="#d62728",
                label=f"rolling mean (w={smooth})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("temperature (°C)")
    ax.set_title(f"body temperature — {temperature_csv.name}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def plot_tail_baseline(
    temperature_csv: Path,
    out_png: Path,
    *,
    fps: float | None = None,
    smooth: int = 15,
    conf_threshold: float = 0.3,
    show: bool = False,
) -> Path:
    """尾部基准方案：主体温 + 眼均值 + 置信度（副轴）。"""
    import matplotlib.pyplot as plt

    df = _load(temperature_csv)
    if "body_temperature" not in df.columns:
        raise KeyError("需要 body_temperature 列")

    x, xlabel = _frame_to_seconds(df, fps)
    body = df["body_temperature"].to_numpy(dtype=float)
    conf = (
        df["temperature_confidence"].to_numpy(dtype=float)
        if "temperature_confidence" in df.columns
        else np.ones(len(df))
    )
    eye = (
        df["eye_temperature_mean"].to_numpy(dtype=float)
        if "eye_temperature_mean" in df.columns
        else np.full(len(df), np.nan)
    )
    eye_raw = (
        df["eye_temperature_mean_raw"].to_numpy(dtype=float)
        if "eye_temperature_mean_raw" in df.columns
        else None
    )

    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    ax2 = ax1.twinx()

    ax1.plot(x, body, lw=0.7, color="#888888", alpha=0.6, label="body (tail, raw)")
    if smooth and smooth > 1:
        bs = pd.Series(body).rolling(smooth, min_periods=1, center=True).mean()
        ax1.plot(x, bs.to_numpy(), lw=1.8, color="#d62728", label=f"body smooth (w={smooth})")
    if eye_raw is not None and np.isfinite(eye_raw).any():
        ax1.plot(x, eye_raw, lw=0.5, color="#1f77b4", alpha=0.35, label="eye mean (raw)")
    if np.isfinite(eye).any():
        ax1.plot(x, eye, lw=1.2, color="#1f77b4", alpha=0.9, label="eye mean (time smooth)")
    low = conf < conf_threshold
    if low.any():
        ax1.scatter(
            x[low], body[low], s=8, c="orange", alpha=0.5, label=f"low conf (<{conf_threshold})"
        )

    ax2.fill_between(x, 0, conf, color="#2ca02c", alpha=0.2, label="confidence")
    ax2.plot(x, conf, lw=0.6, color="#2ca02c", alpha=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("confidence")

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("temperature (°C)")
    ax1.set_title(f"tail-baseline temperature — {temperature_csv.name}")
    ax1.grid(alpha=0.3)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def plot_bodyparts(
    temperature_csv: Path,
    out_png: Path,
    *,
    bodyparts: list[str] | None = None,
    fps: float | None = None,
    show: bool = False,
) -> Path:
    """每个 bodypart 一条线，可对比眼睛 vs 尾巴等。"""
    import matplotlib.pyplot as plt

    df = _load(temperature_csv)
    if bodyparts is None:
        bodyparts = [c[: -len("_temperature")] for c in df.columns
                     if c.endswith("_temperature") and c != "body_temperature"]
    if not bodyparts:
        raise ValueError("没有 <bp>_temperature 列可画")

    x, xlabel = _frame_to_seconds(df, fps)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    for bp in bodyparts:
        col = f"{bp}_temperature"
        if col not in df.columns:
            continue
        y = df[col].to_numpy(dtype=float)
        ax.plot(x, y, lw=0.9, label=bp)
    if "body_temperature" in df.columns:
        ax.plot(x, df["body_temperature"].to_numpy(dtype=float),
                lw=1.6, color="black", alpha=0.7, label="body_temperature")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("temperature (°C)")
    ax.set_title(f"per-bodypart temperature — {temperature_csv.name}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_png
