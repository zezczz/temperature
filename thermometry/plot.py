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
