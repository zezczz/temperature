"""读取 DLC 逐帧关键点。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_dlc_csv(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        hint = (
            f"\n文件不存在: {p.resolve()}\n"
            "请使用真实路径，不要用文档里的 top.... 占位符。\n"
            "可运行: python -m respiration list\n"
            "或: python -m respiration run --stamp 2026-05-17-14-44-35"
        )
        raise FileNotFoundError(hint)
    return pd.read_csv(p, header=[0, 1, 2], index_col=0)


def _scorer_for_bodypart(df: pd.DataFrame, bodypart: str) -> str:
    for scorer, bp, _ in df.columns:
        if bp == bodypart:
            return scorer
    raise KeyError(f"DLC csv 中未找到 bodypart: {bodypart}")


def bodypart_xy_likelihood(
    df: pd.DataFrame,
    bodypart: str,
    *,
    scorer: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回与帧索引对齐的 x, y, likelihood（float）。"""
    sc = scorer or _scorer_for_bodypart(df, bodypart)
    x = df[(sc, bodypart, "x")].to_numpy(dtype=float)
    y = df[(sc, bodypart, "y")].to_numpy(dtype=float)
    p = df[(sc, bodypart, "likelihood")].to_numpy(dtype=float)
    return x, y, p
