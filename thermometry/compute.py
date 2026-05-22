"""把 *_thermal_intensity.csv 逐行换算成温度，并按 bodypart 聚合得到体温。

输入 (来自 alignment.dlc sample) 列结构（每个 bodypart 4 列）：

    <bp>_x, <bp>_y, <bp>_intensity, <bp>_likelihood

输出 csv 结构：

    <bp>_temperature, <bp>_likelihood, ..., body_temperature
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from thermometry import calibration as cal
from thermometry.config import (
    CALIBRATION_PATH,
    DEFAULT_AGGREGATION,
    DEFAULT_ESTIMATION_SCHEME,
    DEFAULT_P_CUTOFF,
    DEFAULT_QUANTILE,
    INTENSITY_DIR,
    TEMPERATURE_DIR,
)
from thermometry.tail_baseline import (
    TailBaselineConfig,
    apply_tail_baseline_vectorized,
)

_INTENSITY_SUFFIX = "_intensity"
_LIKELIHOOD_SUFFIX = "_likelihood"
_X_SUFFIX = "_x"
_Y_SUFFIX = "_y"


def list_bodyparts(df: pd.DataFrame) -> list[str]:
    """从列名提取 bodypart 列表，保持原列序。"""
    seen: set[str] = set()
    out: list[str] = []
    for col in df.columns:
        if col.endswith(_INTENSITY_SUFFIX):
            bp = col[: -len(_INTENSITY_SUFFIX)]
            if bp not in seen:
                seen.add(bp)
                out.append(bp)
    return out


def _aggregate(arr: np.ndarray, aggregator: str, quantile: float) -> np.ndarray:
    """对 (n_frames, n_bodyparts) 的温度矩阵按行聚合。"""
    if arr.size == 0:
        return np.full(arr.shape[0], np.nan)
    all_nan = np.isnan(arr).all(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if aggregator == "max":
            agg = np.nanmax(arr, axis=1)
        elif aggregator == "mean":
            agg = np.nanmean(arr, axis=1)
        elif aggregator == "median":
            agg = np.nanmedian(arr, axis=1)
        elif aggregator == "quantile":
            agg = np.nanquantile(arr, quantile, axis=1)
        else:
            raise ValueError(f"未知聚合方式: {aggregator}")
    return np.where(all_nan, np.nan, agg)


def apply_to_dataframe(
    df: pd.DataFrame,
    calibration: cal.Calibration,
    *,
    bodyparts: Iterable[str] | None = None,
    aggregator: str = DEFAULT_AGGREGATION,
    quantile: float = DEFAULT_QUANTILE,
    p_cutoff: float = DEFAULT_P_CUTOFF,
    keep_xy: bool = True,
    ffill_body: bool = False,
    scheme: str = DEFAULT_ESTIMATION_SCHEME,
    tail_baseline: TailBaselineConfig | None = None,
) -> pd.DataFrame:
    """对单个 intensity DataFrame 做标定，返回温度 DataFrame。

    Parameters
    ----------
    df : 来自 alignment.dlc sample 的 intensity csv（已 set_index('frame')）
    calibration : 已经加载的标定模型
    bodyparts : None 表示使用 csv 中所有 bodypart
    aggregator : 多 bodypart 聚合方式
    p_cutoff : likelihood 低于该值的 bodypart 单帧记 NaN
    keep_xy : 是否保留 x/y 列（便于后续 overlay 视频）
    scheme : ``tail_baseline``（尾温为主体 + 眼均值 + 置信度）或 ``legacy_max``（多部位取 max）
    tail_baseline : 仅 scheme=tail_baseline 时生效的细项配置
    """
    bps = list_bodyparts(df) if bodyparts is None else list(bodyparts)
    if not bps:
        raise ValueError("DataFrame 中没有 *_intensity 列，无法计算温度")

    out = pd.DataFrame(index=df.index)
    temp_cols: list[str] = []
    for bp in bps:
        ic = bp + _INTENSITY_SUFFIX
        lc = bp + _LIKELIHOOD_SUFFIX
        if ic not in df.columns:
            continue

        intensity = df[ic].to_numpy(dtype=float)
        temperature = np.asarray(calibration.apply(intensity), dtype=float)

        nan_mask = np.isnan(intensity)
        if p_cutoff > 0 and lc in df.columns:
            lk = df[lc].to_numpy(dtype=float)
            nan_mask |= lk < p_cutoff
        temperature[nan_mask] = np.nan

        temp_col = bp + "_temperature"
        out[temp_col] = temperature
        temp_cols.append(temp_col)
        if keep_xy:
            xc = bp + _X_SUFFIX
            yc = bp + _Y_SUFFIX
            if xc in df.columns:
                out[xc] = df[xc]
            if yc in df.columns:
                out[yc] = df[yc]
        if lc in df.columns:
            out[lc] = df[lc]

    if scheme == "tail_baseline":
        tb_cfg = tail_baseline or TailBaselineConfig(p_cutoff=p_cutoff)
        out = apply_tail_baseline_vectorized(out, cfg=tb_cfg)
        if ffill_body:
            out["body_temperature"] = out["body_temperature"].ffill()
        return out

    if temp_cols:
        arr = out[temp_cols].to_numpy(dtype=float)
        body = pd.Series(_aggregate(arr, aggregator, quantile), index=out.index)
        if ffill_body:
            body = body.ffill()
        out["body_temperature"] = body
        out["estimation_scheme"] = "legacy_max"
    else:
        out["body_temperature"] = np.nan

    return out


def derive_temperature_path(intensity_csv: Path) -> Path:
    name = intensity_csv.name
    if name.endswith("_thermal_intensity.csv"):
        new = name[: -len("_thermal_intensity.csv")] + "_temperature.csv"
    else:
        new = intensity_csv.stem + "_temperature.csv"
    return intensity_csv.with_name(new)


def process_csv(
    intensity_csv: Path,
    out_csv: Path | None = None,
    calibration_path: Path | None = None,
    **kwargs,
) -> Path:
    intensity_csv = Path(intensity_csv)
    out_csv = Path(out_csv) if out_csv else derive_temperature_path(intensity_csv)
    model = cal.load(calibration_path or CALIBRATION_PATH)
    df = pd.read_csv(intensity_csv, index_col=0)
    res = apply_to_dataframe(df, model, **kwargs)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv)
    return out_csv


def batch(
    intensity_dir: Path | None = None,
    out_dir: Path | None = None,
    calibration_path: Path | None = None,
    *,
    pattern: str = "*_thermal_intensity.csv",
    **kwargs,
) -> list[Path]:
    """扫描 intensity_dir 下所有 intensity csv 并批量处理。"""
    intensity_dir = Path(intensity_dir) if intensity_dir else INTENSITY_DIR
    out_dir = Path(out_dir) if out_dir else TEMPERATURE_DIR
    outs: list[Path] = []
    for csv_path in sorted(intensity_dir.glob(pattern)):
        out_csv = out_dir / derive_temperature_path(csv_path).name
        process_csv(csv_path, out_csv, calibration_path, **kwargs)
        outs.append(out_csv)
    return outs


def measure_bodypart(
    intensity_csv: Path,
    bodypart: str,
    *,
    start: int | None = None,
    end: int | None = None,
    p_cutoff: float = DEFAULT_P_CUTOFF,
) -> dict:
    """读取某段帧区间、某个 bodypart 的 intensity 描述统计。

    用来给 anchors.json 准备真实 intensity 数值（再配上人眼真实温度即可拟合）。
    """
    df = pd.read_csv(intensity_csv, index_col=0)
    ic = bodypart + _INTENSITY_SUFFIX
    if ic not in df.columns:
        raise KeyError(f"{ic} 不在 {intensity_csv.name} 中。可用 bodypart: {list_bodyparts(df)}")
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df.index >= start
    if end is not None:
        mask &= df.index <= end
    lc = bodypart + _LIKELIHOOD_SUFFIX
    if lc in df.columns and p_cutoff > 0:
        mask &= df[lc] >= p_cutoff
    s = df.loc[mask, ic].dropna()
    if s.empty:
        return {
            "bodypart": bodypart,
            "n_frames": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    return {
        "bodypart": bodypart,
        "n_frames": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "p10": float(s.quantile(0.1)),
        "p90": float(s.quantile(0.9)),
    }
