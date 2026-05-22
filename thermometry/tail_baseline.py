"""
以 tail_base 为基准的体温估计 + 双眼平均 + 综合置信度。

设计思路
--------
1. **主体温** ``body_temperature`` = ``tail_base`` 标定温度（稳定、DLC 置信度最高）。
2. **眼部参考** ``eye_temperature_mean_raw`` = 逐帧眼均温；``eye_temperature_mean`` = 对 raw 做
   ``eye_smooth_sec`` 秒滚动平均（默认 1 s @ 60 fps → 60 帧窗口）。
3. **置信度** ``temperature_confidence`` ∈ [0, 1]，由三部分相乘后再与尾部权重混合：

   - ``conf_tail``：尾部 DLC likelihood（追踪是否可靠）
   - ``conf_eye``：眼部追踪 × 覆盖率 × 与尾部温差一致性
   - 最终 ``confidence = conf_tail * (w_tail + w_eye * conf_eye)``，默认 w_tail=0.65, w_eye=0.35

眼部与尾部一致性：``exp(-|eye_mean - tail| / sigma)``，默认 sigma=2°C。
温差过大说明眼点飘到背景或映射失败，置信度下降。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thermometry.config import (
    DEFAULT_THERMAL_FPS,
    TAIL_BASELINE_AGREEMENT_SIGMA,
    TAIL_BASELINE_EYE_PARTS,
    TAIL_BASELINE_EYE_SMOOTH_SEC,
    TAIL_BASELINE_TAIL_BP,
    TAIL_BASELINE_TAIL_WEIGHT,
    TAIL_BASELINE_W_EYE,
)

_LIKELIHOOD_SUFFIX = "_likelihood"
_TEMPERATURE_SUFFIX = "_temperature"


@dataclass(frozen=True)
class TailBaselineConfig:
    tail_bodypart: str = TAIL_BASELINE_TAIL_BP
    eye_bodyparts: tuple[str, ...] = TAIL_BASELINE_EYE_PARTS
    tail_weight: float = TAIL_BASELINE_TAIL_WEIGHT
    eye_weight: float = TAIL_BASELINE_W_EYE
    agreement_sigma_c: float = TAIL_BASELINE_AGREEMENT_SIGMA
    p_cutoff: float = 0.0  # 在 apply 前各点温度已按 p_cutoff 置 NaN 时可设 0
    eye_smooth_sec: float = TAIL_BASELINE_EYE_SMOOTH_SEC
    fps: float = DEFAULT_THERMAL_FPS


def _eye_smooth_window(cfg: TailBaselineConfig) -> int:
    if cfg.eye_smooth_sec <= 0 or cfg.fps <= 0:
        return 1
    return max(1, int(round(cfg.eye_smooth_sec * cfg.fps)))


def smooth_eye_temperature(
    eye_mean_raw: np.ndarray,
    *,
    cfg: TailBaselineConfig,
) -> np.ndarray:
    """对眼均温做时间滚动平均（center=True，边界用已有帧凑窗口）。"""
    w = _eye_smooth_window(cfg)
    if w <= 1:
        return eye_mean_raw.copy()
    return (
        pd.Series(eye_mean_raw)
        .rolling(w, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=float)
    )


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def compute_frame_confidence(
    tail_t: float,
    tail_lk: float,
    eye_ts: list[float],
    eye_lks: list[float],
    *,
    cfg: TailBaselineConfig,
) -> tuple[float, float, float, float, float, float]:
    """单帧计算。返回 (body_T, eye_mean, confidence, conf_tail, conf_eye, conf_agreement)。"""
    tail_valid = np.isfinite(tail_t) and (cfg.p_cutoff <= 0 or tail_lk >= cfg.p_cutoff)
    conf_tail = _clip01(np.array([tail_lk if np.isfinite(tail_lk) else 0.0]))[0]

    valid_eye_t: list[float] = []
    valid_eye_lk: list[float] = []
    for t, lk in zip(eye_ts, eye_lks):
        if not np.isfinite(t):
            continue
        if cfg.p_cutoff > 0 and (not np.isfinite(lk) or lk < cfg.p_cutoff):
            continue
        valid_eye_t.append(float(t))
        valid_eye_lk.append(float(lk) if np.isfinite(lk) else 0.0)

    n_eye = len(valid_eye_t)
    eye_mean = float(np.mean(valid_eye_t)) if n_eye else float("nan")
    conf_eye_track = float(np.mean(valid_eye_lk)) if valid_eye_lk else 0.0
    conf_eye_coverage = n_eye / max(len(cfg.eye_bodyparts), 1)

    if tail_valid and n_eye and np.isfinite(eye_mean):
        delta = abs(eye_mean - float(tail_t))
        conf_agreement = float(np.exp(-delta / max(cfg.agreement_sigma_c, 0.1)))
    else:
        conf_agreement = 0.0 if n_eye else 1.0  # 无眼时不因一致性扣分

    conf_eye = conf_eye_track * conf_eye_coverage * conf_agreement

    if not tail_valid:
        return float("nan"), eye_mean, 0.0, conf_tail, conf_eye, conf_agreement

    body_t = float(tail_t)
    w_tail = cfg.tail_weight
    w_eye = cfg.eye_weight
    if n_eye == 0:
        confidence = conf_tail * w_tail
    else:
        confidence = conf_tail * (w_tail + w_eye * conf_eye)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return body_t, eye_mean, confidence, conf_tail, conf_eye, conf_agreement


def apply_tail_baseline_columns(
    temp_df: pd.DataFrame,
    *,
    cfg: TailBaselineConfig | None = None,
) -> pd.DataFrame:
    """在已有各 bodypart *_temperature 列的 DataFrame 上追加尾部基准列。

    需要列：``{tail}_temperature``, ``{tail}_likelihood``,
    以及各眼的 ``{eye}_temperature``, ``{eye}_likelihood``。
    """
    cfg = cfg or TailBaselineConfig()
    out = temp_df.copy()
    n = len(out)

    tail_tc = cfg.tail_bodypart + _TEMPERATURE_SUFFIX
    tail_lc = cfg.tail_bodypart + _LIKELIHOOD_SUFFIX
    if tail_tc not in out.columns:
        raise KeyError(f"缺少 {tail_tc}，请先跑 thermometry apply")

    tail_t = out[tail_tc].to_numpy(dtype=float)
    tail_lk = (
        out[tail_lc].to_numpy(dtype=float)
        if tail_lc in out.columns
        else np.full(n, np.nan)
    )

    eye_t_cols = [bp + _TEMPERATURE_SUFFIX for bp in cfg.eye_bodyparts]
    eye_l_cols = [bp + _LIKELIHOOD_SUFFIX for bp in cfg.eye_bodyparts]

    body = np.full(n, np.nan)
    eye_mean = np.full(n, np.nan)
    confidence = np.zeros(n)
    conf_tail = np.zeros(n)
    conf_eye = np.zeros(n)
    conf_agreement = np.zeros(n)

    for i in range(n):
        eye_ts = [
            float(out[c].iloc[i]) if c in out.columns else float("nan")
            for c in eye_t_cols
        ]
        eye_lks = [
            float(out[c].iloc[i]) if c in out.columns else float("nan")
            for c in eye_l_cols
        ]
        b, em, cf, ct, ce, ca = compute_frame_confidence(
            float(tail_t[i]),
            float(tail_lk[i]) if np.isfinite(tail_lk[i]) else 0.0,
            eye_ts,
            eye_lks,
            cfg=cfg,
        )
        body[i] = b
        eye_mean[i] = em
        confidence[i] = cf
        conf_tail[i] = ct
        conf_eye[i] = ce
        conf_agreement[i] = ca

    out["body_temperature"] = body
    out["eye_temperature_mean"] = eye_mean
    out["temperature_confidence"] = confidence
    out["conf_tail"] = conf_tail
    out["conf_eye"] = conf_eye
    out["conf_agreement"] = conf_agreement
    out["estimation_scheme"] = "tail_baseline"

    return out


def apply_tail_baseline_vectorized(
    temp_df: pd.DataFrame,
    *,
    cfg: TailBaselineConfig | None = None,
) -> pd.DataFrame:
    """向量化版本（与逐帧逻辑等价，更快）。"""
    cfg = cfg or TailBaselineConfig()
    out = temp_df.copy()
    n = len(out)

    tail_tc = cfg.tail_bodypart + _TEMPERATURE_SUFFIX
    tail_lc = cfg.tail_bodypart + _LIKELIHOOD_SUFFIX
    tail_t = out[tail_tc].to_numpy(dtype=float)
    tail_lk = (
        out[tail_lc].to_numpy(dtype=float)
        if tail_lc in out.columns
        else np.zeros(n)
    )
    conf_tail = _clip01(np.nan_to_num(tail_lk, nan=0.0))

    eye_t_list = []
    eye_l_list = []
    for bp in cfg.eye_bodyparts:
        tc, lc = bp + _TEMPERATURE_SUFFIX, bp + _LIKELIHOOD_SUFFIX
        eye_t_list.append(out[tc].to_numpy(dtype=float) if tc in out.columns else np.full(n, np.nan))
        eye_l_list.append(
            out[lc].to_numpy(dtype=float) if lc in out.columns else np.full(n, np.nan)
        )
    eye_stack = np.stack(eye_t_list, axis=1) if eye_t_list else np.empty((n, 0))
    lk_stack = np.stack(eye_l_list, axis=1) if eye_l_list else np.empty((n, 0))

    n_eyes = eye_stack.shape[1]
    eye_valid = np.isfinite(eye_stack)
    if cfg.p_cutoff > 0 and lk_stack.size:
        eye_valid &= np.isfinite(lk_stack) & (lk_stack >= cfg.p_cutoff)

    n_valid = eye_valid.sum(axis=1).astype(float)
    eye_mean_raw = np.where(
        n_valid > 0,
        np.nansum(np.where(eye_valid, eye_stack, np.nan), axis=1) / np.maximum(n_valid, 1),
        np.nan,
    )
    eye_mean = smooth_eye_temperature(eye_mean_raw, cfg=cfg)

    with np.errstate(invalid="ignore"):
        lk_masked = np.where(eye_valid, lk_stack, np.nan)
        n_lk = np.sum(np.isfinite(lk_masked), axis=1)
        conf_eye_track = np.nansum(lk_masked, axis=1) / np.maximum(n_lk, 1)
    conf_eye_track = np.where(n_lk > 0, conf_eye_track, 0.0)
    conf_eye_coverage = n_valid / max(n_eyes, 1)

    tail_valid = np.isfinite(tail_t)
    eye_for_conf = np.isfinite(eye_mean)
    delta = np.abs(eye_mean - tail_t)
    conf_agreement = np.where(
        tail_valid & eye_for_conf,
        np.exp(-delta / max(cfg.agreement_sigma_c, 0.1)),
        np.where(eye_for_conf, 0.0, 1.0),
    )
    conf_eye = conf_eye_track * conf_eye_coverage * conf_agreement

    w_tail, w_eye = cfg.tail_weight, cfg.eye_weight
    confidence = conf_tail * np.where(
        eye_for_conf,
        w_tail + w_eye * conf_eye,
        w_tail,
    )
    confidence = np.where(tail_valid, np.clip(confidence, 0, 1), 0.0)

    out["body_temperature"] = np.where(tail_valid, tail_t, np.nan)
    out["eye_temperature_mean_raw"] = eye_mean_raw
    out["eye_temperature_mean"] = eye_mean
    out["temperature_confidence"] = confidence
    out["conf_tail"] = conf_tail
    out["conf_eye"] = conf_eye
    out["conf_agreement"] = conf_agreement
    out["estimation_scheme"] = "tail_baseline"

    return out
