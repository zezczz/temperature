"""带通滤波 + FFT，估计整体与瞬时呼吸频率。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


@dataclass
class RespirationResult:
    fps: float
    global_bpm: float
    global_peak_hz: float
    median_inst_bpm: float
    filtered: np.ndarray
    time_s: np.ndarray
    inst_time_s: np.ndarray
    inst_bpm: np.ndarray
    inst_hz: np.ndarray
    freqs_hz: np.ndarray
    spectrum: np.ndarray
    valid_frac: float


def _interp_nan_short_gaps(y: np.ndarray, valid: np.ndarray | None, max_gap: int) -> np.ndarray:
    """仅对短无效段线性插值；长段保持 NaN。"""
    out = y.copy()
    n = len(out)
    if valid is None:
        return _interp_nan(out)
    i = 0
    while i < n:
        if valid[i] and np.isfinite(out[i]):
            i += 1
            continue
        j = i
        while j < n and (not valid[j] or not np.isfinite(out[j])):
            j += 1
        gap = j - i
        if gap <= max_gap and i > 0 and j < n and np.isfinite(out[i - 1]) and np.isfinite(out[j]):
            out[i:j] = np.linspace(out[i - 1], out[j], gap + 2)[1:-1]
        i = j
    # 首尾仍用最近有效值
    mask = np.isfinite(out)
    if mask.sum() >= 2:
        idx = np.where(mask)[0]
        out[: idx[0]] = out[idx[0]]
        out[idx[-1] + 1 :] = out[idx[-1]]
    return out


def _robust_clip(y: np.ndarray, valid: np.ndarray | None, k: float) -> np.ndarray:
    """按 median + k*MAD 截断异常尖峰。"""
    out = y.copy()
    mask = np.isfinite(out)
    if valid is not None:
        mask &= valid.astype(bool)
    if mask.sum() < 8:
        return out
    ref = out[mask]
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    if mad < 1e-9:
        mad = float(np.std(ref)) or 1e-9
    cap = med + k * mad
    out[mask & (out > cap)] = cap
    return out


def _median_filter_1d(y: np.ndarray, k: int) -> np.ndarray:
    if k < 3 or k % 2 == 0:
        return y
    from scipy.ndimage import median_filter

    filled = y.copy()
    mask = np.isfinite(filled)
    if mask.sum() < k:
        return y
    tmp = filled.copy()
    tmp[~mask] = np.nanmedian(filled[mask])
    sm = median_filter(tmp, size=k, mode="nearest")
    out = y.copy()
    out[mask] = sm[mask]
    return out


def bandpass_respiration(
    y: np.ndarray,
    fps: float,
    *,
    f_min: float,
    f_max: float,
    order: int,
) -> np.ndarray:
    nyq = 0.5 * fps
    lo = max(f_min / nyq, 1e-4)
    hi = min(f_max / nyq, 0.99)
    if lo >= hi:
        raise ValueError(f"无效频段: f_min={f_min}, f_max={f_max}, fps={fps}")
    b, a = signal.butter(order, [lo, hi], btype="band")
    return signal.filtfilt(b, a, y)


def global_fft_peak(
    y: np.ndarray,
    fps: float,
    *,
    f_min: float,
    f_max: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """全段 FFT，返回 (peak_hz, bpm, freqs, |spectrum|)。"""
    n = len(y)
    if n < 8:
        return float("nan"), float("nan"), np.array([]), np.array([])
    win = signal.windows.hann(n)
    spec = np.fft.rfft(y * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mag = np.abs(spec)
    band = (freqs >= f_min) & (freqs <= f_max)
    if not band.any():
        return float("nan"), float("nan"), freqs, mag
    idx = int(np.argmax(mag[band]))
    peak_hz = float(freqs[band][idx])
    return peak_hz, peak_hz * 60.0, freqs, mag


def sliding_fft_bpm(
    y: np.ndarray,
    fps: float,
    *,
    f_min: float,
    f_max: float,
    window_sec: float,
    hop_sec: float,
    valid: np.ndarray | None = None,
    min_valid_frac: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """滑动窗 FFT 主频 → 瞬时呼吸率（次/分）。"""
    n = len(y)
    win = max(int(round(window_sec * fps)), int(fps * 2))
    hop = max(int(round(hop_sec * fps)), 1)
    if n < win:
        return np.array([]), np.array([]), np.array([])

    centers: list[int] = []
    bpm_list: list[float] = []
    hz_list: list[float] = []
    for start in range(0, n - win + 1, hop):
        if valid is not None:
            vf = float(np.mean(valid[start : start + win]))
            if vf < min_valid_frac:
                continue
        seg = y[start : start + win]
        if np.isnan(seg).mean() > 0.25:
            continue
        seg = np.nan_to_num(seg, nan=0.0) - np.nanmean(seg)
        w = signal.windows.hann(len(seg))
        spec = np.fft.rfft(seg * w)
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fps)
        mag = np.abs(spec)
        band = (freqs >= f_min) & (freqs <= f_max)
        if not band.any():
            continue
        idx = int(np.argmax(mag[band]))
        peak = float(freqs[band][idx])
        centers.append(start + win // 2)
        hz_list.append(peak)
        bpm_list.append(peak * 60.0)

    if not centers:
        return np.array([]), np.array([]), np.array([])

    t = np.asarray(centers, dtype=float) / fps
    return t, np.asarray(bpm_list), np.asarray(hz_list)


def analyze_signal_csv(
    signal_csv: Path,
    *,
    fps: float,
    f_min: float,
    f_max: float,
    filter_order: int,
    fft_window_sec: float,
    fft_hop_sec: float,
) -> RespirationResult:
    from respiration.config import (
        MAX_INTERP_GAP_SEC,
        MIN_VALID_FRAC_IN_WINDOW,
        MOTION_MAD_CLIP_K,
        MOTION_MEDIAN_FILTER,
    )

    df = pd.read_csv(signal_csv, index_col=0)
    if "motion" in df.columns:
        raw = df["motion"].to_numpy(dtype=float)
    elif "raw" in df.columns:
        raw = df["raw"].to_numpy(dtype=float)
    else:
        raise KeyError("signal csv 需包含 motion 列（或旧版 raw 列）")
    valid = df["valid"].to_numpy(dtype=bool) if "valid" in df.columns else None
    if "time_s" in df.columns:
        time_s = df["time_s"].to_numpy(dtype=float)
    else:
        time_s = df.index.to_numpy(dtype=float) / fps

    valid_frac = float(valid.mean()) if valid is not None else float(np.isfinite(raw).mean())
    max_gap = max(1, int(round(MAX_INTERP_GAP_SEC * fps)))

    y = raw.copy()
    if valid is not None:
        y[~valid] = np.nan
    y = _robust_clip(y, valid, MOTION_MAD_CLIP_K)
    y = _interp_nan_short_gaps(y, valid, max_gap)
    y = _median_filter_1d(y, MOTION_MEDIAN_FILTER)
    finite = np.isfinite(y)
    if finite.sum() < 8:
        raise ValueError("有效 motion 帧过少，无法分析")
    fill_val = float(np.median(y[finite]))
    y = np.where(finite, y, fill_val)
    y = signal.detrend(y, type="linear")
    filtered = bandpass_respiration(
        y, fps, f_min=f_min, f_max=f_max, order=filter_order,
    )

    peak_hz, global_bpm, freqs, mag = global_fft_peak(
        filtered, fps, f_min=f_min, f_max=f_max,
    )
    t_inst, inst_bpm, inst_hz = sliding_fft_bpm(
        filtered,
        fps,
        f_min=f_min,
        f_max=f_max,
        window_sec=fft_window_sec,
        hop_sec=fft_hop_sec,
        valid=valid,
        min_valid_frac=MIN_VALID_FRAC_IN_WINDOW,
    )
    median_inst = float(np.median(inst_bpm)) if len(inst_bpm) else float("nan")

    return RespirationResult(
        fps=fps,
        global_bpm=global_bpm,
        global_peak_hz=peak_hz,
        median_inst_bpm=median_inst,
        filtered=filtered,
        time_s=time_s,
        inst_time_s=t_inst,
        inst_bpm=inst_bpm,
        inst_hz=inst_hz,
        freqs_hz=freqs,
        spectrum=mag,
        valid_frac=valid_frac,
    )


def save_analysis(
    result: RespirationResult,
    signal_csv: Path,
    out_prefix: Path,
) -> tuple[Path, Path, Path]:
    """写出瞬时率、滤波序列与摘要。"""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rate_path = out_prefix.with_name(out_prefix.name + "_instant_rate.csv")
    pd.DataFrame(
        {
            "time_s": result.inst_time_s,
            "breaths_per_min": result.inst_bpm,
            "freq_hz": result.inst_hz,
        },
        index=pd.RangeIndex(len(result.inst_bpm), name="window"),
    ).to_csv(rate_path)

    filtered_path = out_prefix.with_name(out_prefix.name + "_filtered.csv")
    sig_df = pd.read_csv(signal_csv, index_col=0)
    out_sig = sig_df.copy()
    n = min(len(out_sig), len(result.filtered))
    out_sig = out_sig.iloc[:n].copy()
    out_sig["filtered"] = result.filtered[:n]
    out_sig.to_csv(filtered_path)

    summary_path = out_prefix.with_name(out_prefix.name + "_summary.txt")
    summary_path.write_text(
        f"signal_csv={signal_csv}\n"
        f"fps={result.fps}\n"
        f"valid_frac={result.valid_frac}\n"
        f"global_peak_hz={result.global_peak_hz}\n"
        f"global_breaths_per_min={result.global_bpm}\n"
        f"median_instant_breaths_per_min={result.median_inst_bpm}\n"
        f"instant_rate_csv={rate_path}\n"
        f"filtered_csv={filtered_path}\n",
        encoding="utf-8",
    )
    return rate_path, filtered_path, summary_path
