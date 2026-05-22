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
    filtered: np.ndarray
    time_s: np.ndarray
    inst_time_s: np.ndarray
    inst_bpm: np.ndarray
    inst_hz: np.ndarray
    freqs_hz: np.ndarray
    spectrum: np.ndarray


def _interp_nan(y: np.ndarray) -> np.ndarray:
    """线性插值填补 NaN，首尾用最近有效值。"""
    x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 4:
        return np.zeros_like(y)
    out = y.copy()
    out[~mask] = np.interp(x[~mask], x[mask], y[mask])
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """滑动窗 FFT 主频 → 瞬时呼吸率（次/分）。"""
    n = len(y)
    win = max(int(round(window_sec * fps)), int(fps * 2))
    hop = max(int(round(hop_sec * fps)), 1)
    if n < win:
        t = np.array([])
        return t, np.array([]), np.array([])

    centers: list[int] = []
    bpm_list: list[float] = []
    hz_list: list[float] = []
    for start in range(0, n - win + 1, hop):
        seg = y[start : start + win]
        seg = seg - np.mean(seg)
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
    df = pd.read_csv(signal_csv, index_col=0)
    if "motion" in df.columns:
        raw = df["motion"].to_numpy(dtype=float)
    elif "raw" in df.columns:
        raw = df["raw"].to_numpy(dtype=float)
    else:
        raise KeyError("signal csv 需包含 motion 列（或旧版 raw 列）")
    if "time_s" in df.columns:
        time_s = df["time_s"].to_numpy(dtype=float)
    else:
        time_s = df.index.to_numpy(dtype=float) / fps

    filled = _interp_nan(raw)
    filled = signal.detrend(filled, type="linear")
    filtered = bandpass_respiration(
        filled, fps, f_min=f_min, f_max=f_max, order=filter_order,
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
    )

    return RespirationResult(
        fps=fps,
        global_bpm=global_bpm,
        global_peak_hz=peak_hz,
        filtered=filtered,
        time_s=time_s,
        inst_time_s=t_inst,
        inst_bpm=inst_bpm,
        inst_hz=inst_hz,
        freqs_hz=freqs,
        spectrum=mag,
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
        f"global_peak_hz={result.global_peak_hz}\n"
        f"global_breaths_per_min={result.global_bpm}\n"
        f"instant_rate_csv={rate_path}\n"
        f"filtered_csv={filtered_path}\n",
        encoding="utf-8",
    )
    return rate_path, filtered_path, summary_path
