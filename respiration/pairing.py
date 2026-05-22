"""彩色视频与 DLC 推理 csv 按时间戳配对。"""
from __future__ import annotations

from pathlib import Path

from respiration.config import COLOR_DIR, DLC_RESULTS_DIR

_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def _stamp_from_top_name(name: str) -> str | None:
    """top2026-05-17-14-44-35.mp4 -> 2026-05-17-14-44-35"""
    stem = Path(name).stem
    if not stem.startswith("top"):
        return None
    return stem[3:]


def list_pairs() -> list[tuple[str, Path, Path]]:
    """返回 [(stamp, video, dlc_csv), ...]，仅包含两边都存在的条目。"""
    if not DLC_RESULTS_DIR.is_dir():
        return []

    dlc_by_stamp: dict[str, Path] = {}
    for csv_path in sorted(DLC_RESULTS_DIR.glob("top*DLC_*.csv")):
        stamp = _stamp_from_top_name(csv_path.name.split("DLC_", 1)[0])
        if stamp:
            dlc_by_stamp[stamp] = csv_path

    pairs: list[tuple[str, Path, Path]] = []
    if not COLOR_DIR.is_dir():
        return []

    for stamp, dlc_path in sorted(dlc_by_stamp.items()):
        video = find_color_video(stamp)
        if video is not None:
            pairs.append((stamp, video, dlc_path))
    return pairs


def find_color_video(stamp: str) -> Path | None:
    """stamp 如 2026-05-17-14-44-35；支持 top{stamp}.mp4 或带空格旧名。"""
    if not COLOR_DIR.is_dir():
        return None
    candidates = [
        COLOR_DIR / f"top{stamp}.mp4",
        COLOR_DIR / f"top{stamp.replace('-', ' ')}.mp4",
    ]
    for p in candidates:
        if p.is_file():
            return p
    for suf in _VIDEO_SUFFIXES:
        hits = sorted(COLOR_DIR.glob(f"top*{stamp}*{suf}"))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            exact = [h for h in hits if _stamp_from_top_name(h.name) == stamp]
            if len(exact) == 1:
                return exact[0]
    return None


def find_dlc_csv(stamp: str) -> Path | None:
    if not DLC_RESULTS_DIR.is_dir():
        return None
    hits = sorted(DLC_RESULTS_DIR.glob(f"top{stamp}DLC_*.csv"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return hits[-1]
    return None


def resolve_pair(
    *,
    stamp: str | None = None,
    video: Path | None = None,
    dlc_csv: Path | None = None,
) -> tuple[Path, Path, str]:
    """解析视频与 DLC 路径；优先显式参数，否则用 stamp 查找。"""
    if video is not None and dlc_csv is not None:
        v, d = Path(video), Path(dlc_csv)
        if not v.is_file():
            raise FileNotFoundError(f"视频不存在: {v.resolve()}")
        if not d.is_file():
            raise FileNotFoundError(f"DLC csv 不存在: {d.resolve()}")
        st = _stamp_from_top_name(v.name) or v.stem
        return v, d, st

    if not stamp:
        raise ValueError("请指定 --stamp，或同时指定 --video 与 --dlc-csv")

    st = stamp.strip()
    if st.startswith("top"):
        st = st[3:]
    v = find_color_video(st)
    d = find_dlc_csv(st)
    if v is None:
        raise FileNotFoundError(
            f"未在 {COLOR_DIR} 找到 stamp={st} 的彩色视频（期望 top{st}.mp4）"
        )
    if d is None:
        raise FileNotFoundError(
            f"未在 {DLC_RESULTS_DIR} 找到 stamp={st} 的 DLC csv（期望 top{st}DLC_*.csv）"
        )
    return v, d, st
