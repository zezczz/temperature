"""按录制时间戳匹配 color / thermal 视频。"""
from __future__ import annotations

import re
from pathlib import Path

from alignment.config import COLOR_DIR, COLOR_PREFIX, THERMAL_DIR, THERMAL_PREFIX

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def _extract_stamp(name: str, prefix: str) -> str | None:
    # top2026-05-17 14-44-35 / top2026-05-17-14-44-35 / temp...
    body = name
    if body.lower().startswith(prefix.lower()):
        body = body[len(prefix) :]
    body = body.lstrip("-_ ")
    m = re.search(r"(\d{4}-\d{2}-\d{2}[\s-]\d{2}-\d{2}-\d{2})", body)
    if not m:
        return None
    return re.sub(r"[\s-]+", "-", m.group(1))


def list_videos(folder: Path, prefix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        stamp = _extract_stamp(p.stem, prefix)
        if stamp:
            out[stamp] = p
    return out


def pair_videos(
    color_dir: Path = COLOR_DIR,
    thermal_dir: Path = THERMAL_DIR,
) -> list[tuple[str, Path, Path]]:
    colors = list_videos(color_dir, COLOR_PREFIX)
    thermals = list_videos(thermal_dir, THERMAL_PREFIX)
    pairs: list[tuple[str, Path, Path]] = []
    for stamp in sorted(set(colors) & set(thermals)):
        pairs.append((stamp, colors[stamp], thermals[stamp]))
    return pairs
