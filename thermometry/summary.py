"""聚合每段视频的体温统计。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from thermometry.config import TEMPERATURE_DIR


def summarize_csv(path: Path) -> dict:
    df = pd.read_csv(path, index_col=0)
    row: dict = {"file": path.name, "n_frames": int(len(df))}
    cols = [c for c in df.columns if c.endswith("_temperature") or c == "body_temperature"]
    for col in cols:
        s = df[col].dropna()
        if s.empty:
            row[f"{col}_mean"] = float("nan")
            row[f"{col}_median"] = float("nan")
            row[f"{col}_max"] = float("nan")
            row[f"{col}_p90"] = float("nan")
            row[f"{col}_n"] = 0
            continue
        row[f"{col}_mean"] = float(s.mean())
        row[f"{col}_median"] = float(s.median())
        row[f"{col}_max"] = float(s.max())
        row[f"{col}_p90"] = float(s.quantile(0.9))
        row[f"{col}_n"] = int(s.size)
    return row


def summarize_dir(
    directory: Path | None = None,
    pattern: str = "*_temperature.csv",
) -> pd.DataFrame:
    directory = Path(directory) if directory else TEMPERATURE_DIR
    rows: list[dict] = []
    for p in sorted(directory.glob(pattern)):
        rows.append(summarize_csv(p))
    return pd.DataFrame(rows)
