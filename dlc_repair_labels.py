"""
修复简易标注生成的 CollectedData，使其符合 DLC train-dataset 要求。

用法:
    python dlc_repair_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import dlc_config as cfg


def _import_dlc():
    try:
        import deeplabcut
        from deeplabcut.utils import auxiliaryfunctions
    except ImportError as exc:
        print("需要 deeplabcut:", exc, file=sys.stderr)
        raise SystemExit(1) from exc
    return deeplabcut, auxiliaryfunctions


def _config_path() -> Path:
    if cfg.CONFIG_PATH_CACHE.is_file():
        p = Path(cfg.CONFIG_PATH_CACHE.read_text(encoding="utf-8").strip())
        if p.is_file():
            return p
    raise FileNotFoundError("未找到 config.yaml")


def _label_folder(project_dir: Path) -> Path:
    if cfg.LABEL_FRAMES_DIR and Path(cfg.LABEL_FRAMES_DIR).is_dir():
        return Path(cfg.LABEL_FRAMES_DIR)
    labeled = project_dir / "labeled-data"
    for d in labeled.iterdir():
        if d.is_dir() and not d.name.endswith("_labeled") and any(d.glob("*.png")):
            return d
    raise FileNotFoundError("未找到标注文件夹")


def _parse_raw_csv(csv_path: Path, bodyparts: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """从当前（可能损坏的）CSV 按行解析：每行 6 个数字 → 3 个关键点。"""
    text = csv_path.read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in text[3:] if ln.strip()]
    out: dict[str, dict[str, tuple[float, float]]] = {}
    n_coords = len(bodyparts) * 2

    for line in data_lines:
        parts = line.split(",")
        if len(parts) < 1 + n_coords:
            continue
        img_key = parts[0].strip()
        if not img_key.endswith(".png"):
            continue
        nums = []
        for p in parts[1 : 1 + n_coords]:
            p = p.strip()
            if not p:
                nums.append(np.nan)
            else:
                try:
                    nums.append(float(p))
                except ValueError:
                    nums.append(np.nan)
        if len(nums) != n_coords:
            continue
        pts: dict[str, tuple[float, float]] = {}
        for i, bp in enumerate(bodyparts):
            x, y = nums[i * 2], nums[i * 2 + 1]
            if np.isfinite(x) and np.isfinite(y):
                pts[bp] = (x, y)
        if pts:
            out[img_key] = pts
    return out


def _save_dlc_csv(
    csv_path: Path,
    scorer: str,
    bodyparts: list[str],
    annotations: dict[str, dict[str, tuple[float, float]]],
) -> None:
    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    rows = {}
    for img_key, pts in sorted(annotations.items()):
        if not str(img_key).endswith(".png"):
            continue
        row = []
        for bp in bodyparts:
            if bp in pts:
                row.extend([pts[bp][0], pts[bp][1]])
            else:
                row.extend([np.nan, np.nan])
        rows[str(img_key)] = row
    df = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    # 不要 index.name='image'，否则会多出无效行
    df.to_csv(csv_path)


def main() -> None:
    deeplabcut, auxiliaryfunctions = _import_dlc()
    config_path = _config_path()
    dlc_cfg = auxiliaryfunctions.read_config(str(config_path))
    project_dir = Path(dlc_cfg["project_path"])
    scorer = dlc_cfg["scorer"]
    bodyparts = list(dlc_cfg["bodyparts"])

    folder = _label_folder(project_dir)
    csv_path = folder / f"CollectedData_{scorer}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    annotations = _parse_raw_csv(csv_path, bodyparts)
    if not annotations:
        raise SystemExit("未能解析到有效标注，请检查 CSV。")

    backup = csv_path.with_suffix(".csv.bak")
    backup.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    _save_dlc_csv(csv_path, scorer, bodyparts, annotations)
    deeplabcut.convertcsv2h5(str(config_path), scorer=scorer, userfeedback=False)

    complete = sum(
        1
        for pts in annotations.values()
        if all(bp in pts for bp in bodyparts)
    )
    print(f"已修复: {csv_path}")
    print(f"  备份: {backup}")
    print(f"  有效图片行: {len(annotations)}")
    print(f"  3 点全标齐: {complete} 帧")
    print(f"  未标全的帧不会进入训练（DLC 会自动跳过 NaN）")
    print("\n下一步: python dlc_workflow.py train-dataset")


if __name__ == "__main__":
    main()
