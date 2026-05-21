"""
不依赖 napari 的简易标注工具（OpenCV）。
输出 DLC 标准 CollectedData CSV/H5，可用 deeplabcut.check_labels 检查。

用法:
    python dlc_label_simple.py
    python dlc_label_simple.py --folder "E:\\...\\labeled-data\\top2026-05-17-14-44-35"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
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
    if cfg.CONFIG_PATH and Path(cfg.CONFIG_PATH).is_file():
        return Path(cfg.CONFIG_PATH)
    if cfg.CONFIG_PATH_CACHE.is_file():
        p = Path(cfg.CONFIG_PATH_CACHE.read_text(encoding="utf-8").strip())
        if p.is_file():
            return p
    raise FileNotFoundError("未找到 config.yaml，请先运行 create / prepare-label")


def _is_labeling_folder(folder: Path) -> bool:
    """排除 check_labels 生成的 *_labeled 预览目录。"""
    name = folder.name
    if name.endswith("_labeled"):
        return False
    # 待标注目录里应有原始抽帧 png
    return any(folder.glob("*.png"))


def _find_image_folder(project_dir: Path, folder_arg: str | None) -> Path:
    if folder_arg:
        p = Path(folder_arg)
        if not p.is_dir():
            raise FileNotFoundError(p)
        if p.name.endswith("_labeled"):
            raise SystemExit(
                f"不要用 *_labeled 目录标注: {p}\n"
                "请选不含 _labeled 的文件夹（与视频同名的那个）。"
            )
        return p.resolve()

    labeled = project_dir / "labeled-data"
    all_dirs = [d for d in labeled.iterdir() if d.is_dir()] if labeled.is_dir() else []
    dirs = [d for d in all_dirs if _is_labeling_folder(d)]
    if not dirs:
        raise FileNotFoundError(
            f"{labeled} 下没有可标注文件夹（请先 extract 抽帧；勿选 *_labeled）"
        )
    if len(dirs) == 1:
        return dirs[0].resolve()

    # 优先含 CollectedData 的目录（继续标注时常用）
    with_data = [d for d in dirs if list(d.glob("CollectedData_*.csv"))]
    if len(with_data) == 1:
        return with_data[0].resolve()

    print("存在多个待标注文件夹，请用 --folder 指定（不要选 *_labeled）:")
    for d in dirs:
        print(f"  {d}")
    raise SystemExit(1)


def _relative_image_key(project_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(project_dir)
    return rel.as_posix()


def _load_existing(csv_path: Path, scorer: str, bodyparts: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for idx in df.index.astype(str):
        if not idx.endswith(".png"):
            continue
        pts: dict[str, tuple[float, float]] = {}
        for bp in bodyparts:
            try:
                x = float(df[(scorer, bp, "x")][idx])
                y = float(df[(scorer, bp, "y")][idx])
                if np.isfinite(x) and np.isfinite(y):
                    pts[bp] = (x, y)
            except (KeyError, TypeError, ValueError):
                pass
        out[idx] = pts
    return out


def _save_csv(
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
        row = []
        for bp in bodyparts:
            if bp in pts:
                row.extend([pts[bp][0], pts[bp][1]])
            else:
                row.extend([np.nan, np.nan])
        rows[img_key] = row
    df = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    # 勿设 index.name='image'，否则 CSV 会多出无效行导致 train-dataset 报错
    df.to_csv(csv_path)


def run(folder: Path | None) -> None:
    deeplabcut, auxiliaryfunctions = _import_dlc()
    config_path = _config_path()
    dlc_cfg = auxiliaryfunctions.read_config(str(config_path))
    project_dir = Path(dlc_cfg["project_path"])
    scorer = dlc_cfg["scorer"]
    bodyparts = cfg.get_bodyparts()

    image_dir = _find_image_folder(project_dir, str(folder) if folder else None)
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"{image_dir} 中没有 .png 帧")

    csv_path = image_dir / f"CollectedData_{scorer}.csv"
    annotations = _load_existing(csv_path, scorer, bodyparts)

    print("=== 简易标注（OpenCV，无需 napari）===")
    print(f"文件夹: {image_dir}")
    print(f"帧数: {len(images)} | 关键点: {', '.join(bodyparts)}")
    print("按键: 1-9 选点 | 左键标点 | n 下一帧 | p 上一帧 | s 保存 | q 退出")
    print()

    win = "DLC label (simple)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    idx = 0
    bp_i = 0
    state = {"bp_i": 0}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and param is not None:
            pts_map, bp_name = param
            pts_map[bp_name] = (float(x), float(y))

    while 0 <= idx < len(images):
        img_path = images[idx]
        img_key = _relative_image_key(project_dir, img_path)
        frame = cv2.imread(str(img_path))
        if frame is None:
            idx += 1
            continue

        pts = annotations.setdefault(img_key, {})

        while True:
            bp_i = state["bp_i"]
            bp = bodyparts[bp_i]
            cv2.setMouseCallback(win, _on_mouse, (pts, bp))

            vis = frame.copy()
            for name, (x, y) in pts.items():
                color = (0, 255, 0) if name == bp else (0, 200, 255)
                cv2.circle(vis, (int(x), int(y)), 6, color, -1)
                cv2.putText(vis, name, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            status = (
                f"[{idx + 1}/{len(images)}] {img_path.name} | "
                f"当前点: {bp} ({bp_i + 1}/{len(bodyparts)}) | s保存 q退出"
            )
            cv2.putText(vis, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                _save_csv(csv_path, scorer, bodyparts, annotations)
                deeplabcut.convertcsv2h5(str(config_path), scorer=scorer, userfeedback=False)
                cv2.destroyAllWindows()
                print(f"\n已保存: {csv_path}")
                print("已转换 H5。可运行: python dlc_workflow.py check")
                return

            if key == ord("s"):
                _save_csv(csv_path, scorer, bodyparts, annotations)
                print(f"已保存 ({idx + 1}/{len(images)}) -> {csv_path.name}")

            if key == ord("n"):
                idx += 1
                break
            if key == ord("p"):
                idx = max(0, idx - 1)
                break

            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                bi = key - ord("1")
                if bi < len(bodyparts):
                    state["bp_i"] = bi

            if key == 9:  # Tab
                state["bp_i"] = (state["bp_i"] + 1) % len(bodyparts)

    _save_csv(csv_path, scorer, bodyparts, annotations)
    deeplabcut.convertcsv2h5(str(config_path), scorer=scorer, userfeedback=False)
    cv2.destroyAllWindows()
    print(f"\n全部完成: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV 简易标注（不依赖 napari）")
    parser.add_argument("--folder", type=str, default=None, help="labeled-data 下的帧文件夹")
    args = parser.parse_args()
    run(Path(args.folder) if args.folder else None)


if __name__ == "__main__":
    main()
