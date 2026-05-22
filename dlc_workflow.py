"""
DeepLabCut 标定工作流（创建工程 → 抽帧 → 人工标注 → 检查）。

用法（在已安装 deeplabcut 的 conda 环境中）:
    python dlc_workflow.py merge-videos    # 合并 data/color 分段为一条训练视频
    python dlc_workflow.py create          # 创建工程（仅首次）
    python dlc_workflow.py sync-bodyparts  # 把 dlc_config 里的关键点写入 config.yaml
    python dlc_workflow.py prepare-label   # merge + sync + 抽帧（推荐标定前执行）
    python dlc_workflow.py label           # 打开标注 GUI
    python dlc_workflow.py check           # 检查标注
    python dlc_workflow.py train-dataset   # 生成训练集
    python dlc_workflow.py train           # 训练网络
    python dlc_workflow.py evaluate        # 评估精度
    python dlc_workflow.py analyze         # 推理（结果写入 data/dlc_results/）
    python dlc_workflow.py plot            # 生成可视化视频（同上目录）
    python dlc_workflow.py organize-results  # 把混在 data/color 里的旧结果搬过去

单只动物 · 默认 3 点：left_eye, right_eye, tail_base
可选第 4 点：在 dlc_config.py 设 INCLUDE_TAIL_MIDDLE = True 后运行 sync-bodyparts
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import dlc_gui_bootstrap

dlc_gui_bootstrap.apply()

import dlc_config as cfg

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}


def _import_dlc():
    try:
        import deeplabcut  # noqa: E402
    except ImportError as exc:
        print(
            "未找到 deeplabcut。请先激活安装 DLC 的 conda 环境，例如:\n"
            "  conda activate DEEPLABCUT\n"
            "  python dlc_workflow.py create",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return deeplabcut


def normalize_video_filenames(directory: Path) -> None:
    """将目录内带空格的视频名改为 DLC/YAML 安全名，并去掉与安全名重复的副本。"""
    if not directory.is_dir():
        return
    exclude = getattr(cfg, "VIDEO_EXCLUDE_NAMES", frozenset())
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if path.name in exclude:
            continue
        safe_name = cfg.sanitize_video_filename(path.name)
        if safe_name == path.name:
            continue
        safe_path = path.parent / safe_name
        if safe_path.exists():
            print(f"删除重复视频: {path.name}（保留 {safe_name}）")
            path.unlink()
        else:
            path.rename(safe_path)
            print(f"重命名视频: {path.name} -> {safe_name}")


def _merged_video_path(data_dir: Path) -> Path:
    return data_dir / cfg.MERGED_VIDEO_NAME


def _is_merged_video(path: Path) -> bool:
    return path.name == cfg.MERGED_VIDEO_NAME


def find_source_clips(data_dir: Path) -> list[Path]:
    """data/color 下的原始分段视频（不含合并后的训练长视频）。"""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    normalize_video_filenames(data_dir)
    exclude = set(getattr(cfg, "VIDEO_EXCLUDE_NAMES", frozenset()))
    exclude.add(cfg.MERGED_VIDEO_NAME)
    seen_safe: set[str] = set()
    clips: list[Path] = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if p.name in exclude or _is_merged_video(p):
            continue
        safe_key = cfg.sanitize_video_filename(p.name)
        if safe_key in seen_safe:
            continue
        seen_safe.add(safe_key)
        clips.append(p.resolve())
    if not clips:
        raise FileNotFoundError(
            f"在 {data_dir} 下未找到分段视频，支持后缀: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )
    return clips


def find_videos(data_dir: Path) -> list[str]:
    """供 DLC 建工程/抽帧/训练：默认同源多段先合成一条。"""
    clips = find_source_clips(data_dir)
    if getattr(cfg, "USE_MERGED_VIDEO_FOR_TRAINING", False):
        merged = ensure_merged_video(data_dir, clips)
        return [str(merged.resolve())]
    return [str(p) for p in clips]


def find_analyze_videos(data_dir: Path) -> list[str]:
    """推理/可视化：默认仍用各原始分段，不用合并长视频。"""
    if getattr(cfg, "ANALYZE_USE_MERGED_VIDEO", False):
        return find_videos(data_dir)
    return [str(p) for p in find_source_clips(data_dir)]


def _find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for candidate in (
        Path(r"E:\Tools\ffmpeg-2026-05-13-git-a327bc0561-essentials_build\bin\ffmpeg.exe"),
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("未找到 ffmpeg，请安装并加入 PATH")


def ensure_merged_video(data_dir: Path, clips: list[Path] | None = None) -> Path:
    """按时间顺序无损拼接分段视频，生成一条训练用长视频。"""
    clips = clips or find_source_clips(data_dir)
    merged = _merged_video_path(data_dir)
    newest_clip_mtime = max(p.stat().st_mtime for p in clips)
    if merged.is_file() and merged.stat().st_mtime >= newest_clip_mtime:
        print(f"已存在最新合并视频，跳过: {merged}")
        return merged

    list_file = data_dir / ".merge_concat_list.txt"
    lines = []
    for clip in clips:
        path = str(clip.resolve()).replace("\\", "/").replace("'", "''")
        lines.append(f"file '{path}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ffmpeg = _find_ffmpeg()
    print(f"正在合并 {len(clips)} 段视频 -> {merged.name}")
    for clip in clips:
        print(f"  + {clip.name}")
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(merged),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 合并失败（exit {proc.returncode}）:\n{(proc.stderr or proc.stdout)[-2000:]}"
        )
    print(f"合并完成: {merged} ({merged.stat().st_size / 1024 / 1024:.1f} MB)")
    return merged


def _ensure_inference_dir() -> Path:
    out = cfg.INFERENCE_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _print_inference_layout() -> None:
    print("\n推理结果目录:")
    print(f"  {cfg.INFERENCE_OUTPUT_DIR}")
    print("    · *DLC*.csv / *.h5  — 坐标")
    print("    · *_labeled.mp4     — 可视化视频")
    print(f"  原始视频: {cfg.DATA_DIR}")


def _is_dlc_result_file(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    if _is_merged_video(path):
        return False
    if "DLC" in name:
        return True
    if path.suffix.lower() in {".h5", ".pickle"}:
        return True
    if path.suffix.lower() == ".mp4" and ("labeled" in lower or "superimposed" in lower):
        return True
    return False


def organize_legacy_results_in_color() -> None:
    """将误写在 data/color 下的 DLC 结果移到 data/dlc_results/。"""
    out = _ensure_inference_dir()
    moved = 0
    for path in sorted(cfg.DATA_DIR.iterdir()):
        if not path.is_file() or not _is_dlc_result_file(path):
            continue
        dest = out / path.name
        if dest.exists():
            print(f"跳过（目标已存在）: {path.name}")
            continue
        path.rename(dest)
        print(f"已移动: {path.name}")
        moved += 1
    print(f"\n共整理 {moved} 个文件。")
    _print_inference_layout()


def step_merge_videos() -> Path:
    merged = ensure_merged_video(cfg.DATA_DIR)
    clips = find_source_clips(cfg.DATA_DIR)
    print(
        f"\n训练将使用 1 条合并视频；抽帧目标 {cfg.num_frames_to_extract(len(clips))} 帧"
        f"（{len(clips)} 段 × {cfg.FRAMES_PER_SOURCE_VIDEO} 帧/段）"
    )
    return merged


def _write_video_sets_block(config_yaml: Path, video_sets: dict[str, dict[str, str]]) -> None:
    """直接重写 config.yaml 的 video_sets 段（edit_config 无法读取已损坏的 yaml）。"""
    lines = [
        "video_sets:",
        *(
            f"  {path}:\n    crop: {info['crop']}"
            for path, info in video_sets.items()
        ),
    ]
    block = "\n".join(lines) + "\n"
    text = config_yaml.read_text(encoding="utf-8")
    start = text.index("video_sets:")
    end = text.index("bodyparts:")
    config_yaml.write_text(text[:start] + block + text[end:], encoding="utf-8")


def repair_config_video_sets(config_yaml: Path) -> None:
    """按 videos/ 目录重建 video_sets，修复因文件名含空格导致的坏 config.yaml。"""
    project_dir = config_yaml.parent
    videos_dir = project_dir / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"未找到 videos 目录: {videos_dir}")

    normalize_video_filenames(videos_dir)
    video_sets = {
        str(p.resolve()): {"crop": "0, 1920, 0, 1080"}
        for p in sorted(videos_dir.glob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    }
    if not video_sets:
        raise FileNotFoundError(f"{videos_dir} 下没有视频文件")

    try:
        deeplabcut = _import_dlc()
        deeplabcut.auxiliaryfunctions.read_plainconfig(str(config_yaml))
        deeplabcut.auxiliaryfunctions.edit_config(str(config_yaml), {"video_sets": video_sets})
    except Exception:
        _write_video_sets_block(config_yaml, video_sets)
    print(f"已修复 config.yaml 中 {len(video_sets)} 个视频路径")


def _save_config_cache(path: Path) -> None:
    cfg.CONFIG_PATH_CACHE.write_text(str(path.resolve()), encoding="utf-8")


def _load_config_cache() -> Path | None:
    if not cfg.CONFIG_PATH_CACHE.is_file():
        return None
    cached = Path(cfg.CONFIG_PATH_CACHE.read_text(encoding="utf-8").strip())
    return cached if cached.is_file() else None


def config_path() -> Path:
    """解析 config.yaml。DLC 3.x 目录形如 ProjectName-experimenter-日期，与 2.x 不同。"""
    if cfg.CONFIG_PATH is not None:
        if cfg.CONFIG_PATH.is_file():
            return cfg.CONFIG_PATH.resolve()
        raise FileNotFoundError(f"dlc_config.CONFIG_PATH 不存在: {cfg.CONFIG_PATH}")

    cached = _load_config_cache()
    if cached is not None:
        return cached

    if not cfg.DLC_PROJECT_DIR.is_dir():
        raise FileNotFoundError(
            f"未找到工程目录 {cfg.DLC_PROJECT_DIR}\n请先运行: python dlc_workflow.py create"
        )

    candidates = sorted(cfg.DLC_PROJECT_DIR.rglob("config.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"在 {cfg.DLC_PROJECT_DIR} 下未找到 config.yaml\n请先运行: python dlc_workflow.py create"
        )

    name_key = cfg.PROJECT_NAME.lower()
    exp_key = cfg.EXPERIMENTER.lower()
    matched = [
        p
        for p in candidates
        if name_key in p.parent.name.lower() and exp_key in p.parent.name.lower()
    ]
    pool = matched if matched else candidates
    chosen = max(pool, key=lambda p: p.stat().st_mtime)
    _save_config_cache(chosen)
    return chosen.resolve()


def _project_config_dict() -> dict:
    clip_count = len(find_source_clips(cfg.DATA_DIR))
    frames = cfg.num_frames_to_extract(clip_count)
    return {
        "bodyparts": cfg.get_bodyparts(),
        "skeleton": cfg.get_skeleton(),
        "numframes2pick": frames,
        "multianimalproject": cfg.MULTIANIMAL,
    }


def step_sync_bodyparts(deeplabcut, config: str) -> None:
    parts = cfg.get_bodyparts()
    deeplabcut.auxiliaryfunctions.edit_config(config, _project_config_dict())
    print("已写入 config.yaml:")
    print(f"  单只动物: {not cfg.MULTIANIMAL} (multianimalproject={cfg.MULTIANIMAL})")
    print(f"  关键点 ({len(parts)}): {', '.join(parts)}")
    if not cfg.INCLUDE_TAIL_MIDDLE:
        print("  未启用 tail_middle；需要时在 dlc_config.py 设 INCLUDE_TAIL_MIDDLE=True 后再次 sync")


def step_create(deeplabcut) -> str:
    normalize_video_filenames(cfg.DATA_DIR)
    if getattr(cfg, "USE_MERGED_VIDEO_FOR_TRAINING", False):
        step_merge_videos()
    videos = find_videos(cfg.DATA_DIR)
    cfg.DLC_PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    print("将使用以下视频创建工程:")
    for v in videos:
        print(f"  - {v}")

    config = deeplabcut.create_new_project(
        cfg.PROJECT_NAME,
        cfg.EXPERIMENTER,
        videos,
        working_directory=str(cfg.DLC_PROJECT_DIR),
        copy_videos=cfg.COPY_VIDEOS,
        multianimal=cfg.MULTIANIMAL,
    )

    deeplabcut.auxiliaryfunctions.edit_config(config, _project_config_dict())

    config_path_resolved = Path(config).resolve()
    _save_config_cache(config_path_resolved)

    print(f"\n工程已创建，配置文件:\n  {config_path_resolved}")
    print("\n下一步: python dlc_workflow.py extract")
    return str(config_path_resolved)


def step_extract(deeplabcut, config: str) -> None:
    step_sync_bodyparts(deeplabcut, config)
    deeplabcut.extract_frames(
        config,
        mode=cfg.EXTRACT_MODE,
        algo=cfg.EXTRACT_ALGO,
        crop=False,
        userfeedback=False,
    )
    print("\n抽帧完成。下一步: python dlc_workflow.py label")


def _labeled_frame_dirs(config_yaml: Path) -> list[Path]:
    labeled_root = config_yaml.parent / "labeled-data"
    if not labeled_root.is_dir():
        return []
    return [d for d in labeled_root.iterdir() if d.is_dir()]


def _count_png_frames(config_yaml: Path) -> tuple[int, list[Path]]:
    folders = _labeled_frame_dirs(config_yaml)
    counts = [(d, len(list(d.glob("*.png")))) for d in folders]
    total = sum(n for _, n in counts)
    return total, [d for d, n in counts if n > 0]


def step_check_label_ready(config_yaml: Path) -> None:
    total, folders = _count_png_frames(config_yaml)
    if total == 0:
        raise FileNotFoundError(
            "labeled-data 下没有抽出的 .png 帧。\n"
            "请先运行: python dlc_workflow.py prepare-label"
        )
    print(f"待标注图片: {total} 张（{len(folders)} 个文件夹）")
    for folder in folders:
        print(f"  - {folder}")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _try_import_version(mod_name: str, label: str, timeout_sec: int = 90) -> bool:
    """在子进程中导入模块，避免 diagnose 主进程被卡死。"""
    _log(f"正在检测 {label}（最多等待 {timeout_sec}s）...")
    code = (
        "import dlc_gui_bootstrap; dlc_gui_bootstrap.apply()\n"
        f"import {mod_name} as m\n"
        "print(getattr(m, '__version__', 'ok'))\n"
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(cfg.PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        _log(f"  ✗ {label}: 导入超时（>{timeout_sec}s）—— 这就是 label 卡住的常见原因")
        return False

    elapsed = time.time() - t0
    if proc.returncode == 0:
        ver = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else "ok"
        _log(f"  ✓ {label}: {ver} （{elapsed:.1f}s）")
        return True

    err = (proc.stderr or proc.stdout or "未知错误").strip()
    _log(f"  ✗ {label}: 导入失败\n    {err[:500]}")
    return False


def step_diagnose(deeplabcut) -> None:
    """检查 napari 标注 GUI 依赖，不打开完整标注窗口。"""
    _log("=== DeepLabCut / GUI 环境检查 ===\n")
    _log(f"deeplabcut: {deeplabcut.__version__}")
    _log(f"环境变量: QT_API={sys.environ.get('QT_API')}, "
         f"NAPARI_DISABLE_PLUGIN_DISCOVERY={sys.environ.get('NAPARI_DISABLE_PLUGIN_DISCOVERY')}")

    spec = importlib.util.find_spec("napari_deeplabcut")
    if spec is None:
        _log("\nnapari-deeplabcut: 未安装 → pip install napari-deeplabcut")
    else:
        _log(f"\nnapari-deeplabcut 路径: {spec.origin}")

    ok_qt = _try_import_version("qtpy", "qtpy", timeout_sec=30)
    if ok_qt:
        try:
            import qtpy

            _log(f"  Qt 后端: {qtpy.API_NAME}")
            if qtpy.API_NAME == "PySide6":
                import PySide6

                _log(f"  PySide6: {PySide6.__version__}")
        except Exception as exc:
            _log(f"  Qt 详情读取失败: {exc}")

    ok_napari = _try_import_version("napari", "napari", timeout_sec=120)
    ok_ndlc = _try_import_version("napari_deeplabcut", "napari-deeplabcut", timeout_sec=120)

    try:
        config = config_path()
        step_check_label_ready(config)
    except FileNotFoundError as exc:
        _log(f"\n数据检查: {exc}")

    _log("\n=== 结论 ===")
    if not ok_ndlc:
        _log(
            "napari-deeplabcut 无法正常导入，label 会一直卡在「启用 GUI」。\n"
            "请在当前 conda 环境中运行修复脚本后重试:\n"
            "  .\\fix_napari_gui.ps1\n"
            "或手动:\n"
            '  pip install "PySide6==6.10.2" "shiboken6==6.10.2" '
            '"napari==0.6.6" "napari-deeplabcut==0.2.1.8"\n'
            "仍失败可尝试升级 DLC:\n"
            '  pip install --upgrade "deeplabcut[gui]>=3.0.0rc14"'
        )
    elif ok_napari and ok_ndlc:
        _log(
            "依赖导入正常。运行 label 时终端会阻塞直到关闭 napari，属正常。\n"
            "首次打开 napari 仍可能需 1～3 分钟，请看任务栏。"
        )


def step_label(deeplabcut, config: str) -> None:
    config_yaml = Path(config)
    step_check_label_ready(config_yaml)
    total, _ = _count_png_frames(config_yaml)

    parts = cfg.get_bodyparts()
    print("\n=== 启动 napari 标注 GUI ===")
    print("【正常现象】")
    print("  1. 终端会停在这里不动，直到你关闭 napari 窗口（不是卡死）。")
    print("  2. 第一次打开可能要 1～5 分钟，正在加载 Qt / napari / 图片。")
    print("  3. 窗口可能在后台：请看任务栏「napari」图标，或 Alt+Tab 切换。")
    print(f"\n待标 {total} 帧 × {len(parts)} 点: {' → '.join(parts)}")
    print("操作: 键盘切换当前点 | 左键落点 | 下一帧 | 全部完成后关闭 napari\n")

    deeplabcut.label_frames(config)
    print("\n标注 GUI 已关闭。若已保存，可运行: python dlc_workflow.py check")


def step_check(deeplabcut, config: str) -> None:
    deeplabcut.check_labels(config, draw_skeleton=True)


def step_train_dataset(deeplabcut, config: str) -> None:
    deeplabcut.create_training_dataset(config, net_type="resnet_50")
    print("\n训练集已生成。下一步:")
    print("  python dlc_workflow.py train")


def step_train(deeplabcut, config: str) -> None:
    print(f"开始训练（最多 {cfg.TRAIN_MAXITERS} 轮，有 GPU 会快很多）...")
    print("训练过程中终端会持续输出 loss，属正常现象。\n")
    deeplabcut.train_network(config, maxiters=cfg.TRAIN_MAXITERS)
    print("\n训练完成。下一步:")
    print("  python dlc_workflow.py evaluate")
    print("  python dlc_workflow.py analyze")


def step_evaluate(deeplabcut, config: str) -> None:
    deeplabcut.evaluate_network(config, plotting=True)
    print("\n评估完成。查看 dlc-models 下 evaluation-results 中的图表。")
    print("若误差偏大，可补标更多帧后重新 train-dataset → train。")


def step_analyze(deeplabcut, config: str) -> None:
    videos = find_analyze_videos(cfg.DATA_DIR) if cfg.ANALYZE_VIDEOS_FROM_DATA else []
    if not videos:
        raise FileNotFoundError(f"在 {cfg.DATA_DIR} 下未找到待分析视频")
    out = _ensure_inference_dir()
    print("将对以下视频做姿态估计:")
    for v in videos:
        print(f"  - {v}")
    deeplabcut.analyze_videos(
        config,
        videos,
        save_as_csv=True,
        destfolder=str(out),
    )
    print("\n分析完成。")
    _print_inference_layout()
    print("下一步: python dlc_workflow.py plot")


def step_plot(deeplabcut, config: str) -> None:
    videos = find_analyze_videos(cfg.DATA_DIR) if cfg.ANALYZE_VIDEOS_FROM_DATA else []
    if not videos:
        raise FileNotFoundError(f"在 {cfg.DATA_DIR} 下未找到视频")
    out = _ensure_inference_dir()
    deeplabcut.create_labeled_video(
        config,
        videos,
        destfolder=str(out),
    )
    print("\n可视化视频已生成。")
    _print_inference_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLabCut 标定工作流")
    parser.add_argument(
        "step",
        choices=[
            "merge-videos",
            "create",
            "sync-bodyparts",
            "extract",
            "prepare-label",
            "diagnose",
            "label",
            "label-simple",
            "check",
            "repair-labels",
            "repair-config",
            "train-dataset",
            "train",
            "evaluate",
            "analyze",
            "plot",
            "organize-results",
            "all",
        ],
        help="要执行的步骤",
    )
    args = parser.parse_args()
    deeplabcut = _import_dlc()

    if args.step == "merge-videos":
        step_merge_videos()
        return
    if args.step == "create":
        step_create(deeplabcut)
        return

    config = str(config_path())

    if args.step == "sync-bodyparts":
        step_sync_bodyparts(deeplabcut, config)
    elif args.step == "extract":
        step_extract(deeplabcut, config)
    elif args.step == "prepare-label":
        if getattr(cfg, "USE_MERGED_VIDEO_FOR_TRAINING", False):
            step_merge_videos()
        repair_config_video_sets(Path(config))
        step_sync_bodyparts(deeplabcut, config)
        step_extract(deeplabcut, config)
        print("\n标定准备完成。下一步: python dlc_workflow.py label")
    elif args.step == "diagnose":
        step_diagnose(deeplabcut)
        return
    elif args.step == "label":
        step_label(deeplabcut, config)
    elif args.step == "label-simple":
        import subprocess

        script = cfg.PROJECT_ROOT / "dlc_label_simple.py"
        cmd = [sys.executable, str(script)]
        if cfg.LABEL_FRAMES_DIR is not None:
            cmd.extend(["--folder", str(cfg.LABEL_FRAMES_DIR)])
        raise SystemExit(subprocess.call(cmd))
    elif args.step == "check":
        step_check(deeplabcut, config)
    elif args.step == "repair-labels":
        import subprocess

        script = cfg.PROJECT_ROOT / "dlc_repair_labels.py"
        raise SystemExit(subprocess.call([sys.executable, str(script)]))
    elif args.step == "repair-config":
        repair_config_video_sets(config_path())
        print("\n修复完成。可重新运行: python dlc_workflow.py prepare-label")
    elif args.step == "train-dataset":
        step_train_dataset(deeplabcut, config)
    elif args.step == "train":
        step_train(deeplabcut, config)
    elif args.step == "evaluate":
        step_evaluate(deeplabcut, config)
    elif args.step == "analyze":
        step_analyze(deeplabcut, config)
    elif args.step == "plot":
        step_plot(deeplabcut, config)
    elif args.step == "organize-results":
        organize_legacy_results_in_color()
    elif args.step == "all":
        try:
            config = str(config_path())
        except FileNotFoundError:
            config = step_create(deeplabcut)
        step_extract(deeplabcut, config)
        print("\n已创建工程并抽帧。请运行: python dlc_workflow.py label")


if __name__ == "__main__":
    main()
