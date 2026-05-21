"""
DeepLabCut 项目配置 — 单只动物，俯视追踪双眼与尾巴。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "color"
DLC_PROJECT_DIR = PROJECT_ROOT / "dlc_project"

CONFIG_PATH: Path | None = None
CONFIG_PATH_CACHE = PROJECT_ROOT / ".dlc_config_path"

PROJECT_NAME = "TemperatureTopView"
EXPERIMENTER = "lab"

# 单只动物（False）；同屏多只才设 True
MULTIANIMAL = False

# 是否标注尾巴中部（第 4 个点）。建议：先 False 练通流程，稳定后再改 True 并 sync-bodyparts
INCLUDE_TAIL_MIDDLE = False

# 核心 3 点：左眼、右眼、尾根（俯视时眼睛在头部两侧）
BODYPARTS_CORE = [
    "left_eye",
    "right_eye",
    "tail_base",
]

BODYPARTS_OPTIONAL = [
    "tail_middle",
]


def get_bodyparts() -> list[str]:
    parts = list(BODYPARTS_CORE)
    if INCLUDE_TAIL_MIDDLE:
        parts.append(BODYPARTS_OPTIONAL[0])
    return parts


def get_skeleton() -> list[list[str]]:
    """骨架连线，仅用于标注/检查时的可视化。"""
    sk = [
        ["left_eye", "right_eye"],
        ["left_eye", "tail_base"],
        ["right_eye", "tail_base"],
    ]
    if INCLUDE_TAIL_MIDDLE:
        sk.append(["tail_base", "tail_middle"])
    return sk


# 兼容旧脚本中的名称
BODYPARTS = get_bodyparts()
SKELETON = get_skeleton()

# create 时复制视频到工程 videos/（不移动 data/color 原文件）
COPY_VIDEOS = True

# 已统一为无空格文件名后可为空；仅用于显式跳过个别文件
VIDEO_EXCLUDE_NAMES: frozenset[str] = frozenset()


def sanitize_video_filename(name: str) -> str:
    """top2026-05-17 14-44-35.mp4 -> top2026-05-17-14-44-35.mp4（避免 config.yaml 解析失败）。"""
    return name.replace(" ", "-")


# 标定/训练：把 data/color 下多段同源视频先合成一条再建 DLC 工程（推荐同一实验批次）
USE_MERGED_VIDEO_FOR_TRAINING = True
MERGED_VIDEO_NAME = "top2026-05-17-merged.mp4"

# 合成训练视频时：按「每段原视频抽多少帧」换算总抽帧数（12 段 × 12 帧 = 144）
FRAMES_PER_SOURCE_VIDEO = 12

# 不合成时：每个视频各自抽帧的数量
NUM_FRAMES_TO_EXTRACT = 12


def num_frames_to_extract(source_clip_count: int) -> int:
    if USE_MERGED_VIDEO_FOR_TRAINING and source_clip_count > 0:
        return FRAMES_PER_SOURCE_VIDEO * source_clip_count
    return NUM_FRAMES_TO_EXTRACT

EXTRACT_MODE = "automatic"
EXTRACT_ALGO = "kmeans"

# PyTorch 引擎下与 pytorch_config.yaml 的 epochs 对齐
TRAIN_MAXITERS = 200

# 推理/出图：仍对 data/color 下各原始分段视频分析（不含合并后的训练用长视频）
ANALYZE_VIDEOS_FROM_DATA = True
ANALYZE_USE_MERGED_VIDEO = False

# 多视频标定时由脚本自动选择 labeled-data 子目录
LABEL_FRAMES_DIR: Path | None = None
