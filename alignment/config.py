"""双相机对齐：路径与分辨率常量。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COLOR_DIR = DATA_DIR / "color"
THERMAL_DIR = DATA_DIR / "thermal"
OUTPUT_DIR = DATA_DIR / "aligned"

TRANSFORM_PATH = Path(__file__).resolve().parent / "transform.json"

COLOR_SIZE = (1920, 1080)  # width, height
THERMAL_SIZE = (1440, 1080)

COLOR_PREFIX = "top"
THERMAL_PREFIX = "temp"

# 标定默认使用的视频对（时间戳一致）
DEFAULT_CALIB_COLOR = COLOR_DIR / "top2026-05-17 14-44-35.mp4"
DEFAULT_CALIB_THERMAL = THERMAL_DIR / "temp2026-05-17 14-44-35.mp4"
DEFAULT_CALIB_FRAME = 100
