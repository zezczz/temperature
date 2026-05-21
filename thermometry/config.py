"""thermometry 模块的路径与默认参数。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# alignment.dlc sample 产出的 intensity csv 默认在这里
INTENSITY_DIR = DATA_DIR / "aligned" / "dlc"
# 温度 csv 输出位置（与 intensity csv 同目录，便于配对查找）
TEMPERATURE_DIR = DATA_DIR / "aligned" / "dlc"
# 汇总表与叠加视频
SUMMARY_DIR = DATA_DIR / "aligned" / "temperature"
OVERLAY_DIR = DATA_DIR / "aligned" / "temperature_overlay"

MODULE_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = MODULE_DIR / "calibration.json"
ANCHORS_PATH = MODULE_DIR / "anchors.json"

# 默认聚合方式：把若干 bodypart 的温度聚合成一个"小鼠体温"
# max | mean | median | quantile
DEFAULT_AGGREGATION = "max"
DEFAULT_QUANTILE = 0.9

# 默认置信度阈值（低于该 likelihood 的 bodypart 不参与计算）
DEFAULT_P_CUTOFF = 0.5
