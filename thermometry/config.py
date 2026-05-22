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

# --- tail_baseline 估计方案 ---
TAIL_BASELINE_TAIL_BP = "tail_base"
TAIL_BASELINE_EYE_PARTS = ("left_eye", "right_eye")
TAIL_BASELINE_TAIL_WEIGHT = 0.65  # 置信度里尾部追踪占权
TAIL_BASELINE_W_EYE = 0.35  # 眼部一致性可提升的置信度上限
TAIL_BASELINE_AGREEMENT_SIGMA = 2.0  # 眼温与尾温差 >2°C 时置信度明显衰减 (°C)
TAIL_BASELINE_EYE_SMOOTH_SEC = 1.0  # 眼均温时间平滑窗口（秒）
DEFAULT_THERMAL_FPS = 60.0  # 热成像视频帧率，用于把「秒」换算成滚动窗口帧数
DEFAULT_ESTIMATION_SCHEME = "tail_baseline"  # tail_baseline | legacy_max
