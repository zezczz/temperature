"""respiration 模块路径与默认参数。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COLOR_DIR = DATA_DIR / "color"
DLC_RESULTS_DIR = DATA_DIR / "dlc_results"

# 胸腔 ROI 信号与呼吸率结果
OUTPUT_DIR = DATA_DIR / "aligned" / "respiration"
SIGNAL_DIR = OUTPUT_DIR
RATE_DIR = OUTPUT_DIR

# 用双眼中点 → 尾根 定义身体轴向；胸腔中心沿该轴的比例（0=眼、1=尾）
DEFAULT_CHEST_FRACTION = 0.38
# ROI 半宽 ≈ 眼间距 × 该系数；半长 ≈ 眼–尾距离 × 该系数
DEFAULT_ROI_WIDTH_SCALE = 0.55
DEFAULT_ROI_LENGTH_SCALE = 0.30

DEFAULT_P_CUTOFF = 0.5
DEFAULT_FPS = 25.0

# 呼吸率搜索范围（次/分）→ 带通滤波与 FFT 主峰搜索共用
DEFAULT_BPM_MIN = 60
DEFAULT_BPM_MAX = 250


def bpm_to_hz(bpm: float) -> float:
    return bpm / 60.0


DEFAULT_F_MIN_HZ = bpm_to_hz(DEFAULT_BPM_MIN)   # 1.0 Hz
DEFAULT_F_MAX_HZ = bpm_to_hz(DEFAULT_BPM_MAX)   # ≈ 4.17 Hz
DEFAULT_FILTER_ORDER = 4

# 运动序列稳健化：抑制 phaseCorrelate 尖峰、短 gap 才插值
MOTION_MAD_CLIP_K = 5.0          # 超过 median + k*MAD 的 motion 截断
MAX_INTERP_GAP_SEC = 0.15        # 更长无效段不插值，避免伪造信号
MIN_VALID_FRAC_IN_WINDOW = 0.75  # 滑动 FFT 窗内至少 75% 帧 valid
HEAVE_MAX_SHIFT_PX = 3.0         # heave 单帧位移上限（对齐 patch 像素）
HEAVE_MIN_RESPONSE = 0.05        # phaseCorrelate 响应过低视为不可信
MOTION_MEDIAN_FILTER = 3         # 分析前中值滤波窗口（帧，奇数）

# 滑动 FFT：窗长需覆盖数个呼吸周期
DEFAULT_FFT_WINDOW_SEC = 6.0
DEFAULT_FFT_HOP_SEC = 0.5

# 谱图与叠加视频
PLOT_DIR = OUTPUT_DIR / "plots"
OVERLAY_DIR = OUTPUT_DIR / "overlay"

# 运动信号：对齐 ROI 固定尺寸；metric = mad | heave | combo
DEFAULT_MOTION_METRIC = "mad"
MOTION_PATCH_WIDTH = 64
MOTION_PATCH_HEIGHT = 32
