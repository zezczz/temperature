# Thermometry — 从 DLC + 热成像反算小鼠体温

紧接 `alignment/` 模块之后使用。`alignment.dlc sample` 已经把 DLC 在彩色画面上的关键点反向映射到热成像，并采样得到每个 bodypart 位置的 **伪彩灰度强度** `intensity (0~255)`。这里要做的是：

1. 用一组「灰度 ↔ 真实温度」的 **anchor**，拟合一个 `intensity → 温度 (°C)` 的标定函数；
2. 把每一帧每个 bodypart 的 intensity 换算成温度；
3. 按一定策略（默认 `max`）把多个部位聚合成一个「小鼠体温」估计；
4. 可选地把温度数字叠加在 thermal 视频上做可视化。

> 之所以需要 anchor，是因为我们手头只有 8-bit 伪彩 mp4，不带绝对辐射温度信息；只有给它至少两个「已知温度对应的 intensity」点，才能反解出一条 `intensity → 温度` 曲线。

---

## 1. 目录结构

```
thermometry/
├── __init__.py
├── __main__.py                # python -m thermometry 入口，等价于 thermometry.cli
├── config.py                  # 路径、默认聚合方式、默认 likelihood cutoff
├── calibration.py             # LinearCalibration / PiecewiseCalibration + 拟合 + JSON I/O
├── compute.py                 # apply_to_dataframe / process_csv / batch / measure_bodypart
├── summary.py                 # 每段视频的体温统计
├── overlay.py                 # 在 thermal 视频上叠加 bodypart 圆点 + 温度数字
├── cli.py                     # argparse 子命令分发
├── calibration.json           # 当前生效的标定参数（fit 子命令会覆盖）
├── anchors.example.json       # anchor 文件模板
└── README.md                  # 速查（本文件是详细版）

data/aligned/
├── dlc/                       # alignment 与 thermometry 共用
│   ├── <stamp>_thermal_intensity.csv      ← 输入：alignment.dlc sample 产出
│   └── <stamp>_temperature.csv            ← 输出：apply / batch 产出
├── temperature/
│   └── temperature_summary.csv            ← summary 产出
└── temperature_overlay/
    └── <stamp>_temp_overlay.mp4            ← overlay 产出（可选）
```

---

## 2. 数据流

```
data/aligned/dlc/<stamp>_thermal_intensity.csv     ← alignment.dlc sample
        │
        │   python -m thermometry apply / batch
        ▼
data/aligned/dlc/<stamp>_temperature.csv           ← 每帧、每 bodypart 的 °C
        │                                            （含 body_temperature 聚合列）
        │
        ├── python -m thermometry summary
        │       └─> data/aligned/temperature/temperature_summary.csv
        │
        └── python -m thermometry overlay
                └─> data/aligned/temperature_overlay/<...>.mp4
```

---

## 3. 工作原理

### 3.1 输入端：intensity 是什么

`alignment.dlc sample` 在每一帧 thermal mp4 上做了这件事：

1. `cv2.cvtColor(BGR2GRAY)` 把伪彩转灰度（线性混合 `0.114·B + 0.587·G + 0.299·R`）
2. 在 DLC 关键点反映射回 thermal 之后的坐标 `(x, y)` 处，取 `(2r+1)×(2r+1)` 邻域均值

所以每行得到一个浮点 `intensity ∈ [0, 255]`。这是个 **相对热度** 指标，越亮通常越热，但 mp4 的伪彩映射不一定是严格线性的（IRON、Rainbow 等调色板就明显非线性）。

### 3.2 标定模型

我们提供两种把 `intensity → 温度` 拟合的方式（在 `thermometry/calibration.py`）：

| 模型 | 公式 | 需要的 anchor 数 | 适用场景 |
| --- | --- | --- | --- |
| `linear` | `T = a·I + b` | ≥ 2 | 伪彩接近灰度阶梯，且工作温度范围窄（如生理体温 35~40°C），首选 |
| `piecewise` | 分段线性插值 `np.interp` | ≥ 2，建议 4~6 | 伪彩明显非线性（IRON/Rainbow），或工作范围跨度大 |

拟合方式：

* `linear` 走 `np.polyfit(I, T, 1)`，最小二乘
* `piecewise` 直接把 anchor 当作插值节点，端点之外的输入会被截断（保守，避免外推爆炸）

两个模型都实现统一的 `apply(intensity)` 接口，下游 `compute.py` 不关心是哪一种。

### 3.3 聚合到「小鼠体温」

DLC 通常有多个 bodypart（眼角、尾根、鼻头等），每帧得到一个温度向量。`thermometry` 在 `compute.apply_to_dataframe` 里再做一次聚合，得到一列 `body_temperature`：

| `--aggregator` | 含义 | 何时用 |
| --- | --- | --- |
| `max` *(默认)* | 取该帧所有有效 bodypart 的最高温度 | 想用「最热点」近似核心温度（眼角、耳廓往往最接近） |
| `mean` | 平均 | 各部位都比较稳定，求一个综合指标 |
| `median` | 中位 | 对异常 bodypart（标点漂到背景）更鲁棒 |
| `quantile` | `--quantile` 分位数 | 想抑制极端值又比 median 更激进 |

聚合时受两个过滤影响：

* `--p-cutoff`（默认 0.5）：单帧 likelihood 低于阈值的 bodypart 视为 NaN
* intensity 本身为 NaN 的也直接跳过（一般是反映射到画面外）

一帧里所有 bodypart 都被过滤的话，`body_temperature` 也写 NaN。

### 3.4 标定文件 `calibration.json`

实际生效的标定参数存在 `thermometry/calibration.json` 里。两种形态：

```json
{
  "version": 1,
  "mode": "linear",
  "linear": {"a": 0.0179, "b": 35.74},
  "meta": {"anchors_file": "...", "n_anchors": 3, "fit_mode": "linear"}
}
```

```json
{
  "version": 1,
  "mode": "piecewise",
  "piecewise": {
    "intensities":  [60.0, 90.0, 120.0, 160.0],
    "temperatures": [34.0, 37.0,  38.5,  40.0]
  }
}
```

`meta` 字段只是给人看的，不参与计算。

---

## 4. 准备环境

依赖 `numpy / pandas / opencv-python`，项目里已有现成的 conda 环境：

```powershell
conda activate DEEPLABCUT
cd E:\projects\for_study\temperature
```

所有命令都以 **项目根目录** 为工作目录运行（模块以 `python -m thermometry ...` 形式调用，否则 import 会失败）。

---

## 5. 标定流程（首次使用 / 相机或视场被动过）

整个流程一句话：**用 `measure` 找 intensity，把 `(intensity, true_temperature)` 配对写进 `anchors.json`，再 `fit`**。

### 5.1 第一步：找 anchor 候选区间

挑几段你知道小鼠真实体温（用直肠温度计、红外测温枪、或 `data/color/温度记录.txt` 这种笔记里写好的数）的时间段。对每段：

* 时间区间换算成 **帧号区间**（按 `cv2.CAP_PROP_FPS` 或 mp4 帧数估算）
* 选一个 **稳定可见、信号干净** 的 bodypart（推荐 `tail_base`、`left_eye`、`right_eye`，鼻头容易受呼吸气流干扰）

用 `measure` 子命令把那段时间该 bodypart 的 intensity 统计出来：

```powershell
python -m thermometry measure `
    --csv data\aligned\dlc\top2026-05-17-14-44-35_thermal_intensity.csv `
    --bodypart tail_base `
    --start 0 --end 199 `
    --p-cutoff 0.25
```

输出示例：

```
file     : data\aligned\dlc\top2026-05-17-14-44-35_thermal_intensity.csv
bodypart : tail_base
frames   : 123
mean     : 97.449
median   : 95.184
std      : 10.974
min..max : 81.245 .. 120.020
p10..p90 : 85.008 .. 113.763
```

通常用 **median** 作为这段时间该 bodypart 的代表 intensity，比 mean 更稳。

### 5.2 第二步：写 `anchors.json`

把 `thermometry/anchors.example.json` 复制为 `thermometry/anchors.json`，按格式填入：

```json
{
  "mode": "linear",
  "anchors": [
    {"intensity":  85.0, "temperature": 37.2, "source": "tail_base 14:56:01, 温度记录"},
    {"intensity":  95.0, "temperature": 37.6, "source": "tail_base 14:44:30, 温度记录"},
    {"intensity": 130.0, "temperature": 38.5, "source": "left_eye 15:08:20, 红外枪"}
  ]
}
```

经验建议：

* **至少 2 条**且 intensity 不同（线性拟合最低要求）
* **3~5 条** 通常已经够稳，最好覆盖你关心温度范围的两端
* 跨多个 bodypart / 多个视频段是允许的——前提是同一个相机参数下采到的
* 模糊温度（如「37.3~5」）取中值

### 5.3 第三步：拟合

```powershell
python -m thermometry fit
```

输出示例：

```
linear: T = 0.017857 * I + 35.735714
  I=  82.000  T_true=37.200  T_pred=37.200  Δ=+0.000
  I= 110.000  T_true=37.700  T_pred=37.700  Δ=+0.000
已写入: E:\projects\for_study\temperature\thermometry\calibration.json
```

如果想强制走分段插值：

```powershell
python -m thermometry fit --mode piecewise
```

每次 `fit` 都会覆盖 `calibration.json`，老参数会丢；想留底用 git 或 `--out` 写到别处。

### 5.4 检验

`fit` 输出里的 `Δ` 是每个 anchor 的残差，`linear` 模式下应该在 ±0.2 °C 之内才算合理；偏差大说明：

* 不同 anchor 来自不同视场 / 相机距离 / 发射率，不能放一个模型里
* 伪彩调色板高度非线性，改用 `piecewise`
* anchor 本身记错了（温度记录笔记和帧号对不上是最常见的）

---

## 6. 使用

### 6.1 占位标定（联调用）

```powershell
python -m thermometry init                 # 已存在则跳过
python -m thermometry init --force         # 覆盖
```

写入 `T = 0.05·I + 32.0` 的占位线性标定，只用来跑通整个流程，**绝对不要用这个标定下结论**。

### 6.2 单文件换算

```powershell
python -m thermometry apply `
    --csv data\aligned\dlc\top2026-05-17-14-44-35_thermal_intensity.csv
```

输出 → 同目录的 `top2026-05-17-14-44-35_temperature.csv`。

如果只想用部分 bodypart：

```powershell
python -m thermometry apply `
    --csv ... `
    --bodyparts left_eye right_eye `
    --aggregator mean
```

DLC 训练得不太好、likelihood 普遍偏低时，可以把 cutoff 放宽：

```powershell
python -m thermometry apply --csv ... --p-cutoff 0.25
```

### 6.3 批量换算

```powershell
python -m thermometry batch                # 扫描 INTENSITY_DIR 下所有 *_thermal_intensity.csv
python -m thermometry batch --p-cutoff 0.25
```

`batch` 接受所有 `apply` 的参数。

### 6.4 汇总每段视频

```powershell
python -m thermometry summary
```

输出 → `data/aligned/temperature/temperature_summary.csv`，控制台会打印每段视频的体温统计（mean / median / max / p90）。

### 6.5 叠加视频（可选）

```powershell
python -m thermometry overlay `
    --intensity   data\aligned\dlc\top2026-05-17-14-44-35_thermal_intensity.csv `
    --temperature data\aligned\dlc\top2026-05-17-14-44-35_temperature.csv `
    --thermal     "data\thermal\temp2026-05-17 14-44-35.mp4"
```

输出 → `data/aligned/temperature_overlay/temp2026-05-17 14-44-35_temp_overlay.mp4`。

画面上每个 bodypart 一个彩色圆点 + `bp: 37.65C` 标签，左上角再有一行 `Tbody = 37.43 C` 总体温。常用参数：`--p-cutoff`、`--dot-radius`、`--max-frames`（调试用，先跑前几百帧）。

---

## 7. 输入 / 输出 CSV 字段

### 7.1 输入（来自 `alignment.dlc sample`）

`<stamp>_thermal_intensity.csv`，宽表，索引列 `frame`：

| 列 | 含义 |
| --- | --- |
| `<bp>_x`, `<bp>_y` | 该 bodypart 在 **thermal 坐标系** 下的像素坐标（已经反映射） |
| `<bp>_intensity` | 灰度强度，0~255 浮点 |
| `<bp>_likelihood` | DLC 输出的置信度，0~1 |

### 7.2 输出（`apply` / `batch`）

`<stamp>_temperature.csv`，索引列 `frame`：

| 列 | 含义 |
| --- | --- |
| `<bp>_temperature` | 用当前 `calibration.json` 算出来的温度，°C；NaN 表示无效 |
| `<bp>_x`, `<bp>_y` | 透传自输入，方便 `overlay` 复用 |
| `<bp>_likelihood` | 透传自输入 |
| `body_temperature` | 所有有效 bodypart 按 `--aggregator` 聚合后的总体温 |

### 7.3 汇总（`summary`）

`temperature_summary.csv`，每行一段视频：

```
file, n_frames, <col>_mean, <col>_median, <col>_max, <col>_p90, <col>_n, ...
```

其中 `<col>` 是 `<bp>_temperature` 或 `body_temperature`。

---

## 8. Python API

最常见的几种用法：

```python
from pathlib import Path

from thermometry import calibration as cal
from thermometry.compute import (
    apply_to_dataframe, process_csv, batch, measure_bodypart,
)
from thermometry.summary import summarize_dir
from thermometry.overlay import render

# 1) 加载现有标定
model = cal.load(Path("thermometry/calibration.json"))
print(model.describe())          # 'linear: T = 0.0179 * I + 35.7357'
print(model.apply(95.0))         # ≈ 37.43

# 2) 直接拟合（不写 anchors.json）
anchors = [(85.0, 37.2), (95.0, 37.6), (130.0, 38.5)]
model = cal.fit(anchors, mode="linear")
cal.save(model, Path("thermometry/calibration.json"),
         extra={"note": "ad-hoc"})

# 3) 单文件换算
process_csv(
    Path("data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv"),
    aggregator="max",
    p_cutoff=0.25,
)

# 4) 拿到某段 bodypart 的 intensity 描述统计
stats = measure_bodypart(
    Path("data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv"),
    bodypart="tail_base",
    start=0, end=199,
    p_cutoff=0.25,
)
# {'bodypart': 'tail_base', 'n_frames': 123, 'median': 95.18, ...}

# 5) 直接在 DataFrame 上做换算（不走 CSV）
import pandas as pd
df = pd.read_csv("...", index_col=0)
result = apply_to_dataframe(df, model, aggregator="median", p_cutoff=0.5)
print(result["body_temperature"].describe())
```

---

## 9. 常见问题

**Q: 一次 `apply` 之后 `body_temperature` 全是 NaN？**
A: 大概率是 DLC likelihood 普遍低于 `--p-cutoff`（默认 0.5）。先用 `measure` 看一下 bodypart 的 likelihood 分布，或者直接 `--p-cutoff 0.25` 甚至 `0` 试一下。

**Q: anchor 残差 Δ 很大（> 0.5 °C）？**
A: 三种原因：(1) 多个 anchor 跨了不同视场 / 距离 / 发射率，环境变量没控制好；(2) 伪彩调色板非线性，改 `--mode piecewise`；(3) anchor 本身记错——`temperature.txt` 里某段写的是「实际接近 37.7」，那 37.7 才是 anchor，不是 mp4 里看到的 37.2。

**Q: 同一只小鼠不同视频段，能复用同一份 `calibration.json` 吗？**
A: 相机参数、距离、伪彩 LUT 都不变的话可以。但跨日 / 跨实验最好重新 `measure` + `fit` 一次——尤其是热相机重启后，自动量程可能变了。

**Q: 我只有一个 anchor 怎么办？**
A: 拟合不出来。最低成本的补救：另外找一段画面里 **稳定可见的低温参考物**（例如室温的金属台、不发热的笼壁），假设它就是室温（如 25 °C），再 `measure` 取那个区域附近某个 bodypart 的 intensity 作为低温 anchor。或者把另一段视频里温度计读数也补上。

**Q: `body_temperature` 取 `max` 经常比真实体温偏高？**
A: 是预期的——`max` 取的是若干 bodypart 中「看上去最热」的那个，往往是眼角/耳廓，会比直肠温度高 0.3~0.5 °C。如果你想直接对标直肠温度，把 anchor 直接选成 **要复刻的那个 bodypart** 的 intensity（例如固定用 tail_base），并把 `--aggregator` 设成 `mean` 或只用单一 bodypart。

**Q: 想把所有视频段统一汇总成一张「体温曲线」？**
A: `summary` 输出是每段一行；要画时间序列就用 pandas 读 `<stamp>_temperature.csv`，按文件名里的时间戳拼起来。每段视频的 frame=0 对应的真实时间在文件名里（如 `14-44-35`），按帧率换算就能拼到统一时间轴。

**Q: 想把不同 bodypart 各画一条曲线对比？**
A: 直接读 `<stamp>_temperature.csv`，每个 `<bp>_temperature` 就是一条曲线。这种用法 `overlay` 视频帮不上忙，得自己写画图脚本。

**Q: overlay 视频里温度数字漂得很厉害？**
A: 单帧 intensity 噪声很大（一两个像素的差就能让温度变 0.2°C）。建议两条路：(1) `alignment.dlc sample --radius 3` 或更大，先在采样阶段做空间平均；(2) `apply` 之后对 csv 做时间平滑（pandas `.rolling(window=5).mean()`），然后用平滑后的列重画 overlay。

**Q: 我有热相机的原始 raw 数据，不想用伪彩 mp4 这套了。**
A: 那直接跳过这个模块——thermometry 的存在前提就是「只剩 mp4」。原始 raw 数据通常带有逐像素辐射温度，按厂家文档把 raw → °C 算出来即可，DLC 关键点照样反映射上去取值。

---

## 10. 历史与设计取舍

* 早期想法是从伪彩 BGR 三通道反推 colormap，再用 colormap 的逆映射得 normalized 强度，最后用「最高温 / 最低温」标定两端。问题是 (1) mp4 H.264 量化让 BGR 不再严格落在 colormap 曲线上；(2) 标定时根本不知道当时画面的最高 / 最低温——红外相机自动量程一直在变。
* 改成现在这套：直接拿 `cvtColor(BGR→GRAY)` 后的灰度，把所有标定问题归结到一条 `intensity → 温度` 曲线上。代价是必须提供至少 2 个有真实温度的 anchor，好处是简单、可解释、几行代码就能 reproduce。
* 默认 `--aggregator max` 是经验选择：DLC 标的几个部位里，眼角附近最接近核心温度，max 通常正好选到它；后续如果只用 tail_base，单独传 `--bodyparts tail_base` 比改 aggregator 更稳。
