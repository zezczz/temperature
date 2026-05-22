# 尾部基准体温估计方案（tail_baseline）

在彩色 + 热成像 + DLC 流程里，**眼部**追踪往往不稳定（likelihood 低、映射坐标易飞出画面），**尾根**通常最稳。本方案把体温主估计改为「以尾部为准」，眼部只作辅助与置信度参考。

---

## 1. 每帧输出什么

| 列名 | 含义 |
| --- | --- |
| `tail_base_temperature` | 尾根处热图采样 → 标定后的温度 (°C) |
| `left_eye_temperature` / `right_eye_temperature` | 各眼温度（可能为 NaN） |
| **`body_temperature`** | **= `tail_base_temperature`**（主体温，不再用 max(眼,尾)） |
| **`eye_temperature_mean_raw`** | 逐帧：有效左/右眼温度的算术平均 |
| **`eye_temperature_mean`** | 对 raw 做 **1 秒**（默认）时间滚动平均，减轻眼部抖动 |
| **`temperature_confidence`** | 综合置信度，0~1，越大越可信 |
| `conf_tail` | 尾部 DLC likelihood（裁剪到 0~1） |
| `conf_eye` | 眼部追踪 × 覆盖率 × 与尾部一致性 |
| `conf_agreement` | 眼均温与尾温是否接近（见下） |

---

## 2. 置信度怎么算

记：

- \(T_{\text{tail}}\) = 尾温，\(L_{\text{tail}}\) = 尾部 likelihood  
- \(T_{\text{eye,raw}} = \text{mean}(\text{有效左眼}, \text{有效右眼})\)  
- \(T_{\text{eye}} = \text{rolling\_mean}(T_{\text{eye,raw}},\, \text{窗口}= \lfloor fps \times \text{eye\_smooth\_sec} \rfloor)\)，默认 60 帧 ≈ 1 s @ 60 fps  
- \(L_{\text{eye}} = \text{mean}(\text{有效眼的 likelihood})\)  
- 覆盖率 = 有效眼数 / 2  

**一致性**（眼温应接近尾温，差太多说明点飘了）：

\[
\text{conf\_agreement} = \exp\left(-\frac{|T_{\text{eye}} - T_{\text{tail}}|}{\sigma}\right)
\]

默认 \(\sigma = 2\,°\text{C}\)（`--agreement-sigma` 可调）。

**眼部综合：**

\[
\text{conf\_eye} = L_{\text{eye}} \times \text{覆盖率} \times \text{conf\_agreement}
\]

**最终置信度：**

\[
\text{confidence} = \text{clip}\bigl(L_{\text{tail}} \times (0.65 + 0.35 \times \text{conf\_eye})\bigr)
\]

若无有效眼，则 \(\text{confidence} = 0.65 \times L_{\text{tail}}\)（只靠尾部）。

尾部无效 → `body_temperature` 与 `confidence` 均为 NaN / 0。

---

## 3. 命令行用法

```powershell
conda activate DEEPLABCUT
cd E:\projects\for_study\temperature

# 从 intensity 生成温度（默认已是 tail_baseline）
python -m thermometry apply `
    --csv data\aligned\dlc\top2026-05-17-14-44-35_thermal_intensity.csv `
    --p-cutoff 0.25 `
    --scheme tail_baseline `
    --eye-smooth-sec 1.0 --fps 60

# 眼尾允许温差更松（3°C）
python -m thermometry apply `
    --csv data\aligned\dlc\top2026-05-17-14-44-35_thermal_intensity.csv `
    --scheme tail_baseline --agreement-sigma 3.0

# 仍用旧方案（多部位取 max）
python -m thermometry apply --csv ... --scheme legacy_max

# 画图：主体温 + 眼均值 + 置信度副轴
python -m thermometry plot `
    --csv data\aligned\dlc\top2026-05-17-14-44-35_temperature.csv `
    --fps 60 --smooth 30
```

输出图：`data/aligned/temperature/plots/<stamp>_tail_baseline.png`  
橙色散点 = 置信度低于 `--conf-threshold`（默认 0.3）的帧。

---

## 4. Python 里读取

```python
import pandas as pd

df = pd.read_csv("data/aligned/dlc/top2026-05-17-14-44-35_temperature.csv", index_col=0)

# 分析时建议过滤低置信度
good = df[df["temperature_confidence"] >= 0.4]
print(good["body_temperature"].describe())
print(good["eye_temperature_mean"].describe())
```

---

## 5. 和旧方案对比

| | `legacy_max` | `tail_baseline`（推荐） |
| --- | --- | --- |
| 主体温 | 各部位温度取 max | **固定为尾温** |
| 眼部 | 参与 max，易拉高/拉低 | 单独 `eye_temperature_mean`，不参与主体温 |
| 置信度 | 无 | `temperature_confidence` + 分项 |
| 适用 | 快速粗看 | 论文/分析、需筛低质量帧 |

---

## 6. 调参建议

| 参数 | 默认 | 何时改 |
| --- | --- | --- |
| `--p-cutoff` | 0.5（apply 里常设 0.25） | DLC 整体置信度偏低时降到 0.2~0.3 |
| `--agreement-sigma` | 2.0 °C | 眼温系统性高于尾温时可略加大到 3 |
| `--eye-smooth-sec` | 1.0 s | 眼均温滚动窗口；`0` = 不平滑 |
| `--fps` | 60 | 与热成像视频一致，用于秒→帧 |
| `TAIL_BASELINE_TAIL_WEIGHT` | 0.65 | 改 `thermometry/config.py` 提高尾部权重 |
| `TAIL_BASELINE_W_EYE` | 0.35 | 与上两者之和不必为 1；公式里 eye 项是加成 |

实现代码：`thermometry/tail_baseline.py`。
