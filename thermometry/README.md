# thermometry — 用 DLC + 热成像反算小鼠体温

本模块紧接 `alignment/` 之后使用。`alignment.dlc sample` 已经把 DLC 在彩色画面上的关键点映射到热成像，并采样得到每个 bodypart 的灰度强度 `intensity (0~255)`。这里要做的是：

1. 用一组「灰度 ↔ 真实温度」的 **anchor** 拟合一个标定函数；
2. 把每一帧每个 bodypart 的 intensity 换算成温度 (°C)；
3. 按一定策略（默认 `max`）把若干部位聚合成一个「小鼠体温」估计；
4. 可选地把温度数字叠加在 thermal 视频上做可视化。

> 之所以需要 anchor，是因为视频是 8-bit 伪彩 mp4，并不带绝对温度信息；只有给它至少两个「已知温度对应的 intensity」点，才能反解。

---

## 数据流

```
data/aligned/dlc/<stamp>_thermal_intensity.csv     ← alignment.dlc sample 的产物
            │
            │  python -m thermometry apply / batch
            ▼
data/aligned/dlc/<stamp>_temperature.csv           ← 每帧、每 bodypart 的温度
            │
            ├── python -m thermometry summary  → data/aligned/temperature/temperature_summary.csv
            └── python -m thermometry overlay  → data/aligned/temperature_overlay/<...>.mp4
```

---

## 快速上手

### 1. 找几个 anchor 点

如果你已经知道某段时间小鼠体温是 37.7°C（用温度计或红外测温枪测），可以用 `measure` 子命令看一下那段时间对应 bodypart 的 intensity 中位数：

```powershell
python -m thermometry measure `
    --csv data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv `
    --bodypart tail_base `
    --start 0 --end 199
```

输出示例：

```
bodypart : tail_base
frames   : 195
mean     : 95.42
median   : 92.18
...
```

把 `(intensity_median, true_temperature)` 这样的二元组写到 `thermometry/anchors.json`（可仿照 `anchors.example.json`），凑齐至少 2 条 intensity 不同的 anchor 即可。

### 2. 拟合标定

```powershell
python -m thermometry fit
```

会在 `thermometry/calibration.json` 写入拟合好的 `T = a * I + b`（或分段插值）。

### 3. 算温度

单文件：

```powershell
python -m thermometry apply --csv data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv
```

批量：

```powershell
python -m thermometry batch
```

### 4. 汇总 & 叠加视频（可选）

```powershell
python -m thermometry summary

python -m thermometry overlay `
    --intensity   data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv `
    --temperature data/aligned/dlc/top2026-05-17-14-44-35_temperature.csv `
    --thermal     "data/thermal/temp2026-05-17 14-44-35.mp4"
```

---

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `config.py` | 路径、默认聚合方式等常量 |
| `calibration.py` | `LinearCalibration` / `PiecewiseCalibration` 模型 + 拟合 + JSON I/O |
| `compute.py` | `apply_to_dataframe` / `process_csv` / `batch` / `measure_bodypart` |
| `summary.py` | 每段视频的体温统计汇总 |
| `overlay.py` | 把 bodypart 圆点和温度数字画到 thermal 视频上 |
| `cli.py` / `__main__.py` | `python -m thermometry ...` 命令行入口 |
| `calibration.json` | 当前生效的标定参数（`fit` 子命令会覆盖） |
| `anchors.example.json` | anchor 文件模板，复制为 `anchors.json` 后填实际数据 |

---

## 输出 CSV 字段约定

`*_temperature.csv` 列（每个 bodypart 都有，加一个总体温列）：

```
frame, <bp>_temperature, <bp>_x, <bp>_y, <bp>_likelihood, ..., body_temperature
```

- `body_temperature` = 按 `--aggregator` 聚合所有有效 bodypart 的温度，默认取 `max`。
- 任何 likelihood < `--p-cutoff`（默认 0.5）或 intensity 缺失的 bodypart 在该帧记 `NaN`。

---

## 关于"准确度"的注意事项

1. 伪彩 mp4 经过 `cvtColor(BGR → GRAY)` 后再做线性拟合，本质上是 **近似**；如果调色板是高度非线性的（IRON、Rainbow），建议改用 `mode: piecewise` 的多个 anchor，效果会明显更好。
2. 想做更严谨的标定，最好在采集时同时记录原始辐射温度文件（如设备原始 raw 数据），而不是只留伪彩 mp4。这里的方法是「在没有 raw 数据时」的折中方案。
3. 室温、相机距离、表面发射率会影响 intensity，跨实验/跨日的视频建议各自做一次 `fit`。
