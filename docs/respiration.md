# 呼吸频率测量 (`respiration/`)

> 与 `thermometry/` 并列的独立模块：用**彩色视频 + DLC 关键点**估计小鼠呼吸频率（次/分），**不依赖热成像**。

---

## 核心要点（请先读）

| 要点 | 说明 |
| --- | --- |
| **测什么** | 胸腔区域的**帧间运动**（起伏），不是亮度、不是热成像温度 |
| **为什么** | 俯视彩色画面里呼吸主要表现为毛发/轮廓/阴影的周期性位移；胸腔在热像上温度对比往往很弱 |
| **需要什么** | ① `data/color/top*.mp4` ② 同一段视频的 DLC 推理 csv（`data/dlc_results/top*DLC_*.csv`） |
| **怎么跑** | `python -m respiration list` 看本机有哪些段 → `python -m respiration run --stamp <时间戳> --plot` |
| **勿用占位符** | 文档示例里的路径必须换成你机器上的真实文件名；**不要**写 `top....mp4` / `top....csv` |

**推荐运动指标**：肉眼能看到胸腔起伏时，优先 `--motion-metric heave`；信号偏弱再试 `combo` 或 `mad`。

---

## 1. 在项目中的位置

```text
彩色视频 (data/color/top*.mp4)
        │
        │  DeepLabCut analyze → data/dlc_results/top*DLC_*.csv
        ▼
python -m respiration run --stamp <时间戳>
        │
        ▼
data/aligned/respiration/   ← 运动序列、呼吸率、图
```

与体温流程的关系：

- **体温**（`thermometry`）：DLC → 映射到 **thermal** → intensity → 温度  
- **呼吸**（`respiration`）：DLC → 留在 **彩色** 视频上取胸腔 ROI 运动 → FFT → 次/分  

两段流程共用同一套 DLC 结果，但呼吸模块**只读彩色 mp4**，不读热成像。

---

## 2. 环境

在已激活的 DLC 环境中（与项目 README 一致，例如 `conda activate DEEPLABCUT`）：

```powershell
pip install scipy
```

其余依赖与项目相同：`opencv-python`、`pandas`、`numpy`、`matplotlib`。

---

## 3. 快速开始（三步）

### 第 1 步：确认数据已就绪

需要同时存在（**同一时间戳**）：

| 类型 | 路径示例 |
| --- | --- |
| 彩色视频 | `data/color/top2026-05-17-14-44-35.mp4` |
| DLC 推理 csv | `data/dlc_results/top2026-05-17-14-44-35DLC_Resnet50_TemperatureTopViewMay20shuffle1_snapshot_best-200.csv` |

若还没有 csv，需先对彩色视频跑 DLC 推理（见 [deeplabcut.md](deeplabcut.md) 中的 `analyze`）。

### 第 2 步：列出本机可运行的段落

```powershell
cd E:\projects\for_study\temperature
python -m respiration list
```

输出示例：

```text
共 12 组（stamp / 视频 / DLC csv）:

  2026-05-17-14-44-35
    video: ...\data\color\top2026-05-17-14-44-35.mp4
    dlc:   top2026-05-17-14-44-35DLC_Resnet50_..._best-200.csv
```

记下你要分析的那一行的 **stamp**（即 `2026-05-17-14-44-35` 这一段，不含 `top` 前缀）。

### 第 3 步：一键分析并出图

```powershell
python -m respiration run `
    --stamp 2026-05-17-14-44-35 `
    --motion-metric heave `
    --plot
```

终端会打印全段呼吸率（次/分）。加 `--plot` 会保存分析图；加 `--video-overlay` 还会生成带 ROI 框的叠加视频。

### 第 4 步（可选）：只对已有结果做可视化

若已经跑过 `run` 但没出图，无需重新读视频做 FFT，可直接：

```powershell
# 静态三联图（运动曲线 + 瞬时呼吸率 + 频谱）
python -m respiration plot --stamp 2026-05-17-14-44-35

# 同时生成彩色叠加视频（绿框=胸腔 ROI，左上角=呼吸率）
python -m respiration plot --stamp 2026-05-17-14-44-35 --video-overlay
```

---

## 4. 可视化输出说明

| 产物 | 路径 | 内容 |
| --- | --- | --- |
| **分析图 PNG** | `data/aligned/respiration/plots/top<stamp>_respiration.png` | ① 原始/滤波运动 ② 瞬时与全段呼吸率 ③ 频谱 + 文字摘要 |
| **叠加视频 mp4** | `data/aligned/respiration/overlay/top<stamp>_respiration_overlay.mp4` | 彩色画面上画胸腔 ROI（绿框）、全段/瞬时次/分 |

叠加视频用途：检查 ROI 是否盖住胸腔、呼吸率文字是否与肉眼起伏一致。

---

## 5. 命令说明

所有命令格式：`python -m respiration <子命令> [参数]`。

### 5.1 `list` — 查看可配对列表

```powershell
python -m respiration list
```

当 `list` 为空时，检查：

- `data/color/` 下是否有 `top*.mp4`
- `data/dlc_results/` 下是否有对应的 `top*DLC_*.csv`

### 5.2 `run` — 推荐：提取运动 + 滤波 + FFT + 可选作图

```powershell
python -m respiration run --stamp <时间戳> [选项]
```

等价于先 `extract` 再 `analyze`，适合日常使用。

**常用选项：**

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--stamp` | — | 时间戳，如 `2026-05-17-14-44-35`（与 `list` 输出一致） |
| `--motion-metric` | `mad` | `heave` 更贴近俯视起伏；`combo` = mad + heave |
| `--plot` | 关 | 保存三联图 PNG |
| `--f-min` / `--f-max` | 1.5 / 5.0 Hz | 呼吸频段，约 90–300 次/分 |
| `--fft-window-sec` | 6.0 | 滑动 FFT 窗长（秒），越长越稳、时间分辨率越差 |

**不用 `--stamp` 时**，必须同时给出真实全路径（可复制 `list` 里的路径，或 Tab 补全）：

```powershell
python -m respiration run `
    --video data/color/top2026-05-17-14-44-35.mp4 `
    --dlc-csv data/dlc_results/top2026-05-17-14-44-35DLC_Resnet50_TemperatureTopViewMay20shuffle1_snapshot_best-200.csv `
    --motion-metric heave `
    --plot
```

### 5.3 `plot` — 仅可视化（不重新 extract）

```powershell
python -m respiration plot --stamp 2026-05-17-14-44-35
python -m respiration plot --stamp 2026-05-17-14-44-35 --video-overlay
python -m respiration plot --stamp 2026-05-17-14-44-35 --show   # 弹出 matplotlib 窗口
```

也可用 `--signal` 指向某个 `*_chest_motion.csv`。

### 5.4 `extract` — 只生成胸腔运动时间序列

适合先检查信号质量，再决定是否调参。

```powershell
python -m respiration extract --stamp 2026-05-17-14-44-35 --motion-metric heave
```

输出：`data/aligned/respiration/top2026-05-17-14-44-35_chest_motion.csv`

### 5.5 `analyze` — 对已生成的 csv 做滤波与 FFT

```powershell
python -m respiration analyze `
    --signal data/aligned/respiration/top2026-05-17-14-44-35_chest_motion.csv `
    --plot
```

可反复改 `--f-min`、`--f-max`、`--fft-window-sec` 重跑，无需重新读视频。

---

## 6. 输出文件

均在 `data/aligned/respiration/`：

| 文件 | 内容 |
| --- | --- |
| `top<stamp>_chest_motion.csv` | 每帧 `motion`（运动强度）、`valid`、ROI 几何列 |
| `top<stamp>_chest_motion.meta.txt` | 对应视频、fps、`motion_metric` 等元数据 |
| `top<stamp>_filtered.csv` | 带通滤波后的 `filtered` 列 |
| `top<stamp>_instant_rate.csv` | 滑动窗瞬时呼吸率：`time_s`、`breaths_per_min`、`freq_hz` |
| `top<stamp>_summary.txt` | 全段主峰频率、全局 bpm 等摘要 |
| `plots/top<stamp>_respiration.png` | 使用 `--plot` 时：原始运动 / 滤波 / 瞬时率 / 频谱 |

**如何读结果：**

- **全段呼吸率**：`run` 结束时终端打印的「全段呼吸率: xx 次/分」  
- **随时间变化**：打开 `*_instant_rate.csv` 或 PNG 中间子图  
- **信号是否正常**：看 PNG 第一行，`motion` 应有明显周期性起伏；若近乎平直，见下文「调参」

---

## 7. 工作原理（简版）

### 7.1 胸腔 ROI 怎么定

复用 DLC 三个点：`left_eye`、`right_eye`、`tail_base`（无需单独标胸腔点）。

1. 双眼中点 → 尾根 定义**身体轴**  
2. 胸腔中心默认在轴上 **38%** 处（`--chest-fraction`，0=头侧，1=尾侧）  
3. 裁切旋转矩形，缩放到 **64×32** 像素 patch，保证帧间可比  

### 7.2 运动信号（`--motion-metric`）

| 指标 | 含义 | 何时用 |
| --- | --- | --- |
| `mad` | 相邻帧 patch 的**平均绝对差** | 默认；对任何像素变化都敏感 |
| `heave` | 垂直于身体轴方向的**位移幅度**（phaseCorrelate） | **推荐**：俯视起伏明显时 |
| `combo` | `mad + heave` | 单指标偏弱时 |

### 7.3 呼吸率怎么算

```text
motion 序列
    → 填补无效帧 + 线性去趋势
    → 带通滤波 [f_min, f_max] Hz
    → 全段 FFT 主峰 → 全局 次/分
    → 滑动窗 FFT 主峰 → 瞬时 次/分 曲线
```

默认频段 **1.5–5 Hz**（约 **90–300 次/分**），可按动物状态调整。

---

## 8. 调参与排错

### 8.1 `FileNotFoundError: ... top....csv`

原因：命令里用了文档占位符或未写全的文件名。

处理：

1. 先 `python -m respiration list`  
2. 用输出里的 **stamp** 或完整路径，不要用 `....`

### 8.2 `list` 显示 0 组

- 缺 mp4：把彩色视频放进 `data/color/`，命名 `top<时间戳>.mp4`  
- 缺 csv：对同一段视频跑 `python dlc_workflow.py analyze`，结果应在 `data/dlc_results/`

### 8.3 运动信号很平、呼吸率明显不对

按顺序尝试：

1. `--motion-metric heave` 或 `combo`  
2. 略增大 ROI：`--roi-width-scale 0.65 --roi-length-scale 0.35`  
3. 微调胸腔位置：`--chest-fraction 0.35`（更靠头）或 `0.42`（更靠尾）  
4. 检查 DLC：`--p-cutoff 0.6`，或回看 DLC 叠加视频确认三点稳定  
5. 收窄或放宽频段：例如 `--f-min 2 --f-max 4`  

### 8.4 与体温模块对比

| | 呼吸 `respiration` | 体温 `thermometry` |
| --- | --- | --- |
| 输入视频 | 彩色 `top*.mp4` | 热成像 `temp*.mp4` |
| 物理量 | 像素运动 | 伪彩强度 → 温度 |
| DLC 用途 | 定位胸腔 ROI | 映射到 thermal 采样 |

---

## 9. 目录结构（代码）

```
respiration/
├── __main__.py          # python -m respiration
├── cli.py               # list / extract / analyze / run
├── config.py            # 默认路径与参数
├── pairing.py           # stamp ↔ 视频 + DLC csv
├── roi.py               # 胸腔几何与对齐 patch
├── extract.py           # 视频 → chest_motion.csv
├── analyze.py           # 滤波 + FFT
├── plot.py              # PNG
└── dlc_io.py            # 读 DLC csv
```

---

## 10. 局限说明

- 头尾整体平移、挣扎会污染 `mad`；`heave` 更专注起伏，但依赖 patch 内有足够纹理。  
- 瞬时率是**滑动窗 FFT 主峰**，不是逐次呼吸的峰值计数。  
- 未单独标定「胸腔」DLC 点，胸腔位置由眼、尾几何推算；若体态特殊，需调 `--chest-fraction`。  
- 旧版亮度信号 `*_chest_signal.csv`（列 `raw`）仍可 `analyze`，建议用新流程重新 `extract` 生成 `*_chest_motion.csv`。

---

## 11. 命令速查表

```powershell
python -m respiration list

python -m respiration run --stamp <时间戳> --motion-metric heave --plot --video-overlay

python -m respiration plot --stamp <时间戳> --video-overlay

python -m respiration extract --stamp <时间戳> --motion-metric heave
python -m respiration analyze --signal data/aligned/respiration/top<时间戳>_chest_motion.csv --plot
```

相关文档：[deeplabcut.md](deeplabcut.md)（DLC 推理）、[thermometry.md](thermometry.md)（体温）、[algorithm_overview.md](algorithm_overview.md)（总览）。
