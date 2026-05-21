# DeepLabCut 姿态估计与标注

在 **彩色俯视视频**（`data/` 下 `top*.mp4`，1920×1080）上，用 [DeepLabCut 3.0](https://deeplabcut.github.io/DeepLabCut/)（PyTorch 引擎）追踪 **单只动物** 的 3 个关键点，输出逐帧坐标 CSV/H5，供后续 `alignment.dlc` 映射到热成像、做温度分析。

本仓库在官方 DLC 流程外封装了 **一键工作流** 与 **OpenCV 简易标注**（绕过 Windows 上常见的 napari 卡死问题）。

---

## 1. 目录结构

```
temperature/
├── data/
│   ├── top2026-05-17-14-44-35.mp4          # 原始彩色视频（输入）
│   └── top2026-05-17-14-44-35DLC_*.csv     # analyze 输出的逐帧预测（推理结果）
│
├── dlc_project/
│   └── TemperatureTopView-lab-2026-05-18/  # DLC 工程根目录
│       ├── config.yaml                     # 工程配置（bodyparts、抽帧数、pcutoff 等）
│       ├── videos/                         # 复制进来的训练用视频
│       ├── labeled-data/
│       │   ├── top2026-05-17-14-44-35/     # ★ 待标注抽帧 + CollectedData（在这里标）
│       │   └── top2026-05-17-14-44-35_labeled/  # check 生成的预览图（勿在此标注）
│       ├── training-datasets/              # create_training_dataset 产物
│       ├── dlc-models-pytorch/             # 训练权重与日志
│       └── evaluation-results-pytorch/     # evaluate 误差图与 CSV
│
├── dlc_config.py           # 项目级配置（关键点、抽帧数、训练轮数等）
├── dlc_workflow.py         # 主工作流 CLI
├── dlc_label_simple.py     # OpenCV 简易标注（推荐）
├── dlc_repair_labels.py    # 修复 CollectedData 格式
├── dlc_gui_bootstrap.py    # napari 启动前环境变量
├── trace_napari_import.py  # napari 导入分段诊断
├── fix_napari_gui.ps1      # Windows napari 修复脚本
└── .dlc_config_path        # 缓存 config.yaml 绝对路径
```

---

## 2. 环境与版本

| 项 | 说明 |
|----|------|
| Conda 环境 | `DEEPLABCUT`（示例路径 `D:\miniconda3\envs\DEEPLABCUT`） |
| DeepLabCut | **3.0.0rc13**（PyTorch 引擎，`config.yaml` 中 `engine: pytorch`） |
| 网络 | 默认 `resnet_50` |
| 标注 GUI（官方） | napari + napari-deeplabcut（本机 Windows 上常无法导入，见 §8） |
| 标注 GUI（本仓库） | OpenCV（`dlc_label_simple.py`） |

```powershell
conda activate DEEPLABCUT
cd E:\projects\for_study\temperature
python -c "import deeplabcut; print(deeplabcut.__version__)"
```

---

## 3. 关键点定义

**单只动物**（`multianimalproject: false`），俯视视角，以 **动物自身左右** 定义 left/right（与画面左右可能相反）。

| 名称 | 含义 |
|------|------|
| `left_eye` | 左眼 |
| `right_eye` | 右眼 |
| `tail_base` | 尾巴根部 |

可选第 4 点 `tail_middle`：在 `dlc_config.py` 设 `INCLUDE_TAIL_MIDDLE = True`，再执行 `sync-bodyparts`。

配置入口：`dlc_config.py` → 同步到 `dlc_project/.../config.yaml`（`sync-bodyparts` / `extract` 时会自动写入）。

---

## 4. 端到端流程

```
原始视频 (data/*.mp4)
    │
    ├─ create ──────────────► 创建 DLC 工程 + config.yaml
    │
    ├─ extract ─────────────► 从视频抽取代表性 PNG 帧 → labeled-data/<视频名>/
    │
    ├─ label-simple ────────► 人工标点 → CollectedData_lab.csv / .h5
    │
    ├─ repair-labels ───────► （可选）修正 CSV 格式
    │
    ├─ check ───────────────► 可视化检查标注 → *_labeled/ 预览目录
    │
    ├─ train-dataset ───────► 生成 training-datasets/、.mat 等
    │
    ├─ train ───────────────► 训练 ResNet50 → dlc-models-pytorch/
    │
    ├─ evaluate ────────────► 测试集误差
    │
    ├─ analyze ─────────────► 全视频逐帧预测 → data/*DLC*.csv
    │
    └─ plot ────────────────► 带关键点叠加的视频（通常仍在 data/ 同目录）
```

### 4.1 首次使用（从零开始）

```powershell
conda activate DEEPLABCUT
cd E:\projects\for_study\temperature

python dlc_workflow.py create          # 仅首次
python dlc_workflow.py prepare-label   # sync 配置 + 抽帧（等价 sync-bodyparts + extract）
python dlc_workflow.py label-simple    # 人工标注
python dlc_workflow.py check           # 检查
python dlc_workflow.py repair-labels   # 建议跑一次，确保 CSV 合法
python dlc_workflow.py train-dataset
python dlc_workflow.py train
python dlc_workflow.py evaluate
python dlc_workflow.py analyze
python dlc_workflow.py plot
```

### 4.2 在已有标注上继续加帧

1. 在 `dlc_config.py` 增大 `NUM_FRAMES_TO_EXTRACT`（如 100 → 150）。
2. `python dlc_workflow.py extract`（**勿删**已有 PNG 与 `CollectedData_lab.*`）。
3. `python dlc_workflow.py label-simple`（已标帧会保留坐标，按 `n` 跳过或修改）。
4. `repair-labels` → `train-dataset` → `train` → `evaluate` → `analyze` → `plot`。

### 4.3 标注数据量建议

| 目标 | 约需标注帧数（每帧 3 点标全） |
|------|------------------------------|
| 跑通流程 | 50～80 |
| 大部分场景可用 | 100～150 |
| 姿态/光照变化大 | 200+ |

未标全的帧（存在 NaN）在 `train-dataset` 时通常会被跳过；补标难例比重复标相似帧更有效。

---

## 5. 命令参考（`dlc_workflow.py`）

| 命令 | 作用 |
|------|------|
| `create` | 创建工程，将 `data/` 下视频加入项目 |
| `sync-bodyparts` | 把 `dlc_config.py` 中的 bodyparts / skeleton / numframes2pick 写入 `config.yaml` |
| `extract` | 按 kmeans 自动抽帧到 `labeled-data/` |
| `prepare-label` | `sync-bodyparts` + `extract` |
| `diagnose` | 检测 napari / Qt 依赖（子进程 + 超时，避免卡死） |
| `label` | 官方 napari 标注（Windows 上常不可用） |
| `label-simple` | **OpenCV 简易标注（推荐）** |
| `check` | 检查标注，生成 `*_labeled/` 预览 |
| `repair-labels` | 修复 `CollectedData_lab.csv` 并重建 `.h5` |
| `train-dataset` | 生成训练集 |
| `train` | 训练网络（轮数见 `TRAIN_MAXITERS`，默认 250） |
| `evaluate` | 评估并出图 |
| `analyze` | 对 `data/` 下视频推理，输出 CSV |
| `plot` | 生成带关键点轨迹的视频 |

---

## 6. 简易标注（`dlc_label_simple.py`）

不依赖 napari，适合本项目的默认标注方式。

```powershell
python dlc_label_simple.py
# 或指定文件夹（不要选 *_labeled）
python dlc_label_simple.py --folder "E:\projects\for_study\temperature\dlc_project\TemperatureTopView-lab-2026-05-18\labeled-data\top2026-05-17-14-44-35"
```

| 按键 | 功能 |
|------|------|
| `1` `2` `3` | 切换当前关键点（left_eye / right_eye / tail_base） |
| 鼠标左键 | 在当前点位置落点 |
| `n` / `p` | 下一帧 / 上一帧 |
| `s` | 保存到 CSV |
| `q` | 保存并退出，自动 `convertcsv2h5` |

输出文件（在 `labeled-data/<视频名>/` 下）：

- `CollectedData_lab.csv` — 标注源文件
- `CollectedData_lab.h5` — DLC 读取用

---

## 7. 标注文件格式与 `repair-labels`

DLC 要求 CSV 为 **三级表头**（scorer / bodyparts / coords），**行索引为图片相对路径**（如 `labeled-data/top.../img0018.png`）。

简易标注早期版本可能产生：

- 多余行：`image`、`labeled-data`（无 `.png`）→ 导致 `train-dataset` 报错 `join() ... not 'float'`
- 表头缺少 `left_eye`，与 `config.yaml` 不一致

**修复：**

```powershell
python dlc_workflow.py repair-labels
# 或
python dlc_repair_labels.py
```

脚本会：

1. 备份为 `CollectedData_lab.csv.bak`
2. 删除无效行，按 6 列数值还原 3 个关键点
3. 写回标准 CSV 并 `convertcsv2h5(..., userfeedback=False)`

---

## 8. napari 标注与 Windows 故障

DLC 3.0 官方标注走 **napari-deeplabcut**。在 Windows + DLC 3.0.0rc13 上常见现象：

- `label` / `diagnose` 在 `import napari_deeplabcut` 处 **长时间无响应**
- 原因多为 **PySide6 / napari 版本冲突** 或 **conda/pip 混装 Qt DLL**

诊断：

```powershell
python dlc_workflow.py diagnose
python trace_napari_import.py
```

尝试修复（不保证成功）：

```powershell
.\fix_napari_gui.ps1
```

**实践建议：** 标注统一用 `label-simple`；napari 修不好也不影响训练与推理。

---

## 9. 训练与推理产物

### 9.1 训练

- 配置：`dlc_config.py` → `TRAIN_MAXITERS`（默认 250）
- 权重：`dlc_project/.../dlc-models-pytorch/iteration-0/<Task>-trainset95shuffle1/train/`
- 学习曲线：`.../learning_stats.csv`

### 9.2 评估

- 目录：`evaluation-results-pytorch/iteration-0/`
- 含散点图、`*-results.csv` 等

### 9.3 推理（analyze）

- 默认分析 `data/` 下全部 `.mp4`
- 输出示例：`data/top2026-05-17-14-44-35DLC_Resnet50_TemperatureTopViewMay18shuffle1_snapshot_best-30.csv`
- 表头：`scorer` / `bodyparts` / `coords`（含 `x`, `y`, `likelihood`）

### 9.4 可视化视频（plot）

- 与 **输入视频同目录**（默认 `data/`）
- 文件名含 `labeled` / `DLC` / 网络名 / shuffle 等
- 在资源管理器中按 **修改时间** 排序查找最新 mp4

### 9.5 置信度阈值 `pcutoff`

`config.yaml` 中 `pcutoff`（默认 0.6）控制 plot 时是否绘制低置信度点。调低（如 0.1）会 **显示更多点**，但不提高模型精度；离谱预测需 **补标 + 重训**。

---

## 10. 与热成像流水线的衔接

DLC 在 **彩色坐标系** 输出关键点。映射到热成像见 [`docs/alignment.md`](alignment.md) 与 `alignment/dlc.py`：

```powershell
# 彩色 DLC 坐标 → 热图坐标
python -m alignment.dlc convert ^
    --csv data/top2026-05-17-14-44-35DLC_Resnet50_..._best-30.csv ^
    --out data/aligned/dlc/top2026-05-17-14-44-35_thermal_coords.csv

# 在热图视频上采样像素强度（相对热度）
python -m alignment.dlc sample ^
    --csv data/top2026-05-17-14-44-35DLC_Resnet50_..._best-30.csv ^
    --thermal "data/thermal/temp2026-05-17 14-44-35.mp4" ^
    --out data/aligned/dlc/top2026-05-17-14-44-35_thermal_intensity.csv

# 在热图视频上绘制映射后的关键点
python -m alignment.dlc draw ^
    --csv data/top2026-05-17-14-44-35DLC_Resnet50_..._best-30.csv ^
    --thermal "data/thermal/temp2026-05-17 14-44-35.mp4" ^
    --out data/aligned/dlc/top2026-05-17-14-44-35_thermal_labeled.mp4 ^
    --p-cutoff 0.25
```

温度定量与汇总见 [`docs/thermometry.md`](thermometry.md)。

---

## 11. 配置项说明（`dlc_config.py`）

| 变量 | 含义 | 默认 |
|------|------|------|
| `PROJECT_NAME` / `EXPERIMENTER` | 工程命名 | `TemperatureTopView` / `lab` |
| `MULTIANIMAL` | 是否多动物 | `False` |
| `INCLUDE_TAIL_MIDDLE` | 是否启用第 4 关键点 | `False` |
| `NUM_FRAMES_TO_EXTRACT` | 每视频抽帧数 | `100` |
| `EXTRACT_MODE` / `EXTRACT_ALGO` | 抽帧方式 | `automatic` / `kmeans` |
| `TRAIN_MAXITERS` | 训练迭代次数 | `250` |
| `ANALYZE_VIDEOS_FROM_DATA` | analyze/plot 是否用 `data/` 下视频 | `True` |
| `LABEL_FRAMES_DIR` | 简易标注默认帧目录 | 见文件内路径 |
| `CONFIG_PATH` | 手动指定 config.yaml | `None`（自动查找） |

`config.yaml` 路径缓存：`.dlc_config_path`（DLC 3.x 工程目录形如 `TemperatureTopView-lab-2026-05-18`，与 2.x 嵌套结构不同）。

---

## 12. 常见问题

**Q: `train-dataset` 报错 `join() argument must be str ... not 'float'`？**  
A: `CollectedData_lab.csv` 含无效行或表头错误。运行 `python dlc_workflow.py repair-labels` 后重试。

**Q: `label-simple` 提示「存在多个文件夹」？**  
A: 选 **不带** `_labeled` 的目录，即 `top2026-05-17-14-44-35`；`*_labeled` 是 `check` 的预览。

**Q: `extract` 报 `userquality` 未知参数？**  
A: DLC 3.x 已移除该参数；抽帧数由 `config.yaml` 的 `numframes2pick` 控制，本仓库 `extract` 会从 `dlc_config.py` 同步。

**Q: plot 出来的视频几乎没有点？**  
A: 确认打开的是 `*labeled*.mp4`；确认已 `analyze`；尝试临时降低 `pcutoff`；根本办法是补标难帧并重训。

**Q: 能否在旧标注上继续标？**  
A: 可以。不要删除 `CollectedData_lab.*`，`extract` 加新 PNG 后用 `label-simple` 续标。

**Q: 训练要多久？**  
A: 取决于 CPU/GPU 与帧数。无 GPU 时 200～250 轮可能需数十分钟到数小时。

**Q: shuffle1 / shuffle2 / shuffle3 是什么？**  
A: DLC 训练集随机划分与增强的多次重复；默认使用 `shuffle1` 的 `snapshot_best` 权重做 analyze。

---

## 13. 设计说明

| 决策 | 原因 |
|------|------|
| 封装 `dlc_workflow.py` | DLC 3.x 路径/API 与 2.x 不同，统一入口降低踩坑 |
| OpenCV 简易标注 | Windows 上 napari-deeplabcut 导入频繁卡死 |
| `repair-labels` | 简易标注 CSV 与 DLC 严格格式存在差异，需一键修复 |
| `dlc_gui_bootstrap.py` | 设置 `QT_API`、`NAPARI_DISABLE_PLUGIN_DISCOVERY` 减轻 napari 启动负担 |
| 3 点（眼 + 尾根） | 俯视单鼠任务足够；尾中点可选，不强制 |
| PyTorch + ResNet50 | 与 `config.yaml` 中 `engine: pytorch` 一致 |

---

## 14. 相关文档

- [Color / Thermal 双相机对齐](alignment.md)
- [温度计算与汇总](thermometry.md)
- [DeepLabCut 官方文档](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html)
