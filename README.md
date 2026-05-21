# temperature — 用 DLC + 热成像测小鼠体温

把彩色相机上跑出的 DeepLabCut 关键点（眼睛/尾巴等）映射到热成像视频，从伪彩亮度反推出实际温度 (°C)。

## 模块

| 目录 | 作用 |
| --- | --- |
| `alignment/` | 双相机标定 + 彩色 → 热成像几何映射，DLC 点反映射到 thermal |
| `thermometry/` | intensity → 温度 (°C) 标定、逐帧体温计算、汇总与可视化 |
| `docs/` | `alignment.md` / `thermometry.md` / `performance.md` 详细文档 |
| `dlc_*.py` | DLC 项目辅助脚本（标点、修复、工作流） |

## 主要工作流

```text
彩色视频 + DLC 模型
        │
        │  python -m alignment.dlc sample
        ▼
data/aligned/dlc/<stamp>_thermal_intensity.csv
        │
        │  python -m thermometry fit + apply
        ▼
data/aligned/dlc/<stamp>_temperature.csv
        │
        ├── python -m thermometry summary  → 汇总 csv
        ├── python -m thermometry plot     → 折线图 PNG
        └── python -m thermometry overlay  → 叠加视频 mp4
```

详细命令与参数见 `docs/` 下的对应文档。

## 仓库注意

- 数据视频 `data/` 与 DLC 训练项目 `dlc_project/` 都已通过 `.gitignore` 排除，**仓库只跟踪代码**。
- 你需要自行准备：
  - `data/color/top*.mp4`、`data/thermal/temp*.mp4`
  - DLC 训练好的 snapshot（放入你本地 `dlc_project/`）
  - `alignment/transform.json`（运行 `python -m alignment.tune` 标定）
  - `thermometry/anchors.json`（按 `thermometry/anchors.example.json` 填）

## 环境

```powershell
conda activate DEEPLABCUT
# 主要依赖：deeplabcut, opencv-python, pandas, numpy, matplotlib
```
