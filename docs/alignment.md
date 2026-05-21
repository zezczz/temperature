# Color / Thermal 双相机对齐

把 `data/thermal/` 里 **1440×1080** 的热成像视频，逐帧映射到 `data/color/` 里 **1920×1080** 的彩色视频坐标系。两台相机位置固定、夹角很小，所以**只需标定一次**，把得到的几何变换写死在 `alignment/transform.json` 里，之后所有视频/帧共用。

---

## 1. 目录结构

```
alignment/
├── __init__.py
├── __main__.py            # python -m alignment 入口，等价于 alignment.align
├── config.py              # 路径、分辨率、默认标定视频
├── pairing.py             # 按文件名时间戳匹配 color / thermal
├── transforms.py          # 加载/保存/应用变换，正/反坐标映射
├── transform.json         # 标定结果（核心文件，必须存在）
├── tune.py                # 交互式滑块微调（推荐）
├── calibrate.py           # 点击对应点标定
├── align.py               # 预览 / 导出单帧 / 导出叠加视频 / 批量
└── dlc.py                 # 把彩色视频上的 DLC 关键点应用到热成像

data/
├── color/   top<stamp>.mp4       # 1920×1080
├── thermal/ temp<stamp>.mp4      # 1440×1080
└── aligned/                      # 程序输出
    ├── preview/   {stamp}_f{idx}_overlay.jpg
    ├── frames/{stamp}/  color_*.jpg / thermal_aligned_*.jpg / overlay_*.jpg
    └── videos/    {stamp}_overlay.mp4
```

**视频文件命名规则**（`pairing.py` 据此自动配对）：

* color 前缀 `top`，thermal 前缀 `temp`
* 文件名里必须含形如 `2026-05-17 14-44-35` 或 `2026-05-17-14-44-35` 的时间戳
* 时间戳一致的两个视频自动配成一对

---

## 2. 工作原理

热成像 → 彩色 的映射分两步：

```
thermal (1440×1080)
   └── ① 预旋转 prerot ∈ {0, 90, 180, 270}    # 修正两台相机物理朝向差
       └── ② 仿射 / 单应矩阵 M                # 修正缩放、平移、小角度旋转
           └── color (1920×1080)
```

为什么要分两步？早期版本只用一个单应矩阵，但两台相机其中一台是横装、一台是竖装，差了 90°；如果把这个 90° 编进 `M`，滑块就需要在大旋转 + 大平移之间联动，调起来非常反直觉。拆出 `prerot` 之后，`M` 只需处理小量缩放/平移，滑块行为变得线性、易调。

变换矩阵存在 `alignment/transform.json` 里，目前实际使用的版本（示例）：

```json
{
  "mode": "homography",
  "src": "thermal",  "dst": "color",
  "src_size": [1440, 1080], "dst_size": [1920, 1080],
  "prerot": 270,
  "matrix": [
    [0.68, 0.00,  707.0],
    [0.00, 0.61, -124.0],
    [0.00, 0.00,    1.0]
  ]
}
```

> `prerot=270` 表示加载帧后先做一次 `cv2.ROTATE_90_COUNTERCLOCKWISE`。`matrix` 在 `homography` 模式下是 3×3，在 `affine` 模式下是 2×3。

---

## 3. 准备环境

需要 Python + OpenCV。项目里现成可用：

```powershell
conda activate DEEPLABCUT
cd E:\projects\for_study\temperature
```

所有命令都以 **项目根目录** 为工作目录运行（模块以 `python -m alignment.xxx` 形式调用，否则 import 会失败）。

---

## 4. 标定流程（首次使用 / 相机被动过）

> 已经存在合用的 `transform.json` 时可跳过本节，直接看第 5 节。

### 4.1 推荐：滑块微调

```powershell
python -m alignment.tune
```

弹出窗口里有半透明叠加（彩色 + warp 后的热图），上方 6 个滑块：

| 滑块 | 含义 | 说明 |
| --- | --- | --- |
| `prerot*90` | 0 / 1 / 2 / 3 → 0° / 90° / 180° / 270° | 先粗调这个，让两个画面的菱形板朝向一致 |
| `scale_x%` | 水平缩放（百分比） | 上限 300 即放大 3.0× |
| `scale_y%` | 垂直缩放（百分比） | |
| `tx+2000` | 水平平移 + 2000 | 中点 2000 = 0 像素 |
| `ty+1500` | 垂直平移 + 1500 | 中点 1500 = 0 像素 |
| `deg+90` | 微旋转角度 + 90° | 一般只在 ±2° 范围内动 |

操作：

* 拖动滑块实时看叠加效果
* 按 `s` → 立刻写入 `transform.json`，屏幕右上角闪 `SAVED`，**不退出**，可继续微调再 `s`
* 按 `q` / `Esc` → 退出

启动时会自动从 `transform.json` 反推出滑块初值，所以**再次打开会从上一次停下的位置继续**。

可选参数：

```powershell
python -m alignment.tune --frame 1500
python -m alignment.tune --color  <video> --thermal <video> --frame 800
```

### 4.2 备选：点击对应点标定

```powershell
python -m alignment.calibrate                  # 单应（4 点起）
python -m alignment.calibrate --mode affine    # 仿射（恰好 3 点）
```

* 在 `thermal` 窗口依次点击稳定结构（菱形板四角、网格交点等）
* 在 `color` 窗口按**同样顺序**点击对应位置
* 按 `Enter` 计算并预览
* 按 `s` 保存
* 按 `r` 全部清空重点
* 按 `q` 退出

> ⚠️ 当前实现暂未叠加 `prerot`，所以点击标定前请确保 `transform.json` 里 `prerot` 已经设对（用 tune 调一下也行），或者你点的坐标系本身已经把旋转考虑进去。多数情况下 **tune 已经够用**。

---

## 5. 使用

### 5.1 查看可用视频对

```powershell
python -m alignment.align pairs
```

输出形如：

```
2026-05-17-14-44-35
  color:   top2026-05-17 14-44-35.mp4
  thermal: temp2026-05-17 14-44-35.mp4
...
```

### 5.2 单帧叠加预览（最常用，做验证）

```powershell
python -m alignment.align preview --stamp 2026-05-17-14-44-35 --frame 100
python -m alignment.align preview --stamp 2026-05-17-14-44-35 --frame 1500 --alpha 0.4
```

输出 → `data/aligned/preview/{stamp}_f{idx}_overlay.jpg`

`--alpha` 控制热图占比，范围 0~1，默认 0.45。

### 5.3 单帧三视图（color / 对齐后 thermal / overlay 各存一张）

```powershell
python -m alignment.align frame --stamp 2026-05-17-14-44-35 --frame 100
```

输出 → `data/aligned/frames/{stamp}/`

### 5.4 整段视频叠加导出

```powershell
python -m alignment.align video --stamp 2026-05-17-14-44-35
python -m alignment.align video --stamp 2026-05-17-14-44-35 --max-frames 500   # 测试用
```

输出 → `data/aligned/videos/{stamp}_overlay.mp4`

### 5.5 批量处理全部时间戳

```powershell
python -m alignment.align batch                # 跳过已存在的
python -m alignment.align batch --force        # 覆盖重写
```

### 5.6 重置变换文件

```powershell
python -m alignment.align init
```

会把 `transform.json` 写成单位变换。**只在变换文件损坏或要从零重标时用。**

---

## 6. `transform.json` 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `version` | int | 文件格式版本，目前为 1 |
| `mode` | `"homography"` / `"affine"` | 矩阵类型，决定用 `warpPerspective` 还是 `warpAffine` |
| `src` / `dst` | `"thermal"` / `"color"` | 方向标识，目前固定 thermal → color |
| `src_size` | `[w, h]` | 源图（旋转前）尺寸 |
| `dst_size` | `[w, h]` | 目标尺寸 |
| `prerot` | 0 / 90 / 180 / 270 | 应用矩阵**之前**对源图做的整体旋转 |
| `matrix` | `3×3` 或 `2×3` | 实际的几何矩阵 |
| `note` / `calibration_frame` / `point_pairs` | 可选 | 备注，被自动写入但不参与计算 |

---

## 7. 把 DLC 关键点应用到热成像

跑 DLC 是在 **彩色视频** 上做的，得到形如 `top<stamp>DLC_..._best-30.csv` 的关键点结果（`left_eye`、`right_eye`、`tail_base` 等，三列一组：`x, y, likelihood`）。要把这些坐标应用到 **热成像**（取温度代理、可视化叠加等），用 `alignment.dlc` 模块。

底层只有一步：`map_color_point_to_thermal(xc, yc)` —— 把彩色像素坐标反向映射回热图像素坐标（先做矩阵 `H⁻¹`，再做 `prerot` 的反向旋转）。

### 7.1 子命令

```powershell
# 1) 坐标转换：在原 DLC csv 的多级表头结构上，把所有 (x, y) 列改写为 thermal 坐标
python -m alignment.dlc convert `
    --csv data\top2026-05-17-14-44-35DLC_Resnet50_TemperatureTopViewMay18shuffle1_snapshot_best-30.csv `
    --out data\aligned\dlc\top2026-05-17-14-44-35_thermal_coords.csv

# 2) 在 thermal 视频上读出每个关键点位置的像素亮度（伪彩 → 灰度，作为温度的相对指标）
python -m alignment.dlc sample `
    --csv  data\top2026-05-17-14-44-35DLC_...best-30.csv `
    --thermal "data\thermal\temp2026-05-17 14-44-35.mp4" `
    --out  data\aligned\dlc\top2026-05-17-14-44-35_thermal_intensity.csv `
    --radius 3 --p-cutoff 0.25

# 3) 把 DLC 点画到 thermal 视频上，导出标注视频
python -m alignment.dlc draw `
    --csv  data\top2026-05-17-14-44-35DLC_...best-30.csv `
    --thermal "data\thermal\temp2026-05-17 14-44-35.mp4" `
    --out  data\aligned\dlc\top2026-05-17-14-44-35_thermal_labeled.mp4 `
    --p-cutoff 0.25
```

参数说明：

| 参数 | 命令 | 含义 |
| --- | --- | --- |
| `--csv` | 三者通用 | DLC 在彩色视频上跑出的 csv |
| `--thermal` | `sample` / `draw` | 对应的热成像 mp4（按时间戳手动指定） |
| `--out` | 三者通用 | 输出路径（自动建目录） |
| `--radius` | `sample` | 取 `(2r+1)×(2r+1)` 邻域均值，`0` = 单像素 |
| `--p-cutoff` | `sample` / `draw` | 低于该 likelihood 的点：`sample` 记 NaN，`draw` 不画 |
| `--dot-radius` | `draw` | 圆点半径（像素） |
| `--max-frames` | `sample` / `draw` | 限制处理帧数（调试用） |

`sample` 输出的 csv 是宽表（一行一帧），每个关键点 4 列：`<bp>_x, <bp>_y, <bp>_intensity, <bp>_likelihood`，可以直接送进 pandas / Excel 画时间序列。

### 7.2 关于「像素亮度」≠「温度」

`sample` 取的是热成像 mp4 解码后的灰度值（0~255）。如果你的热相机已经把数据烤进伪彩 colormap，灰度只能作为**相对热度**，不能直接换算成 °C。

要拿到真实温度，需要：

* 拿到热相机原始数据（raw / TIFF / .seq 等，而不是 mp4），并按厂家公式 / LUT 换算；或者
* 自己做一次定标：用已知温度物体（如冰水、温水）出现在画面中的强度，拟合 `intensity → °C` 的曲线，再对 `sample` 输出的列做转换。

### 7.3 Python API

```python
from alignment.transforms import (
    load_transform,
    warp_thermal,
    blend_overlay,
    map_thermal_point_to_color,
    map_color_point_to_thermal,
)

tform = load_transform()  # 默认读 alignment/transform.json

# 整张热图 → 彩色坐标系（1920×1080）
aligned_thermal = warp_thermal(thermal_bgr, tform)

# 单点正向映射（热图坐标 → 彩色坐标）
xc, yc = map_thermal_point_to_color(xt, yt, tform)

# 单点反向映射（彩色坐标 → 热图坐标）—— 把 DLC 点搬到 thermal 时用这个
xt, yt = map_color_point_to_thermal(xc, yc, tform)

# 叠加预览
preview = blend_overlay(color_bgr, aligned_thermal, alpha=0.45)
```

---

## 8. 常见问题

**Q: 叠加图里热图完全看不见？**
A: 一般是 `prerot` 没对，warp 后整张图被推出画面。先在 tune 里轮换 `prerot*90` 的 0/1/2/3 找朝向，再调缩放和平移。

**Q: 调到一半按 s，重开后参数好像不太一样？**
A: 滑块只有整数粒度，从矩阵反推时会有 ±0.5 像素 / 1% 缩放的取整误差，正常。重要参数（`prerot`、整数 `tx/ty`）一定准确。

**Q: 不同视频段之间小鼠位置变化大，能用同一个 `transform.json` 吗？**
A: 能。变换只取决于两台相机的相对几何关系，与拍的内容无关，只要相机没被动过。

**Q: 改了相机位置 / 角度怎么办？**
A: 重新跑一次 `python -m alignment.tune`，按 `s` 写入新参数即可。`transform.json` 也可以用 git 备份多个版本。

**Q: 想要更精细的滑块？**
A: 编辑 `alignment/tune.py` 顶部常量 `TX_RANGE`、`TY_RANGE`、`SCALE_MAX` 调整范围；scale 是百分比所以已经 1% 步进，平移是 1px 步进。

**Q: 想要让对齐后画面尺寸不是 1920×1080？**
A: `warp_thermal(thermal_bgr, dst_size=(w, h))` 传新尺寸；命令行版本默认走 `transform.json["dst_size"]`。

**Q: DLC 模型置信度低，`draw` 时关键点不显示？**
A: 把 `--p-cutoff` 调小，例如 `0.25` 或 `0.0`。我们这套 DLC 模型在 0.5 之上的帧很少（详见 csv 中 likelihood 分布），实际使用建议先用 `--max-frames 200` 跑短片段验证。

**Q: `sample` 出来的「亮度」就是温度吗？**
A: 不是直接的 °C，是 mp4 灰度值（0~255），作为**相对热度**指标。要换成 °C，要么用原始热成像数据 + 厂家 LUT，要么做一次定标拟合，见 §7.2。

---

## 9. 历史与设计取舍

* 最初尝试自动特征匹配（ORB + RANSAC、ECC、Phase Correlation、模板匹配）效果都很差——彩色与热成像跨模态信息差异太大，且场景以重复网格为主，几乎没有可靠的判别性特征。
* 改为人工 4 点标定 + 单应矩阵，效果一般，单应里的斜切项干扰判断。
* 最终方案：`prerot` 处理 90° 量级的整体旋转，剩下用「scale + translate + 小角度旋转」的纯仿射处理，调起来最直观，并且单一矩阵 `transform.json` 足够描述。
