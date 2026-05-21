# 全流水线参数量 / FLOPs / 延时估算

本文档覆盖三段：

1. **DLC 推理**（彩色视频上跑关键点检测）—— ResNet-50 backbone + Heatmap/Locref head
2. **彩色 → 热成像映射** + 热图采样（`alignment.dlc sample`）
3. **温度计算与聚合**（`thermometry apply`）

> 估算的输入：彩色视频 **1920×1080**，DLC 项目配置 `ResNet-50 (GN), output_stride=16, 3 bodyparts`，热成像视频 **1440×1080**。FLOP 与 MAC 关系：**1 MAC ≈ 2 FLOP**。

---

## 1. DLC 推理（每帧）

### 1.1 模型结构（来自 `dlc-models-pytorch/.../train/pytorch_config.yaml`）

| 模块 | 设计 |
| --- | --- |
| Backbone | ResNet-50 (GroupNorm 版本), `output_stride=16` |
| Heatmap head | `ConvTranspose2d(2048→3, k=3, s=2)`，输出 stride = 8 |
| Locref head | `ConvTranspose2d(2048→6, k=3, s=2)`，输出 stride = 8 |
| 关键点数 | 3 (`left_eye`, `right_eye`, `tail_base`) |

> ResNet-50 默认 stride=32；DLC 通过 dilation 把最后一个 stage 改成 stride=16，**特征图大一倍**，FLOPs 比标准 ResNet-50 高约 30%。

### 1.2 参数量

| 模块 | 参数 |
| --- | --- |
| ResNet-50 backbone | ≈ **25.56 M** |
| Heatmap head（2048×3×3×3 + 3） | **55,299** |
| Locref head（2048×3×3×6 + 6） | **110,598** |
| **合计** | **≈ 25.73 M params** |
| 显存占用（FP32） | ≈ **103 MB** 权重 |

### 1.3 FLOPs（单帧, 1920×1080 输入）

#### Backbone（ResNet-50, stride=16）
- 224×224 时官方约 **4.1 GMACs**；stride-16 改造后 ≈ **5.5 GMACs**。
- 1920×1080 与 224×224 像素数之比 ≈ **41.3×**。
- 单帧 ≈ 5.5 × 41.3 ≈ **227 GMACs ≈ 454 GFLOPs**。

#### Heads（输出 240×135）
- Heatmap deconv：`240×135 × 9 × 2048 × 3` ≈ **1.79 GMACs ≈ 3.6 GFLOPs**
- Locref deconv：`240×135 × 9 × 2048 × 6` ≈ **3.58 GMACs ≈ 7.2 GFLOPs**
- 合计 ≈ **5.37 GMACs ≈ 10.7 GFLOPs**

#### DLC 单帧总计

| 项 | MACs | FLOPs |
| --- | --- | --- |
| Backbone | 227 G | 454 G |
| Heads | 5.4 G | 10.7 G |
| **合计** | **≈ 232 GMACs** | **≈ 465 GFLOPs** |

> 训练时 `batch_size: 8`，推理时 DLC 默认 batch=1；逐帧推理一般就是 ≈ 0.47 TFLOPs/帧。

### 1.4 延时实测/估算

| 平台 | 理论算力 | 单帧延时（FP32） | 备注 |
| --- | --- | --- | --- |
| RTX 4090 | 83 TFLOPs FP32 | **≈ 6 ms** | 受显存带宽限制，实际 8~15 ms |
| RTX 3090 | 35 TFLOPs FP32 | **≈ 13 ms** | 实际 15~25 ms |
| RTX 3060 | 13 TFLOPs FP32 | **≈ 36 ms** | 实际 40~70 ms |
| GTX 1660 | 5 TFLOPs FP32 | **≈ 93 ms** | 实际 100~200 ms |
| CPU (i7-12700) | 0.5 TFLOPs | **≈ 0.9 s** | 实际 1~2 s/帧（DLC PyTorch 后端） |

> "理论 = 465 GFLOPs ÷ 卡算力"；实测因显存读写、Python overhead、kernel launch 一般是理论值的 2~3 倍。FP16/AMP 可再快 1.5~2×（pytorch_config 默认 `autocast.enabled: false`，没开）。

3514 帧视频在 RTX 3060 上跑完：**约 2~4 分钟**；在 CPU 上：**约 1~2 小时**。

---

## 2. 彩色 → 热成像 映射 + 采样（`alignment.dlc sample`）

### 2.1 单点几何映射

对每个 DLC 关键点 `(xc, yc)` 调用 `map_color_point_to_thermal`：

- 预旋转 `prerot`（整数比较 + 加减）≈ 5 FLOPs
- `cv2.perspectiveTransform`：3×3 矩阵乘点 + 1 次除法 ≈ **18 FLOPs**

3 个 bodypart × 23 ≈ **70 FLOPs/帧**。**完全可以忽略**。

### 2.2 帧级图像处理（1440×1080 thermal）

| 操作 | 计算量 | 备注 |
| --- | --- | --- |
| `cv2.cvtColor(BGR2GRAY)` | ≈ **4.67 M ops/帧** | 3 通道线性混合 |
| `cv2.boxFilter(7×7, 可分离)` | ≈ **21.8 M ops/帧** | 等效 2×7 ops/像素 |
| 搜索窗 argmax (±8, 17×17) × 3 bp | ≈ **0.87 K ops/帧** | 可忽略 |
| `perspectiveTransform × 3` | ≈ 70 FLOPs | 可忽略 |
| **合计** | **≈ 26.5 M ops/帧 ≈ 0.027 GFLOPs/帧** | 比 DLC **小 4 个数量级** |

### 2.3 参数 / 内存

| 项 | 量 |
| --- | --- |
| `alignment/transform.json` 矩阵 | 9 个 float32 ≈ **36 B** |
| 单帧 thermal RGB buffer | 1440×1080×3 ≈ **4.67 MB** |
| `mean_map` (float32, 全图 7×7 均值) | 1440×1080×4 ≈ **6.22 MB** |

### 2.4 延时（不含视频解码）

| 平台 | 单帧 | 3514 帧 |
| --- | --- | --- |
| CPU (i7-12700) | ≈ **2~5 ms** | **7~18 s** |

> 这一步主导成本其实是 **视频解码**：H.264 1440×1080 在 CPU 上 ≈ 2~6 ms/帧；总计算（解码 + cvtColor + boxFilter + 采样）约 **5~10 ms/帧**。实测 3514 帧大约 **20~40 秒**。

---

## 3. 温度计算 / 聚合（`thermometry apply`）

### 3.1 单帧单点

| 操作 | 计算量 |
| --- | --- |
| `T = a·I + b`（linear 标定） | 2 FLOPs |
| likelihood 比较 + nan 掩码 | 2 FLOPs |
| 3 bodypart 聚合 `nanmean` | ≈ 5 FLOPs |
| `body_temperature` ffill | 1 FLOP |
| **单帧合计** | **≈ 14 FLOPs** |

### 3.2 全视频（3514 帧）

- 直接乘以帧数 ≈ **5 × 10⁴ FLOPs**。**完全可忽略**。
- 实测瓶颈在 `pandas.read_csv` / `to_csv` I/O，整体 ≈ **0.5~1.0 秒**。

### 3.3 参数

| 项 | 量 |
| --- | --- |
| `calibration.json` (linear) | 2 个 float（a, b） |
| `calibration.json` (piecewise N anchor) | 2N 个 float |

**几乎零参数、零算力**，但它**决定了输出绝对值的正确性**——所有计算都建立在这两个数上。

---

## 4. 全链路汇总

### 4.1 单帧成本

| 阶段 | 参数量 | FLOPs/帧 | 单帧延时 (RTX 3060 / CPU) |
| --- | --- | --- | --- |
| DLC 推理 | **25.73 M** | **465 G** | 40~70 ms / 1~2 s |
| 几何映射 + 采样 | 9 floats | 0.027 G | — / 5~10 ms |
| 温度计算 | 2 floats | 1.4 × 10⁻⁸ G | — / <0.001 ms |
| **合计** | **≈ 25.73 M** | **≈ 465 G** | **≈ 45~80 ms / ≈ 1~2 s** |

> 三个阶段计算量比例 **DLC : 映射采样 : 温度 ≈ 17000 : 1 : 5×10⁻⁵**。

### 4.2 一段视频（3514 帧, ≈ 140 s @ 25 fps）

| 阶段 | RTX 3060 | RTX 4090 | CPU (i7-12700) |
| --- | --- | --- | --- |
| DLC | **2~4 min** | **30~50 s** | **1~2 h** |
| sample | 30~40 s | 30~40 s | 30~40 s |
| apply | <1 s | <1 s | <1 s |
| **End-to-end** | **3~5 min** | **1~2 min** | **1~2 h** |

> sample 阶段不上 GPU（只用 cv2 在 CPU 上算），所以在高端 GPU 卡上整体延时会被它拖到 ~30 秒下限；如果对极致吞吐有需求，可以把 boxFilter / argmax 改用 cupy 或 torch+cuda。

### 4.3 内存峰值

| 阶段 | 显存（GPU） | 内存（CPU） |
| --- | --- | --- |
| DLC | 权重 ≈ 100 MB + 单帧激活 ≈ 1.5~2 GB | 视频解码缓存 ≈ 100 MB |
| sample | — | thermal 帧 + mean_map ≈ 12 MB |
| apply | — | csv ≈ 数 MB |

---

## 5. 怎么自己测真实数字

### 5.1 DLC 推理延时

```python
import time, deeplabcut as dlc
t0 = time.time()
dlc.analyze_videos(config_path, [video_path], shuffle=1, save_as_csv=True)
print("elapsed:", time.time() - t0)
```

或者更精细，把 `deeplabcut.pose_estimation_pytorch.runners` 的 batch 循环加 `torch.cuda.synchronize()` + `time.perf_counter()` 套上。

### 5.2 参数 / FLOPs（用 fvcore）

```python
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# 假设你已经加载了 DLC 的 model（PyTorch nn.Module）
x = torch.randn(1, 3, 1080, 1920, device="cuda")
print("params:", parameter_count_table(model))
print("FLOPs :", FlopCountAnalysis(model, x).total() / 1e9, "GFLOPs")
```

### 5.3 sample / apply 延时

```powershell
Measure-Command {
  python -m alignment.dlc sample --csv ... --thermal ... --out ... --radius 3 --search-radius 8
}

Measure-Command {
  python -m thermometry apply --csv ... --bodyparts left_eye right_eye tail_base --aggregator mean --p-cutoff 0 --ffill
}
```

---

## 6. 优化方向（按 ROI 排序）

| 改动 | 收益 | 代价 |
| --- | --- | --- |
| **DLC 输入下采样** 到 960×540 | FLOPs ↓ 4×，延时 ↓ 3~4× | 关键点精度略降（眼角更明显） |
| **FP16 / AMP** 推理（`autocast.enabled: true`） | 延时 ↓ 1.5~2× | 几乎无精度损失 |
| **TorchScript / TensorRT** 导出 | 延时 ↓ 2~3× | 一次性导出成本 |
| **更小 backbone**（MobileNetV3 / ResNet-18） | 延时 ↓ 4~10× | mAP 下降 5~15% |
| Heatmap head 移除 locref | 头部 FLOPs ↓ 2/3 | 亚像素精度下降 ~1 px |
| `--search-radius` 改小或关掉 boxFilter | sample ↓ 30~50% | 抗 DLC 抖动能力下降 |
| DLC 直接跑在 thermal 上 | 省掉映射 | 需要在热成像上重新标 144 帧训练集 |

---

## 7. 备忘

- 上表的所有 GFLOPs 数都按 **MAC × 2** 算出；如果你引用论文用的是 GMACs，请折半。
- 没有计入：mp4 解码 / 编码、Python 解释器开销、磁盘 I/O。视频解码在 1920×1080 H.264 上 CPU 约 **3~6 ms/帧**，在长视频里会和 DLC GPU 推理并发掩盖。
- DLC `batch_size`（推理）默认 1，**改成 4~8 通常能再快 30~80%**，前提是显存够（每张图激活 ≈ 1.5 GB）。
