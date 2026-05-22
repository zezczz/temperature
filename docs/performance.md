# 全流水线参数量 / FLOPs / 延时估算

本文档覆盖三段：

1. **DLC 推理**（彩色视频上跑关键点检测）—— ResNet-50 backbone + Heatmap/Locref head
2. **彩色 → 热成像映射** + 热图采样（`alignment.dlc sample`）
3. **温度计算与聚合**（`thermometry apply`）

> 输入：彩色视频 **1920×1080**，DLC 项目配置 `ResNet-50 (GN), output_stride=16, 3 bodyparts`，热成像视频 **1440×1080**。FLOP 与 MAC 关系：**1 MAC = 2 FLOP**。
>
> DLC 部分给的是 **`scripts/bench_dlc.py` 在 RTX 4070 Laptop 上的实测值**；其余阶段 OpenCV/numpy 的估算见 §2、§3。

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

### 1.2 参数量（实测）

`fvcore.parameter_count` 在加载 `snapshot-best-200.pt` 后给出：

| 模块 | 参数 |
| --- | --- |
| Backbone (ResNet-50 GN, stride=16) | **≈ 23.51 M** |
| Heatmap + Locref head | **≈ 0.17 M** |
| **合计** | **23,673,929 ≈ 23.67 M params** |
| 显存占用（FP32 权重） | ≈ **95 MB** |

> 比标准 ResNet-50（25.56 M）少 ≈ 2 M，是 GroupNorm 版本省下了 BN 的 running stats / γβ 参数。

### 1.3 FLOPs（单帧 1920×1080, 实测）

`fvcore.FlopCountAnalysis` 报告：

| 项 | MACs | FLOPs |
| --- | --- | --- |
| Backbone (`backbone.model`) | **260.44 G** | 520.88 G |
| └ `layer4` | 122.39 G | 244.78 G |
| └ `layer3` | 61.38 G | 122.76 G |
| └ `layer2` | 43.11 G | 86.22 G |
| └ `layer1` | 28.52 G | 57.04 G |
| └ stem + 其它 | 5.04 G | 10.08 G |
| Heads (heatmap + locref) | ≈ 1.35 G | 2.70 G |
| **合计** | **≈ 261.79 GMACs** | **≈ 523.58 GFLOPs** |

要点：

- **backbone 占 99.5%**，其中 **`layer4` 单独 47%**（stride=16 让最后一个 stage 多算了一倍）。
- **heads 仅 0.5%**（输出 stride=8，特征图 240×135，3+6 通道）。
- 标准 ResNet-50（stride=32）在 224×224 是 4.1 GMACs；本项目 stride=16 在 224×224 实测 **6.30 GMACs**，多 50%，符合预期。
- 同分辨率改成 batch=N 时 FLOPs/吞吐都按 N 线性增长。

### 1.4 延时（实测）

`scripts/bench_dlc.py` 在 **RTX 4070 Laptop GPU** 上 20 次推理：

| 项 | 值 |
| --- | --- |
| mean | **95.77 ms / 帧** |
| median | 95.79 ms |
| min / max | 95.54 / 96.08 ms |
| std | 0.15 ms |
| throughput | **10.44 FPS** |

3514 帧（≈140 s @ 25 fps）视频：

| 平台 | 单帧 | 全片 | 备注 |
| --- | --- | --- | --- |
| **RTX 4070 Laptop**（本机实测） | **96 ms** | **≈ 5.6 min** | FP32, batch=1 |
| RTX 4090 (估) | ≈ 25~35 ms | ≈ 1.5~2 min | 算力 ≈ 4070 Laptop 的 3~4× |
| RTX 3060 (估) | ≈ 130~180 ms | ≈ 7~10 min | 算力 ≈ 4070 Laptop 的 0.5~0.7× |
| CPU (估) | ≈ 1~2 s | ≈ 1~2 h | DLC PyTorch 后端 |

> 想要更快：开 `autocast.enabled: true`（FP16）通常再快 1.5~2×；下采样到 960×540 → FLOPs ↓ 4× → 帧时间约 25~35 ms。详见 §6。

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

| 阶段 | 参数量 | FLOPs/帧 | 单帧延时 (RTX 4070 Laptop / CPU) |
| --- | --- | --- | --- |
| DLC 推理 | **23.67 M** | **523.6 G** | **95.8 ms** / ≈ 1~2 s |
| 几何映射 + 采样 | 9 floats | 0.027 G | — / 5~10 ms |
| 温度计算 | 2 floats | 1.4 × 10⁻⁸ G | — / <0.001 ms |
| **合计** | **≈ 23.67 M** | **≈ 523.6 G** | **≈ 100~105 ms / ≈ 1~2 s** |

> 三个阶段计算量比例 **DLC : 映射采样 : 温度 ≈ 19000 : 1 : 5×10⁻⁵**。

### 4.2 一段视频（3514 帧, ≈ 140 s @ 25 fps）

| 阶段 | RTX 4070 Laptop（本机实测） | RTX 4090（估） | CPU（估） |
| --- | --- | --- | --- |
| DLC | **≈ 5.6 min** | **1.5~2 min** | **1~2 h** |
| sample | 30~40 s | 30~40 s | 30~40 s |
| apply | <1 s | <1 s | <1 s |
| **End-to-end** | **≈ 6~7 min** | **2~3 min** | **1~2 h** |

> sample 在 CPU 上跑（cv2 / numpy），所以即使换更强的 GPU 也不会更快。如果要把整段视频压到 1 分钟以内，要么 (1) DLC 改 batch + FP16；(2) 把 boxFilter / argmax 也搬到 GPU（cupy / torch）。

### 4.3 内存峰值

| 阶段 | 显存（GPU） | 内存（CPU） |
| --- | --- | --- |
| DLC | 权重 ≈ 100 MB + 单帧激活 ≈ 1.5~2 GB | 视频解码缓存 ≈ 100 MB |
| sample | — | thermal 帧 + mean_map ≈ 12 MB |
| apply | — | csv ≈ 数 MB |

---

## 5. 怎么自己测真实数字

### 5.1 DLC：参数 / FLOPs / 延时一次给齐

仓库自带脚本 `scripts/bench_dlc.py`，会自动找到 `dlc_project/` 里最新的 PyTorch shuffle、加载 `snapshot-best-*.pt`，然后跑 fvcore 数 FLOPs + 实测延时。

```powershell
# 默认 1920×1080, batch=1, 20 次推理, 自动选 cuda
python scripts\bench_dlc.py

# 切到一半分辨率看提速
python scripts\bench_dlc.py --height 540 --width 960 --iters 20

# 看 batch 吞吐
python scripts\bench_dlc.py --batch 4 --iters 10

# 强制 CPU 基线
python scripts\bench_dlc.py --device cpu --iters 3
```

输出会同时给：

- `Params : 23,673,929 (~23.67 M)`
- `MACs/帧 : 261.79 GMACs` / `FLOPs/帧 : 523.58 GFLOPs`
- 模块级 top-10 占比
- 20 次实测的 `mean / median / min / max / std (ms)` 和 `throughput (FPS)`

依赖：`pip install fvcore`（DEEPLABCUT 环境里已加）。

### 5.2 也想测完整 DLC 推理流水（含 I/O / 预处理）

```python
import time, deeplabcut as dlc
t0 = time.time()
dlc.analyze_videos(config_path, [video_path], shuffle=1, save_as_csv=True)
print("elapsed:", time.time() - t0)
```

比 §5.1 的纯网络延时多了视频解码、letterbox、CPU→GPU 传输、后处理 argmax，一般是 1.2~1.5× 倍。

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

- §1 的所有数字都是 **`scripts/bench_dlc.py` 在 RTX 4070 Laptop GPU、FP32、batch=1、1920×1080 输入** 上跑出来的实测；想换设备直接重跑脚本即可。
- GFLOPs 按 **MAC × 2** 折算；引用论文如果用的是 GMACs 请直接看 MACs 列。
- 没有计入：mp4 解码 / 编码、Python 解释器开销、磁盘 I/O。视频解码在 1920×1080 H.264 上 CPU 约 **3~6 ms/帧**，在长视频里会和 DLC GPU 推理并发掩盖。
- DLC `batch_size`（推理）默认 1，**改成 4~8 通常能再快 30~80%**，前提是显存够（每张图激活 ≈ 1.5 GB）。
