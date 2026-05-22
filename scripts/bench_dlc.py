"""测一下 DLC 模型的参数量 / FLOPs / 延时。

用法（在项目根目录）：

    python scripts\bench_dlc.py
    python scripts\bench_dlc.py --height 1080 --width 1920 --device cuda --iters 30
    python scripts\bench_dlc.py --height 540  --width 960  --device cuda --iters 30
    python scripts\bench_dlc.py --device cpu --iters 5

会自动从 dlc_project/ 找到第一个 PyTorch shuffle 的 pytorch_config.yaml 和最新 snapshot。
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import yaml

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
except ImportError as exc:
    raise SystemExit(
        "缺少 fvcore，请先安装：\n"
        "    pip install fvcore"
    ) from exc

from deeplabcut.pose_estimation_pytorch.models import PoseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DLC_ROOT = PROJECT_ROOT / "dlc_project"


def _find_pytorch_train_dir() -> Path:
    cands = list(DLC_ROOT.glob("*/dlc-models-pytorch/iteration-*/*/train"))
    if not cands:
        raise FileNotFoundError(f"在 {DLC_ROOT} 下没有找到 dlc-models-pytorch/.../train")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _pick_snapshot(train_dir: Path) -> Path | None:
    best = sorted(train_dir.glob("snapshot-best-*.pt"))
    if best:
        return best[-1]
    snaps = sorted(train_dir.glob("snapshot-*.pt"))
    return snaps[-1] if snaps else None


def build_model(train_dir: Path):
    cfg_path = train_dir / "pytorch_config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = PoseModel.build(cfg["model"])
    snap = _pick_snapshot(train_dir)
    if snap is not None:
        state = torch.load(snap, map_location="cpu", weights_only=False)
        sd = state.get("model", state)
        try:
            model.load_state_dict(sd, strict=False)
            print(f"已加载权重: {snap.name}")
        except Exception as e:
            print(f"加载权重失败（用随机初始化继续）: {e}")
    return model, cfg


@torch.no_grad()
def measure_latency(model: torch.nn.Module, x: torch.Tensor, iters: int) -> dict:
    is_cuda = x.is_cuda
    # warmup
    for _ in range(3):
        _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.pstdev(times),
        "n": iters,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--train-dir", type=Path, default=None,
                   help="DLC train 目录，默认自动找最新 PyTorch shuffle")
    args = p.parse_args()

    train_dir = args.train_dir or _find_pytorch_train_dir()
    print(f"train_dir: {train_dir}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device   : {device}")
    if device == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")

    model, cfg = build_model(train_dir)
    model.eval().to(device)

    # 参数量
    n_params = parameter_count(model)[""]
    print(f"\nParams   : {n_params:,}   (~{n_params / 1e6:.2f} M)")

    # FLOPs
    x = torch.randn(args.batch, 3, args.height, args.width, device=device)
    print(f"input    : {tuple(x.shape)}  dtype={x.dtype}")
    flops = FlopCountAnalysis(model, x)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    total_macs = flops.total()
    print(f"MACs/帧  : {total_macs / 1e9:.2f} GMACs")
    print(f"FLOPs/帧 : {2 * total_macs / 1e9:.2f} GFLOPs  (按 1 MAC = 2 FLOPs)")

    # 模块级 top-10
    print("\nTop modules by MACs:")
    by_mod = sorted(flops.by_module().items(), key=lambda kv: kv[1], reverse=True)
    for name, val in by_mod[:10]:
        if not name:
            continue
        print(f"  {val / 1e9:8.2f} GMACs   {name}")

    # 延时
    print(f"\nLatency over {args.iters} runs (batch={args.batch}):")
    stats = measure_latency(model, x, args.iters)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:10s}: {v:8.2f}")
        else:
            print(f"  {k:10s}: {v}")
    print(f"  throughput: {1000.0 * args.batch / stats['mean_ms']:.2f} FPS")


if __name__ == "__main__":
    main()
