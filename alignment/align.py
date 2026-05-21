"""
将热成像对齐到彩色坐标系，并导出叠加预览或对齐视频。

用法:
    python -m alignment.align preview --stamp 2026-05-17-14-44-35
    python -m alignment.align frame --stamp 2026-05-17-14-44-35 --index 100
    python -m alignment.align video --stamp 2026-05-17-14-44-35
    python -m alignment.align batch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from alignment.config import COLOR_SIZE, OUTPUT_DIR
from alignment.pairing import pair_videos
from alignment.transforms import blend_overlay, load_transform, save_transform, warp_thermal


def read_frame_at(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取 {path} 第 {index} 帧")
    return frame


def find_pair(stamp: str | None):
    pairs = pair_videos()
    if not pairs:
        raise FileNotFoundError("未找到可匹配的 color/thermal 视频对")
    if stamp is None:
        return pairs[0]
    for s, c, t in pairs:
        if s == stamp or stamp in str(c) or stamp in str(t):
            return s, c, t
    raise FileNotFoundError(f"未找到时间戳 {stamp}，可用: {[p[0] for p in pairs]}")


def cmd_preview(args) -> None:
    stamp, color_path, thermal_path = find_pair(args.stamp)
    tform = load_transform()
    idx = args.frame
    color = read_frame_at(color_path, idx)
    thermal = read_frame_at(thermal_path, idx)
    warped = warp_thermal(thermal, tform)
    blend = blend_overlay(color, warped, args.alpha)
    out = OUTPUT_DIR / "preview"
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{stamp}_f{idx}_overlay.jpg"
    cv2.imwrite(str(out_file), blend)
    print(f"已保存: {out_file}")


def cmd_frame(args) -> None:
    stamp, color_path, thermal_path = find_pair(args.stamp)
    tform = load_transform()
    idx = args.frame
    color = read_frame_at(color_path, idx)
    thermal = read_frame_at(thermal_path, idx)
    warped = warp_thermal(thermal, tform)
    out = OUTPUT_DIR / "frames" / stamp
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"color_{idx:06d}.jpg"), color)
    cv2.imwrite(str(out / f"thermal_aligned_{idx:06d}.jpg"), warped)
    cv2.imwrite(str(out / f"overlay_{idx:06d}.jpg"), blend_overlay(color, warped, args.alpha))
    print(f"已写入 {out}")


def _write_video(
    color_path: Path,
    thermal_path: Path,
    out_path: Path,
    tform: dict,
    alpha: float,
    max_frames: int | None,
) -> None:
    cap_c = cv2.VideoCapture(str(color_path))
    cap_t = cv2.VideoCapture(str(thermal_path))
    if not cap_c.isOpened() or not cap_t.isOpened():
        raise RuntimeError("无法打开视频")

    w, h = COLOR_SIZE
    fps = cap_c.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(min(cap_c.get(cv2.CAP_PROP_FRAME_COUNT), cap_t.get(cv2.CAP_PROP_FRAME_COUNT)))
    if max_frames:
        n = min(n, max_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for i in range(n):
        ok_c, fc = cap_c.read()
        ok_t, ft = cap_t.read()
        if not ok_c or not ok_t:
            break
        warped = warp_thermal(ft, tform)
        writer.write(blend_overlay(fc, warped, alpha))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}")
    cap_c.release()
    cap_t.release()
    writer.release()
    print(f"完成: {out_path} ({n} 帧)")


def cmd_video(args) -> None:
    stamp, color_path, thermal_path = find_pair(args.stamp)
    tform = load_transform()
    out = OUTPUT_DIR / "videos" / f"{stamp}_overlay.mp4"
    _write_video(color_path, thermal_path, out, tform, args.alpha, args.max_frames)


def cmd_batch(args) -> None:
    tform = load_transform()
    pairs = pair_videos()
    for stamp, color_path, thermal_path in pairs:
        out = OUTPUT_DIR / "videos" / f"{stamp}_overlay.mp4"
        if out.exists() and not args.force:
            print(f"跳过已存在: {out.name}")
            continue
        print(f"处理 {stamp} ...")
        _write_video(color_path, thermal_path, out, tform, args.alpha, args.max_frames)


def cmd_pairs(_args) -> None:
    for stamp, c, t in pair_videos():
        print(f"{stamp}\n  color:   {c.name}\n  thermal: {t.name}")


def cmd_init(_args) -> None:
    from alignment.transforms import identity_transform

    path = save_transform(identity_transform("homography"))
    print(f"已创建单位变换: {path}，请运行 tune 或 calibrate")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="color / thermal 对齐导出")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preview", help="导出单帧叠加图")
    p.add_argument("--stamp", default=None)
    p.add_argument("--frame", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.45)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("frame", help="导出单帧 color / aligned / overlay")
    p.add_argument("--stamp", default=None)
    p.add_argument("--frame", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.45)
    p.set_defaults(func=cmd_frame)

    p = sub.add_parser("video", help="导出整段叠加视频")
    p.add_argument("--stamp", default=None)
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("batch", help="批量处理所有匹配视频对")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("pairs", help="列出可匹配的视频对")
    p.set_defaults(func=cmd_pairs)

    p = sub.add_parser("init", help="生成空白 transform.json")
    p.set_defaults(func=cmd_init)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
