"""
点击对应点标定（4 点以上单应性 / 3 点仿射）。

用法:
    python -m alignment.calibrate
    python -m alignment.calibrate --mode affine

步骤:
    1. 在「thermal」窗口依次点击与彩色图相同结构的位置（至少 3/4 点）
    2. 在「color」窗口按相同顺序点击对应点
    3. 按 Enter 计算；s 保存；r 重置；q 退出
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from alignment.config import DEFAULT_CALIB_COLOR, DEFAULT_CALIB_FRAME, DEFAULT_CALIB_THERMAL
from alignment.transforms import (
    affine_from_point_pairs,
    homography_from_point_pairs,
    load_transform,
    save_transform,
    warp_thermal,
)


class PointPicker:
    def __init__(self, title: str, image: np.ndarray):
        self.title = title
        self.image = image.copy()
        self.base = image.copy()
        self.points: list[tuple[int, int]] = []
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(title, self._on_mouse)

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self._redraw()

    def _redraw(self):
        self.image = self.base.copy()
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.image, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(self.image, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(self.title, self.image)

    def reset(self):
        self.points.clear()
        self._redraw()

    def show(self):
        self._redraw()


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取帧 {index}: {path}")
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="点击标定 thermal→color")
    parser.add_argument("--color", type=Path, default=DEFAULT_CALIB_COLOR)
    parser.add_argument("--thermal", type=Path, default=DEFAULT_CALIB_THERMAL)
    parser.add_argument("--frame", type=int, default=DEFAULT_CALIB_FRAME)
    parser.add_argument("--mode", choices=("homography", "affine"), default="homography")
    args = parser.parse_args(argv)

    color = read_frame(args.color, args.frame)
    thermal = read_frame(args.thermal, args.frame)
    pick_t = PointPicker("thermal: 按顺序点击", thermal)
    pick_c = PointPicker("color: 按相同顺序点击", color)
    pick_t.show()
    pick_c.show()

    print("thermal 窗口点选后，在 color 窗口点相同结构；Enter 计算，s 保存，r 重置，q 退出")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            pick_t.reset()
            pick_c.reset()
        if key in (13, ord("s")) and len(pick_t.points) >= (3 if args.mode == "affine" else 4):
            if len(pick_t.points) != len(pick_c.points):
                print("两侧点数须相同")
                continue
            pt = np.float32(pick_t.points)
            pc = np.float32(pick_c.points)
            if args.mode == "affine":
                if len(pt) != 3:
                    print("仿射需要恰好 3 对点")
                    continue
                M = affine_from_point_pairs(pt, pc)
                tform = {
                    "version": 1,
                    "mode": "affine",
                    "src": "thermal",
                    "dst": "color",
                    "src_size": [thermal.shape[1], thermal.shape[0]],
                    "dst_size": [color.shape[1], color.shape[0]],
                    "matrix": M,
                    "calibration_frame": args.frame,
                    "point_pairs": {"thermal": pt.tolist(), "color": pc.tolist()},
                }
            else:
                H = homography_from_point_pairs(pt, pc)
                tform = {
                    "version": 1,
                    "mode": "homography",
                    "src": "thermal",
                    "dst": "color",
                    "src_size": [thermal.shape[1], thermal.shape[0]],
                    "dst_size": [color.shape[1], color.shape[0]],
                    "matrix": H,
                    "calibration_frame": args.frame,
                    "point_pairs": {"thermal": pt.tolist(), "color": pc.tolist()},
                }
            warped = warp_thermal(thermal, tform)
            preview = cv2.addWeighted(color, 0.55, warped, 0.45, 0)
            cv2.imshow("preview", preview)
            if key == ord("s"):
                save_transform(tform)
                print("已保存 transform.json")
                break
            print("预览已更新；按 s 保存")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
