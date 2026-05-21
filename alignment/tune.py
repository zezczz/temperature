"""
滑块实时微调热成像→彩色的对齐（推荐先运行本脚本）。

用法（DEEPLABCUT 等含 opencv 的环境）:
    python -m alignment.tune
    python -m alignment.tune --frame 200

操作:
    拖动滑块调整 scale_x / scale_y / tx / ty / rotation
    s  保存到 alignment/transform.json
    q  退出不保存
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from alignment.config import (
    COLOR_SIZE,
    DEFAULT_CALIB_COLOR,
    DEFAULT_CALIB_FRAME,
    DEFAULT_CALIB_THERMAL,
)
from alignment.transforms import affine_from_params, load_transform, save_transform, warp_thermal


def homography_from_params(sx, sy, tx, ty, deg):
    M = affine_from_params(sx, sy, tx, ty, deg)
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = M
    return H


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取帧 {index}: {path}")
    return frame


TX_RANGE = 2000  # tx 滑块半幅，单位像素：tx ∈ [-2000, +2000]
TY_RANGE = 1500
SCALE_MAX = 300  # scale_x% 上限 = 300 即放大到 3.0×


class Tuner:
    def __init__(self, color: np.ndarray, thermal: np.ndarray, init: dict | None):
        self.color = color
        self.thermal = thermal
        self.win = "alignment_tune (s=save, q=quit)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 1280, 720)

        sx, sy, tx, ty, deg = 1.0, 1.0, 0.0, 0.0, 0.0
        prerot = 0
        if init:
            prerot = int(init.get("prerot", 0))
        if init and init.get("mode") == "homography":
            H = np.asarray(init["matrix"], dtype=np.float32)
            sx = float(np.linalg.norm(H[0, :2]))
            sy = float(np.linalg.norm(H[1, :2]))
            tx, ty = float(H[0, 2]), float(H[1, 2])
            deg = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))

        sx = max(0.1, min(sx, SCALE_MAX / 100.0))
        sy = max(0.1, min(sy, SCALE_MAX / 100.0))
        tx = max(-TX_RANGE, min(tx, TX_RANGE))
        ty = max(-TY_RANGE, min(ty, TY_RANGE))
        deg = max(-90.0, min(deg, 90.0))

        cv2.createTrackbar("prerot*90", self.win, prerot // 90, 3, lambda v: None)
        cv2.createTrackbar("scale_x%", self.win, int(sx * 100), SCALE_MAX, lambda v: None)
        cv2.createTrackbar("scale_y%", self.win, int(sy * 100), SCALE_MAX, lambda v: None)
        cv2.createTrackbar("tx+2000", self.win, int(tx + TX_RANGE), 2 * TX_RANGE, lambda v: None)
        cv2.createTrackbar("ty+1500", self.win, int(ty + TY_RANGE), 2 * TY_RANGE, lambda v: None)
        cv2.createTrackbar("deg+90", self.win, int(deg + 90), 180, lambda v: None)

    def params(self):
        prerot = cv2.getTrackbarPos("prerot*90", self.win) * 90
        sx = cv2.getTrackbarPos("scale_x%", self.win) / 100.0
        sy = cv2.getTrackbarPos("scale_y%", self.win) / 100.0
        tx = cv2.getTrackbarPos("tx+2000", self.win) - TX_RANGE
        ty = cv2.getTrackbarPos("ty+1500", self.win) - TY_RANGE
        deg = cv2.getTrackbarPos("deg+90", self.win) - 90
        return prerot, sx, sy, tx, ty, deg

    def loop(self) -> dict | None:
        saved = None
        flash = 0  # 屏幕上 SAVED 字样剩余帧数
        try:
            while True:
                prerot, sx, sy, tx, ty, deg = self.params()
                H = homography_from_params(sx, sy, tx, ty, deg)
                tform = {
                    "version": 1,
                    "mode": "homography",
                    "src": "thermal",
                    "dst": "color",
                    "src_size": [self.thermal.shape[1], self.thermal.shape[0]],
                    "dst_size": [self.color.shape[1], self.color.shape[0]],
                    "prerot": prerot,
                    "matrix": H,
                }
                warped = warp_thermal(self.thermal, tform)
                blend = cv2.addWeighted(self.color, 0.55, warped, 0.45, 0)
                label = (
                    f"prerot={prerot}  sx={sx:.3f} sy={sy:.3f}  "
                    f"tx={tx:.0f} ty={ty:.0f}  rot={deg:.1f}"
                )
                cv2.putText(blend, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if flash > 0:
                    cv2.putText(blend, "SAVED", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    flash -= 1
                cv2.imshow(self.win, blend)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    save_transform(tform)
                    saved = tform
                    flash = 60
                    print(f"已写入 {Path(__file__).resolve().parent / 'transform.json'}")
        finally:
            cv2.destroyAllWindows()
        return saved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="滑块微调 color/thermal 对齐")
    parser.add_argument("--color", type=Path, default=DEFAULT_CALIB_COLOR)
    parser.add_argument("--thermal", type=Path, default=DEFAULT_CALIB_THERMAL)
    parser.add_argument("--frame", type=int, default=DEFAULT_CALIB_FRAME)
    args = parser.parse_args(argv)

    init = None
    try:
        init = load_transform()
    except FileNotFoundError:
        pass

    color = read_frame(args.color, args.frame)
    thermal = read_frame(args.thermal, args.frame)
    tuner = Tuner(color, thermal, init)
    tuner.loop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
