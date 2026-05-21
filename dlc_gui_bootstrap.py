"""
在导入 napari / napari-deeplabcut 之前设置环境变量，减轻 Windows 上插件扫描卡死。
由 dlc_workflow.py 在 label / diagnose 前自动调用。
"""
from __future__ import annotations

import os


def apply() -> None:
    # Qt 后端（须在 import qtpy / napari 之前）
    os.environ.setdefault("QT_API", "pyside6")
    # 减少 napari 启动时扫描全部已安装插件（常见卡死原因）
    os.environ.setdefault("NAPARI_DISABLE_PLUGIN_DISCOVERY", "1")
    # 部分环境下可避免同步阻塞
    os.environ.setdefault("NAPARI_ASYNC", "1")
