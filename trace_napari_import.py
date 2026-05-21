"""分段测试 napari 相关导入，定位卡在哪一步。每步子进程 45s 超时。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMEOUT = 45

os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("NAPARI_DISABLE_PLUGIN_DISCOVERY", "1")


def run_step(name: str, code: str) -> bool:
    print(f"[{name}] ", end="", flush=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=str(ROOT),
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        print(f"超时 (>{TIMEOUT}s) — 卡在此处")
        return False

    if proc.returncode == 0:
        out = (proc.stdout or "").strip().splitlines()
        print("OK", out[-1] if out else "")
        return True

    err = ((proc.stderr or "") + (proc.stdout or "")).strip()[:400]
    print(f"失败: {err}")
    return False


def main() -> int:
    steps = [
        ("qtpy", "import os; os.environ['QT_API']='pyside6'; import qtpy; print(qtpy.API_NAME)"),
        ("PySide6.QtCore", "import os; os.environ['QT_API']='pyside6'; from PySide6.QtCore import QTimer; print('QtCore ok')"),
        ("napari", "import os; os.environ['NAPARI_DISABLE_PLUGIN_DISCOVERY']='1'; import napari; print(napari.__version__)"),
        (
            "napari_deeplabcut",
            "import os; os.environ['QT_API']='pyside6'; os.environ['NAPARI_DISABLE_PLUGIN_DISCOVERY']='1'; "
            "import napari_deeplabcut; print(getattr(napari_deeplabcut,'__version__','ok'))",
        ),
    ]
    print("=== napari 导入分段测试 ===\n")
    ok = True
    for name, code in steps:
        if not run_step(name, code):
            ok = False
            break
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
