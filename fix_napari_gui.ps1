# 修复 Windows 上 napari-deeplabcut 导入卡死（conda 安装 Qt，避免 pip/conda 混装 DLL 冲突）
# 用法: conda activate DEEPLABCUT
#       .\fix_napari_gui.ps1

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$env:QT_API = "pyside6"
$env:NAPARI_DISABLE_PLUGIN_DISCOVERY = "1"

Write-Host "=== Step 1: 卸载 pip 安装的 Qt（避免与 conda DLL 冲突）===" -ForegroundColor Cyan
pip uninstall -y PySide6 shiboken6 PyQt5 PyQt5-Qt5 PyQt6 PyQt6-Qt6 2>$null

Write-Host "`n=== Step 2: 用 conda-forge 安装 PySide6 ===" -ForegroundColor Cyan
conda install -y -c conda-forge "pyside6>=6.6,<6.11" "shiboken6>=6.6,<6.11"

Write-Host "`n=== Step 3: 安装 napari 标注栈（pip）===" -ForegroundColor Cyan
pip install --upgrade "qtpy>=2.4.0" "napari==0.6.6" "napari-deeplabcut==0.2.1.8"

Write-Host "`n=== Step 4: 可选 — 移除易冲突的 napari 第三方插件 ===" -ForegroundColor Cyan
$plugins = @(
    "napari-animation", "napari-console", "napari-svg", "cellpose-napari",
    "napari-skimage-regionprops", "napari-mcp"
)
foreach ($p in $plugins) {
    pip uninstall -y $p 2>$null
}

Write-Host "`n=== Step 5: 分段测试导入（每步 45s）===" -ForegroundColor Cyan
python "$PSScriptRoot\trace_napari_import.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nnapari 仍无法使用。请改用 OpenCV 简易标注（不依赖 napari）:" -ForegroundColor Yellow
    Write-Host "  python dlc_label_simple.py" -ForegroundColor Yellow
    Write-Host "`n或新建干净环境（官方推荐）:" -ForegroundColor Yellow
    Write-Host "  conda create -n DLC-label python=3.11 -y" -ForegroundColor Yellow
    Write-Host "  conda activate DLC-label" -ForegroundColor Yellow
    Write-Host "  conda install -y -c conda-forge pyside6 ffmpeg" -ForegroundColor Yellow
    Write-Host '  pip install "deeplabcut[gui]>=3.0.0rc14"' -ForegroundColor Yellow
    exit 1
}

Write-Host "`n修复成功。可运行:" -ForegroundColor Green
Write-Host "  python dlc_workflow.py label" -ForegroundColor Green
