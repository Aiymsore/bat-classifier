# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：蝙蝠声学分类桌面 App（onedir，窗口程序）。

产物：dist/BatClassifier/BatClassifier.exe 及同目录依赖库。
首次运行 exe 会在其旁自动创建 workspace/ 工作区。

构建：.venv/Scripts/python.exe -m PyInstaller bat_app.spec --noconfirm
"""

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# 科学计算栈缺少标准 hook 的部分（librosa/numba/llvmlite 尤其），
# 用 collect_all 连数据文件、动态库和子模块一起收进来。
for _package in (
    "librosa",
    "soundfile",
    "sklearn",
    "scipy",
    "numba",
    "llvmlite",
    "joblib",
    "matplotlib",
    "pandas",
):
    _datas, _binaries, _hidden = collect_all(_package)
    datas += _datas
    binaries += _binaries
    hiddenimports += _hidden

a = Analysis(
    ["bat_classifier_app.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BatClassifier",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BatClassifier",
)
