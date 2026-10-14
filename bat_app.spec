# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：蝙蝠声学分类桌面 App（onedir，窗口程序）。

产物：dist/BatClassifier/BatClassifier.exe 及同目录依赖库。
首次运行 exe 会在其旁自动创建 workspace/ 工作区。

构建：.venv/Scripts/python.exe -m PyInstaller bat_app.spec --noconfirm
"""

from PyInstaller.utils.hooks import collect_all

def is_runtime_file(path):
    parts = str(path).replace('\\', '/').split('/')
    return not any(part in {'tests', 'test', '__pycache__'} for part in parts) and not parts[-1].startswith('test_')


def is_runtime_module(name):
    return not any(part in {'tests', 'test'} or part.startswith('test_') for part in name.split('.'))


datas = []
binaries = []
hiddenimports = []

# 科学计算栈缺少标准 hook 的部分（librosa/numba/llvmlite 尤其），
# 用 collect_all 连数据文件、动态库和子模块一起收进来。
for _package in (
    "librosa",
    "soundfile",
    "scipy",
    "numba",
    "llvmlite",
    "matplotlib",
    "pandas",
):
    _datas, _binaries, _hidden = collect_all(
        _package, filter_submodules=is_runtime_module,
        exclude_datas=["**/tests/**", "**/test/**", "**/test_*.py", "**/__pycache__/**"],
    )
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

# Dependency hooks can add package test trees again; filter final collection too.
a.datas = [entry for entry in a.datas if is_runtime_file(entry[0])]
# Build-host tooling can add unrelated native DLLs to PATH; these must not override Windows/Qt runtimes.
a.binaries = [entry for entry in a.binaries
              if is_runtime_file(entry[0]) and "codex-primary-runtime" not in entry[1].lower()]
a.pure = [entry for entry in a.pure if is_runtime_module(entry[0])]

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
