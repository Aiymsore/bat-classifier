@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem  未打包源码版启动器：双击即可打开 GUI，无需 build_exe.bat。
rem  改了 src\ 下的代码，重新双击本文件即生效（无需任何打包）。
rem  如需查看报错信息，在本目录命令行执行：
rem      .venv\Scripts\python.exe bat_classifier_app.py
rem ============================================================

if not exist ".venv\Scripts\pythonw.exe" (
  echo [ERROR] Python venv not found: .venv\Scripts\pythonw.exe
  echo Create it first: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

rem pythonw 启动：不弹黑色控制台窗口
start "" ".venv\Scripts\pythonw.exe" "bat_classifier_app.py"
