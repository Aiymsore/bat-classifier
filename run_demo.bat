@echo off
setlocal
cd /d "%~dp0"
python demo.py --self-test
if errorlevel 1 (
  echo.
  echo Demo failed. Check the error message above.
) else (
  echo.
  echo Demo completed successfully.
)
pause
