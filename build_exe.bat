@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Python venv not found: .venv\Scripts\python.exe
  echo Create it first: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)

echo Installing PyInstaller if missing...
.venv\Scripts\python.exe -c "import PyInstaller" 2>nul || .venv\Scripts\python.exe -m pip install pyinstaller

echo Building BatClassifier (onedir)...
rem PyInstaller 会清空 dist\BatClassifier，先把用户工作区移出去再还原。
if exist "dist\BatClassifier\workspace" (
  if exist "workspace_backup_tmp" rmdir /s /q "workspace_backup_tmp"
  move /y "dist\BatClassifier\workspace" "workspace_backup_tmp" >nul
)
.venv\Scripts\python.exe -m PyInstaller bat_app.spec --noconfirm
if errorlevel 1 (
  echo.
  echo [ERROR] Build failed. See output above.
  if exist "workspace_backup_tmp" move /y "workspace_backup_tmp" "dist\BatClassifier\workspace" >nul
  exit /b 1
)
if exist "workspace_backup_tmp" (
  rmdir /s /q "dist\BatClassifier\workspace" 2>nul
  move /y "workspace_backup_tmp" "dist\BatClassifier\workspace" >nul
  echo Restored user workspace into dist\BatClassifier\workspace
)

echo.
echo Build OK: dist\BatClassifier\BatClassifier.exe
echo Note: run the exe in dist\BatClassifier\ (NOT the one inside build\ - that is an intermediate file).
start "" explorer "dist\BatClassifier"
