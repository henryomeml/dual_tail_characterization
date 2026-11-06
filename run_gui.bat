@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Project environment missing. Run: py -m venv .venv
 echo Then: .venv\Scripts\python.exe -m pip install -e .
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m dual_tail_characterization.gui
if errorlevel 1 (
 echo GUI startup failed. Review the error above.
 pause
 exit /b 1
)
