@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" ListenNote.py --live
pause
