@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not defined FACE_DETECTOR set "FACE_DETECTOR=opencv"
echo Open http://127.0.0.1:8000 - model full_v3 - detector %FACE_DETECTOR%
echo Press Ctrl+C to stop.
venv\Scripts\python.exe -m uvicorn ai.api:app --host 127.0.0.1 --port 8000
