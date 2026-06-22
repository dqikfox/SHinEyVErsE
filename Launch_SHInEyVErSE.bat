@echo off
title SHInEyVErSE
cd /d "C:\Users\Shadow\SHInEyVErSE"
start "" /min cmd /c "timeout /t 4 >nul && start http://127.0.0.1:7860"
"C:\Users\Shadow\SHInEyVErSE\venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 7860
pause