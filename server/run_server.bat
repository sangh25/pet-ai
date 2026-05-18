@echo off
call .venv\Scripts\activate.bat
set PET_AI_DEVICE=cpu
set PET_AI_IMG_SIZE=224
set PET_AI_FAST_MAX_DOGS=3
set PET_AI_FAST_YOLO_IMGSZ=416
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
