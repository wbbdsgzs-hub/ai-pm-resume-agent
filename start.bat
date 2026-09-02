@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 AI PM 求职助手...
python server.py
pause
