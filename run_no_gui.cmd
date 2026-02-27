@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m qr_to_pos.server %*
