@echo off
cd /d C:\rl4co_pc
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\RunPCCurriculumHighGammaRetry.ps1"
exit /b %errorlevel%
