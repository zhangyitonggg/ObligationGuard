@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
python -m obligationguard check --paper configs\paper.toml --runtime configs\runtime.toml
exit /b %ERRORLEVEL%
