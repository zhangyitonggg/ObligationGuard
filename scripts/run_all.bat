@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
if "%OBLIGATIONGUARD_RUN_ROOT%"=="" (
  echo Set OBLIGATIONGUARD_RUN_ROOT to the experiment output directory.
  exit /b 2
)
set "OG_RUNTIME=configs\runtime.toml"
if not "%~1"=="" set "OG_RUNTIME=%~1"
python -m obligationguard positive --paper configs\paper.toml --runtime "%OG_RUNTIME%"
exit /b %ERRORLEVEL%
