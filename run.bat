@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" goto existing_env
set /p PROJECT_PYTHON=<"%PROJECT_ROOT%.python-version"
py -%PROJECT_PYTHON% "%PROJECT_ROOT%scripts\run.py" %*
exit /b %errorlevel%

:existing_env
"%PROJECT_ROOT%.venv\Scripts\python.exe" "%PROJECT_ROOT%scripts\run.py" %*
exit /b %errorlevel%
