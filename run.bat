@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set /p PROJECT_PYTHON=<"%PROJECT_ROOT%.python-version"
py -%PROJECT_PYTHON% "%PROJECT_ROOT%scripts\run.py" %*
exit /b %errorlevel%
