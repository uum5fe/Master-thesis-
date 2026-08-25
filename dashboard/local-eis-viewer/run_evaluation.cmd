@echo off
REM Process raw FAMOS recordings on Windows.
REM
REM Exists because "python" on a fresh Windows PATH is often the Microsoft
REM Store placeholder, which answers "Python konnte nicht gefunden werden"
REM instead of running anything. This tries the interpreters that actually
REM work, in the order that is most likely to be the right one.
REM
REM   run_evaluation.cmd --list
REM   run_evaluation.cmd --all --stage-local
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_evaluation.py %*
    goto :eof
)

py -3 --version >nul 2>&1
if %errorlevel%==0 (
    py -3 run_evaluation.py %*
    goto :eof
)

REM The Python Install Manager (python.org's current installer) puts its shims
REM in %LOCALAPPDATA%\Python\bin. The Microsoft Store placeholder sits earlier
REM on PATH than that on a normal machine, so "python" answers "konnte nicht
REM gefunden werden" while a perfectly good interpreter is installed. Probe the
REM real location before giving up.
if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
    "%LOCALAPPDATA%\Python\bin\python.exe" run_evaluation.py %*
    goto :eof
)

python --version >nul 2>&1
if %errorlevel%==0 (
    python run_evaluation.py %*
    goto :eof
)

echo.
echo   No working Python was found.
echo.
echo   "python" on this machine is most likely the Microsoft Store
echo   placeholder. Any of these fixes it:
echo.
echo     1. Install Python from python.org and tick "Add python.exe to PATH".
echo        (already installed? try %LOCALAPPDATA%\Python\bin\python.exe)
echo     2. Or turn the placeholder off:
echo          Settings ^> Apps ^> Advanced app settings ^>
echo          App execution aliases ^> switch off python.exe and python3.exe
echo     3. Or make a virtual environment here, which this script will then use:
echo          py -3 -m venv .venv
echo          .venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
exit /b 1
