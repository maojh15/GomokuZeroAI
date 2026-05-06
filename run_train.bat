@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "CONDA_ENV=torch-cu130py12"
set "CONFIG_FILE=result_15x15/train_config.yaml"
set "LOG_FILE=result_15x15/log.txt"

call conda activate "%CONDA_ENV%"
if errorlevel 1 (
    echo Failed to activate conda environment %CONDA_ENV%.
    exit /b 1
)

if exist "%LOG_FILE%" del "%LOG_FILE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$configFile = $env:CONFIG_FILE; $logFile = $env:LOG_FILE; python -u train.py --config $configFile 2>&1 | ForEach-Object { $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $_; $line; Add-Content -LiteralPath $logFile -Value $line -Encoding UTF8 }; exit $LASTEXITCODE"
set "TRAIN_EXIT=%ERRORLEVEL%"

echo.
if "%TRAIN_EXIT%"=="0" (
    echo Training completed. Log saved to %LOG_FILE%
) else (
    echo Training failed with exit code %TRAIN_EXIT%. See %LOG_FILE%
)
exit /b %TRAIN_EXIT%
