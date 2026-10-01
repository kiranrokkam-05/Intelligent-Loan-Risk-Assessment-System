@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "PROJECT_DIR=%~dp0"
if not defined PORT set "PORT=8000"
if not defined ENVIRONMENT set "ENVIRONMENT=development"

echo ============================================================
echo Starting RiskLens ML Project
echo ============================================================
echo Project directory: %PROJECT_DIR%

set "PYTHON="
if exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    call "%PROJECT_DIR%.venv\Scripts\activate.bat"
    set "PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
) else if exist "%PROJECT_DIR%venv\Scripts\python.exe" (
    call "%PROJECT_DIR%venv\Scripts\activate.bat"
    set "PYTHON=%PROJECT_DIR%venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python was not found and no virtual environment exists.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

"%PYTHON%" --version
if errorlevel 1 (
    echo ERROR: The selected Python interpreter could not be started.
    pause
    exit /b 1
)

echo.
echo Checking required dependencies...
"%PYTHON%" -c "import flask, imblearn, joblib, numpy, pandas, sklearn, scipy, xgboost"
if errorlevel 1 (
    echo Required dependencies are missing. Installing from requirements.txt...
    "%PYTHON%" -m pip install -r "%PROJECT_DIR%requirements.txt"
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
)
"%PYTHON%" -m pip check
if errorlevel 1 (
    echo ERROR: Python dependency verification failed.
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%preprocessing\loan_dataset.csv" (
    echo ERROR: Required dataset is missing: preprocessing\loan_dataset.csv
    pause
    exit /b 1
)
if not exist "%PROJECT_DIR%static\riskLens.html" (
    echo ERROR: Existing frontend is missing: static\riskLens.html
    pause
    exit /b 1
)

if /i "%ENVIRONMENT%"=="development" (
    echo.
    echo Running the existing training pipeline for local development...
    "%PYTHON%" "%PROJECT_DIR%models\Model_Logic.py"
    if errorlevel 1 (
        echo ERROR: ML training pipeline failed.
        pause
        exit /b 1
    )
) else (
    echo.
    echo Production mode: training is disabled; using the approved artifact.
)
if not exist "%PROJECT_DIR%models\model.joblib" (
    echo ERROR: Training completed without creating models\model.joblib
    pause
    exit /b 1
)

echo.
echo Running automated tests...
"%PYTHON%" -m unittest -v test_system.py
if errorlevel 1 (
    echo ERROR: Automated tests failed.
    pause
    exit /b 1
)

echo.
echo Running held-out model evaluation...
"%PYTHON%" -m evaluation.evaluate_model
if errorlevel 1 (
    echo ERROR: Model evaluation failed.
    pause
    exit /b 1
)

echo.
echo Running model-risk and deployment governance checks...
"%PYTHON%" "%PROJECT_DIR%governance_checks.py"
if errorlevel 1 (
    echo ERROR: Governance checks failed; refusing to start the backend.
    pause
    exit /b 1
)

echo.
echo Starting the Flask backend...
start "RiskLens Backend" /b "%PYTHON%" "%PROJECT_DIR%app.py"
set "HEALTH_CHECK="
for /l %%N in (1,1,15) do (
    if not defined HEALTH_CHECK (
        timeout /t 1 /nobreak >nul
        "%PYTHON%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:%PORT%/api/health', timeout=2).read()" >nul 2>&1
        if not errorlevel 1 set "HEALTH_CHECK=ok"
    )
)
if not defined HEALTH_CHECK (
    echo ERROR: Flask backend did not become healthy at http://127.0.0.1:%PORT%.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo PROJECT COMPLETED
echo ============================================================
echo Backend: http://localhost:%PORT%/
echo Evaluation metrics were printed above in this terminal.
echo Keep this window open while using the frontend.
pause
exit /b 0
