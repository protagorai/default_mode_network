@echo off
REM Development Environment Setup Script for SDMN Framework (Windows)
REM This script sets up the development environment on Windows with native tools

setlocal EnableDelayedExpansion

echo [INFO] Setting up SDMN Framework development environment on Windows...

REM Check if Python is installed
echo [INFO] Checking Python installation...

REM Try python first, then py command
python --version >nul 2>&1
if !errorlevel! neq 0 (
    py --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Python is not installed or not in PATH
        echo [INFO] Please install Python 3.8+ from https://python.org/downloads/
        echo [INFO] Or use winget: winget install Python.Python.3
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)

REM Get Python version and validate
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% is installed

REM Basic version check (check if it's at least 3.8)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    if %%a LSS 3 (
        echo [ERROR] Python version too old. Need Python 3.8+
        exit /b 1
    )
    if %%a EQU 3 if %%b LSS 8 (
        echo [ERROR] Python version too old. Need Python 3.8+
        exit /b 1
    )
)

REM Check if Poetry is installed
echo [INFO] Checking Poetry installation...
poetry --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] Poetry is not installed. Installing Poetry...
    
    REM Try different installation methods
    curl -sSL https://install.python-poetry.org | %PYTHON_CMD% -
    if !errorlevel! neq 0 (
        echo [INFO] Curl failed, trying pip installation...
        %PYTHON_CMD% -m pip install poetry
    )
    
    REM Add Poetry to PATH for current session
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\AppData\Roaming\Python\Scripts;%PATH%"
    
    REM Test Poetry installation
    poetry --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Poetry installation failed
        echo [INFO] Please install Poetry manually: https://python-poetry.org/docs/#installation
        echo [INFO] Or use: pip install poetry
        exit /b 1
    )
)

echo [SUCCESS] Poetry is installed

REM Configure Poetry
echo [INFO] Configuring Poetry...
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

REM Install dependencies
echo [INFO] Installing project dependencies...
poetry install --with dev,test
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install dependencies
    echo [INFO] This might be due to missing Visual C++ Build Tools
    echo [INFO] Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo [INFO] Or install minimal dependencies: poetry install --only=main
    pause
)

REM Setup pre-commit hooks
echo [INFO] Setting up pre-commit hooks...
poetry run pre-commit install

REM Create necessary directories
echo [INFO] Creating development directories...
if not exist "logs" mkdir logs
if not exist "data\checkpoints" mkdir data\checkpoints
if not exist "data\results" mkdir data\results
if not exist "docs\build" mkdir docs\build

REM Install package in editable mode
echo [INFO] Installing package in editable mode...
poetry run pip install -e .

REM Run tests to verify installation
echo [INFO] Running tests to verify installation...
poetry run pytest tests\ -v --tb=short
if !errorlevel! neq 0 (
    echo [WARNING] Some tests failed, but the development environment is set up
) else (
    echo [SUCCESS] All tests passed!
)

REM Setup Jupyter kernel
echo [INFO] Setting up Jupyter kernel...
poetry run python -m ipykernel install --user --name sdmn-dev --display-name "SDMN Development"

echo.
echo [SUCCESS] Development environment setup complete!
echo.
echo Next steps:
echo   1. Activate environment: poetry shell
echo   2. Run tests: poetry run pytest
echo   3. Start development: poetry run python -m sdmn --help
echo   4. Run examples: poetry run python examples\quickstart_simulation.py
echo   5. Start Jupyter: poetry run jupyter lab
echo.
echo Development commands:
echo   poetry run pytest                    # Run tests
echo   poetry run black src\                # Format code
echo   poetry run mypy src\                 # Type checking
echo   poetry run pre-commit run --all     # Run all checks
echo   poetry run sphinx-build docs docs\build  # Build docs
echo.
echo [INFO] Development environment ready!

REM Create Windows activation script
echo [INFO] Creating Windows activation script...
echo @echo off > activate_sdmn.bat
echo REM Activate SDMN development environment >> activate_sdmn.bat
echo poetry shell >> activate_sdmn.bat
echo set SDMN_ENV=development >> activate_sdmn.bat
echo echo SDMN development environment activated >> activate_sdmn.bat
echo echo Use 'python -m sdmn --help' to get started >> activate_sdmn.bat

echo [SUCCESS] Created activate_sdmn.bat for easy activation

pause
