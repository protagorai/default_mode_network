@echo off
REM Production Environment Setup Script for SDMN Framework (Windows)
REM This script sets up the production environment on Windows

setlocal EnableDelayedExpansion

echo [INFO] Setting up SDMN Framework production environment on Windows...

REM Parse command line arguments
set INSTALL_TYPE=pip
set PYTHON_VERSION=3.9

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--poetry" (
    set INSTALL_TYPE=poetry
    shift
    goto :parse_args
)
if "%~1"=="--python" (
    set PYTHON_VERSION=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help

echo [ERROR] Unknown option: %~1
echo Use --help for usage information
exit /b 1

:show_help
echo Usage: %0 [--poetry] [--python VERSION] [--help]
echo.
echo Options:
echo   --poetry          Use Poetry for dependency management (default: pip)
echo   --python VERSION  Specify Python version (default: 3.9)
echo   --help, -h        Show this help message
exit /b 0

:args_done

REM Check if Python is installed
echo [INFO] Checking Python installation...

REM Try python first, then py command
python --version >nul 2>&1
if !errorlevel! neq 0 (
    py --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Python is not installed
        echo [INFO] Install from: https://python.org/downloads/
        echo [INFO] Or use: winget install Python.Python.3
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)

for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set CURRENT_PYTHON_VERSION=%%i
echo [SUCCESS] Python %CURRENT_PYTHON_VERSION% is installed

REM Create production directories (Windows equivalent of /opt/sdmn)
echo [INFO] Creating production directories...
set SDMN_ROOT=%PROGRAMDATA%\SDMN
if not exist "%SDMN_ROOT%" mkdir "%SDMN_ROOT%"
if not exist "%SDMN_ROOT%\logs" mkdir "%SDMN_ROOT%\logs"
if not exist "%SDMN_ROOT%\data" mkdir "%SDMN_ROOT%\data"
if not exist "%SDMN_ROOT%\data\checkpoints" mkdir "%SDMN_ROOT%\data\checkpoints"
if not exist "%SDMN_ROOT%\data\results" mkdir "%SDMN_ROOT%\data\results"
if not exist "%SDMN_ROOT%\config" mkdir "%SDMN_ROOT%\config"
if not exist "config" mkdir config

if "%INSTALL_TYPE%"=="poetry" (
    REM Poetry installation
    echo [INFO] Checking Poetry installation...
    poetry --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [WARNING] Poetry is not installed. Installing Poetry...
        curl -sSL https://install.python-poetry.org | %PYTHON_CMD% -
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
        
        poetry --version >nul 2>&1
        if !errorlevel! neq 0 (
            echo [ERROR] Poetry installation failed
            exit /b 1
        )
    )

    echo [INFO] Installing SDMN package with Poetry...
    poetry config virtualenvs.create true
    poetry install --only=main
    
    REM Create Windows activation script
    echo [INFO] Creating activation script...
    echo @echo off > activate_sdmn.bat
    echo REM Activate SDMN production environment >> activate_sdmn.bat
    echo for /f "tokens=*" %%%%i in ('poetry env info --path') do set VENV_PATH=%%%%i >> activate_sdmn.bat
    echo set "PATH=%%VENV_PATH%%\Scripts;%%PATH%%" >> activate_sdmn.bat
    echo set SDMN_ENV=production >> activate_sdmn.bat
    echo set PYTHONPATH=%CD%\src;%%PYTHONPATH%% >> activate_sdmn.bat
    echo echo SDMN production environment activated >> activate_sdmn.bat
    echo echo Use 'sdmn --help' to get started >> activate_sdmn.bat
    
    echo [SUCCESS] Created activate_sdmn.bat script

) else (
    REM pip installation
    echo [INFO] Setting up Python virtual environment...
    %PYTHON_CMD% -m venv venv_sdmn
    call venv_sdmn\Scripts\activate.bat

    echo [INFO] Upgrading pip...
    %PYTHON_CMD% -m pip install --upgrade pip setuptools wheel

    echo [INFO] Installing SDMN package...
    pip install -e .

    REM Create Windows activation script
    echo [INFO] Creating activation script...
    echo @echo off > activate_sdmn.bat
    echo REM Activate SDMN production environment >> activate_sdmn.bat
    echo call venv_sdmn\Scripts\activate.bat >> activate_sdmn.bat
    echo set SDMN_ENV=production >> activate_sdmn.bat
    echo set PYTHONPATH=%CD%\src;%%PYTHONPATH%% >> activate_sdmn.bat
    echo echo SDMN production environment activated >> activate_sdmn.bat
    echo echo Use 'sdmn --help' to get started >> activate_sdmn.bat
    
    echo [SUCCESS] Created activate_sdmn.bat script
)

REM Create production configuration
echo [INFO] Creating production configuration...
(
echo # SDMN Production Configuration for Windows
echo simulation:
echo   default_dt: 0.1
echo   default_max_time: 1000.0
echo   checkpoint_interval: 1000
echo   enable_logging: true
echo   log_level: "INFO"
echo.
echo logging:
echo   log_dir: "%SDMN_ROOT%\logs"
echo   log_format: "%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s"
echo   log_rotation: true
echo   max_log_size: "100MB"
echo   backup_count: 5
echo.
echo data:
echo   checkpoint_dir: "%SDMN_ROOT%\data\checkpoints"
echo   results_dir: "%SDMN_ROOT%\data\results"
echo   cleanup_old_files: true
echo   max_checkpoint_age_days: 30
echo.
echo performance:
echo   max_memory_usage: "8GB"
echo   max_cpu_threads: 4
echo   enable_gpu: false
echo.
echo network:
echo   default_topology: "small_world"
echo   default_neuron_type: "lif"
echo   default_population_size: 1000
echo.
echo monitoring:
echo   enable_performance_monitoring: true
echo   enable_memory_monitoring: true
echo   monitoring_interval: 60  # seconds
) > config\production.yaml

REM Create Windows service installation script (requires admin)
echo [INFO] Creating Windows service script...
(
echo @echo off
echo REM Install SDMN as Windows Service (Run as Administrator)
echo echo Installing SDMN as Windows Service...
echo.
echo REM Using NSSM (Non-Sucking Service Manager) if available
echo where nssm ^>nul 2^>^&1
echo if %%errorlevel%% neq 0 (
echo     echo [WARNING] NSSM not found. Install with: winget install nssm
echo     echo [INFO] Manual service creation required
echo     pause
echo     exit /b 1
echo ^)
echo.
echo nssm install SDMN "%CD%\venv_sdmn\Scripts\python.exe"
echo nssm set SDMN Parameters "-m sdmn simulate --config config\production.yaml"
echo nssm set SDMN DisplayName "SDMN Framework Service"
echo nssm set SDMN Description "Synthetic Default Mode Network Framework"
echo nssm set SDMN Start SERVICE_AUTO_START
echo nssm set SDMN AppDirectory "%CD%"
echo nssm set SDMN AppEnvironmentExtra SDMN_ENV=production PYTHONPATH=%CD%\src
echo.
echo echo [SUCCESS] SDMN service installed
echo echo Use 'net start SDMN' to start the service
echo echo Use 'net stop SDMN' to stop the service
echo pause
) > install_windows_service.bat

REM Test installation
echo [INFO] Testing installation...
if "%INSTALL_TYPE%"=="poetry" (
    poetry run %PYTHON_CMD% -c "import sdmn; print('SDMN version:', sdmn.__version__)"
) else (
    call venv_sdmn\Scripts\activate.bat
    %PYTHON_CMD% -c "import sdmn; print('SDMN version:', sdmn.__version__)"
)

if !errorlevel! neq 0 (
    echo [ERROR] Installation test failed!
    echo [INFO] Try installing with simpler dependencies:
    echo [INFO] poetry install --only=main
    pause
    exit /b 1
)

echo [SUCCESS] Installation test passed!

REM Create startup script
echo [INFO] Creating startup script...
(
echo @echo off
echo REM Start SDMN simulation with production settings
echo.
echo REM Load configuration
echo call activate_sdmn.bat
echo.
echo REM Run simulation with production config
echo %PYTHON_CMD% -m sdmn simulate ^
echo     --config config\production.yaml ^
echo     --output "%SDMN_ROOT%\data\results" ^
echo     --duration 10000 ^
echo     --neurons 1000 ^
echo     --topology small_world ^
echo     %%*
) > start_sdmn.bat

echo.
echo [SUCCESS] Production environment setup complete!
echo.
echo Production setup summary:
echo   Installation type: %INSTALL_TYPE%
echo   Python version: %CURRENT_PYTHON_VERSION%
echo   Config file: config\production.yaml
echo   Data directory: %SDMN_ROOT%\data
echo   Logs directory: %SDMN_ROOT%\logs
echo.
echo Usage:
echo   activate_sdmn.bat           # Activate environment
echo   start_sdmn.bat              # Run simulation with defaults
echo   python -m sdmn simulate --help  # Show simulation options
echo   python -m sdmn info         # Show package information
echo.
echo Windows Service (Run install_windows_service.bat as Administrator):
echo   install_windows_service.bat # Install as Windows service
echo   net start SDMN              # Start service
echo   net stop SDMN               # Stop service
echo.

echo [INFO] Press any key to continue...
pause >nul
