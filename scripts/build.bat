@echo off
REM Build script for SDMN Framework Docker container (Windows)
REM Compatible with both Docker Desktop and Podman Desktop

setlocal EnableDelayedExpansion

REM Configuration
set IMAGE_NAME=sdmn-framework
set IMAGE_TAG=%1
if "%IMAGE_TAG%"=="" set IMAGE_TAG=latest
set CONTAINER_ENGINE=%CONTAINER_ENGINE%
if "%CONTAINER_ENGINE%"=="" set CONTAINER_ENGINE=docker
set BUILD_CONTEXT=.

echo =========================================
echo SDMN Framework Container Build Script
echo =========================================

REM Check if container engine is available
echo [INFO] Checking container engine...
%CONTAINER_ENGINE% --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] %CONTAINER_ENGINE% is not installed or not in PATH
    echo [INFO] For Docker: Install Docker Desktop from https://docker.com/products/docker-desktop
    echo [INFO] For Podman: Install Podman Desktop from https://podman-desktop.io/
    exit /b 1
)

echo [SUCCESS] Using container engine: %CONTAINER_ENGINE%

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "data" mkdir data
if not exist "output" mkdir output
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
if not exist "examples" mkdir examples

REM Build the container image
echo [INFO] Building SDMN Framework image: %IMAGE_NAME%:%IMAGE_TAG%

REM Check if Dockerfile exists
if not exist "Dockerfile" (
    echo [ERROR] Dockerfile not found in current directory
    exit /b 1
)

REM Build arguments
set BUILD_ARGS=
if not "%PYTHON_VERSION%"=="" (
    set BUILD_ARGS=!BUILD_ARGS! --build-arg PYTHON_VERSION=%PYTHON_VERSION%
)

REM Build the image
echo [INFO] Building image...
%CONTAINER_ENGINE% build !BUILD_ARGS! -t %IMAGE_NAME%:%IMAGE_TAG% -f Dockerfile %BUILD_CONTEXT%

if !errorlevel! neq 0 (
    echo [ERROR] Failed to build image
    echo [INFO] Common issues:
    echo   - Docker Desktop not running
    echo   - Insufficient disk space
    echo   - Network connectivity issues
    exit /b 1
)

echo [SUCCESS] Successfully built image: %IMAGE_NAME%:%IMAGE_TAG%

REM Clean up old images (optional)
if "%CLEANUP%"=="true" (
    echo [INFO] Cleaning up old images...
    %CONTAINER_ENGINE% image prune -f >nul 2>&1
    for /f "delims=" %%i in ('%CONTAINER_ENGINE% images -f "dangling=true" -q 2^>nul') do (
        if not "%%i"=="" %CONTAINER_ENGINE% rmi %%i >nul 2>&1
    )
    echo [SUCCESS] Cleanup completed
)

REM Show image information
echo [INFO] Image information:
%CONTAINER_ENGINE% images | findstr "%IMAGE_NAME%"

echo [INFO] Image size:
for /f %%i in ('%CONTAINER_ENGINE% inspect %IMAGE_NAME%:%IMAGE_TAG% --format "{{.Size}}"') do (
    echo %%i bytes
)

echo.
echo [SUCCESS] Build completed successfully!
echo.
echo Next steps:
echo   1. Run interactive shell:    %CONTAINER_ENGINE% run -it %IMAGE_NAME%:%IMAGE_TAG%
echo   2. Start Jupyter Lab:        %CONTAINER_ENGINE% run -p 8888:8888 %IMAGE_NAME%:%IMAGE_TAG%
echo   3. Run example simulation:   %CONTAINER_ENGINE% run %IMAGE_NAME%:%IMAGE_TAG% evaluation
echo   4. Use docker-compose:       docker-compose -f docker\docker-compose.yml up
echo.
echo Use scripts\run.bat for easier container management

pause
