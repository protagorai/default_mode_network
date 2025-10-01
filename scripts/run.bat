@echo off
REM Run script for SDMN Framework Docker container (Windows)
REM Compatible with both Docker Desktop and Podman Desktop

setlocal EnableDelayedExpansion

REM Configuration
set IMAGE_NAME=sdmn-framework
set IMAGE_TAG=%IMAGE_TAG%
if "%IMAGE_TAG%"=="" set IMAGE_TAG=latest
set CONTAINER_ENGINE=%CONTAINER_ENGINE%
if "%CONTAINER_ENGINE%"=="" set CONTAINER_ENGINE=docker
set CONTAINER_NAME=sdmn-instance

REM Parse command line arguments
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=help

REM Helper function to check container engine
:check_container_engine
%CONTAINER_ENGINE% --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] %CONTAINER_ENGINE% is not installed or not in PATH
    echo [INFO] Install Docker Desktop or Podman Desktop
    exit /b 1
)
echo [INFO] Using container engine: %CONTAINER_ENGINE%
goto :eof

REM Helper function to check image
:check_image
%CONTAINER_ENGINE% images | findstr "%IMAGE_NAME%" | findstr "%IMAGE_TAG%" >nul
if !errorlevel! neq 0 (
    echo [WARNING] Image %IMAGE_NAME%:%IMAGE_TAG% not found locally
    echo [INFO] Building image first...
    call scripts\build.bat %IMAGE_TAG%
    if !errorlevel! neq 0 exit /b 1
)
goto :eof

REM Common run options
:get_common_options
set COMMON_OPTIONS=--rm --name %CONTAINER_NAME%-%RANDOM% -v "%CD%\data:/app/data" -v "%CD%\output:/app/output" -v "%CD%\checkpoints:/app/checkpoints" -v "%CD%\logs:/app/logs"
goto :eof

REM Run interactive shell
:run_shell
echo [INFO] Starting interactive shell...
call :get_common_options
%CONTAINER_ENGINE% run -it %COMMON_OPTIONS% %IMAGE_NAME%:%IMAGE_TAG%
goto :eof

REM Run Jupyter Lab
:run_jupyter
set PORT=%2
if "%PORT%"=="" set PORT=8888
echo [INFO] Starting Jupyter Lab on port %PORT%...
echo [INFO] Access at: http://localhost:%PORT%
call :get_common_options
%CONTAINER_ENGINE% run -it %COMMON_OPTIONS% -p %PORT%:8888 %IMAGE_NAME%:%IMAGE_TAG% jupyter
goto :eof

REM Run example simulation
:run_simulation
echo [INFO] Running example simulation...
call :get_common_options
%CONTAINER_ENGINE% run %COMMON_OPTIONS% %IMAGE_NAME%:%IMAGE_TAG% evaluation
goto :eof

REM Run test suite
:run_tests
echo [INFO] Running test suite...
call :get_common_options
%CONTAINER_ENGINE% run %COMMON_OPTIONS% %IMAGE_NAME%:%IMAGE_TAG% tests
goto :eof

REM Run development mode
:run_dev
echo [INFO] Starting development mode with source code mounted...
call :get_common_options
%CONTAINER_ENGINE% run -it %COMMON_OPTIONS% -v "%CD%\src:/app/src" -v "%CD%\examples:/app/examples" -v "%CD%\tests:/app/tests" %IMAGE_NAME%:%IMAGE_TAG%
goto :eof

REM Run with custom command
:run_custom
echo [INFO] Running custom command...
call :get_common_options
shift
set CUSTOM_CMD=%*
%CONTAINER_ENGINE% run -it %COMMON_OPTIONS% %IMAGE_NAME%:%IMAGE_TAG% %CUSTOM_CMD%
goto :eof

REM Stop running containers
:stop_containers
echo [INFO] Stopping SDMN containers...
for /f %%i in ('%CONTAINER_ENGINE% ps --format "table {{.Names}}" ^| findstr "sdmn-"') do (
    echo [INFO] Stopping container: %%i
    %CONTAINER_ENGINE% stop %%i
)
echo [SUCCESS] All SDMN containers stopped
goto :eof

REM Clean up containers and images
:cleanup
echo [INFO] Cleaning up SDMN containers and images...

REM Stop containers first
call :stop_containers

REM Remove containers
for /f %%i in ('%CONTAINER_ENGINE% ps -a --format "table {{.Names}}" ^| findstr "sdmn-"') do (
    echo [INFO] Removing container: %%i
    %CONTAINER_ENGINE% rm %%i
)

REM Remove images if requested
if "%2"=="--images" (
    %CONTAINER_ENGINE% rmi %IMAGE_NAME%:%IMAGE_TAG% 2>nul
    echo [SUCCESS] Removed image: %IMAGE_NAME%:%IMAGE_TAG%
)

REM Clean up dangling images and volumes
%CONTAINER_ENGINE% image prune -f >nul 2>&1
%CONTAINER_ENGINE% volume prune -f >nul 2>&1

echo [SUCCESS] Cleanup completed
goto :eof

REM Use docker-compose
:run_compose
set SERVICE=%2
if "%SERVICE%"=="" set SERVICE=sdmn-framework

where docker-compose >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] docker-compose not found
    echo [INFO] Install Docker Desktop or add docker-compose to PATH
    exit /b 1
)

echo [INFO] Using docker-compose to start service: %SERVICE%
cd docker
docker-compose up %SERVICE%
cd ..
goto :eof

REM Show help
:show_help
echo SDMN Framework Container Runner (Windows)
echo Usage: %0 ^<command^> [options]
echo.
echo Commands:
echo   shell              Start interactive shell
echo   jupyter [port]     Start Jupyter Lab (default port: 8888)
echo   simulation         Run example simulation
echo   test               Run test suite
echo   dev                Development mode with source mounted
echo   compose [service]  Use docker-compose (default: sdmn-framework)
echo   stop               Stop running SDMN containers
echo   cleanup [--images] Clean up containers and optionally images
echo   custom ^<cmd^>       Run custom command in container
echo.
echo Environment Variables:
echo   CONTAINER_ENGINE   Container engine to use (default: docker)
echo   IMAGE_TAG         Image tag to use (default: latest)
echo.
echo Examples:
echo   %0 shell                    # Interactive shell
echo   %0 jupyter 8889             # Jupyter Lab on port 8889
echo   %0 custom python --version  # Run python --version
echo   set CONTAINER_ENGINE=podman ^& %0 shell  # Use Podman
goto :eof

REM Main execution logic
if not exist "Dockerfile" (
    echo [ERROR] Please run this script from the project root directory
    exit /b 1
)

REM Create necessary directories
if not exist "data" mkdir data
if not exist "output" mkdir output
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs

REM Route to appropriate function
if "%COMMAND%"=="shell" (
    call :check_container_engine
    call :check_image
    call :run_shell
) else if "%COMMAND%"=="jupyter" (
    call :check_container_engine
    call :check_image
    call :run_jupyter
) else if "%COMMAND%"=="simulation" (
    call :check_container_engine
    call :check_image
    call :run_simulation
) else if "%COMMAND%"=="test" (
    call :check_container_engine
    call :check_image
    call :run_tests
) else if "%COMMAND%"=="dev" (
    call :check_container_engine
    call :check_image
    call :run_dev
) else if "%COMMAND%"=="compose" (
    call :run_compose
) else if "%COMMAND%"=="stop" (
    call :check_container_engine
    call :stop_containers
) else if "%COMMAND%"=="cleanup" (
    call :check_container_engine
    call :cleanup
) else if "%COMMAND%"=="custom" (
    call :check_container_engine
    call :check_image
    call :run_custom
) else if "%COMMAND%"=="help" (
    call :show_help
) else if "%COMMAND%"=="--help" (
    call :show_help
) else if "%COMMAND%"=="-h" (
    call :show_help
) else (
    echo [ERROR] Unknown command: %COMMAND%
    echo.
    call :show_help
    exit /b 1
)
