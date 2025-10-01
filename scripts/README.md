# SDMN Framework Scripts

This directory contains all setup, build, and deployment scripts for the SDMN Framework. Each script serves a specific purpose in different deployment scenarios.

## Script Overview

### 🏠 **Local Environment Setup**

#### **Cross-Platform**
- **`setup_platform.sh`** - 🎯 **Smart platform detection and setup assistant**
- **`setup.sh`** - Generic interactive setup assistant

#### **Linux/Unix**
- **`setup_development.sh`** - Linux development environment
- **`setup_production.sh`** - Linux production environment

#### **macOS (Homebrew Optimized)**
- **`setup_development_macos.sh`** - macOS development with Homebrew
- **`setup_production_macos.sh`** - macOS production with launchd

#### **Windows Native**
- **`setup_development.bat`** - Windows development (PowerShell/CMD)
- **`setup_production.bat`** - Windows production with Windows services

### 🐳 **Containerized Deployment**

#### **Linux/macOS**
- **`build.sh`** - Builds Docker/Podman container images
- **`run.sh`** - Runs containers in various modes

#### **Windows**
- **`build.bat`** - Windows container building
- **`run.bat`** - Windows container management

### ✅ **Verification & Testing**
- **`verify_installation.py`** - Cross-platform package verification
- **`verify_installation.ps1`** - Windows PowerShell verification
- **`verify_structure.py`** - Pre-install structure verification

---

## Deployment Strategies

### Strategy 1: Local Development
**Best for:** Active development, debugging, IDE integration

```bash
# Setup
./scripts/setup_development.sh

# Usage
poetry shell
poetry run pytest
python examples/quickstart_simulation.py
```

**What it does:**
- Installs Poetry and dependencies
- Sets up pre-commit hooks
- Installs package in editable mode (`pip install -e .`)
- Configures development tools (black, mypy, etc.)
- Creates development directories

---

### Strategy 2: Local Production
**Best for:** Production deployment on dedicated servers

```bash
# Setup
./scripts/setup_production.sh

# Usage  
./activate_sdmn.sh
python -m sdmn simulate --config production.yaml
systemctl start sdmn  # If using systemd
```

**What it does:**
- Sets up production environment (Poetry or pip)
- Creates systemd service files
- Configures production directories
- Sets up activation scripts
- Optimizes for production usage

---

### Strategy 3: Containerized Development
**Best for:** Consistent environments, team development

```bash
# Build
./scripts/build.sh

# Usage
./scripts/run.sh dev          # Development with source mounted
./scripts/run.sh jupyter      # Jupyter Lab
./scripts/run.sh shell        # Interactive shell
```

**What it does:**
- Builds isolated container environment
- Mounts source code for live editing
- Provides consistent Python/dependency versions
- Includes all development tools

---

### Strategy 4: Containerized Production  
**Best for:** Cloud deployment, Kubernetes, scaling

```bash
# Build
./scripts/build.sh v1.0.0

# Usage
./scripts/run.sh simulation   # Run simulations
./scripts/run.sh compose      # Use docker-compose
docker run sdmn-framework:v1.0.0 evaluation
```

**What it does:**
- Creates production-ready container images
- Optimized for deployment and scaling
- Includes only runtime dependencies
- Ready for orchestration platforms

---

## Platform-Specific Quick Start

### 🎯 **Smart Setup (Recommended)**
```bash
# Automatically detects your platform and shows best options
./scripts/setup_platform.sh
```

### 🐧 **Linux Users**
```bash
# Development
./scripts/setup_development.sh
python scripts/verify_installation.py

# Production  
./scripts/setup_production.sh
systemctl start sdmn
```

### 🍎 **macOS Users**
```bash
# Development (Homebrew optimized)
./scripts/setup_development_macos.sh
python scripts/verify_installation.py

# Production (launchd service)
./scripts/setup_production_macos.sh
launchctl start org.sdmn.framework
```

### 🪟 **Windows Users**

#### **PowerShell/CMD (Recommended)**
```cmd
REM Development
scripts\setup_development.bat
python scripts\verify_installation.py

REM Production
scripts\setup_production.bat
net start SDMN
```

#### **WSL/Git Bash**
```bash
# Development
./scripts/setup_development.sh
python scripts/verify_installation.py
```

### 🐳 **Container Users (All Platforms)**
```bash
# Linux/macOS
./scripts/build.sh
./scripts/run.sh dev

# Windows
scripts\build.bat
scripts\run.bat dev
```

### For Production Deployment
```bash
# Option A: Local Production
./scripts/setup_production.sh

# Option B: Containerized Production
./scripts/build.sh
./scripts/run.sh simulation
```

### For Container Development
```bash
# Build and run in development mode
./scripts/build.sh
./scripts/run.sh dev
```

---

## Complete Script Matrix

| Platform | Development | Production | Container Build | Container Run | Verification |
|----------|-------------|------------|-----------------|---------------|--------------|
| **🐧 Linux** | `setup_development.sh` | `setup_production.sh` | `build.sh` | `run.sh` | `verify_installation.py` |
| **🍎 macOS** | `setup_development_macos.sh` | `setup_production_macos.sh` | `build.sh` | `run.sh` | `verify_installation.py` |
| **🪟 Windows** | `setup_development.bat` | `setup_production.bat` | `build.bat` | `run.bat` | `verify_installation.ps1` |
| **🌐 Cross-Platform** | `setup_platform.sh` | `setup.sh` | `build.sh` | `run.sh` | `verify_structure.py` |

### **🎯 Recommended Entry Points:**
- **Smart Setup**: `./scripts/setup_platform.sh` (detects platform automatically)
- **Generic Setup**: `./scripts/setup.sh` (works everywhere)

---

## Script Details

### `setup_platform.sh` ⭐
- **Purpose**: Smart platform detection and setup assistant
- **Features**: Auto-detects OS, shows platform-specific recommendations
- **Platforms**: Linux, macOS, Windows (bash), Generic Unix
- **Result**: Guides user to optimal setup script

### `setup_development.sh`
- **Purpose**: Complete local development environment
- **Dependencies**: Python 3.8+, curl (for Poetry)
- **Creates**: Virtual environment, installs dependencies, pre-commit hooks
- **Result**: Ready for `poetry shell` and development

### `setup_production.sh` 
- **Purpose**: Production deployment on host system
- **Options**: `--poetry` or `--python VERSION`
- **Creates**: Service files, production config, activation scripts
- **Result**: Ready for production usage

### `build.sh`
- **Purpose**: Build Docker containers
- **Options**: Image tag, cleanup, container engine selection
- **Supports**: Docker and Podman
- **Result**: Built container image ready for deployment

### `run.sh`
- **Purpose**: Run containers in various modes
- **Modes**: shell, jupyter, simulation, test, gui, dev
- **Features**: Volume mounting, port forwarding, development mode
- **Result**: Running container with specified mode

### **Platform-Specific Scripts**

### `setup_development_macos.sh`
- **Purpose**: macOS-optimized development environment
- **Dependencies**: Homebrew, Python 3.8+ (installs via brew)
- **Features**: VS Code integration, app bundles, shell aliases, Jupyter extensions
- **macOS Services**: launchd integration, native app launcher
- **Result**: Full macOS development environment

### `setup_production_macos.sh`
- **Purpose**: macOS production deployment  
- **Features**: launchd services, Homebrew Python, native macOS integration
- **Services**: Creates proper macOS launch agents
- **Result**: Production-ready macOS deployment

### `setup_development.bat`
- **Purpose**: Native Windows development environment
- **Dependencies**: Python 3.8+, Windows tools
- **Features**: Windows services, native activation scripts
- **Tools**: Integrates with winget, chocolatey package managers
- **Result**: Native Windows development setup

### `setup_production.bat`
- **Purpose**: Windows production deployment
- **Features**: Windows services (NSSM), native batch scripts
- **Services**: Creates proper Windows services
- **Result**: Production-ready Windows deployment

### `build.bat` / `run.bat`
- **Purpose**: Windows container management
- **Features**: Native Windows batch commands, Docker Desktop integration
- **Compatibility**: Works with both Docker Desktop and Podman Desktop
- **Result**: Seamless container experience on Windows

### `verify_installation.py`
- **Purpose**: Cross-platform package verification
- **Tests**: Import tests, functionality tests, CLI availability
- **Platforms**: Linux, macOS, Windows (Python)
- **Usage**: Run after any installation method
- **Result**: Confirmation that package works correctly

### `verify_installation.ps1`
- **Purpose**: Windows PowerShell package verification
- **Tests**: Package imports, Windows-specific features, container engines
- **Features**: Native PowerShell output, Windows environment checks
- **Usage**: Run from PowerShell after Windows installation
- **Result**: Windows-specific verification and troubleshooting

---

## Environment Variables

### Common Variables
- `CONTAINER_ENGINE`: `docker` or `podman` (default: `podman`)
- `PYTHON_VERSION`: Python version for containers
- `CLEANUP`: Clean up old images after build (`true`/`false`)

### Setup Scripts
- `POETRY_VERSION`: Poetry version to install
- `IMAGE_TAG`: Container image tag

---

## Platform-Specific Troubleshooting

### 🐧 **Linux Issues**
1. **Poetry not found**: Add `$HOME/.local/bin` to PATH
2. **Permission errors**: Check directory permissions, use `sudo` for `/opt` directories
3. **systemd issues**: Use `sudo systemctl` commands, check service logs

### 🍎 **macOS Issues**  
1. **Homebrew not found**: Install with `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. **Python version conflicts**: Use `brew install python@3.11` and update PATH
3. **launchd issues**: Use `launchctl` commands, check `~/Library/LaunchAgents/`
4. **Apple Silicon**: Ensure Homebrew is in `/opt/homebrew/bin`

### 🪟 **Windows Issues**

#### **PowerShell/CMD**
1. **Python not found**: Install from python.org or use `winget install Python.Python.3`
2. **Poetry installation**: Use `pip install poetry` if curl fails
3. **Visual C++ Build Tools**: Install from Visual Studio installer for compiled packages
4. **Windows services**: Use `net start/stop` or Services.msc
5. **Path issues**: Restart PowerShell/CMD after Python installation

#### **WSL/Git Bash**
1. **WSL Python**: Use `sudo apt install python3-pip python3-venv`
2. **Git Bash**: Ensure proper Unix tools are available
3. **Path separators**: Use forward slashes in WSL, backslashes in CMD

### 🐳 **Container Issues (All Platforms)**
1. **Docker Desktop**: Ensure Docker Desktop is running (Windows/macOS)
2. **Docker daemon**: Start Docker service on Linux
3. **Podman**: Use `podman-desktop` or `podman machine start`
4. **Port conflicts**: Change ports with scripts (e.g., `8889` instead of `8888`)
5. **Volume mounting**: Check path format for your platform

### 📦 **Package Import Issues (All Platforms)**
1. **Import errors**: 
   - Run platform-specific verification script
   - Check virtual environment activation
   - Verify package installation: `pip list | grep sdmn`
2. **Missing dependencies**: 
   - Re-run appropriate setup script
   - Check dependency compilation (especially on Windows)
3. **Path issues**: 
   - Ensure package installed in editable mode (`pip install -e .`)
   - Check PYTHONPATH environment variable

---

## Contributing

When adding new scripts:

1. **Follow naming convention**: `verb_noun.sh` (e.g., `setup_development.sh`)
2. **Add help messages**: Include `--help` option
3. **Use consistent colors**: Follow existing color scheme
4. **Error handling**: Use `set -e` and proper error messages
5. **Document here**: Update this README with new script details

---

*All scripts are designed to be **complementary**, not redundant. Choose the deployment strategy that best fits your use case.*
