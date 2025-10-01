# SDMN Framework Installation Guide

This guide provides comprehensive installation instructions for the Synthetic Default Mode Network (SDMN) Framework using **Poetry** as the unified dependency management system.

## 🚀 Quick Install

### For End Users
```bash
pip install synthetic-default-mode-network
```

### For Developers
```bash
git clone <repository-url>
cd synthetic-default-mode-network
./scripts/setup_platform.sh  # Auto-detects your platform
```

---

## 📦 Dependency Management (Poetry Only)

The project uses **Poetry exclusively** for dependency management. All dependencies are defined in `pyproject.toml`:

### Dependency Groups

| Group | Purpose | Install Command |
|-------|---------|-----------------|
| **main** | Core runtime dependencies | `poetry install --only=main` |
| **dev** | Development tools | `poetry install --with dev` |
| **test** | Testing framework | `poetry install --with test` |
| **data** | Data science tools | `poetry install --with data` |
| **web** | Web interfaces | `poetry install --with web` |
| **neurosim** | Neural simulators | `poetry install --with neurosim` |

### Optional Features

Heavy or platform-specific dependencies are marked as optional:

| Feature | Dependencies | Install |
|---------|-------------|---------|
| **All features** | All optional deps | `poetry install --extras all` |
| **Visualization** | plotly, seaborn | `poetry install --extras visualization` |
| **Performance** | numba, profilers | `poetry install --extras performance` |
| **Data science** | pandas, h5py, stats | `poetry install --extras data` |
| **GUI** | PyQt5 | `poetry install --extras gui` |

---

## 🖥️ Platform-Specific Installation

### 🐧 Linux
```bash
# Development
./scripts/setup_development.sh

# Production  
./scripts/setup_production.sh

# Verify
python scripts/verify_installation.py
```

### 🍎 macOS
```bash
# Development (Homebrew optimized)
./scripts/setup_development_macos.sh

# Production (launchd services)
./scripts/setup_production_macos.sh

# Verify
python scripts/verify_installation.py
```

### 🪟 Windows

#### PowerShell/CMD (Recommended)
```cmd
REM Development
scripts\setup_development.bat

REM Production
scripts\setup_production.bat  

REM Verify
python scripts\verify_installation.py
```

#### WSL/Git Bash
```bash
./scripts/setup_development.sh
python scripts/verify_installation.py
```

---

## 🐳 Container Installation

### Linux/macOS
```bash
./scripts/build.sh
./scripts/run.sh dev
```

### Windows
```cmd
scripts\build.bat
scripts\run.bat dev
```

---

## 🎯 Installation Options

### Minimal Installation (Core Only)
```bash
poetry install --only=main
# or: pip install synthetic-default-mode-network
```

### Full Development Environment
```bash
poetry install --with dev,test,data,web --extras all
```

### Custom Installation
```bash
# Core + specific features
poetry install --with dev --extras "visualization,performance"

# Production with data science
poetry install --only=main --extras data
```

---

## 🔧 Manual Installation

If automated scripts don't work:

### 1. Install Poetry
```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
# or: pip install poetry
```

### 2. Install SDMN
```bash
poetry install --with dev,test
poetry run pip install -e .
```

### 3. Verify Installation
```bash
python scripts/verify_installation.py
python -m sdmn info
python examples/01_basic_neuron_demo.py
```

---

## 🚨 Troubleshooting

### Common Issues

#### Windows: Visual C++ Build Tools Required
```cmd
REM Some packages need compilation on Windows
REM Install Visual Studio Build Tools or use pre-compiled packages
poetry install --only=main  # Skip problematic dev dependencies
```

#### macOS: Homebrew Dependencies
```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Use Homebrew-optimized script
./scripts/setup_development_macos.sh
```

#### Linux: Missing System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-dev build-essential

# RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

### Dependency Resolution Issues

#### Skip Problematic Optional Dependencies
```bash
poetry install --only=main
poetry install --with dev --extras "visualization"  # Skip heavy deps
```

#### Use Pip Fallback
```bash
pip install -e .                    # Core only
pip install -e .[visualization]     # With specific features
```

---

## 📋 Verification Checklist

After installation, verify these work:

```bash
✓ python -c "import sdmn; print(sdmn.__version__)"
✓ python -m sdmn info
✓ python -m sdmn simulate --help
✓ python examples/01_basic_neuron_demo.py
✓ python scripts/verify_installation.py
```

---

## 🔄 Migration from Legacy

If you have the old installation:

1. **Remove old files**: `requirements.txt`, `dev-requirements.txt` → moved to `legacy/`
2. **Use Poetry**: All dependency management now in `pyproject.toml`
3. **Update commands**: Replace `pip install -r requirements.txt` with `poetry install`

### Legacy Compatibility

Old pip-based installation still works:
```bash
pip install -e .[all]  # Install with all optional features
```

But Poetry is recommended for development.
