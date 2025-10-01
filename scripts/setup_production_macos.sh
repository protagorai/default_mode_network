#!/bin/bash
# Production Environment Setup Script for SDMN Framework (macOS)
# This script sets up the production environment on macOS using Homebrew and launchd

set -e  # Exit on any error

echo "🍎 Setting up SDMN Framework production environment on macOS..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
INSTALL_TYPE="poetry"
PYTHON_VERSION="3.11"

while [[ $# -gt 0 ]]; do
    case $1 in
        --pip)
            INSTALL_TYPE="pip"
            shift
            ;;
        --poetry)
            INSTALL_TYPE="poetry"
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--poetry|--pip] [--python VERSION] [--help]"
            echo ""
            echo "Options:"
            echo "  --poetry          Use Poetry for dependency management (default)"
            echo "  --pip             Use pip for dependency management"
            echo "  --python VERSION  Specify Python version (default: 3.11)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Homebrew is installed
print_step "Checking Homebrew installation..."
if ! command -v brew &> /dev/null; then
    print_error "Homebrew is not installed. Please install first:"
    echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

print_success "Homebrew is installed"

# Install Python via Homebrew
print_step "Installing Python $PYTHON_VERSION via Homebrew..."
brew install python@$PYTHON_VERSION
brew link python@$PYTHON_VERSION --force

# Update PATH for current session
if [[ $(uname -m) == 'arm64' ]]; then
    export PATH="/opt/homebrew/bin:/opt/homebrew/opt/python@$PYTHON_VERSION/bin:$PATH"
else
    export PATH="/usr/local/bin:/usr/local/opt/python@$PYTHON_VERSION/bin:$PATH"
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $python_version is installed"

# Create production directories (macOS convention)
print_step "Creating production directories..."
SDMN_ROOT="/usr/local/var/sdmn"
sudo mkdir -p "$SDMN_ROOT"/{logs,data/{checkpoints,results},config}
sudo chown -R $(whoami):staff "$SDMN_ROOT"
mkdir -p config

if [ "$INSTALL_TYPE" = "poetry" ]; then
    # Poetry installation
    print_step "Installing Poetry via Homebrew..."
    if ! command -v poetry &> /dev/null; then
        brew install poetry
    fi

    print_step "Installing SDMN package with Poetry..."
    poetry config virtualenvs.create true
    poetry install --only=main
    
    # Create macOS activation script
    cat > activate_sdmn.sh << 'EOF'
#!/bin/bash
# Activate SDMN production environment on macOS
eval "$(poetry env info --path)/bin/activate"
export SDMN_ENV="production"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
echo "🍎 SDMN production environment activated on macOS"
echo "Use 'sdmn --help' to get started"
EOF
    chmod +x activate_sdmn.sh
    print_success "Created activate_sdmn.sh script"

else
    # pip installation with Homebrew Python
    print_step "Setting up Python virtual environment..."
    python3 -m venv venv_sdmn
    source venv_sdmn/bin/activate

    print_step "Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    print_step "Installing SDMN package..."
    pip install -e .

    # Create macOS activation script
    cat > activate_sdmn.sh << 'EOF'
#!/bin/bash
# Activate SDMN production environment on macOS
source venv_sdmn/bin/activate
export SDMN_ENV="production"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
echo "🍎 SDMN production environment activated on macOS"
echo "Use 'sdmn --help' to get started"
EOF
    chmod +x activate_sdmn.sh
    print_success "Created activate_sdmn.sh script"
fi

# Create macOS-specific production configuration
print_step "Creating macOS production configuration..."
cat > config/production_macos.yaml << EOF
# SDMN Production Configuration for macOS
simulation:
  default_dt: 0.1
  default_max_time: 1000.0
  checkpoint_interval: 1000
  enable_logging: true
  log_level: "INFO"

logging:
  log_dir: "$SDMN_ROOT/logs"
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_rotation: true
  max_log_size: "100MB"
  backup_count: 5

data:
  checkpoint_dir: "$SDMN_ROOT/data/checkpoints"
  results_dir: "$SDMN_ROOT/data/results"
  cleanup_old_files: true
  max_checkpoint_age_days: 30

performance:
  max_memory_usage: "8GB"
  max_cpu_threads: 4
  enable_gpu: false
  # macOS-specific optimizations
  use_accelerate_framework: true
  metal_performance_shaders: false

network:
  default_topology: "small_world"
  default_neuron_type: "lif"
  default_population_size: 1000

monitoring:
  enable_performance_monitoring: true
  enable_memory_monitoring: true
  monitoring_interval: 60  # seconds
  # macOS-specific monitoring
  enable_macos_activity_monitor: true
EOF

# Create launchd service (macOS equivalent of systemd)
print_step "Creating launchd service plist..."
PLIST_PATH="$HOME/Library/LaunchAgents/org.sdmn.framework.plist"

# Set Python path based on installation type
if [ "$INSTALL_TYPE" = "poetry" ]; then
    PYTHON_PATH="$(poetry env info --path)/bin/python"
else
    PYTHON_PATH="$(pwd)/venv_sdmn/bin/python"
fi

cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>org.sdmn.framework</string>
    <key>WorkingDirectory</key>
    <string>$(pwd)</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_PATH</string>
        <string>-m</string>
        <string>sdmn</string>
        <string>simulate</string>
        <string>--config</string>
        <string>config/production_macos.yaml</string>
    </array>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$SDMN_ROOT/logs/sdmn.log</string>
    <key>StandardErrorPath</key>
    <string>$SDMN_ROOT/logs/sdmn.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>SDMN_ENV</key>
        <string>production</string>
        <key>PYTHONPATH</key>
        <string>$(pwd)/src</string>
    </dict>
</dict>
</plist>
EOF

print_success "Created launchd service plist"

# Test installation
print_step "Testing installation..."
if [ "$INSTALL_TYPE" = "poetry" ]; then
    if poetry run python -c "import sdmn; print('SDMN version:', sdmn.__version__)"; then
        print_success "Installation test passed!"
    else
        print_error "Installation test failed!"
        exit 1
    fi
else
    source venv_sdmn/bin/activate
    if python -c "import sdmn; print('SDMN version:', sdmn.__version__)"; then
        print_success "Installation test passed!"
    else
        print_error "Installation test failed!"
        exit 1
    fi
fi

# Create macOS startup script
print_step "Creating startup script..."
cat > start_sdmn.sh << 'EOF'
#!/bin/bash
# Start SDMN simulation with macOS production settings

# Load configuration
source activate_sdmn.sh

# Run simulation with macOS production config
python -m sdmn simulate \
    --config config/production_macos.yaml \
    --output "$SDMN_ROOT/data/results" \
    --duration 10000 \
    --neurons 1000 \
    --topology small_world \
    "$@"
EOF
chmod +x start_sdmn.sh

print_success "✅ macOS production environment setup complete!"
echo ""
echo -e "${BLUE}Production setup summary:${NC}"
echo "  Installation type: $INSTALL_TYPE"
echo "  Python version: $python_version"
echo "  Config file: config/production_macos.yaml"
echo "  Data directory: $SDMN_ROOT/data"
echo "  Logs directory: $SDMN_ROOT/logs"
echo ""
echo -e "${BLUE}Usage:${NC}"
echo "  ./activate_sdmn.sh           # Activate environment"
echo "  ./start_sdmn.sh              # Run simulation with defaults"
echo "  python -m sdmn simulate --help  # Show simulation options"
echo "  python -m sdmn info          # Show package information"
echo ""
echo -e "${BLUE}macOS Service Management:${NC}"
echo "  launchctl load $PLIST_PATH"
echo "  launchctl start org.sdmn.framework"
echo "  launchctl stop org.sdmn.framework"
echo "  launchctl unload $PLIST_PATH"
echo ""
echo -e "${BLUE}macOS-specific features:${NC}"
echo "• App bundle: ~/Applications/SDMN Framework.app"
echo "• Shell aliases: sdmn-dev, sdmn-jupyter, sdmn-test"
echo "• Homebrew integration for dependencies"
echo "• Native macOS service integration"
echo ""
