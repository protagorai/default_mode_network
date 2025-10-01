#!/bin/bash

# Production Environment Setup Script for SDMN Framework
# This script sets up the production environment with minimal dependencies
# for running SDMN simulations in production.

set -e  # Exit on any error

echo "🎯 Setting up SDMN Framework production environment..."

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
INSTALL_TYPE="pip"
PYTHON_VERSION="3.9"

while [[ $# -gt 0 ]]; do
    case $1 in
        --poetry)
            INSTALL_TYPE="poetry"
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--poetry] [--python VERSION] [--help]"
            echo ""
            echo "Options:"
            echo "  --poetry          Use Poetry for dependency management (default: pip)"
            echo "  --python VERSION  Specify Python version (default: 3.9)"
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

# Check if Python is installed
print_step "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    print_error "Python $python_version is installed, but Python $required_version or higher is required."
    exit 1
fi

print_success "Python $python_version is installed"

# Create production directories
print_step "Creating production directories..."
mkdir -p /opt/sdmn/logs
mkdir -p /opt/sdmn/data/checkpoints
mkdir -p /opt/sdmn/data/results
mkdir -p /opt/sdmn/config

# Set permissions
if [ -w "/opt" ]; then
    chmod 755 /opt/sdmn
    chmod 755 /opt/sdmn/logs
    chmod 755 /opt/sdmn/data
    chmod 755 /opt/sdmn/config
fi

if [ "$INSTALL_TYPE" = "poetry" ]; then
    # Poetry installation
    print_step "Checking Poetry installation..."
    if ! command -v poetry &> /dev/null; then
        print_warning "Poetry is not installed. Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        
        if ! command -v poetry &> /dev/null; then
            print_error "Poetry installation failed. Please install Poetry manually."
            exit 1
        fi
    fi

    print_step "Installing SDMN package with Poetry..."
    poetry config virtualenvs.create true
    poetry install --only=main --no-dev
    
    # Create activation script
    cat > activate_sdmn.sh << 'EOF'
#!/bin/bash
# Activate SDMN virtual environment
export PATH="$(poetry env info --path)/bin:$PATH"
export SDMN_ENV="production"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
echo "SDMN production environment activated"
echo "Use 'sdmn --help' to get started"
EOF
    chmod +x activate_sdmn.sh
    print_success "Created activate_sdmn.sh script"

else
    # pip installation
    print_step "Setting up Python virtual environment..."
    python3 -m venv venv_sdmn
    source venv_sdmn/bin/activate

    print_step "Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    print_step "Installing SDMN package..."
    pip install -e .

    # Create activation script
    cat > activate_sdmn.sh << 'EOF'
#!/bin/bash
# Activate SDMN virtual environment
source venv_sdmn/bin/activate
export SDMN_ENV="production"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
echo "SDMN production environment activated"
echo "Use 'sdmn --help' to get started"
EOF
    chmod +x activate_sdmn.sh
    print_success "Created activate_sdmn.sh script"
fi

# Create production configuration
print_step "Creating production configuration..."
cat > config/production.yaml << 'EOF'
# SDMN Production Configuration
simulation:
  default_dt: 0.1
  default_max_time: 1000.0
  checkpoint_interval: 1000
  enable_logging: true
  log_level: "INFO"

logging:
  log_dir: "/opt/sdmn/logs"
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_rotation: true
  max_log_size: "100MB"
  backup_count: 5

data:
  checkpoint_dir: "/opt/sdmn/data/checkpoints"
  results_dir: "/opt/sdmn/data/results"
  cleanup_old_files: true
  max_checkpoint_age_days: 30

performance:
  max_memory_usage: "8GB"
  max_cpu_threads: 4
  enable_gpu: false

network:
  default_topology: "small_world"
  default_neuron_type: "lif"
  default_population_size: 1000

monitoring:
  enable_performance_monitoring: true
  enable_memory_monitoring: true
  monitoring_interval: 60  # seconds
EOF

# Create systemd service file (if systemd is available)
if command -v systemctl &> /dev/null; then
    print_step "Creating systemd service file..."
    
    # Create service file template
    cat > sdmn.service << EOF
[Unit]
Description=SDMN Framework Service
After=network.target

[Service]
Type=simple
User=sdmn
Group=sdmn
WorkingDirectory=$(pwd)
Environment=SDMN_ENV=production
Environment=PYTHONPATH=$(pwd)/src
ExecStart=$(pwd)/venv_sdmn/bin/python -m sdmn simulate --config config/production.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    print_success "Created sdmn.service file (copy to /etc/systemd/system/ to enable)"
fi

# Run basic test to verify installation
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

# Create startup script
print_step "Creating startup script..."
cat > start_sdmn.sh << 'EOF'
#!/bin/bash
# Start SDMN simulation with production settings

# Load configuration
source activate_sdmn.sh

# Run simulation with production config
python -m sdmn simulate \
    --config config/production.yaml \
    --output /opt/sdmn/data/results \
    --duration 10000 \
    --neurons 1000 \
    --topology small_world \
    "$@"
EOF
chmod +x start_sdmn.sh

print_success "✅ Production environment setup complete!"
echo ""
echo -e "${BLUE}Production setup summary:${NC}"
echo "  Installation type: $INSTALL_TYPE"
echo "  Python version: $python_version"
echo "  Config file: config/production.yaml"
echo "  Data directory: /opt/sdmn/data"
echo "  Logs directory: /opt/sdmn/logs"
echo ""
echo -e "${BLUE}Usage:${NC}"
echo "  ./activate_sdmn.sh           # Activate environment"
echo "  ./start_sdmn.sh              # Run simulation with defaults"
echo "  sdmn simulate --help         # Show simulation options"
echo "  sdmn info                    # Show package information"
echo ""
echo -e "${BLUE}Service management (if systemd available):${NC}"
echo "  sudo cp sdmn.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable sdmn"
echo "  sudo systemctl start sdmn"
echo ""
