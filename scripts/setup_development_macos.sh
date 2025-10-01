#!/bin/bash
# Development Environment Setup Script for SDMN Framework (macOS with Homebrew)
# This script sets up the development environment on macOS using Homebrew

set -e  # Exit on any error

echo "🍎 Setting up SDMN Framework development environment on macOS..."

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

# Check if Homebrew is installed
print_step "Checking Homebrew installation..."
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew is not installed. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew installation failed. Please install manually: https://brew.sh"
        exit 1
    fi
fi

print_success "Homebrew is installed"

# Update Homebrew
print_step "Updating Homebrew..."
brew update

# Install Python via Homebrew if not present or old version
print_step "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_warning "Python 3 not found. Installing via Homebrew..."
    brew install python@3.11
else
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
        print_warning "Python $python_version is too old. Installing Python 3.11 via Homebrew..."
        brew install python@3.11
        # Make sure we're using the brew Python
        export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    fi
fi

print_success "Python $python_version is installed"

# Install additional development tools via Homebrew
print_step "Installing development tools via Homebrew..."
brew_packages=(
    "git"
    "curl"
    "wget"
    "jq"
    "tree"
    "htop"
    "graphviz"  # For network visualization
)

for package in "${brew_packages[@]}"; do
    if ! brew list "$package" &> /dev/null; then
        print_step "Installing $package..."
        brew install "$package"
    fi
done

print_success "Development tools installed"

# Install Poetry
print_step "Checking Poetry installation..."
if ! command -v poetry &> /dev/null; then
    print_warning "Poetry is not installed. Installing via Homebrew..."
    brew install poetry
    
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry installation failed"
        exit 1
    fi
fi

print_success "Poetry is installed"

# Configure Poetry for macOS
print_step "Configuring Poetry for macOS..."
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

# macOS-specific configurations
poetry config virtualenvs.prefer-active-python true
poetry config installer.parallel true

# Install dependencies
print_step "Installing project dependencies..."
poetry install --with dev,test

# macOS-specific: Install additional development dependencies
print_step "Installing macOS-specific development tools..."
if ! brew list "pre-commit" &> /dev/null; then
    brew install pre-commit
fi

# Setup pre-commit hooks
print_step "Setting up pre-commit hooks..."
poetry run pre-commit install

# macOS-specific: Install additional hooks
poetry run pre-commit install --hook-type commit-msg
poetry run pre-commit install --hook-type pre-push

# Run code formatting
print_step "Formatting code with black and isort..."
poetry run black src/ tests/ examples/ || print_warning "Black formatting encountered issues"
poetry run isort src/ tests/ examples/ || print_warning "isort formatting encountered issues"

# Run type checking
print_step "Running type checks with mypy..."
poetry run mypy src/ --ignore-missing-imports || print_warning "MyPy found some type issues"

# Create necessary directories with proper macOS permissions
print_step "Creating development directories..."
mkdir -p logs data/{checkpoints,results} docs/build
chmod 755 logs data data/checkpoints data/results docs/build

# Install package in editable mode
print_step "Installing package in editable mode..."
poetry run pip install -e .

# macOS-specific: Setup Jupyter with extensions
print_step "Setting up Jupyter Lab with macOS optimizations..."

# Install useful Jupyter extensions first
poetry run pip install jupyterlab-git jupyterlab-variableinspector

# Generate config and get directory
poetry run jupyter lab --generate-config 2>/dev/null || true
JUPYTER_CONFIG_DIR=$(poetry run jupyter --config-dir 2>/dev/null || echo "$HOME/.jupyter")

# Ensure config directory exists
mkdir -p "$JUPYTER_CONFIG_DIR"

# Create macOS-specific Jupyter config
cat > "$JUPYTER_CONFIG_DIR/jupyter_lab_config.py" << 'EOF'
# Jupyter Lab configuration for SDMN development on macOS
c.ServerApp.notebook_dir = '.'
c.ServerApp.open_browser = True
c.ServerApp.port = 8888
c.ServerApp.ip = 'localhost'
c.LabApp.default_url = '/lab'

# Enable extensions
c.LabApp.extensions_in_dev_mode = True
c.LabApp.collaborative = False
EOF

# Setup Jupyter kernel
print_step "Setting up Jupyter kernel..."
poetry run python -m ipykernel install --user --name sdmn-dev --display-name "SDMN Development"

# Run tests to verify installation
print_step "Running tests to verify installation..."
if poetry run pytest tests/ -v --tb=short; then
    print_success "All tests passed!"
else
    print_warning "Some tests failed, but the development environment is set up"
fi

# macOS-specific: Create macOS application bundle (optional)
print_step "Creating macOS app launcher..."
APP_DIR="$HOME/Applications/SDMN Framework.app"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

cat > "$APP_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>SDMN Framework</string>
    <key>CFBundleIdentifier</key>
    <string>org.sdmn.framework</string>
    <key>CFBundleName</key>
    <string>SDMN Framework</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
</dict>
</plist>
EOF

cat > "$APP_DIR/Contents/MacOS/SDMN Framework" << EOF
#!/bin/bash
cd "$(dirname "$0")/../../.."
source "$HOME/.zshrc" 2>/dev/null || source "$HOME/.bash_profile" 2>/dev/null || true
cd "$(pwd)"
poetry shell
EOF

chmod +x "$APP_DIR/Contents/MacOS/SDMN Framework"

# macOS-specific: Create development tools integration
print_step "Setting up macOS development integration..."

# Create VS Code workspace file
if command -v code &> /dev/null; then
    cat > "sdmn-framework.code-workspace" << 'EOF'
{
    "folders": [
        {
            "path": "."
        }
    ],
    "settings": {
        "python.defaultInterpreterPath": "./.venv/bin/python",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": false,
        "python.linting.flake8Enabled": true,
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length=88"],
        "python.sortImports.args": ["--profile", "black"],
        "files.exclude": {
            "**/__pycache__": true,
            "**/.pytest_cache": true,
            "**/.mypy_cache": true
        }
    },
    "extensions": {
        "recommendations": [
            "ms-python.python",
            "ms-python.black-formatter",
            "ms-python.isort",
            "ms-python.mypy-type-checker",
            "ms-toolsai.jupyter"
        ]
    }
}
EOF
    print_success "Created VS Code workspace configuration"
fi

# macOS-specific: Setup shell integration
print_step "Setting up shell integration..."

# Add to shell profile
SHELL_RC="$HOME/.zshrc"
if [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bash_profile"
fi

# Add SDMN alias
if ! grep -q "alias sdmn-dev" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# SDMN Framework development alias" >> "$SHELL_RC"
    echo "alias sdmn-dev='cd $(pwd) && poetry shell'" >> "$SHELL_RC"
    echo "alias sdmn-jupyter='cd $(pwd) && poetry run jupyter lab'" >> "$SHELL_RC"
    echo "alias sdmn-test='cd $(pwd) && poetry run pytest'" >> "$SHELL_RC"
    print_success "Added shell aliases to $SHELL_RC"
fi

# macOS-specific: Setup launch agent for development server (optional)
print_step "Creating development launcher..."
PLIST_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$PLIST_DIR"

cat > "$PLIST_DIR/org.sdmn.framework.dev.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>org.sdmn.framework.dev</string>
    <key>WorkingDirectory</key>
    <string>$(pwd)</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which poetry)</string>
        <string>run</string>
        <string>jupyter</string>
        <string>lab</string>
        <string>--port=8888</string>
        <string>--no-browser</string>
    </array>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>$(pwd)/logs/jupyter.log</string>
    <key>StandardErrorPath</key>
    <string>$(pwd)/logs/jupyter.error.log</string>
</dict>
</plist>
EOF

print_success "Created launch agent (use 'launchctl load' to enable)"

print_success "✅ macOS development environment setup complete!"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Restart terminal or run: source $SHELL_RC"
echo "2. Activate environment: poetry shell (or use alias: sdmn-dev)"
echo "3. Run tests: poetry run pytest (or: sdmn-test)"
echo "4. Start Jupyter: poetry run jupyter lab (or: sdmn-jupyter)"
echo "5. Open VS Code: code sdmn-framework.code-workspace"
echo ""
echo -e "${BLUE}macOS-specific features:${NC}"
echo "• App launcher: ~/Applications/SDMN Framework.app"
echo "• Shell aliases: sdmn-dev, sdmn-jupyter, sdmn-test"
echo "• Launch agent: ~/Library/LaunchAgents/org.sdmn.framework.dev.plist"
echo "• VS Code workspace: sdmn-framework.code-workspace"
echo ""
echo -e "${BLUE}Development commands:${NC}"
echo "  poetry run pytest                    # Run tests"
echo "  poetry run black src/                # Format code"
echo "  poetry run mypy src/                 # Type checking"
echo "  poetry run pre-commit run --all     # Run all checks"
echo "  brew services start jupyter         # Start Jupyter as service"
echo ""
echo "🎉 Ready for development on macOS!"
