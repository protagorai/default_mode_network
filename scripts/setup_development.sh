#!/bin/bash

# Development Environment Setup Script for SDMN Framework
# This script sets up the development environment with all necessary dependencies,
# pre-commit hooks, and development tools.

set -e  # Exit on any error

echo "🚀 Setting up SDMN Framework development environment..."

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

# Check if Poetry is installed
print_step "Checking Poetry installation..."
if ! command -v poetry &> /dev/null; then
    print_warning "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry installation failed. Please install Poetry manually: https://python-poetry.org/docs/#installation"
        exit 1
    fi
fi

print_success "Poetry is installed"

# Configure Poetry
print_step "Configuring Poetry..."
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

# Install dependencies
print_step "Installing project dependencies..."
poetry install --with dev,test,data,web

# Activate virtual environment and run additional setup
print_step "Setting up pre-commit hooks..."
poetry run pre-commit install

# Run code formatting
print_step "Formatting code with black and isort..."
poetry run black src/ tests/ examples/ || print_warning "Black formatting encountered issues (this is normal for new projects)"
poetry run isort src/ tests/ examples/ || print_warning "isort formatting encountered issues (this is normal for new projects)"

# Run type checking
print_step "Running type checks with mypy..."
poetry run mypy src/ --ignore-missing-imports || print_warning "MyPy found some type issues (this is normal for new projects)"

# Create necessary directories
print_step "Creating development directories..."
mkdir -p logs
mkdir -p data/checkpoints
mkdir -p data/results
mkdir -p docs/build

# Install package in editable mode
print_step "Installing package in editable mode..."
poetry run pip install -e .

# Run tests to verify installation
print_step "Running tests to verify installation..."
if poetry run pytest tests/ -v --tb=short; then
    print_success "All tests passed!"
else
    print_warning "Some tests failed, but the development environment is set up"
fi

# Generate documentation
print_step "Generating documentation..."
cd docs && poetry run sphinx-quickstart -q --sep -p "SDMN" -a "SDMN Team" -v "0.1.0" --ext-autodoc --ext-viewcode --makefile --no-batchfile . || cd ..
cd ..

# Set up Jupyter kernel
print_step "Setting up Jupyter kernel..."
poetry run python -m ipykernel install --user --name sdmn-dev --display-name "SDMN Development"

print_success "✅ Development environment setup complete!"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Activate the environment: poetry shell"
echo "2. Run tests: poetry run pytest"
echo "3. Start development: poetry run python -m sdmn --help"
echo "4. Run examples: poetry run python examples/quickstart_simulation.py"
echo "5. Start Jupyter: poetry run jupyter lab"
echo ""
echo -e "${BLUE}Development commands:${NC}"
echo "  poetry run pytest                    # Run tests"
echo "  poetry run black src/                # Format code"
echo "  poetry run mypy src/                 # Type checking"
echo "  poetry run pre-commit run --all     # Run all checks"
echo "  poetry run sphinx-build docs docs/build  # Build docs"
echo ""
