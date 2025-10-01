#!/bin/bash
# Master Setup Script for SDMN Framework
# This script helps users choose the right deployment strategy

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "=========================================="
    echo "   SDMN Framework Setup Assistant"
    echo "=========================================="
    echo -e "${NC}"
}

print_option() {
    echo -e "${BLUE}$1)${NC} ${GREEN}$2${NC}"
    echo -e "   $3"
    echo ""
}

print_info() {
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

show_deployment_options() {
    echo -e "${CYAN}Choose your deployment strategy:${NC}"
    echo ""
    
    print_option "1" "Local Development" \
        "Best for active development, debugging, IDE integration"
    
    print_option "2" "Local Production" \
        "Best for production deployment on dedicated servers"
    
    print_option "3" "Containerized Development" \
        "Best for consistent environments, team development"
    
    print_option "4" "Containerized Production" \
        "Best for cloud deployment, Kubernetes, scaling"
    
    print_option "5" "Show Detailed Comparison" \
        "Compare all deployment strategies"
    
    print_option "6" "Exit" \
        "Exit without setting up"
}

show_detailed_comparison() {
    echo -e "${CYAN}Deployment Strategy Comparison:${NC}"
    echo ""
    echo "┌─────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐"
    echo "│ Aspect              │ Local Development    │ Local Production     │ Containerized        │"
    echo "├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤"
    echo "│ Setup Time          │ ~2-3 minutes         │ ~3-5 minutes         │ ~5-10 minutes        │"
    echo "│ Disk Space          │ ~500MB               │ ~300MB               │ ~1-2GB               │"
    echo "│ IDE Integration     │ Excellent            │ Good                 │ Good (with mounts)   │"
    echo "│ Debugging           │ Native               │ Native               │ Container logs       │"
    echo "│ Isolation           │ None                 │ Virtual env only     │ Complete             │"
    echo "│ Portability         │ OS dependent         │ OS dependent         │ Fully portable       │"
    echo "│ Performance         │ Native               │ Native               │ Near-native          │"
    echo "│ Deployment Ready    │ No                   │ Yes                  │ Yes                  │"
    echo "└─────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘"
    echo ""
    
    echo -e "${YELLOW}Recommendations:${NC}"
    echo "• ${GREEN}Developers${NC}: Choose Local Development (#1)"
    echo "• ${GREEN}DevOps/Production${NC}: Choose Local Production (#2) or Containerized Production (#4)"  
    echo "• ${GREEN}Teams/CI/CD${NC}: Choose Containerized options (#3 or #4)"
    echo "• ${GREEN}Cloud/Kubernetes${NC}: Choose Containerized Production (#4)"
    echo ""
}

setup_local_development() {
    print_info "Setting up local development environment..."
    echo ""
    
    if [ ! -f "scripts/setup_development.sh" ]; then
        print_error "Development setup script not found!"
        return 1
    fi
    
    print_info "This will:"
    echo "  • Install Poetry and dependencies"
    echo "  • Set up pre-commit hooks"
    echo "  • Install package in editable mode"
    echo "  • Configure development tools"
    echo ""
    
    read -p "Continue? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./scripts/setup_development.sh
        
        if [ $? -eq 0 ]; then
            print_success "Development environment setup complete!"
            echo ""
            echo "Next steps:"
            echo "  poetry shell                    # Activate virtual environment"
            echo "  python examples/quickstart_simulation.py  # Test installation"
            echo "  python -m sdmn info             # Check package info"
        fi
    else
        print_info "Setup cancelled"
    fi
}

setup_local_production() {
    print_info "Setting up local production environment..."
    echo ""
    
    if [ ! -f "scripts/setup_production.sh" ]; then
        print_error "Production setup script not found!"
        return 1
    fi
    
    print_info "This will:"
    echo "  • Set up production Python environment"
    echo "  • Create systemd service files"
    echo "  • Configure production directories"
    echo "  • Create activation scripts"
    echo ""
    
    echo "Choose installation method:"
    echo "  1) Poetry (recommended)"
    echo "  2) pip"
    read -p "Choice [1]: " choice
    choice=${choice:-1}
    
    case $choice in
        1)
            ./scripts/setup_production.sh --poetry
            ;;
        2)
            read -p "Python version [3.9]: " py_version
            py_version=${py_version:-3.9}
            ./scripts/setup_production.sh --python $py_version
            ;;
        *)
            print_error "Invalid choice"
            return 1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        print_success "Production environment setup complete!"
        echo ""
        echo "Next steps:"
        echo "  ./activate_sdmn.sh               # Activate environment"
        echo "  python -m sdmn simulate --help   # Test CLI"
        echo "  sudo systemctl start sdmn        # Start service (if systemd)"
    fi
}

setup_containerized() {
    local mode=$1
    print_info "Setting up containerized $mode environment..."
    echo ""
    
    if [ ! -f "scripts/build.sh" ] || [ ! -f "scripts/run.sh" ]; then
        print_error "Container scripts not found!"
        return 1
    fi
    
    # Check for container engine
    if command -v podman &> /dev/null; then
        engine="podman"
    elif command -v docker &> /dev/null; then
        engine="docker"
    else
        print_error "Neither Docker nor Podman found!"
        echo "Please install Docker or Podman first."
        return 1
    fi
    
    print_info "Using container engine: $engine"
    print_info "This will:"
    echo "  • Build SDMN framework container image"
    echo "  • Set up container runtime environment"
    echo "  • Create data directories"
    echo ""
    
    read -p "Continue? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        export CONTAINER_ENGINE=$engine
        ./scripts/build.sh
        
        if [ $? -eq 0 ]; then
            print_success "Container environment setup complete!"
            echo ""
            
            if [ "$mode" = "development" ]; then
                echo "Next steps (Development):"
                echo "  ./scripts/run.sh dev             # Development with source mounted"
                echo "  ./scripts/run.sh jupyter         # Start Jupyter Lab"
                echo "  ./scripts/run.sh shell           # Interactive shell"
            else
                echo "Next steps (Production):"
                echo "  ./scripts/run.sh simulation      # Run simulation"
                echo "  ./scripts/run.sh compose         # Use docker-compose"
                echo "  $engine run sdmn-framework:latest evaluation"
            fi
        fi
    else
        print_info "Setup cancelled"
    fi
}

check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        return 1
    fi
    
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ]; then 
        print_error "Python 3.8+ required, but found $python_version"
        return 1
    fi
    
    print_success "Python $python_version found"
    
    # Check git
    if ! command -v git &> /dev/null; then
        print_warning "Git not found (recommended for development)"
    fi
    
    # Check curl
    if ! command -v curl &> /dev/null; then
        print_warning "curl not found (needed for Poetry installation)"
    fi
    
    return 0
}

main() {
    print_header
    
    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ] || [ ! -d "scripts" ]; then
        print_error "Please run this script from the SDMN project root directory"
        exit 1
    fi
    
    # Check prerequisites
    if ! check_prerequisites; then
        print_error "Prerequisites check failed"
        exit 1
    fi
    
    echo ""
    
    while true; do
        show_deployment_options
        read -p "Enter your choice [1-6]: " choice
        
        case $choice in
            1)
                echo ""
                setup_local_development
                break
                ;;
            2)
                echo ""
                setup_local_production
                break
                ;;
            3)
                echo ""
                setup_containerized "development"
                break
                ;;
            4)
                echo ""
                setup_containerized "production"
                break
                ;;
            5)
                echo ""
                show_detailed_comparison
                echo ""
                read -p "Press Enter to continue..."
                echo ""
                ;;
            6)
                print_info "Exiting without setup"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please enter 1-6."
                echo ""
                ;;
        esac
    done
    
    echo ""
    print_info "Setup complete! Run 'python scripts/verify_installation.py' to verify."
}

# Show help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "SDMN Framework Setup Assistant"
    echo ""
    echo "This interactive script helps you choose and set up the right"
    echo "deployment strategy for the SDMN Framework."
    echo ""
    echo "Available deployment strategies:"
    echo "  1. Local Development    - For active development"
    echo "  2. Local Production     - For production on dedicated servers"  
    echo "  3. Containerized Dev    - For consistent team development"
    echo "  4. Containerized Prod   - For cloud/Kubernetes deployment"
    echo ""
    echo "Usage: $0 [--help]"
    echo ""
    echo "The script will guide you through:"
    echo "  • Checking prerequisites"
    echo "  • Choosing deployment strategy"
    echo "  • Running appropriate setup scripts"
    echo "  • Verifying installation"
    exit 0
fi

# Run main function
main
