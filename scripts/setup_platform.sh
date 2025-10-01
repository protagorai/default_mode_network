#!/bin/bash
# Platform-Aware Setup Script for SDMN Framework
# This script detects the platform and offers appropriate setup options

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "================================================"
    echo "   SDMN Framework Platform-Aware Setup"
    echo "================================================"
    echo -e "${NC}"
}

print_platform_info() {
    echo -e "${PURPLE}Platform Information:${NC}"
    echo "  OS: $1"
    echo "  Architecture: $2" 
    echo "  Shell: $3"
    echo ""
}

detect_platform() {
    local os_type=""
    local arch=""
    local shell_type=""
    
    # Detect OS
    case "$OSTYPE" in
        linux-gnu*)
            os_type="Linux"
            ;;
        darwin*)
            os_type="macOS"
            ;;
        cygwin*|msys*|win32*)
            os_type="Windows (Bash)"
            ;;
        *)
            os_type="Unknown ($OSTYPE)"
            ;;
    esac
    
    # Detect architecture
    arch=$(uname -m 2>/dev/null || echo "unknown")
    
    # Detect shell
    if [ -n "$ZSH_VERSION" ]; then
        shell_type="zsh"
    elif [ -n "$BASH_VERSION" ]; then
        shell_type="bash"
    else
        shell_type=$(basename "${SHELL:-unknown}")
    fi
    
    echo "$os_type|$arch|$shell_type"
}

show_platform_options() {
    local platform_info="$1"
    local os_type=$(echo "$platform_info" | cut -d'|' -f1)
    
    echo -e "${CYAN}Available setup options for $os_type:${NC}"
    echo ""
    
    case "$os_type" in
        "Linux")
            echo -e "${GREEN}1) Linux Development${NC} - ./scripts/setup_development.sh"
            echo "   Full development environment with Poetry, pre-commit, testing"
            echo ""
            echo -e "${GREEN}2) Linux Production${NC} - ./scripts/setup_production.sh"
            echo "   Production deployment with systemd service integration"
            echo ""
            echo -e "${GREEN}3) Containerized (Docker/Podman)${NC} - ./scripts/build.sh"
            echo "   Container-based deployment for Linux servers"
            echo ""
            ;;
        "macOS")
            echo -e "${GREEN}1) macOS Development (Homebrew)${NC} - ./scripts/setup_development_macos.sh"
            echo "   macOS-optimized development with Homebrew, VS Code, app bundles"
            echo ""
            echo -e "${GREEN}2) macOS Production (Homebrew)${NC} - ./scripts/setup_production_macos.sh"
            echo "   Production deployment with launchd service integration"
            echo ""
            echo -e "${GREEN}3) Generic Development${NC} - ./scripts/setup_development.sh"
            echo "   Standard development setup (works on macOS)"
            echo ""
            echo -e "${GREEN}4) Containerized (Docker)${NC} - ./scripts/build.sh"
            echo "   Container-based deployment for macOS"
            echo ""
            ;;
        "Windows (Bash)")
            echo -e "${GREEN}1) Generic Development${NC} - ./scripts/setup_development.sh"
            echo "   Development setup for WSL/Git Bash"
            echo ""
            echo -e "${GREEN}2) Generic Production${NC} - ./scripts/setup_production.sh"  
            echo "   Production setup for WSL/Git Bash"
            echo ""
            echo -e "${GREEN}3) Containerized (Docker)${NC} - ./scripts/build.sh"
            echo "   Container-based deployment"
            echo ""
            echo -e "${YELLOW}Note: For native Windows, use PowerShell scripts:${NC}"
            echo "   scripts/setup_development.bat"
            echo "   scripts/setup_production.bat"
            echo "   scripts/build.bat"
            echo ""
            ;;
        *)
            echo -e "${YELLOW}Generic options (should work on most Unix-like systems):${NC}"
            echo ""
            echo -e "${GREEN}1) Generic Development${NC} - ./scripts/setup_development.sh"
            echo "   Standard development setup"
            echo ""
            echo -e "${GREEN}2) Generic Production${NC} - ./scripts/setup_production.sh"
            echo "   Standard production setup"
            echo ""
            echo -e "${GREEN}3) Containerized${NC} - ./scripts/build.sh"
            echo "   Container-based deployment"
            echo ""
            ;;
    esac
    
    echo -e "${GREEN}5) Show all available scripts${NC}"
    echo ""
    echo -e "${GREEN}6) Exit${NC}"
}

show_all_scripts() {
    echo -e "${CYAN}All Available Scripts:${NC}"
    echo ""
    
    echo -e "${PURPLE}Development Setup:${NC}"
    echo "  ./scripts/setup_development.sh      # Linux/Unix development"
    echo "  ./scripts/setup_development_macos.sh # macOS with Homebrew"
    echo "  ./scripts/setup_development.bat     # Windows native"
    echo ""
    
    echo -e "${PURPLE}Production Setup:${NC}"
    echo "  ./scripts/setup_production.sh       # Linux/Unix production"
    echo "  ./scripts/setup_production_macos.sh # macOS with launchd"
    echo "  ./scripts/setup_production.bat      # Windows native"
    echo ""
    
    echo -e "${PURPLE}Container Management:${NC}"
    echo "  ./scripts/build.sh                  # Linux/macOS container build"
    echo "  ./scripts/run.sh                    # Linux/macOS container run"
    echo "  ./scripts/build.bat                 # Windows container build"
    echo "  ./scripts/run.bat                   # Windows container run"
    echo ""
    
    echo -e "${PURPLE}Verification:${NC}"
    echo "  python scripts/verify_installation.py  # Verify package install"
    echo "  python scripts/verify_structure.py     # Verify package structure"
    echo ""
    
    echo -e "${PURPLE}Platform Detection:${NC}"
    echo "  ./scripts/setup_platform.sh         # This script"
    echo "  ./scripts/setup.sh                  # Generic interactive setup"
    echo ""
}

run_platform_setup() {
    local platform_info="$1"
    local choice="$2"
    local os_type=$(echo "$platform_info" | cut -d'|' -f1)
    
    case "$os_type" in
        "Linux")
            case "$choice" in
                1) ./scripts/setup_development.sh ;;
                2) ./scripts/setup_production.sh ;;
                3) ./scripts/build.sh ;;
                *) echo "[ERROR] Invalid choice"; return 1 ;;
            esac
            ;;
        "macOS")
            case "$choice" in
                1) ./scripts/setup_development_macos.sh ;;
                2) ./scripts/setup_production_macos.sh ;;
                3) ./scripts/setup_development.sh ;;
                4) ./scripts/build.sh ;;
                *) echo "[ERROR] Invalid choice"; return 1 ;;
            esac
            ;;
        "Windows (Bash)")
            case "$choice" in
                1) ./scripts/setup_development.sh ;;
                2) ./scripts/setup_production.sh ;;
                3) ./scripts/build.sh ;;
                *) echo "[ERROR] Invalid choice"; return 1 ;;
            esac
            ;;
        *)
            case "$choice" in
                1) ./scripts/setup_development.sh ;;
                2) ./scripts/setup_production.sh ;;
                3) ./scripts/build.sh ;;
                *) echo "[ERROR] Invalid choice"; return 1 ;;
            esac
            ;;
    esac
}

main() {
    print_header
    
    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ] || [ ! -d "scripts" ]; then
        echo "[ERROR] Please run this script from the SDMN project root directory"
        exit 1
    fi
    
    # Detect platform
    platform_info=$(detect_platform)
    os_type=$(echo "$platform_info" | cut -d'|' -f1)
    arch=$(echo "$platform_info" | cut -d'|' -f2)
    shell_type=$(echo "$platform_info" | cut -d'|' -f3)
    
    print_platform_info "$os_type" "$arch" "$shell_type"
    
    # Show platform-specific recommendations
    if [ "$os_type" = "Windows (Bash)" ]; then
        echo -e "${YELLOW}💡 Recommendation for Windows users:${NC}"
        echo "For best experience, use native Windows scripts:"
        echo "  scripts/setup_development.bat    # PowerShell/CMD"
        echo "  scripts/setup_production.bat     # PowerShell/CMD"
        echo "  scripts/build.bat                # PowerShell/CMD"
        echo ""
        echo "Or continue with bash-compatible scripts below."
        echo ""
    elif [ "$os_type" = "macOS" ]; then
        echo -e "${YELLOW}💡 Recommendation for macOS users:${NC}"
        echo "Use Homebrew-optimized scripts for best experience:"
        echo "  scripts/setup_development_macos.sh"
        echo "  scripts/setup_production_macos.sh"
        echo ""
    fi
    
    while true; do
        show_platform_options "$platform_info"
        read -p "Enter your choice: " choice
        
        case "$choice" in
            [1-4])
                echo ""
                if run_platform_setup "$platform_info" "$choice"; then
                    echo ""
                    echo -e "${GREEN}[SUCCESS] Setup completed!${NC}"
                    echo "Run 'python scripts/verify_installation.py' to verify."
                    break
                else
                    echo ""
                    echo -e "${RED}[ERROR] Setup failed!${NC}"
                    echo "Please check the error messages above."
                    break
                fi
                ;;
            5)
                echo ""
                show_all_scripts
                echo ""
                read -p "Press Enter to continue..."
                echo ""
                ;;
            6)
                echo ""
                echo "[INFO] Exiting without setup"
                exit 0
                ;;
            *)
                echo ""
                echo "[ERROR] Invalid choice. Please enter a valid option."
                echo ""
                ;;
        esac
    done
}

# Show help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "SDMN Framework Platform-Aware Setup"
    echo ""
    echo "This script detects your platform and recommends the best"
    echo "setup approach for your operating system."
    echo ""
    echo "Supported platforms:"
    echo "  • Linux    - Native bash scripts with systemd"
    echo "  • macOS    - Homebrew-optimized scripts with launchd"  
    echo "  • Windows  - Native .bat scripts or bash-compatible"
    echo ""
    echo "The script will:"
    echo "  • Detect your platform automatically"
    echo "  • Show platform-specific recommendations"
    echo "  • Offer appropriate setup scripts"
    echo "  • Guide you through the setup process"
    echo ""
    echo "Usage: $0 [--help]"
    exit 0
fi

# Run main function
main
