#!/bin/bash
# Run script for SDMN Framework Docker container
# Compatible with both Docker and Podman

set -e

# Configuration
IMAGE_NAME="sdmn-framework"
IMAGE_TAG=${IMAGE_TAG:-"latest"}
CONTAINER_ENGINE=${CONTAINER_ENGINE:-"podman"}
CONTAINER_NAME="sdmn-instance"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if container engine is available
check_container_engine() {
    if ! command -v $CONTAINER_ENGINE &> /dev/null; then
        log_error "$CONTAINER_ENGINE is not installed or not in PATH"
        exit 1
    fi
    
    log_info "Using container engine: $CONTAINER_ENGINE"
}

# Check if image exists
check_image() {
    if ! $CONTAINER_ENGINE images | grep -q "$IMAGE_NAME.*$IMAGE_TAG"; then
        log_warning "Image $IMAGE_NAME:$IMAGE_TAG not found locally"
        log_info "Building image first..."
        ./scripts/build.sh $IMAGE_TAG
    fi
}

# Common run options
get_common_options() {
    echo "--rm \
          --name $CONTAINER_NAME-$(date +%s) \
          -v $(pwd)/data:/app/data:rw \
          -v $(pwd)/output:/app/output:rw \
          -v $(pwd)/checkpoints:/app/checkpoints:rw \
          -v $(pwd)/logs:/app/logs:rw"
}

# Run interactive shell
run_shell() {
    log_info "Starting interactive shell..."
    
    $CONTAINER_ENGINE run -it \
        $(get_common_options) \
        $IMAGE_NAME:$IMAGE_TAG shell
}

# Run Jupyter Lab
run_jupyter() {
    local port=${1:-8888}
    log_info "Starting Jupyter Lab on port $port..."
    log_info "Access at: http://localhost:$port"
    
    $CONTAINER_ENGINE run -it \
        $(get_common_options) \
        -p $port:8888 \
        $IMAGE_NAME:$IMAGE_TAG jupyter
}

# Run example simulation
run_simulation() {
    log_info "Running example simulation..."
    
    $CONTAINER_ENGINE run \
        $(get_common_options) \
        $IMAGE_NAME:$IMAGE_TAG simulation
}

# Run test suite
run_tests() {
    log_info "Running test suite..."
    
    $CONTAINER_ENGINE run \
        $(get_common_options) \
        $IMAGE_NAME:$IMAGE_TAG test
}

# Run GUI application
run_gui() {
    log_info "Starting network assembly GUI..."
    
    # Check if X11 is available (Linux)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # X11 forwarding for Linux
        DISPLAY_OPTS="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro"
        
        # Allow X11 connections
        xhost +local:docker >/dev/null 2>&1 || true
    else
        # For macOS/Windows, would need additional setup (XQuartz, VcXsrv, etc.)
        DISPLAY_OPTS=""
        log_warning "GUI support on non-Linux systems requires additional X11 server setup"
    fi
    
    $CONTAINER_ENGINE run -it \
        $(get_common_options) \
        $DISPLAY_OPTS \
        $IMAGE_NAME:$IMAGE_TAG gui
}

# Run development mode with source code mounted
run_dev() {
    log_info "Starting development mode with source code mounted..."
    
    $CONTAINER_ENGINE run -it \
        $(get_common_options) \
        -v $(pwd)/src:/app/src:rw \
        -v $(pwd)/examples:/app/examples:rw \
        -v $(pwd)/tests:/app/tests:rw \
        $IMAGE_NAME:$IMAGE_TAG shell
}

# Run with custom command
run_custom() {
    local command="$@"
    log_info "Running custom command: $command"
    
    $CONTAINER_ENGINE run -it \
        $(get_common_options) \
        $IMAGE_NAME:$IMAGE_TAG $command
}

# Use docker-compose/podman-compose
run_compose() {
    local service=${1:-"sdmn-framework"}
    local compose_cmd
    
    if command -v podman-compose &> /dev/null; then
        compose_cmd="podman-compose"
    elif command -v docker-compose &> /dev/null; then
        compose_cmd="docker-compose"
    else
        log_error "Neither podman-compose nor docker-compose found"
        exit 1
    fi
    
    log_info "Using $compose_cmd to start service: $service"
    
    cd docker
    $compose_cmd up $service
}

# Stop running containers
stop_containers() {
    log_info "Stopping SDMN containers..."
    
    # Stop containers with sdmn- prefix
    running_containers=$($CONTAINER_ENGINE ps --format "{{.Names}}" | grep "sdmn-" || true)
    
    if [ ! -z "$running_containers" ]; then
        echo "$running_containers" | xargs $CONTAINER_ENGINE stop
        log_success "Stopped containers: $running_containers"
    else
        log_info "No running SDMN containers found"
    fi
}

# Clean up containers and images
cleanup() {
    log_info "Cleaning up SDMN containers and images..."
    
    # Stop containers
    stop_containers
    
    # Remove containers
    all_containers=$($CONTAINER_ENGINE ps -a --format "{{.Names}}" | grep "sdmn-" || true)
    if [ ! -z "$all_containers" ]; then
        echo "$all_containers" | xargs $CONTAINER_ENGINE rm
        log_success "Removed containers: $all_containers"
    fi
    
    # Remove images if requested
    if [ "$1" = "--images" ]; then
        $CONTAINER_ENGINE rmi $IMAGE_NAME:$IMAGE_TAG 2>/dev/null || true
        log_success "Removed image: $IMAGE_NAME:$IMAGE_TAG"
    fi
    
    # Clean up dangling images and volumes
    $CONTAINER_ENGINE image prune -f >/dev/null 2>&1 || true
    $CONTAINER_ENGINE volume prune -f >/dev/null 2>&1 || true
}

# Show help message
show_help() {
    echo "SDMN Framework Container Runner"
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  shell              Start interactive shell"
    echo "  jupyter [port]     Start Jupyter Lab (default port: 8888)"
    echo "  simulation         Run example simulation"
    echo "  test               Run test suite"
    echo "  gui                Start network assembly GUI"
    echo "  dev                Development mode with source mounted"
    echo "  compose [service]  Use docker-compose (default service: sdmn-framework)"
    echo "  stop               Stop running SDMN containers"
    echo "  cleanup [--images] Clean up containers and optionally images"
    echo "  custom <cmd>       Run custom command in container"
    echo ""
    echo "Environment Variables:"
    echo "  CONTAINER_ENGINE   Container engine to use (default: podman)"
    echo "  IMAGE_TAG         Image tag to use (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 shell                    # Interactive shell"
    echo "  $0 jupyter 8889             # Jupyter Lab on port 8889"
    echo "  $0 custom python --version  # Run python --version"
    echo "  CONTAINER_ENGINE=docker $0 shell  # Use Docker instead of Podman"
}

# Main execution
main() {
    local command=${1:-"help"}
    
    case $command in
        shell)
            check_container_engine
            check_image
            run_shell
            ;;
        jupyter)
            check_container_engine
            check_image
            run_jupyter $2
            ;;
        simulation)
            check_container_engine
            check_image
            run_simulation
            ;;
        test)
            check_container_engine
            check_image
            run_tests
            ;;
        gui)
            check_container_engine
            check_image
            run_gui
            ;;
        dev)
            check_container_engine
            check_image
            run_dev
            ;;
        compose)
            run_compose $2
            ;;
        stop)
            check_container_engine
            stop_containers
            ;;
        cleanup)
            check_container_engine
            cleanup $2
            ;;
        custom)
            shift
            check_container_engine
            check_image
            run_custom "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Ensure we're in the project root
if [ ! -f "Dockerfile" ]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
mkdir -p data output checkpoints logs

# Run main function
main "$@"
