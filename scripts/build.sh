#!/bin/bash
# Build script for SDMN Framework Docker container
# Compatible with both Docker and Podman

set -e

# Configuration
IMAGE_NAME="sdmn-framework"
IMAGE_TAG=${1:-"latest"}
CONTAINER_ENGINE=${CONTAINER_ENGINE:-"podman"}
BUILD_CONTEXT="."

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

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    mkdir -p data output checkpoints logs examples
    chmod 755 data output checkpoints logs examples
}

# Build the container image
build_image() {
    log_info "Building SDMN Framework image: $IMAGE_NAME:$IMAGE_TAG"
    
    # Check if Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        log_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    # Build arguments
    BUILD_ARGS=""
    
    # Add build arguments if specified
    if [ ! -z "$PYTHON_VERSION" ]; then
        BUILD_ARGS="$BUILD_ARGS --build-arg PYTHON_VERSION=$PYTHON_VERSION"
    fi
    
    # Build the image
    $CONTAINER_ENGINE build \
        $BUILD_ARGS \
        -t $IMAGE_NAME:$IMAGE_TAG \
        -f Dockerfile \
        $BUILD_CONTEXT
    
    if [ $? -eq 0 ]; then
        log_success "Successfully built image: $IMAGE_NAME:$IMAGE_TAG"
    else
        log_error "Failed to build image"
        exit 1
    fi
}

# Clean up old images (optional)
cleanup_old_images() {
    if [ "$CLEANUP" = "true" ]; then
        log_info "Cleaning up old images..."
        $CONTAINER_ENGINE image prune -f
        $CONTAINER_ENGINE rmi $($CONTAINER_ENGINE images -f "dangling=true" -q) 2>/dev/null || true
        log_success "Cleanup completed"
    fi
}

# Show image information
show_image_info() {
    log_info "Image information:"
    $CONTAINER_ENGINE images | grep $IMAGE_NAME | head -5
    
    log_info "Image size:"
    $CONTAINER_ENGINE inspect $IMAGE_NAME:$IMAGE_TAG --format='{{.Size}}' | numfmt --to=iec --suffix=B
}

# Main execution
main() {
    echo "========================================="
    echo "SDMN Framework Container Build Script"
    echo "========================================="
    
    check_container_engine
    create_directories
    build_image
    cleanup_old_images
    show_image_info
    
    echo ""
    log_success "Build completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Run interactive shell:    $CONTAINER_ENGINE run -it $IMAGE_NAME:$IMAGE_TAG shell"
    echo "  2. Start Jupyter Lab:        $CONTAINER_ENGINE run -p 8888:8888 $IMAGE_NAME:$IMAGE_TAG jupyter"
    echo "  3. Run example simulation:   $CONTAINER_ENGINE run $IMAGE_NAME:$IMAGE_TAG simulation"
    echo "  4. Use docker-compose:       podman-compose -f docker/docker-compose.yml up"
}

# Help message
show_help() {
    echo "Usage: $0 [TAG] [OPTIONS]"
    echo ""
    echo "Build SDMN Framework container image"
    echo ""
    echo "Arguments:"
    echo "  TAG                    Image tag (default: latest)"
    echo ""
    echo "Environment Variables:"
    echo "  CONTAINER_ENGINE       Container engine to use (default: podman)"
    echo "  PYTHON_VERSION         Python version to use in container"
    echo "  CLEANUP               Clean up old images after build (true/false)"
    echo ""
    echo "Examples:"
    echo "  $0                     # Build with tag 'latest'"
    echo "  $0 v1.0.0             # Build with tag 'v1.0.0'"
    echo "  CLEANUP=true $0        # Build and cleanup old images"
    echo "  CONTAINER_ENGINE=docker $0  # Use Docker instead of Podman"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --cleanup)
            export CLEANUP=true
            shift
            ;;
        --docker)
            export CONTAINER_ENGINE=docker
            shift
            ;;
        --podman)
            export CONTAINER_ENGINE=podman
            shift
            ;;
        *)
            if [ -z "$IMAGE_TAG" ] || [ "$IMAGE_TAG" = "latest" ]; then
                IMAGE_TAG="$1"
            else
                log_error "Unknown argument: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Run main function
main
