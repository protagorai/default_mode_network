# Dockerfile for Synthetic Default Mode Network (SDMN) Framework
# Based on Linux LTS with Python 3.11 for neural network simulation

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SDMN_HOME=/app/sdmn

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and build essentials
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    build-essential \
    # Scientific computing dependencies
    gfortran \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    # Graphics and visualization
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    # GUI support (for network assembly interface)
    python3-tk \
    xvfb \
    x11-apps \
    # Network and development tools
    git \
    wget \
    curl \
    vim \
    htop \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash sdmn && \
    chown -R sdmn:sdmn /app

# Switch to sdmn user
USER sdmn

# Set up Python virtual environment
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install Python package manager
RUN pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY --chown=sdmn:sdmn requirements.txt dev-requirements.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt && \
    pip install -r dev-requirements.txt

# Copy source code and documentation
COPY --chown=sdmn:sdmn src/ ./src/
COPY --chown=sdmn:sdmn docs/ ./docs/
COPY --chown=sdmn:sdmn scripts/ ./scripts/
COPY --chown=sdmn:sdmn tests/ ./tests/
COPY --chown=sdmn:sdmn examples/ ./examples/
COPY --chown=sdmn:sdmn README.md ./

# Create necessary directories
RUN mkdir -p /app/data \
             /app/output \
             /app/checkpoints \
             /app/logs \
             /app/models

# Set Python path
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create entry point script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Activate virtual environment\n\
source /app/venv/bin/activate\n\
\n\
# Set display for GUI applications (if X11 forwarding is available)\n\
export DISPLAY=${DISPLAY:-:99}\n\
\n\
# Start Xvfb if no display is available\n\
if [ "$1" = "gui" ] && [ -z "$DISPLAY_SET" ]; then\n\
    Xvfb :99 -screen 0 1024x768x24 &\n\
    export DISPLAY=:99\n\
    export DISPLAY_SET=1\n\
fi\n\
\n\
# Execute the requested command\n\
case "$1" in\n\
    "jupyter")\n\
        echo "Starting Jupyter Lab..."\n\
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app\n\
        ;;\n\
    "simulation")\n\
        echo "Running SDMN simulation..."\n\
        python3 -m examples.quickstart_simulation\n\
        ;;\n\
    "test")\n\
        echo "Running tests..."\n\
        python3 -m pytest tests/ -v\n\
        ;;\n\
    "gui")\n\
        echo "Starting network assembly GUI..."\n\
        python3 -m scripts.network_gui\n\
        ;;\n\
    "shell")\n\
        echo "Starting interactive shell..."\n\
        /bin/bash\n\
        ;;\n\
    *)\n\
        echo "Available commands:"\n\
        echo "  jupyter    - Start Jupyter Lab for interactive development"\n\
        echo "  simulation - Run example simulation"\n\
        echo "  test       - Run test suite"\n\
        echo "  gui        - Start network assembly GUI"\n\
        echo "  shell      - Interactive shell"\n\
        echo ""\n\
        echo "Custom command: $@"\n\
        exec "$@"\n\
        ;;\n\
esac' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 8888 8080 6006

# Set up volume mount points
VOLUME ["/app/data", "/app/output", "/app/checkpoints", "/app/logs"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import src.core; print('SDMN Framework loaded successfully')" || exit 1

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["shell"]

# Labels for container metadata
LABEL maintainer="SDMN Research Team"
LABEL version="1.0.0"
LABEL description="Synthetic Default Mode Network Framework - Biologically-inspired spiking neural networks"
LABEL org.opencontainers.image.title="SDMN Framework"
LABEL org.opencontainers.image.description="Research framework for synthetic default mode networks"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.created="2024-01-01"
LABEL org.opencontainers.image.source="https://github.com/example/sdmn-framework"
LABEL org.opencontainers.image.licenses="MIT"
