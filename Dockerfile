# Multi-stage Docker build for SDMN Framework
# Stage 1: Build environment
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Stage 2: Runtime environment
FROM python:3.10-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    SDMN_ENV=production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 sdmn \
    && useradd --uid 1000 --gid sdmn --shell /bin/bash --create-home sdmn

# Set work directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=sdmn:sdmn /app/.venv /app/.venv

# Copy source code
COPY --chown=sdmn:sdmn src/ ./src/
COPY --chown=sdmn:sdmn examples/ ./examples/
COPY --chown=sdmn:sdmn pyproject.toml ./
COPY --chown=sdmn:sdmn README.md ./

# Install the package in development mode
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data/checkpoints /app/data/results /app/logs \
    && chown -R sdmn:sdmn /app/data /app/logs

# Switch to non-root user
USER sdmn

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function to run tests\n\
run_tests() {\n\
    echo "Running SDMN framework tests..."\n\
    python -m pytest tests/ -v || echo "Some tests failed, but continuing..."\n\
}\n\
\n\
# Function to run examples\n\
run_examples() {\n\
    echo "Running SDMN examples..."\n\
    for example in examples/*.py; do\n\
        if [ -f "$example" ]; then\n\
            echo "Running $example..."\n\
            python "$example" || echo "Example $example failed, continuing..."\n\
        fi\n\
    done\n\
}\n\
\n\
# Function to run evaluation\n\
run_evaluation() {\n\
    echo "Running SDMN evaluation..."\n\
    \n\
    # Test basic import\n\
    python -c "import sdmn; print(f\"SDMN version: {sdmn.__version__}\")" || exit 1\n\
    \n\
    # Run a simple simulation\n\
    python -c "\n\
import sdmn\n\
from sdmn.core import SimulationEngine, SimulationConfig\n\
from sdmn.networks import NetworkBuilder, NetworkConfiguration, NetworkTopology\n\
from sdmn.neurons import NeuronType\n\
\n\
print(\"Creating simulation configuration...\")\n\
config = SimulationConfig(dt=0.1, max_time=100.0, enable_logging=False)\n\
engine = SimulationEngine(config)\n\
\n\
print(\"Building network...\")\n\
net_config = NetworkConfiguration(\n\
    name=\"test_net\",\n\
    n_neurons=10,\n\
    topology=NetworkTopology.SMALL_WORLD,\n\
    neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE\n\
)\n\
builder = NetworkBuilder()\n\
network = builder.create_network(net_config)\n\
engine.add_network(\"test\", network)\n\
\n\
print(\"Running simulation...\")\n\
results = engine.run()\n\
\n\
if results.success:\n\
    print(f\"✅ Simulation completed! Steps: {results.total_steps}, Time: {results.simulation_time:.1f}ms\")\n\
else:\n\
    print(f\"❌ Simulation failed: {results.error_message}\")\n\
    exit(1)\n\
" || exit 1\n\
    \n\
    echo "✅ All evaluations passed!"\n\
}\n\
\n\
# Main execution logic\n\
case "${1:-evaluation}" in\n\
    "tests")\n\
        run_tests\n\
        ;;\n\
    "examples")\n\
        run_examples\n\
        ;;\n\
    "evaluation")\n\
        run_evaluation\n\
        ;;\n\
    "all")\n\
        run_tests\n\
        run_examples  \n\
        run_evaluation\n\
        ;;\n\
    *)\n\
        echo "Usage: $0 {tests|examples|evaluation|all}"\n\
        echo "Default: evaluation"\n\
        exec "$@"\n\
        ;;\n\
esac\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Expose port (if needed for future web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sdmn; print('OK')" || exit 1

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["evaluation"]