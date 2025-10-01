# Legacy Files

This directory contains old configuration files that have been migrated to Poetry for better dependency management.

## Migrated Files

- **`requirements.txt`** → Moved to `pyproject.toml` [tool.poetry.dependencies]
- **`dev-requirements.txt`** → Moved to `pyproject.toml` [tool.poetry.group.dev.dependencies]

## Migration Notes

All dependencies from these files have been consolidated into `pyproject.toml` with proper grouping:

- **Main dependencies**: Required for running SDMN
- **Dev dependencies**: Development tools (testing, linting, docs)
- **Optional dependencies**: Heavy packages marked as optional
- **Extras**: Installable feature groups

## How to Install

### New Way (Poetry - Recommended)
```bash
poetry install --with dev,test          # Full development
poetry install --only=main              # Core only
poetry install --with dev --extras all  # Everything
```

### Old Way (pip - Still Works)
```bash
pip install -e .                        # Core only
pip install -e .[all]                   # With optional features
pip install -e .[visualization,data]    # Specific features
```

The old requirements files are kept here for reference but should not be used for new installations.
