# nuScenes Experiments Workflow
# Install just: cargo install just OR brew install just

# Default recipe (show help)
default:
    @just --list

# Setup virtual environment with uv
setup-venv:
    @echo "Creating virtual environment with uv..."
    uv venv
    @echo "Virtual environment created at .venv/"

# Install dependencies
install:
    @echo "Installing dependencies..."
    uv pip install -e .
    @echo "Dependencies installed!"

# Full setup (venv + install)
setup: setup-venv install
    @echo "Setup complete! Activate with: source .venv/bin/activate"

# Validate virtual environment
validate-venv:
    @echo "Validating virtual environment..."
    @python --version
    @python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
    @python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    @echo "✓ Virtual environment is valid"

# Download nuScenes mini dataset
download-data:
    @echo "Downloading nuScenes mini dataset..."
    @python scripts/download_data.py

# Validate data
validate-data:
    @echo "Validating nuScenes data..."
    @python scripts/validate_data.py

# Full validation (venv + data)
validate: validate-venv validate-data
    @echo "✓ All validations passed!"

# Setup MLflow experiments
setup-mlflow:
    @echo "Setting up MLflow experiments..."
    @python experiments/setup_mlflow.py

# Run data exploration experiment
exp-explore:
    @echo "Running data exploration experiment..."
    @python experiments/explore.py

# Run 3D detection test experiment
exp-detection:
    @echo "Running 3D detection experiment..."
    @python experiments/test_inference.py

# Run all experiments
exp-all: exp-explore exp-detection
    @echo "✓ All experiments completed!"

# Clean up
clean:
    @echo "Cleaning up..."
    rm -rf .venv
    rm -rf *.egg-info
    rm -rf build dist
    rm -rf __pycache__
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    @echo "Cleanup complete!"

# Clean data
clean-data:
    @echo "Removing downloaded data..."
    rm -rf data/
    @echo "Data removed!"

# Full clean
clean-all: clean clean-data
    @echo "✓ Full cleanup complete!"

# Open MLflow UI (info)
mlflow:
    @echo "MLflow UI should be running at: http://localhost:5001"
    @echo "If not, ask your admin to start the MLflow server"

# Quick start (setup + validate + mlflow setup)
quickstart: setup validate setup-mlflow
    @echo ""
    @echo "✓ Quickstart complete!"
    @echo ""
    @echo "Next steps:"
    @echo "  1. Activate venv: source .venv/bin/activate"
    @echo "  2. Download data: just download-data"
    @echo "  3. Run experiments: just exp-all"
    @echo "  4. View results: http://localhost:5001"
