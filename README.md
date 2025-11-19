# nuScenes Devkit Experiments

3D object detection and lane detection experiments using nuScenes dataset.

## Features

- **Sensor Data Support**: Cameras (6x), LiDAR, Radar, IMU/GPS
- **Annotations**: 3D bounding boxes, 23 object categories, tracking IDs
- **Map Data**: HD semantic maps with lane geometry
- **Lightweight Models**: Simple PointNet for 3D detection testing
- **MLflow Integration**: Experiment tracking at port 5001

## Prerequisites

```bash
# Install just (task runner)
cargo install just
# OR
brew install just

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

```bash
# Show all available commands
just

# Setup everything (venv + dependencies + MLflow experiments)
just quickstart

# Activate virtual environment
source .venv/bin/activate

# Download dataset (shows instructions)
just download-data

# Validate setup
just validate

# Run all experiments
just exp-all
```

## Workflow Commands

### Setup & Validation

```bash
just setup           # Create venv + install dependencies
just validate        # Validate venv and data
just validate-venv   # Check Python packages
just validate-data   # Check nuScenes dataset
```

### Data Management

```bash
just download-data   # Show dataset download instructions
just validate-data   # Validate dataset structure
```

### MLflow

```bash
just setup-mlflow    # Create MLflow experiments
just mlflow          # Show MLflow UI URL
```

### Run Experiments

```bash
just exp-explore     # Run data exploration
just exp-detection   # Run 3D detection test
just exp-all         # Run all experiments
```

### Cleanup

```bash
just clean           # Remove venv and build artifacts
just clean-data      # Remove downloaded data
just clean-all       # Full cleanup
```

## Manual Setup (without just)

### Using Docker

```bash
docker build -t nuscenes-exp .
docker run -it -v /path/to/nuscenes/data:/data/nuscenes nuscenes-exp
```

### Local Setup

```bash
# Create venv and install
uv venv
source .venv/bin/activate
uv pip install -e .

# Set environment variables
export NUSCENES_DATAROOT=data/nuscenes
export MLFLOW_TRACKING_URI=http://localhost:5001

# Setup MLflow
python experiments/setup_mlflow.py

# Run experiments
python experiments/explore.py
python experiments/test_inference.py
```

## Project Structure

```
.
├── justfile                   # Workflow automation (just commands)
├── Dockerfile                 # Docker environment
├── pyproject.toml             # Dependencies (uv, mlflow==3.5.1)
├── scripts/
│   ├── download_data.py      # Dataset download helper
│   └── validate_data.py      # Dataset validation
├── experiments/
│   ├── setup_mlflow.py       # MLflow experiment setup
│   ├── explore.py            # Data exploration + MLflow logging
│   └── test_inference.py     # 3D detection test + MLflow tracking
└── data/                     # nuScenes dataset (mount/download)
```

## nuScenes Data Schema

- **Scene**: 20-second driving sequences
- **Sample**: Keyframes at 2Hz with synchronized sensors
- **Sample Data**: Individual sensor captures
- **Ego Pose**: Vehicle position/orientation
- **Annotations**: 3D bounding boxes with attributes

## MLflow Tracking

All experiments automatically log to MLflow at `http://localhost:5001`:

**Data Exploration Logs:**
- Dataset statistics (scenes, samples, annotations)
- Sensor counts by type
- Camera visualizations

**3D Detection Logs:**
- Model architecture and parameters
- Inference confidence scores
- PyTorch model artifacts
- Checkpoint validation results

## Next Steps

1. Integrate real nuScenes LiDAR data
2. Implement lane detection using Map API
3. Train models on mini dataset
4. Add evaluation metrics
5. Compare models in MLflow UI
