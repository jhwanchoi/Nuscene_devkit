# nuScenes Devkit Experiments

3D object detection and lane detection experiments using nuScenes dataset.

## Features

- **Sensor Data Support**: Cameras (6x), LiDAR, Radar, IMU/GPS
- **Annotations**: 3D bounding boxes, 23 object categories, tracking IDs
- **Map Data**: HD semantic maps with lane geometry
- **Lightweight Models**: Simple PointNet for 3D detection testing
- **MLflow Integration**: Experiment tracking at port 5001

## Setup

### Using Docker (Recommended)

```bash
docker build -t nuscenes-exp .
docker run -it -v /path/to/nuscenes/data:/data/nuscenes nuscenes-exp
```

### Local Setup with uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .

# Set environment variables
export NUSCENES_DATAROOT=/path/to/nuscenes
export MLFLOW_TRACKING_URI=http://localhost:5001
```

## Download Dataset

Download the nuScenes mini dataset (10 scenes, ~4GB):
https://www.nuscenes.org/nuscenes#download

Extract to `/data/nuscenes` or set `NUSCENES_DATAROOT`.

## Usage

### Setup MLflow Experiments

```bash
python experiments/setup_mlflow.py
```

This creates the following experiments:
- `nuscenes-data-exploration`: Dataset exploration runs
- `nuscenes-3d-detection`: 3D object detection experiments
- `nuscenes-lane-detection`: Lane detection experiments

View experiments at: http://localhost:5001

### Explore Dataset Features

```bash
python experiments/explore.py
```

This will show:
- Dataset statistics
- Available sensors
- Object categories
- Visualize camera views
- **Logs to MLflow**: metrics, parameters, and visualizations

### Test 3D Object Detection

```bash
python experiments/test_inference.py
```

Simple PointNet model for testing:
- Lightweight architecture (~350K parameters)
- CPU/GPU support
- Checkpoint save/load
- **Logs to MLflow**: model, metrics, and artifacts

## Project Structure

```
.
├── Dockerfile                 # Docker environment
├── pyproject.toml             # Dependencies (uv, mlflow==3.5.1)
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
