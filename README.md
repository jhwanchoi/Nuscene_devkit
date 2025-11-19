# nuScenes Devkit Experiments

3D object detection and lane detection experiments using nuScenes dataset.

## Features

- **Sensor Data Support**: Cameras (6x), LiDAR, Radar, IMU/GPS
- **Annotations**: 3D bounding boxes, 23 object categories, tracking IDs
- **Map Data**: HD semantic maps with lane geometry
- **Lightweight Models**: Simple PointNet for 3D detection testing

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

# Set data path
export NUSCENES_DATAROOT=/path/to/nuscenes
```

## Download Dataset

Download the nuScenes mini dataset (10 scenes, ~4GB):
https://www.nuscenes.org/nuscenes#download

Extract to `/data/nuscenes` or set `NUSCENES_DATAROOT`.

## Usage

### Explore Dataset Features

```bash
python experiments/explore.py
```

This will show:
- Dataset statistics
- Available sensors
- Object categories
- Visualize camera views

### Test 3D Object Detection

```bash
python experiments/test_inference.py
```

Simple PointNet model for testing:
- Lightweight architecture
- CPU/GPU support
- Checkpoint save/load

## Project Structure

```
.
├── Dockerfile              # Docker environment
├── pyproject.toml          # Dependencies (uv)
├── experiments/
│   ├── explore.py         # Data exploration
│   └── test_inference.py  # 3D detection test
└── data/                  # nuScenes dataset (mount/download)
```

## nuScenes Data Schema

- **Scene**: 20-second driving sequences
- **Sample**: Keyframes at 2Hz with synchronized sensors
- **Sample Data**: Individual sensor captures
- **Ego Pose**: Vehicle position/orientation
- **Annotations**: 3D bounding boxes with attributes

## Next Steps

1. Integrate real nuScenes LiDAR data
2. Implement lane detection using Map API
3. Train models on mini dataset
4. Add evaluation metrics
