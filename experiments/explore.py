"""
Explore nuScenes dataset features
- Sensor data (cameras, LiDAR, radar)
- 3D bounding boxes
- Lane/map information
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mlflow


class NuScenesExplorer:
    """Simple nuScenes data explorer"""

    def __init__(self, dataroot: str, version: str = "v1.0-mini"):
        """
        Args:
            dataroot: Path to nuScenes data directory
            version: Dataset version (e.g., 'v1.0-mini', 'v1.0-trainval')
        """
        self.dataroot = Path(dataroot)
        self.version = version
        self.data_path = self.dataroot / version

        # Load metadata
        self.scene = self._load_table('scene')
        self.sample = self._load_table('sample')
        self.sample_data = self._load_table('sample_data')
        self.sensor = self._load_table('sensor')
        self.calibrated_sensor = self._load_table('calibrated_sensor')
        self.ego_pose = self._load_table('ego_pose')
        self.sample_annotation = self._load_table('sample_annotation')
        self.category = self._load_table('category')
        self.attribute = self._load_table('attribute')
        self.map = self._load_table('map')

        print(f"Loaded nuScenes {version}")
        print(f"  Scenes: {len(self.scene)}")
        print(f"  Samples: {len(self.sample)}")
        print(f"  Annotations: {len(self.sample_annotation)}")

    def _load_table(self, table_name: str) -> List[Dict]:
        """Load a metadata table"""
        table_path = self.data_path / f"{table_name}.json"
        if not table_path.exists():
            print(f"Warning: {table_name}.json not found")
            return []

        with open(table_path, 'r') as f:
            data = json.load(f)
        return data

    def get_sensor_types(self) -> Dict[str, int]:
        """Get available sensor types and their counts"""
        sensor_types = {}
        for sensor in self.sensor:
            modality = sensor['modality']
            sensor_types[modality] = sensor_types.get(modality, 0) + 1
        return sensor_types

    def get_camera_names(self) -> List[str]:
        """Get all camera sensor names"""
        cameras = []
        for sensor in self.sensor:
            if sensor['modality'] == 'camera':
                cameras.append(sensor['channel'])
        return cameras

    def get_object_categories(self) -> List[str]:
        """Get all object categories"""
        return [cat['name'] for cat in self.category]

    def get_sample_sensors(self, sample_token: str) -> Dict[str, str]:
        """Get all sensor data for a sample"""
        sample = next((s for s in self.sample if s['token'] == sample_token), None)
        if not sample:
            return {}

        sensor_data = {}
        for key in sample['data'].keys():
            sensor_data[key] = sample['data'][key]
        return sensor_data

    def get_annotations(self, sample_token: str) -> List[Dict]:
        """Get all annotations for a sample"""
        annotations = []
        for ann in self.sample_annotation:
            if ann['sample_token'] == sample_token:
                annotations.append(ann)
        return annotations

    def visualize_sample(self, sample_idx: int = 0):
        """Visualize a sample with all camera views"""
        if sample_idx >= len(self.sample):
            print(f"Sample index {sample_idx} out of range")
            return

        sample = self.sample[sample_idx]
        sample_token = sample['token']

        # Get camera data
        cameras = self.get_camera_names()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sample {sample_idx} - All Camera Views')

        for idx, cam_name in enumerate(cameras[:6]):
            ax = axes[idx // 3, idx % 3]

            # Find sample_data for this camera
            cam_token = sample['data'].get(cam_name)
            if not cam_token:
                ax.axis('off')
                continue

            # Get image path
            cam_data = next((sd for sd in self.sample_data if sd['token'] == cam_token), None)
            if not cam_data:
                ax.axis('off')
                continue

            img_path = self.dataroot / cam_data['filename']
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(cam_name)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'{cam_name}\n(Image not found)',
                       ha='center', va='center')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig('sample_cameras.png', dpi=150, bbox_inches='tight')
        print(f"Saved visualization to sample_cameras.png")
        plt.close()

    def print_summary(self):
        """Print dataset summary"""
        print("\n" + "="*60)
        print("nuScenes Dataset Summary")
        print("="*60)

        print(f"\nDataset Version: {self.version}")
        print(f"Number of Scenes: {len(self.scene)}")
        print(f"Number of Samples: {len(self.sample)}")
        print(f"Number of Annotations: {len(self.sample_annotation)}")

        print("\nSensor Types:")
        for sensor_type, count in self.get_sensor_types().items():
            print(f"  {sensor_type}: {count}")

        print("\nCamera Sensors:")
        for cam in self.get_camera_names():
            print(f"  - {cam}")

        print("\nObject Categories:")
        for cat in self.get_object_categories():
            print(f"  - {cat}")

        print("\nMap Data:")
        print(f"  Available maps: {len(self.map)}")

        print("="*60 + "\n")


def main():
    # Setup MLflow
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5001')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("nuscenes-data-exploration")

    # Example usage
    dataroot = os.environ.get('NUSCENES_DATAROOT', '/data/nuscenes')

    if not os.path.exists(dataroot):
        print(f"Data directory not found: {dataroot}")
        print("Please set NUSCENES_DATAROOT environment variable or mount data volume")
        return

    with mlflow.start_run(run_name="explore_dataset"):
        explorer = NuScenesExplorer(dataroot, version='v1.0-mini')

        # Log dataset statistics
        mlflow.log_param("dataset_version", explorer.version)
        mlflow.log_metric("num_scenes", len(explorer.scene))
        mlflow.log_metric("num_samples", len(explorer.sample))
        mlflow.log_metric("num_annotations", len(explorer.sample_annotation))
        mlflow.log_metric("num_categories", len(explorer.get_object_categories()))

        # Log sensor info
        sensor_types = explorer.get_sensor_types()
        for sensor_type, count in sensor_types.items():
            mlflow.log_metric(f"num_{sensor_type}_sensors", count)

        explorer.print_summary()

        # Visualize first sample
        if len(explorer.sample) > 0:
            print("Visualizing first sample...")
            explorer.visualize_sample(0)

            # Log visualization
            if os.path.exists('sample_cameras.png'):
                mlflow.log_artifact('sample_cameras.png')
                print(f"Logged visualization to MLflow: {mlflow_uri}")


if __name__ == "__main__":
    main()
