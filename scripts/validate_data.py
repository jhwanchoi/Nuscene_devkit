#!/usr/bin/env python3
"""
Validate nuScenes dataset
"""

import os
import sys
from pathlib import Path
import json


def validate_nuscenes_data(data_root: Path, version: str = "v1.0-mini"):
    """
    Validate nuScenes dataset structure

    Args:
        data_root: Root directory of nuScenes data
        version: Dataset version to validate
    """
    print("="*60)
    print("Validating nuScenes Dataset")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Version: {version}")
    print()

    # Check if data root exists
    if not data_root.exists():
        print(f"✗ Data directory not found: {data_root}")
        print()
        print("Run 'just download-data' for download instructions")
        return False

    version_dir = data_root / version

    # Check version directory
    if not version_dir.exists():
        print(f"✗ Version directory not found: {version_dir}")
        print()
        print(f"Expected structure: {data_root}/{version}/")
        return False

    # Required JSON files
    required_files = [
        'scene.json',
        'sample.json',
        'sample_data.json',
        'sample_annotation.json',
        'instance.json',
        'category.json',
        'attribute.json',
        'visibility.json',
        'sensor.json',
        'calibrated_sensor.json',
        'ego_pose.json',
        'log.json',
        'map.json',
    ]

    missing_files = []
    valid_files = []

    for filename in required_files:
        filepath = version_dir / filename
        if filepath.exists():
            # Try to load JSON
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"✓ {filename:30s} ({len(data)} entries)")
                valid_files.append(filename)
            except Exception as e:
                print(f"✗ {filename:30s} (invalid JSON: {e})")
                missing_files.append(filename)
        else:
            print(f"✗ {filename:30s} (not found)")
            missing_files.append(filename)

    print()

    # Check data directories
    data_dirs = ['samples', 'sweeps']
    for dirname in data_dirs:
        dirpath = data_root / dirname
        if dirpath.exists():
            # Count subdirectories
            subdirs = [d for d in dirpath.iterdir() if d.is_dir()]
            print(f"✓ {dirname:20s} ({len(subdirs)} sensor types)")
        else:
            print(f"✗ {dirname:20s} (not found)")

    # Check maps
    maps_dir = data_root / 'maps'
    if maps_dir.exists():
        map_files = list(maps_dir.glob('*.png'))
        print(f"✓ maps                 ({len(map_files)} map images)")
    else:
        print(f"✗ maps                 (not found)")

    print()
    print("="*60)

    if missing_files:
        print(f"✗ Validation failed: {len(missing_files)} files missing")
        return False
    else:
        print("✓ Validation passed: Dataset is valid!")
        return True


def main():
    # Default data directory
    data_root = os.environ.get('NUSCENES_DATAROOT', 'data/nuscenes')
    data_dir = Path(data_root)

    success = validate_nuscenes_data(data_dir, version='v1.0-mini')

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
