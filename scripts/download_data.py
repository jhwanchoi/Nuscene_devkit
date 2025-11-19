#!/usr/bin/env python3
"""
Download nuScenes mini dataset
"""

import os
import sys
from pathlib import Path
import urllib.request
import tarfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_nuscenes_mini(data_dir: Path):
    """
    Download nuScenes mini dataset

    Note: This is a placeholder. The actual nuScenes dataset requires:
    1. Account registration at https://www.nuscenes.org/
    2. Acceptance of terms
    3. Manual download or use of official download script

    For now, this script provides instructions.
    """
    print("="*60)
    print("nuScenes Mini Dataset Download")
    print("="*60)
    print()
    print("Due to licensing requirements, the nuScenes dataset must be downloaded manually.")
    print()
    print("Steps:")
    print("1. Go to: https://www.nuscenes.org/nuscenes#download")
    print("2. Register/Login")
    print("3. Download 'v1.0-mini' dataset (~4GB)")
    print("   - Metadata: v1.0-mini.tgz")
    print("   - File blobs: Trainval part 1-10 (mini uses subset)")
    print()
    print(f"4. Extract to: {data_dir.absolute()}/")
    print()
    print("Expected structure:")
    print(f"  {data_dir}/")
    print("    └── v1.0-mini/")
    print("        ├── maps/")
    print("        ├── samples/")
    print("        ├── sweeps/")
    print("        ├── v1.0-mini/")
    print("        │   ├── scene.json")
    print("        │   ├── sample.json")
    print("        │   └── ...")
    print()
    print("Alternative: Use nuScenes devkit download script")
    print("  pip install nuscenes-devkit")
    print("  python -m nuscenes.download --version v1.0-mini --dataroot ./data/nuscenes")
    print()
    print("="*60)

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Created data directory: {data_dir}")


def main():
    # Default data directory
    data_root = os.environ.get('NUSCENES_DATAROOT', 'data/nuscenes')
    data_dir = Path(data_root)

    print(f"Data directory: {data_dir}")
    print()

    download_nuscenes_mini(data_dir)


if __name__ == "__main__":
    main()
