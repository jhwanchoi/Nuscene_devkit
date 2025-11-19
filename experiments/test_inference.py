"""
Simple 3D object detection inference test
Uses a lightweight model for testing purposes
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class SimplePointNet(nn.Module):
    """Simplified PointNet for 3D object detection testing"""

    def __init__(self, num_classes: int = 10, num_points: int = 1024):
        super().__init__()
        self.num_points = num_points

        # Feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Classification head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud
        Returns:
            class_logits: (B, num_classes)
        """
        # Transpose to (B, 3, N)
        x = x.transpose(2, 1)

        # Feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = torch.max(x, 2)[0]

        # Classification
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class Simple3DDetector:
    """Simple 3D object detector for testing"""

    def __init__(self, num_classes: int = 10, device: str = 'cpu'):
        self.device = device
        self.model = SimplePointNet(num_classes=num_classes).to(device)
        self.model.eval()

        # nuScenes categories (simplified)
        self.categories = [
            'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
            'pedestrian', 'bicycle', 'motorcycle', 'traffic_cone', 'barrier'
        ]

        print(f"Initialized Simple3DDetector on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def preprocess_pointcloud(self, points: np.ndarray, num_points: int = 1024) -> torch.Tensor:
        """
        Preprocess point cloud data

        Args:
            points: (N, 3) numpy array
            num_points: number of points to sample

        Returns:
            (1, num_points, 3) tensor
        """
        # Random sampling
        if points.shape[0] > num_points:
            indices = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < num_points:
            # Pad with zeros
            padding = np.zeros((num_points - points.shape[0], 3))
            points = np.vstack([points, padding])

        # Normalize
        points = points - np.mean(points, axis=0)
        points = points / (np.max(np.abs(points)) + 1e-6)

        return torch.FloatTensor(points).unsqueeze(0)

    def predict(self, points: np.ndarray) -> Tuple[str, float]:
        """
        Run inference on point cloud

        Args:
            points: (N, 3) numpy array

        Returns:
            predicted_class: str
            confidence: float
        """
        # Preprocess
        x = self.preprocess_pointcloud(points).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        predicted_class = self.categories[pred_idx]
        return predicted_class, confidence

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'categories': self.categories,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.categories = checkpoint['categories']
        print(f"Loaded checkpoint from {path}")


def test_random_pointcloud():
    """Test with random point cloud"""
    print("\n" + "="*60)
    print("Testing Simple 3D Object Detector")
    print("="*60)

    # Create detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    detector = Simple3DDetector(num_classes=10, device=device)

    # Generate random point cloud (simulate LiDAR data)
    print("\nGenerating random point cloud...")
    num_points = 2048
    points = np.random.randn(num_points, 3).astype(np.float32)

    # Run inference
    print("Running inference...")
    pred_class, confidence = detector.predict(points)

    print(f"\nResults:")
    print(f"  Predicted class: {pred_class}")
    print(f"  Confidence: {confidence:.4f}")

    # Test checkpoint save/load
    print("\nTesting checkpoint save/load...")
    checkpoint_path = "simple_detector.pth"
    detector.save_checkpoint(checkpoint_path)

    # Create new detector and load
    new_detector = Simple3DDetector(num_classes=10, device=device)
    new_detector.load_checkpoint(checkpoint_path)

    # Verify same prediction
    pred_class2, confidence2 = new_detector.predict(points)
    assert pred_class == pred_class2
    assert abs(confidence - confidence2) < 1e-5
    print("Checkpoint save/load test passed!")

    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60 + "\n")


def main():
    """Main test function"""
    test_random_pointcloud()

    print("Next steps:")
    print("1. Install nuScenes devkit: pip install nuscenes-devkit")
    print("2. Download mini dataset from nuScenes website")
    print("3. Integrate with real nuScenes LiDAR data")
    print("4. Train on real data for actual detection")


if __name__ == "__main__":
    main()
