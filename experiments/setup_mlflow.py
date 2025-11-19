"""
Setup MLflow experiments for nuScenes project
"""

import os
import mlflow


def setup_experiments(tracking_uri: str = "http://localhost:5001"):
    """
    Setup MLflow experiments

    Args:
        tracking_uri: MLflow tracking server URI
    """
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Connected to MLflow: {tracking_uri}")

    # Create experiments
    experiments = [
        {
            "name": "nuscenes-data-exploration",
            "description": "Dataset exploration and visualization experiments"
        },
        {
            "name": "nuscenes-3d-detection",
            "description": "3D object detection model experiments"
        },
        {
            "name": "nuscenes-lane-detection",
            "description": "Lane detection experiments"
        },
    ]

    for exp in experiments:
        try:
            exp_id = mlflow.create_experiment(
                name=exp["name"],
                tags={"project": "nuscenes", "description": exp["description"]}
            )
            print(f"Created experiment: {exp['name']} (ID: {exp_id})")
        except Exception as e:
            # Experiment might already exist
            existing_exp = mlflow.get_experiment_by_name(exp["name"])
            if existing_exp:
                print(f"Experiment already exists: {exp['name']} (ID: {existing_exp.experiment_id})")
            else:
                print(f"Error creating experiment {exp['name']}: {e}")

    print(f"\nSetup complete! View experiments at: {tracking_uri}")


def main():
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5001')
    setup_experiments(tracking_uri)


if __name__ == "__main__":
    main()
