"""YOLOv8 Traffic Light Detection Training Script

This script trains a YOLOv8 model for traffic light detection using Roboflow dataset.
It includes data loading, model training, validation, and inference capabilities.
"""

import os
import cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import yaml


# Load environment variables from .env file
load_dotenv()


def download_dataset(api_key, workspace, project, version):
    """Download dataset from Roboflow"""
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8")
    print(f"✓ Dataset downloaded to: {dataset.location}")
    return dataset


def load_data_config(config_path):
    """Load YOLO dataset configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(data_yaml, epochs=100, imgsz=640, batch=16, device=0):
    """Train YOLOv8 model"""
    model = YOLO('yolov8n.pt')  # Load nano model
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        save=True,
        patience=20,  # Early stopping
        project='runs/detect',
        name='train'
    )
    
    print("✓ Training completed!")
    return model, results


def validate_model(model, data_yaml):
    """Validate model performance"""
    metrics = model.val(data=data_yaml)
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    return metrics


def predict(model, source, conf=0.25):
    """Run inference on image or video"""
    results = model.predict(source=source, conf=conf)
    return results


if __name__ == "__main__":
    # Configuration
    API_KEY = os.getenv("ROBOFLOW_API_KEY")
    WORKSPACE = "trafficsignaldetection-lti5q"
    PROJECT = "traffic_signal_detection"
    VERSION = 1
    
    # Download dataset
    dataset = download_dataset(API_KEY, WORKSPACE, PROJECT, VERSION)
    
    # Train model
    model, results = train_model(
        data_yaml=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0
    )
    
    # Validate
    metrics = validate_model(model, f"{dataset.location}/data.yaml")
    
    print("\n✓ Training pipeline completed successfully!")
