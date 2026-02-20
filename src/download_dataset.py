"""Dataset Download Script

This script downloads the traffic light detection dataset from Roboflow.
It automatically handles authentication using the ROBOFLOW_API_KEY environment variable.
"""

import os
from dotenv import load_dotenv
from roboflow import Roboflow


# Load environment variables from .env file
load_dotenv()


def download_dataset(api_key, workspace, project, version, output_format="yolov8"):
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
        output_format: Export format (default: yolov8)
    
    Returns:
        Dataset object with location information
    """
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download(output_format)
    
    print(f"\u2713 Dataset downloaded successfully!")
    print(f"Location: {dataset.location}")
    print(f"Format: {output_format}")
    
    return dataset


if __name__ == "__main__":
    # Configuration from environment variables
    API_KEY = os.getenv("ROBOFLOW_API_KEY")
    WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "trafficsignaldetection-lti5q")
    PROJECT = os.getenv("ROBOFLOW_PROJECT", "traffic_signal_detection")
    VERSION = 1
    
    if not API_KEY:
        print("Error: ROBOFLOW_API_KEY not found in environment variables.")
        print("Please set up your .env file with: ROBOFLOW_API_KEY=your_api_key")
        exit(1)
    
    # Download dataset
    print(f"Downloading dataset from Roboflow...")
    print(f"Workspace: {WORKSPACE}")
    print(f"Project: {PROJECT}")
    print(f"Version: {VERSION}\n")
    
    dataset = download_dataset(API_KEY, WORKSPACE, PROJECT, VERSION)
    
    print(f"\nDataset is ready for training!")
    print(f"Use this path in your training script: {dataset.location}/data.yaml")
