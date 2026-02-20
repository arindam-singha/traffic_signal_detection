# traffic_signal_detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-brightgreen)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**YOLOv8 Real-Time Traffic Light Detection**

A production-ready deep learning project for detecting and classifying traffic light states (Red, Green, Yellow) using YOLOv8 and Roboflow.

[Features](#-features) • [Dataset](#-dataset) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Results](#-results)

</div>

---

## 🎯 Overview

This project implements a state-of-the-art traffic light detection system using YOLOv8 nano model, optimized for real-time inference on edge devices and GPUs. It detects and classifies traffic light states in images and video streams with high accuracy.

**Perfect for:**
- Autonomous vehicle perception pipelines
- Smart traffic management systems  
- Computer vision portfolio projects
- Edge device deployment (Jetson, TPU)

---

## ✨ Features

- **🚀 Real-time Detection**: GPU-accelerated inference with <100ms latency
- **🎨 Multi-class Classification**: Detects Red, Green, and Yellow lights
- **📊 1,280+ Annotated Images**: High-quality curated Roboflow dataset
- **⚡ Lightweight Model**: YOLOv8 Nano for edge deployment
- **🔧 Production Ready**: Optimized with preprocessing and post-processing
- **📦 Modern Stack**: Uses `uv` for fast, deterministic dependency management
- **🧪 Reproducible**: Easy setup and consistent results across machines

---

## 📊 Dataset

- **Source**: [Roboflow Traffic Signal Dataset](https://roboflow.com/trafficsignaldetection-lti5q/traffic-signal-m5bdo-trspi)
- **Total Images**: 1,280 annotated samples
- **Classes**: 3 (Red Light, Green Light, Yellow Light)
- **Split**: 
  - Train: 896 images (70%)
  - Val: 192 images (15%)
  - Test: 192 images (15%)
- **Annotation Format**: YOLOv8 format (normalized bounding boxes)
- **Resolution**: Variable (auto-resized to 640×640 during training)

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.10 or higher
- **CUDA** (optional): For GPU acceleration
- **Git**: For cloning the repository

### Using `uv` (Recommended)

`uv` is a blazing-fast Python package installer that provides deterministic builds and simpler dependency management.

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Clone and setup:**
```bash
git clone https://github.com/arindam-singha/traffic_signal_detection.git
cd traffic_signal_detection

# Create virtual environment and sync dependencies
uv sync
```

**Or install specific packages:**
```bash
uv pip install ultralytics roboflow opencv-python torch torchvision
```

---

## ⚡ Quick Start

### 1. Download Dataset

```bash
python download_dataset.py
```

This downloads the latest version from Roboflow to `traffic_signal_detection-1/` folder.

### 2. Train the Model

```bash
python training.py
```

Key training parameters:
- **Model**: YOLOv8 Nano
- **Epochs**: 10 (adjust as needed)
- **Image Size**: 512×512
- **Batch Size**: Auto (depends on GPU memory)

Training outputs are saved to `runs/detect/train/`

### 3. Run Inference

**On a single image:**
```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
results = model.predict(source='path/to/image.jpg', conf=0.25)
```

**On a video:**
```python
results = model.predict(source='path/to/video.mp4', conf=0.25)
```

**Visualize results:**
```python
for r in results:
    r.show()
```

---

## 📈 Results & Metrics

| Metric | Value |
|--------|-------|
| **mAP@0.5** | - |
| **mAP@0.5:0.95** | - |
| **Precision** | - |
| **Recall** | - |
| **Inference Time** | ~30ms (GPU) |

*Note: Update these metrics after first training run*

### Sample Detections

Training results including loss curves and validation metrics are saved to:
```
runs/detect/train/
├── weights/
│   ├── best.pt          # Best model
│   └── last.pt          # Last checkpoint
├── results.png          # Training curves
└── confusion_matrix.png # Confusion matrix
```

---

## 📁 Project Structure

```
traffic_signal_detection/
├── download_dataset.py          # Roboflow dataset downloader
├── download_dataset.ipynb       # Jupyter version of downloader
├── training.py                  # Training script
├── training.ipynb               # Jupyter notebook for interactive training
├── README.md                    # This file
├── LICENSE                      # Apache 2.0 License
│
└── runs/                        # Training outputs (gitignored)
    └── detect/
        └── train/
            ├── weights/
            ├── results.png
            └── confusion_matrix.png
```

---

## 🔧 Usage Guide

### Downloading Dataset Programmatically

```python
from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("trafficsignaldetection-lti5q").project("traffic_signal_detection")
dataset = project.version(1).download("yolov8")

print(f"Dataset location: {dataset.location}")
```

### Training with Custom Parameters

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load nano model

results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,           # Early stopping
    device=0,              # GPU ID
    save=True,
    project='runs/detect',
    name='train'
)
```

### Validation

```python
metrics = model.val(data='path/to/data.yaml')
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

---

## 🛠️ Environment Setup

### Using .env File

Create `.env` in project root:
```
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=trafficsignaldetection-lti5q
ROBOFLOW_PROJECT=traffic_signal_detection
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
```

---

## 🎓 Learning Resources

- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Roboflow Docs**: https://docs.roboflow.com/
- **uv Documentation**: https://astral.sh/uv/
- **Traffic Light Detection**: Research papers on vehicle perception

---

## 📝 Notebook Examples

The project includes Jupyter notebooks for interactive exploration:

- **`download_dataset.ipynb`**: Step-by-step dataset download with visualization
- **`training.ipynb`**: Full training pipeline with metrics and visualization

Launch Jupyter:
```bash
uv run jupyter notebook
```

---

## ⚠️ Important Notes

1. **API Key Security**: Never commit API keys. Use environment variables or `.env` files
2. **GPU Requirements**: For faster training, CUDA-enabled GPU is recommended
3. **Dataset Path**: The Roboflow downloader creates a `traffic_signal_detection-1/` folder
4. **Model Weights**: Pre-trained YOLOv8 weights (~6MB) are downloaded automatically on first run

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Arindam Singha**
- GitHub: [@arindam-singha](https://github.com/arindam-singha)
- Location: West Bengal, India
- Background: Lead Robotics Engineer | AI/ML Researcher | PhD in Robotics

---

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the excellent YOLO implementation
- **Roboflow**: For dataset management and preprocessing tools
- **Dataset Contributors**: Community for annotating traffic light images
- **uv**: Astral for fast Python package management

---

## 📞 Support & Questions

For issues or questions:
- Open an [Issue](https://github.com/arindam-singha/traffic_signal_detection/issues)
- Check [Discussions](https://github.com/arindam-singha/traffic_signal_detection/discussions)
- Review [YOLOv8 FAQ](https://docs.ultralytics.com/help/faq/)

---

<div align="center">

Made with ❤️ for autonomous vehicles & computer vision

⭐ If this helps, please star the repository!

</div>
