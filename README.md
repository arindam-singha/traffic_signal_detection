# traffic_signal_detection
# YOLOv8 Traffic Light Detection

A production-ready deep learning project for real-time traffic light detection using YOLOv8 and Roboflow.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results](#results)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [License](#license)

## 🎯 Overview
Detects and classifies traffic light states (Red, Green, Yellow) in real-time from video/image feeds using YOLOv8 nano model. Achieves **[your_mAP]** mAP on validation set.

## ✨ Features
- **Real-time Detection**: GPU-accelerated inference
- **Multi-class Classification**: Red, Green, Yellow light detection
- **1280+ Annotated Images**: Curated Roboflow dataset
- **Production Ready**: Optimized for deployment
- **Easy Integration**: Python API + CLI tools

## 📊 Dataset
- **Source**: [Roboflow Dataset Link](https://roboflow.com/trafficsignaldetection-lti5q/traffic-signal-m5bdo-trspi)
- **Images**: 1,280 annotated images
- **Classes**: 3 (Red, Green, Yellow)
- **Train/Val/Test Split**: 70/15/15

## 🚀 Installation
```bash
git clone https://github.com/arindam-singha/traffic_signal_detection
cd traffic_signal_detection
pip install -r requirements.txt