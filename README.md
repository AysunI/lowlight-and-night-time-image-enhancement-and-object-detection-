# lowlight-and-night-time-image-enhancement and object detection
This repository contains the implementation and experiments for my MSc dissertation, which investigates how low-light image enhancement affects object detection performance.

# Getting started

# Repository Structure


Faster-RCNN/ – Faster R-CNN training and evaluation.

YOLOv8_original/ – Baseline YOLOv8 on raw images.

YOLOv8_zerodce/ – YOLOv8 on Zero-DCE++ enhanced images.

YOLOv8_nafnet/ – YOLOv8 on NAFNet enhanced images.

Zero-DCE++/ – Zero-DCE++ enhancement method.

NafNet/ – NAFNet enhancement method.

EDA.ipynb – Exploratory data analysis.

Scripts (.py files) – dataset preparation, annotation conversion, splitting, and consistency checks.


# Dataset
The experiments use the ExDark dataset (7,363 low-light images, 12 classes).
Due to size constraints, the dataset and model results are hosted separately on [https://drive.google.com/drive/folders/1o9iWtsA4BxezOBXrv3Z6u5NjYpCpNUsb?usp=drive_link].

# Usage

Clone repo and install dependencies (pip install -r requirements.txt).
Download dataset from Google Drive.
Run enhancement methods (Zero-DCE++, NAFNet).
Train/evaluate detectors (Faster R-CNN, YOLOv8).


# Requirements

Python 3.8.20
PyTorch ≥ 1.12
CUDA 11.8


# Results

Enhancement improves visual interpretability.
Detection accuracy does not always improve, sometimes decreases.
Shows trade-off between human- and machine-perceived quality.


# Acknowledgement
Developed as part of my MSc dissertation at the University of Birmingham.
