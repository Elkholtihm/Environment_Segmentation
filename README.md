# Environment_Segmentation

## Overview
This project include implementing, training and evaluating three models U-net, DeepLabv3+ and yolo on cityscape dataset.

## repository structure 
DeepLabv3plus/  
│  
├── model/  
│   ├── deeplabv3plus.py        # Implementation of the model architecture  
│   ├── metrics.py              # Metrics calculation (Accuracy, mIoU, F1 Score, etc.)  
│   ├── prediction.py           # Prediction functions and visualization  
│   └── train.py                # Training script  
│  
└── processing/  
    └── data_processing.py      # Data preprocessing script  

YOLO/  
│  


U-Net/  
│  
└── image-segmentation-unet.pynb  # Implementation and training of the U-Net model  



# Repository Content  

This repository contains the implementations of three image segmentation models: **DeepLabv3+**, **YOLO**, and **U-Net**, organized into three distinct folders. Here's an overview:  

---

## 1. DeepLabv3+  
This folder is structured as follows:  
- **model/**: Contains the modules related to the DeepLabv3+ model.  
  - `deeplabv3plus.py`: Implementation of the DeepLabv3+ model architecture (from scratch and imported).  
  - `metrics.py`: Implementation of three key metrics:  
    - **Accuracy**  
    - **Mean Intersection over Union (mIoU)**  
    - **Recall**, **F1 Score**, and **Precision**  
  - `prediction.py`: Functions for prediction and result visualization.  
  - `train.py`: Training function to fine-tune the model with data.  

- **processing/**: Contains scripts for data preprocessing.  
  - `data_processing.py`: Functions to prepare and preprocess data before training.  

---

## 2. YOLO  

---

## 3. U-Net  
This folder contains:  
- `image-segmentation-unet.pynb`: Jupyter Notebook script implementing and training the U-Net model on the Cityscape dataset.  
