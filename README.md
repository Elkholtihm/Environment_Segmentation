# Environment_Segmentation

## Overview
This project involves implementing, training, and evaluating three models: **U-Net**, **DeepLabv3+**, and **YOLO** on the Cityscape dataset. The aim is to compare these models in terms of performance metrics such as accuracy, mean Intersection over Union (mIoU), and F1 score.

---

## 📂 Repository Structure

### DeepLabv3plus/
```
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
```

### YOLO/
```
YOLO/  
│   ├── results                 # This contains the training files results  
│   ├── process_data.py         # This is a python file to process data (convert masks to txt annotation data)  
│   ├── test.py                 # Use model to test  
│   └── train.py                # Training script  
│   └── image-segmentation-yolo.pynb  # Implementation and training of the YOLO model  
```

### U-Net/
```
U-Net/  
│  
├── image-segmentation-unet.pynb  # Implementation and training of the U-Net model  
└── report_deepL.pdf              # Project report  
```

---

## Repository Content
This repository contains implementations of three image segmentation models: **DeepLabv3+**, **YOLO**, and **U-Net**, organized into three distinct folders as described above. Each folder includes scripts for model implementation, training, and evaluation.

---

## Collaboration
This project was created in collaboration with **[Anouar Bouzhar](https://github.com/anouarbouzhar)** and **[Zakariae Yahya](https://github.com/zakariaeyahya)**.

---

## Used Technologies
The project leverages the following technologies:

- ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
- ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

---

## Project Files and Links

- **Project Report**: [View Report](https://drive.google.com/file/d/1YLYr8bskG49I0BWl_W2FsBR5lS_qE6iD/view?usp=drive_link)  
  - all project files including training weights, datasets, other resources.
