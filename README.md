
# 🚗 License Plate Detection and Recognition Using Deep Learning

This repository contains two end-to-end deep learning pipelines for detecting and recognizing vehicle license plates from images using modern computer vision techniques.

## 📌 Project Overview

The goal is to compare and evaluate two approaches for license plate recognition:

1. **CNN-Based Custom Detector**  
   A deep convolutional neural network inspired by ResNeXt and Feature Pyramid Networks (FPN) to detect license plates.

2. **YOLOv8 + EasyOCR Pipeline**  
   A modular pipeline combining the YOLOv8 object detector (Ultralytics) with EasyOCR to detect and read license plate text efficiently.

---

## 🧠 Notebooks

| Notebook | Description |
|---------|-------------|
| `CNN_License_Plate.ipynb` | Custom-built CNN for license plate bounding box regression using PyTorch and a combined IoU + MSE loss. |
| `detection-ocr-lpr.ipynb` | Uses YOLOv8 for license plate detection and EasyOCR for text recognition. Highly modular and suitable for real-world deployment. |

---

## 📁 Dataset

Both notebooks utilize the **Large License Plate Dataset** available on [Kaggle](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset).

Dataset structure:
```
/images/train
/images/val
/images/test
/labels/train
/labels/val
/labels/test
```

---

## ⚙️ Setup Instructions

```bash
# Clone this repository
git clone https://github.com/your-username/license-plate-recognition.git
cd license-plate-recognition

# (Optional) Set up virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Required Libraries

- PyTorch
- Ultralytics YOLO
- EasyOCR
- OpenCV
- tqdm
- matplotlib
- numpy
- pandas

> Note: `ultralytics`, `easyocr`, and `opencv-python-headless` are automatically installed in the notebook.

---

## 🚀 How to Run

### 🔧 CNN-Based Detector

```python
# Run the notebook step by step
jupyter notebook CNN_License_Plate.ipynb
```

### 🔧 YOLO + EasyOCR Pipeline

```python
# Run the notebook step by step
jupyter notebook detection-ocr-lpr.ipynb
```

---

## 📊 Results Summary

| Metric                | CNN-Based Detector         | YOLO + EasyOCR Pipeline     |
|----------------------|----------------------------|-----------------------------|
| Detection Accuracy   | ~85% IoU                   | >90% on clean datasets      |
| Text Recognition     | Not included               | >92% precision (EasyOCR)    |
| Inference Speed      | Moderate                   | Fast                        |
| Deployment Ready     | Requires integration       | Plug-and-play               |

---

## 📸 Visual Output

- Ground truth and predicted bounding boxes are visualized.
- OCR text is rendered on detected plates in the YOLO pipeline.

---

## 🧾 References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [Feature Pyramid Networks - CVPR 2017](https://arxiv.org/abs/1612.03144)
- [ResNeXt Paper](https://arxiv.org/abs/1611.05431)
- [PyTorch Official Site](https://pytorch.org)
- [OpenCV Docs](https://docs.opencv.org)

---

## 🙌 Contributing

Contributions are welcome! If you want to improve the models, optimize training, or expand the dataset, feel free to submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
