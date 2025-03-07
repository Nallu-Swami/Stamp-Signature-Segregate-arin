# Stamp Detection with YOLOv12

## Overview
This project implements a stamp detection system using YOLOv12 object detection model. The system is designed to identify and locate stamps in document images with high accuracy, enabling automated document processing applications.

## Features
- **High Performance Detection**: Trained YOLOv12 model achieves mAP50 of 0.825 and mAP50-95 of 0.448
- **Complete Training Pipeline**: Includes data preparation, model training, evaluation, and inference
- **Streamlit Web Application**: User-friendly interface for uploading and analyzing documents

## Dataset
The dataset comprises document images containing stamps, created and managed through Roboflow:
- 1,422 training images
- 82 validation images
- Test set for model evaluation

## Model Performance
- **mAP 50:95**: 0.448
- **mAP 50**: 0.825
- **mAP 75**: 0.418

## Project Structure
```
project_root/
├── Colab-NoteBooks/           # Jupyter notebooks for model training
│   └── stamp-detect-crop.ipynb # Main training notebook
├── webapp/                    # Streamlit application files
│   └── streamlit-yolo-app.txt # Web app structure
└── lightning_logs/           # Training logs and checkpoints
```

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/stamp-detection.git
cd stamp-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Training
```python
from ultralytics import YOLO

# Load the model
model = YOLO('yolov12s.yaml')

# Train the model
results = model.train(data='path/to/data.yaml', epochs=100)
```

### Inference
```python
import supervision as sv
from ultralytics import YOLO

# Load the trained model
model = YOLO('path/to/best.pt')

# Run inference on an image
results = model(image_path)[0]

# Process detections
detections = sv.Detections.from_ultralytics(results).with_nms()

# Visualize results
box_annotator = sv.BoxAnnotator()
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
```

### Web Application
Run the Streamlit app:
```bash
cd webapp
streamlit run main.py
```

## Requirements
- Python 3.8+
- PyTorch
- Ultralytics YOLOv12
- Roboflow
- Supervision
- Streamlit (for web app)

## License
[Specify your license]