# Minecraftors StampExtractor

## AIQoD Hackathon '25 | Problem Statement 3

A comprehensive solution for extracting overlapping stamps and signatures from documents using YOLO object detection. This project is the result of collective efforts by Shivam, Arin, Varad, and Yasir for the AIQoD hackathon organized at VIT Chennai.

## 📋 Overview

Minecraftors StampExtractor is a document processing tool designed to:

- Upload PNG documents for analysis
- Process images using a pre-trained YOLO model
- Annotate detected objects with bounding boxes and labels
- Extract detected objects as separate images
- Provide a simple and intuitive web interface


![Workflow Diagram](image_assests/image.png)

## 🚀 Features

- **High Performance Detection**: Trained YOLOv12 model achieves mAP50 of 0.825 and mAP50-95 of 0.448
- **Complete Training Pipeline**: Includes data preparation, model training, evaluation, and inference
- **Streamlit Web Application**: User-friendly interface for uploading and analyzing documents

## 📊 Model Performance

- **mAP 50:95**: 0.448
- **mAP 50**: 0.825
- **mAP 75**: 0.418
### Final Model Metrics:
> Test Loss: 0.1441  
> Test IoU: 0.6779  
> Test Accuracy: 0.7385  

## 🔄 Workflow

### Step 1: Document Upload Interface
![Upload Interface](image_assests/WhatsApp%20Image%202025-03-07%20at%2010.36.55.jpeg)

### Step 2: Document Processing
![Upload Interface](image_assests/WhatsApp%20Image%202025-03-07%20at%2011.16.49.jpeg)

### Step 3: Semantic Segmentation for Precise Differentiation
![img](image_assests/4.jpg)


### Step 4 : Semantic Segmentaion and Signature Removal Using  Color Segmentation and Masking
![img](image_assests/3.jpg)

## Workflow Demonstration

### Step 1.1: Input Document
![Input Document](image_assests/Screenshot%202025-03-06%20234344.png)

### Step 1.2: Stamp and Sign Recognition
![Recognition Results](image_assests/output.png)

### Step 1.3: ROI Extraction
![Extracted Object 1](image_assests/object_0.png)
![Extracted Object 2](image_assests/object_3.png)

### Step 2: Signature Segmentation
![Signature Segmentation 1](image_assests/Screenshot%202025-03-07%20104506.png)
![Signature Segmentation 2](image_assests/Screenshot%202025-03-07%20104528.png)

### Step 3.1: Signature Seperation
![Signature_sep](image_assests/WhatsApp%20Image%202025-03-07%20at%2013.27.23.jpeg)  

Signature Heatmap  
![Signature_Heatmap](image_assests/WhatsApp%20Image%202025-03-07%20at%2013.34.00.jpeg)


### Step 3.2: Stamp Separation
![Stamp Separation](image_assests/Screenshot%202025-03-07%20104815.png)
![Signatre_Seperation](image_assests/tert2.JPG)
![Stamp_sep](image_assests/WhatsApp%20Image%202025-03-07%20at%2013.28.09.jpeg)

## 📁 Dataset

The dataset comprises document images containing stamps, created and managed through Roboflow:
- 1,422 training images
- 82 validation images
- Test set for model evaluation

## Metrics

![training_n_validation](image_assests/WhatsApp%20Image%202025-03-07%20at%2011.05.15.jpeg)
![final_metrics](image_assests/WhatsApp%20Image%202025-03-07%20at%2011.05.42.jpeg)


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👨‍💻 Authors

- **Kumar Shivam** *22BCE1804*
- **Arin Yadav** *22BCE5043*
- **Varad Dherange** *22BCE1298*
- **Yasir Ahmad** *22MIA1064*

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
