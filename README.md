# ♻️ E-Waste Detection Using Deep Learning

An AI-powered image classification application that identifies electronic waste (e-waste) from images using **MobileNetV3Large**, TensorFlow, and Streamlit. The project aims to make electronic waste identification easier and encourage responsible waste segregation and recycling.

## 📌 Project Overview

Electronic waste includes discarded electronic devices and electrical equipment such as batteries, mobile phones, keyboards, printers, televisions, and circuit boards. Improper disposal of e-waste can cause environmental pollution and expose people to hazardous substances.

This project uses a deep learning model to classify an uploaded image into one of **10 e-waste categories**. It can support individuals, households, educational institutions, and waste-management initiatives in identifying electronic waste before sending it for appropriate collection or recycling.

🚀 Live Demo

Try the E-Waste Detection application here:

🔗 Live Application: https://e-wastes-detection.streamlit.app/

Upload an image of an electronic item to get its predicted e-waste category and confidence score.

## 🎯 Objectives

- Automatically classify electronic waste using image recognition.
- Simplify the identification of common discarded electronic items.
- Support waste segregation and responsible e-waste disposal.
- Demonstrate the practical application of transfer learning in environmental sustainability.
- Provide an accessible web interface for image-based predictions.

## ✨ Key Features

- **10-class classification:** Recognizes 10 categories of electronic waste.
- **Deep learning:** Uses the pretrained MobileNetV3Large architecture.
- **Transfer learning:** Reuses pretrained visual features and trains a task-specific classification head.
- **Image-based prediction:** Accepts an uploaded image for classification.
- **Confidence scores:** Displays the model's prediction confidence.
- **Interactive web application:** Provides a simple interface built with Streamlit.
- **Performance evaluation:** Includes accuracy, loss, precision, recall, F1-score, and confusion-matrix analysis.

## 🗂️ E-Waste Categories

The model classifies images into the following categories:

1. Battery
2. Keyboard
3. Microwave
4. Mobile
5. Mouse
6. PCB (Printed Circuit Board)
7. Player
8. Printer
9. Television
10. Washing Machine

## 🧠 Model Architecture

The project uses **MobileNetV3Large**, a convolutional neural network architecture designed for efficient image recognition.

The model follows this structure:

1. **MobileNetV3Large:** Extracts visual features from input images. The pretrained backbone is frozen during training.
2. **Global Average Pooling 2D:** Reduces the spatial feature maps to a feature vector.
3. **Dense layer:** Learns task-specific representations with 256 units.
4. **Dropout:** Helps reduce overfitting during training.
5. **Output layer:** Produces scores for the 10 e-waste classes.

### Model Summary

| Metric | Value |
|---|---:|
| Architecture | MobileNetV3Large |
| Input image size | 224 × 224 |
| Output classes | 10 |
| Total parameters | 3,742,112 |
| Trainable parameters | 248,586 |
| Non-trainable parameters | 2,996,352 |
| Training epochs | 10 |

## 📊 Model Performance

The model was trained for 10 epochs and evaluated on a separate test dataset containing 300 images, with 30 images per class.

### Training Results

| Epoch | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 84.62% | 96.00% |
| 2 | 95.58% | 95.33% |
| 3 | 97.87% | 96.33% |
| 4 | 98.58% | 95.67% |
| 5 | 99.00% | 96.67% |
| 6 | 99.58% | 96.67% |
| 7 | 99.54% | 96.33% |
| 8 | 100.00% | 96.67% |
| 9 | 99.87% | 96.67% |
| 10 | 99.75% | 97.00% |

### Test Evaluation

| Metric | Result |
|---|---:|
| Test Accuracy | **95.67%** |
| Test Loss | 0.1188 |
| Correct predictions | 287 / 300 |
| Macro-average Precision | 0.96 |
| Macro-average Recall | 0.96 |
| Macro-average F1-score | 0.96 |

### Class-wise Performance

| E-Waste Category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Battery | 0.96 | 0.90 | 0.93 |
| Keyboard | 1.00 | 0.97 | 0.98 |
| Microwave | 0.93 | 0.90 | 0.92 |
| Mobile | 1.00 | 1.00 | 1.00 |
| Mouse | 0.97 | 1.00 | 0.98 |
| PCB | 0.94 | 0.97 | 0.95 |
| Player | 0.93 | 0.90 | 0.92 |
| Printer | 1.00 | 1.00 | 1.00 |
| Television | 0.88 | 0.93 | 0.90 |
| Washing Machine | 0.97 | 1.00 | 0.98 |

*Note: These results reflect the supplied test dataset. Real-world performance may differ for images with unusual angles, poor lighting, damaged objects, cluttered backgrounds, or items outside the 10 trained classes.*

## 🛠️ Technology Stack

- **Programming language:** Python
- **Deep learning:** TensorFlow, Keras
- **Model architecture:** MobileNetV3Large
- **Numerical processing:** NumPy
- **Image processing:** Pillow (PIL)
- **Web application:** Streamlit
- **Development environment:** Google Colab, Visual Studio Code
- **Dataset:** Kaggle E-Waste Image Dataset
- **Version control:** Git and GitHub

## 📁 Project Structure

```text
E-Waste-Detection/
│
├── app.py
├── MobileNetV3_model.keras
├── class_names.json
├── requirements.txt
└── README.md
```

- `app.py` — Streamlit application for uploading images and displaying predictions.
- `MobileNetV3_model.keras` — Saved trained Keras model.
- `class_names.json` — Class labels in the model's training order.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.


## 🌍 Real-Life Applications

This project can support e-waste awareness and identification in several everyday situations.

### 1. Household waste segregation

People often have old chargers, keyboards, mobile phones, batteries, and other electronic items at home. The application can help identify the category of an item before it is separated from ordinary household waste.

### 2. Schools and colleges

Educational institutions can use the application during environmental-awareness activities, e-waste collection drives, and sustainability campaigns to demonstrate how AI can assist waste identification.

### 3. E-waste collection centres

Collection centres may use image classification as an initial sorting aid for common electronic items, helping staff organize items into categories for further handling.

### 4. Recycling facilities

The model could be integrated into a larger waste-sorting workflow to assist with preliminary visual classification. Industrial use would require additional testing, monitoring, and integration with appropriate sorting equipment.

### 5. Offices and IT departments

Organizations replacing old computers and peripherals can use the application to help categorize items such as keyboards, mice, printers, and circuit boards during electronic equipment collection.

### 6. Environmental-awareness campaigns

The application can help people recognize common types of e-waste and encourage them to use authorized collection and recycling channels instead of disposing of electronic equipment with general waste.

### 7. Smart waste-management systems

The model could serve as a component of a future AI-based waste-management system that combines cameras, classification models, and collection workflows.

**Important limitation:** This model identifies an item's visual category; it does not determine whether an item is recyclable, hazardous, reusable, or safe to handle. Batteries and other electronic waste should be handled according to local disposal guidance and taken to appropriate collection facilities. The application is an identification aid, not a replacement for professional waste assessment.

## 📦 Dataset

The project uses the E-Waste Image Dataset available on Kaggle.

**Dataset link:** https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset

The dataset contains images belonging to 10 electronic-waste categories and is organized into training, validation, and testing subsets.

The test evaluation described above used 300 images, with 30 images for each category.

## 🚀 Deployment

The Streamlit application can be deployed using **Streamlit Community Cloud**.

1. Push the application code, trained model, class labels, and `requirements.txt` to GitHub.
2. Visit https://share.streamlit.io/.
3. Sign in with GitHub and create a new app.
4. Select the repository, branch, and `app.py` entry point.
5. Deploy the application and test it using sample images.

The final application URL can be added here after deployment:

**Live Demo:** `(https://e-wastes-detection.streamlit.app/)`

## 🔮 Future Improvements

- Expand the dataset to include more e-waste categories and real-world images.
- Improve classification performance for visually similar items.
- Add Grad-CAM or another explainability technique to visualize relevant image regions.
- Explore object detection for identifying multiple electronic items in one image.
- Add multilingual support for wider accessibility.
- Integrate location-based information about authorized e-waste collection centres.
- Explore edge deployment for use on low-resource devices.
- Evaluate the model on diverse real-world images before practical or large-scale deployment.



## 📄 Disclaimer

This project is developed for educational and environmental-awareness purposes. Predictions are based on visual patterns learned from the training dataset and should not be treated as a definitive assessment of an item's composition, hazard level, or recycling eligibility.

## 🌱 Conclusion

The E-Waste Detection project demonstrates how deep learning and transfer learning can be applied to an environmental challenge. By providing an accessible way to identify common electronic-waste categories, it can support awareness, preliminary sorting, and responsible disposal practices.

**Let's use AI not only to solve technical problems, but also to build a more sustainable future.** ♻️
