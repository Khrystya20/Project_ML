# Dolphin Fin Recognition Project

## Project Description

This project provides a Machine Learning solution for recognizing dolphin fins. 
The main goal is to develop a tool that automates the process of identifying Dolphin #1 from a dataset of various dolphin images.

### Key Features
- **Binary Classification**: Identify if an image belongs to Dolphin #1 or not.
- **Architecture**: Built on a fine-tuned ResNet18 model.
- **Automated Workflow**: Includes data preprocessing, model training, evaluation, and prediction.

---

## Data Overview

- **Dataset**:
  - 41 images of Dolphin #1.
  - 1171 images of other dolphins.
- **Data Preprocessing**:
  - Images are stored in separate directories: `dataset/1_dolphin` and `dataset/other_dolphins`.
  - Dataset is split into:
    - **Training Set**: 80%
    - **Validation Set**: 10%
    - **Test Set**: 10%
  - Image preprocessing includes resizing, normalization, and creating PyTorch-compatible datasets and loaders.

---

## How to Run the Program

### Prerequisites

**Python Environment**:
   - Python 3.8+
   - Install dependencies from `requirements.txt` using:
```bash
pip install -r requirements.txt
```

### Training Script

To train the model, run:
```bash
python model.py --dataset_folder dataset
```
This script:
 - Splits the dataset into training, validation, and test sets.
 - Trains the ResNet18-based binary classification model.
 - Saves the trained model to dolphin_binary_classification.pth

### Model Evaluation and Metrics

To compute metrics and evaluate the model on the test set, run:
```bash
python calculate_metrics.py
```
This script:

 - Loads the trained model.
 - Computes metrics such as accuracy, precision, recall, and F1 scores.
 - Displays a confusion matrix.

### Web Application

To predict dolphin fin classifications through a web interface:

1. Start the Flask application:
```bash
python app.py
```
2. Open your browser and navigate to http://127.0.0.1:5000.
3. Upload an image to receive predictions.

### Results 
**Metrics**:
 - Accuracy: 99.19%

 - Precision (Dolphin #1): 100.00%
 - Recall (Dolphin #1): 80.00%
 - F1 Score (Dolphin #1): 88.89%

 - Precision (Other dolphins): 99.16%
 - Recall (Other dolphins): 100.00%
 - F1 Score (Other dolphins): 99.58%

**Confusion Matrix**:

| **Actual \ Predicted** | **Not Dolphin #1** | **Dolphin #1** |
|-------------------------|--------------------|----------------|
| **Not Dolphin #1**      | 118                | 0              |
| **Dolphin #1**          | 1                  | 4              |

### Key Benefits
 - Accelerates dolphin monitoring by automating the fin recognition process.
 - Leverages a robust pre-trained architecture (ResNet18) for efficient feature extraction.
 - Provides a user-friendly web interface for real-time predictions.
