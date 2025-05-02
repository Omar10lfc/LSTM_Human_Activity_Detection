# Human Activity Recognition with LSTM (WISDM Dataset)

This project implements a deep learning pipeline for human activity recognition using smartphone accelerometer data from the WISDM dataset. The model leverages an LSTM (Long Short-Term Memory) neural network to classify activities such as walking, jogging, sitting, standing, upstairs, and downstairs.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [References](#references)

---

## Overview

Human Activity Recognition (HAR) is a key problem in ubiquitous computing and wearable sensor analytics. This project demonstrates how to preprocess raw accelerometer data, segment it into windows, and train an LSTM-based classifier to recognize different physical activities.

## Dataset

- **Source:** [WISDM Lab](http://www.cis.fordham.edu/wisdm/)
- **Description:** The WISDM dataset contains labeled accelerometer data collected from smartphones carried by 36 users performing six activities.
- **Activities:** Walking, Jogging, Upstairs, Downstairs, Sitting, Standing

## Installation

1. Clone this repository and navigate to the project directory.
2. Install the required Python packages:
    ```bash
    pip install numpy pandas matplotlib scikit-learn torch scipy
    ```
    
## Usage

1. **Data Preprocessing and EDA:**  
   - cleaned raw accelerometer data, handles missing values, and engineers features such as acceleration magnitude.
   - Visualized the activity distribution and checked for outliers and removed them.
   - Splited data into training, validation, and test sets; applied normalization and label encoding.

2. **Window Segmentation:**  
   - Segment the time-series data into overlapping windows.
   - Only keep windows containing a single activity.

3. **Data Augmentation:**
   - Applied jittering, scaling, permutation, and random rotation to improve model robustness.
  
4. **Class Imbalance Handling:**
   - Used weighted sampling to solve this issue.
     
6. **Model Training:**  
   - Convert data to PyTorch tensors.
   - Define and train the LSTM model.
   - Includes early stopping and tracks loss/accuracy over epochs.

7. **Evaluation & Visualization:**  
   - Plot loss and accuracy curves.
   - Report accuracy, precision, recall, F1-score and confusion matrix.
   - Used t-SNE to visualize LSTM hidden states.
   - Ploted ROC-AUC curves for each activity class.

8. **Run the Notebook:**  
   Open `Human_Activity_Recognition_with_LSTM.ipynb` in Jupyter Notebook and execute the cells step by step.

## Model Architecture

- Implemented a multi-layer bidirectional LSTM using PyTorch for sequence modeling.

## Results

- **High Classification Performance:** Achieves strong accuracy and F1-scores across all activity classes.
- **Robust Generalization:** Model generalizes well to unseen test data, as shown by ROC-AUC and t-SNE visualizations.
- **Effective Feature Learning:** LSTM hidden states show clear class separation in t-SNE plot.

## References

- Jennifer R. Kwapisz, Gary M. Weiss and Samuel A. Moore (2010).  
  "Activity Recognition using Cell Phone Accelerometers,"  
  Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data (at KDD-10), Washington DC.  
  [Paper Link](http://www.cis.fordham.edu/wisdm/public_files/sensorKDD-2010.pdf)

- WISDM Lab: http://www.cis.fordham.edu/wisdm/

---

**Note:**  
This project is for educational and research purposes. Please cite the original WISDM paper if you use this dataset or code in your work.
