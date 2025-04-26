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

1. **Data Preprocessing:**  
   - Load the raw WISDM data.
   - Clean and preprocess the data (handle missing values, convert types, normalize features).
   - Encode activity labels numerically.

2. **Window Segmentation:**  
   - Segment the time-series data into overlapping windows.
   - Only keep windows containing a single activity.

3. **Model Training:**  
   - Convert data to PyTorch tensors.
   - Define and train the LSTM model.
   - Track training and validation loss/accuracy.

4. **Evaluation & Visualization:**  
   - Plot loss and accuracy curves.
   - Visualize predictions vs. actual labels.
   - Use t-SNE to visualize LSTM hidden states.
   - Plot ROC-AUC curves for each activity class.

5. **Run the Notebook:**  
   Open `LSTM_Human_Activity_Detection.ipynb` in Jupyter Notebook and execute the cells step by step.

## Model Architecture

- **Input:** Windowed accelerometer data (x, y, z, magnitude)
- **Model:** LSTM layers followed by fully connected layers
- **Output:** Activity class probabilities

## Results

- **Accuracy:** High classification accuracy on the test set.
- **ROC-AUC:** Near-perfect ROC-AUC scores for all activity classes.
- **Visualizations:** Clear separation of activities in t-SNE and ROC-AUC plots.

## References

- Jennifer R. Kwapisz, Gary M. Weiss and Samuel A. Moore (2010).  
  "Activity Recognition using Cell Phone Accelerometers,"  
  Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data (at KDD-10), Washington DC.  
  [Paper Link](http://www.cis.fordham.edu/wisdm/public_files/sensorKDD-2010.pdf)

- WISDM Lab: http://www.cis.fordham.edu/wisdm/

---

**Note:**  
This project is for educational and research purposes. Please cite the original WISDM paper if you use this dataset or code in your work.
