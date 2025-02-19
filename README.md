# SAFER-Predictor

## Overview
**SAFER-Predictor (Sparse Adversarial Training Framework for Robust Traffic Prediction under Missing and Noisy Data)** is a traffic flow prediction framework designed to enhance robustness under conditions of missing or noisy data. It employs adversarial training and sparse learning techniques to improve the accuracy and stability of predictions in real-world scenarios.
This work builds upon previous research, particularly in scenarios where [T-GCN](https://github.com/lehaifeng/T-GCN/tree/master) encounters challenges due to missing or noisy observation data.

## Repository Structure
This repository is organized as follows:

- [**models/**](models): Contains various deep learning models used for traffic flow prediction.
- [**tasks/**](tasks): Includes scripts for running different prediction tasks, such as data preprocessing and model evaluation.
- [**utils/**](utils): Provides utility functions, including data loading, preprocessing, and evaluation metrics.
- [**main.py**](main.py): The main entry point for training and evaluating the prediction models.
- [**test_soft.sh**](test_soft.sh): A shell script for running basic tests to verify the setup and dependencies.
### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/SAFER-Predictor.git
   cd SAFER-Predictor
   ```
2. Run the main script:
   ```bash
   python main.py
   ```
3. Test the setup using:
   ```bash
   sh test_soft.sh
   ```
