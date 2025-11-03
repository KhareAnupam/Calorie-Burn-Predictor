# 🏋️‍♂️ Calorie Burn Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/Frameworks-PyTorch%2C%20XGBoost%2C%20CatBoost-orange)
![ML](https://img.shields.io/badge/Machine%20Learning-Calorie%20Prediction-yellow)

---

## 📖 Introduction
**Calorie Burn Predictor** is a machine learning project that estimates the number of calories burned during physical exercise based on physiological and activity data.  
The project compares multiple ML approaches — **CatBoost**, **XGBoost**, and **PyTorch-based neural networks** — to determine the most accurate model for calorie prediction.

---

## 📚 Table of Contents
- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Descriptions](#-model-descriptions)
  - [CatBoost Model](#catboost-model)
  - [XGBoost Model](#xgboost-model)
  - [PyTorch Model](#pytorch-model)
- [Model Comparison](#-model-comparison)
- [Features](#-features)
- [Dependencies](#-dependencies)
- [Troubleshooting](#-troubleshooting)
- [Contributors](#-contributors)
- [License](#-license)

---

## 📊 Dataset
The project uses two main CSV files:
- **`calories.csv`** — Contains calorie data for exercises.
- **`exercise.csv`** — Includes exercise details such as duration, heart rate, and other relevant metrics.

These datasets are merged and processed to train and evaluate different ML models.

---

## 📁 Project Structure
Calorie Burn predictor/
├── calories.csv
├── exercise.csv
├── compare_models.py
├── ML_catboost.py
├── XGBoost.py
├── pytorch.py
├── catboost_info/
│ ├── catboost_training.json
│ ├── learn_error.tsv
│ ├── test_error.tsv
│ └── ...
├── .gradio/
│ └── flagged/
│ └── dataset1.csv
└── .gitignore

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/calorie-burn-predictor.git
   cd calorie-burn-predictor
2. Create and activate a virtual environment:   
  python -m venv venv
  .\venv\Scripts\activate   # On Windows
  # OR
  source venv/bin/activate  # On macOS/Linux

3. Install dependencies:
  pip install pandas scikit-learn torch xgboost catboost gradio

Usage

Each model script can be run independently to train and evaluate the model.

# Run CatBoost model
python ML_catboost.py

# Run XGBoost model
python XGBoost.py

# Run PyTorch model
python pytorch.py


Each script trains the model on the calorie–exercise dataset and outputs the Mean Absolute Error (MAE) as a performance metric.

🧠 Model Descriptions
🐈 CatBoost Model

File: ML_catboost.py

Implements the CatBoost Regressor, an efficient gradient boosting algorithm optimized for categorical features.

Provides fast training and strong performance with minimal parameter tuning.

Evaluation metric: Mean Absolute Error (MAE).

Stores detailed training logs in the catboost_info/ directory.

🌳 XGBoost Model

File: XGBoost.py

Uses the XGBoost Regressor for gradient boosting on decision trees.

Highly tunable and well-suited for structured tabular data.

Compares favorably in speed and performance across experiments.

Outputs evaluation metrics directly to the console for analysis.

🔥 PyTorch Model

File: pytorch.py

Implements a feed-forward neural network using the PyTorch framework.

Allows deeper learning of nonlinear relationships in the dataset.

Uses standard training loops with loss functions such as MSELoss or L1Loss.

Reports MAE at the end of training for easy comparison with other models.

