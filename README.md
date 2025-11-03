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
