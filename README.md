<p align="center">
  <img src="assets/zivita-logo.png" width="180">
</p>

<h1 align="center">Zivita Battery Intelligence</h1>

<p align="center">
AI-powered Battery Health Estimation for Second-Life Lithium-ion Batteries
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-success)
![Battery Analytics](https://img.shields.io/badge/Battery-Analytics-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

</p>

---

# Overview

The rapid growth of electric vehicles has resulted in an increasing number of lithium-ion batteries reaching the end of their automotive life while still retaining a significant portion of their usable capacity. These batteries possess considerable potential for **Second-Life Energy Storage Systems (SLBESS)** if their health can be accurately assessed.

The challenge lies in identifying which retired batteries remain suitable for reuse. Conventional battery testing methods are time-consuming, expensive, and require specialized laboratory equipment.

This project develops an **AI-powered battery intelligence pipeline** capable of estimating the **State of Health (SOH)** of lithium-ion batteries using operational measurements such as voltage, current, temperature, discharge duration, and internal resistance.

The work forms the first step toward Zivita's vision of building an intelligent battery grading and diagnostics platform for second-life battery applications.

---

# Motivation

Millions of EV batteries are expected to retire over the coming decade.

Most of these batteries still retain **70–90%** of their original capacity and can be repurposed for stationary energy storage if their health is accurately determined.

Current battery evaluation techniques generally involve:

- Long laboratory tests
- Manual inspection
- Capacity testing
- Electrochemical measurements

These processes are expensive and difficult to scale.

Artificial Intelligence provides an opportunity to estimate battery health directly from measurable operating parameters, enabling faster and more economical battery grading.

---

# Project Objectives

The current objectives of this repository are:

- Parse raw NASA battery datasets
- Extract battery health features
- Generate machine learning datasets
- Train an AI model for SOH estimation
- Evaluate prediction performance

Future objectives include:

- Battery Cell Grading
- State of Charge (SOC) Estimation
- Remaining Useful Life (RUL) Prediction
- Digital Battery Twin
- AI-assisted Battery Management Systems

---

# Project Pipeline

```
NASA Battery Dataset
        │
        ▼
MATLAB (.mat) Files
        │
        ▼
Feature Extraction
        │
        ▼
Structured CSV Dataset
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Model
        │
        ▼
SOH Prediction
        │
        ▼
Battery Cell Grading (Future)
```

---

# Dataset

This project utilizes the **NASA Ames Prognostics Center of Excellence Battery Aging Dataset**.

The dataset contains battery aging experiments performed under controlled laboratory conditions.

## Dataset Characteristics

- Multiple lithium-ion batteries
- Charge cycles
- Discharge cycles
- Electrochemical Impedance Spectroscopy (EIS)
- Different temperatures
- Different discharge currents
- Complete battery aging until End-of-Life

### Battery Measurements

Each cycle records:

- Battery Voltage
- Battery Current
- Temperature
- Capacity
- Electrolyte Resistance (Re)
- Charge Transfer Resistance (Rct)
- Discharge Time

The batteries were repeatedly cycled until approximately **30% capacity fade**.

---

# Feature Engineering

The MATLAB files were processed to generate a structured machine learning dataset.

The following features were extracted for every discharge cycle.

| Feature | Description |
|----------|-------------|
| Battery | Battery Identifier |
| Cycle | Charge-Discharge Cycle Number |
| Capacity_Ah | Measured Capacity |
| SOH | Calculated State of Health |
| AvgVoltage | Average Terminal Voltage |
| MinVoltage | Minimum Voltage |
| MaxVoltage | Maximum Voltage |
| AvgCurrent | Average Current |
| AvgTemp | Average Temperature |
| MaxTemp | Maximum Temperature |
| Duration | Discharge Duration |
| Re | Electrolyte Resistance |
| Rct | Charge Transfer Resistance |

Final Dataset Size

- **2769 battery discharge cycles**

---

# State of Health (SOH)

Battery State of Health represents how much usable capacity remains relative to a new battery.

The SOH is calculated as

\[
SOH=\frac{Current\ Capacity}{Rated\ Capacity}\times100
\]

where

Rated Capacity = **2 Ah**

Example

| Capacity | SOH |
|----------|------|
|2.00 Ah|100%|
|1.80 Ah|90%|
|1.60 Ah|80%|
|1.40 Ah|70% (End of Life)|

---

# Machine Learning Model

The extracted dataset was used to train a supervised regression model.

## Model

Random Forest Regressor

Random Forest is an ensemble learning algorithm consisting of multiple decision trees.

Each tree independently predicts battery health, and the final prediction is obtained by averaging the outputs of all trees.

Advantages:

- Handles nonlinear battery behavior
- Resistant to overfitting
- Robust to noisy measurements
- Works well with mixed battery parameters
- Provides feature importance analysis

---

# Training Pipeline

```
MATLAB Files

↓

Feature Extraction

↓

CSV Dataset

↓

Cleaning Missing Values

↓

Feature Selection

↓

Train-Test Split

↓

Random Forest Regression

↓

SOH Prediction

↓

Performance Evaluation
```

---

# Model Inputs

The model uses

- Average Voltage
- Minimum Voltage
- Maximum Voltage
- Average Current
- Average Temperature
- Maximum Temperature
- Discharge Duration
- Electrolyte Resistance (Re)
- Charge Transfer Resistance (Rct)
- Cycle Number

Target Variable

```
SOH (%)
```

---

# Results

The trained model achieved

| Metric | Value |
|---------|--------|
|Mean Absolute Error (MAE)|0.816|
|R² Score|0.968|

---

## Interpretation

### Mean Absolute Error

MAE indicates the average prediction error.

An MAE of **0.816** means that the predicted battery health differs from the actual health by less than **1 percentage point** on average.

---

### R² Score

The R² score measures how well the model explains battery degradation.

An R² value of **0.968** indicates that approximately **96.8% of the variation in battery health** is captured by the model.

These results demonstrate that the model successfully learns meaningful degradation patterns from the extracted battery parameters.

---

# Repository Structure

```
zivita-battery-intelligence/

│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── assets/
│      zivita-logo.png
│
├── data/
│      raw/
│      processed/
│          battery_dataset.csv
│
├── src/
│      extract_dataset.py
│      train_soh_model.py
│      inspect_dataset.py
│
├── notebooks/
│      Battery_Exploration.ipynb
│
├── models/
│      random_forest_soh.pkl
│
├── results/
│      feature_importance.png
│      prediction_results.png
│
└── docs/
       methodology.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/zivita-battery-intelligence.git

cd zivita-battery-intelligence
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

```
Python 3.11

NumPy

Pandas

SciPy

Scikit-Learn

Matplotlib

Joblib
```

---

# Running the Project

Extract the dataset

```bash
python src/extract_dataset.py
```

Train the model

```bash
python src/train_soh_model.py
```

Evaluate

```bash
python src/evaluate_model.py
```

---

# Current Progress

✅ MATLAB dataset parsing

✅ Feature extraction

✅ Dataset generation

✅ Machine learning pipeline

✅ SOH prediction model

⬜ Battery grading engine

⬜ SOC estimation

⬜ Remaining Useful Life prediction

⬜ Battery digital twin

⬜ Embedded deployment

---

# Future Roadmap

## Phase 1

✔ Dataset Parsing

✔ Feature Engineering

✔ SOH Prediction

---

## Phase 2

Battery Cell Grading

Automatically classify retired cells into

- Grade A
- Grade B
- Grade C
- Reject

---

## Phase 3

State of Charge Estimation

Estimate the remaining charge of batteries during operation.

---

## Phase 4

Remaining Useful Life Prediction

Predict the remaining operational lifespan of retired batteries.

---

## Phase 5

Battery Digital Twin

Develop a digital representation of every battery for continuous monitoring.

---

## Phase 6

Embedded Deployment

Deploy trained AI models onto Battery Management Systems (BMS) for real-time diagnostics.

---

# Current Limitations

- Only NASA battery datasets are used.
- Single-cell laboratory data.
- Offline prediction only.
- No embedded deployment.
- No real-world EV battery validation yet.
- Cell grading and SOC estimation are under development.

---

# Applications

Potential applications include

- Second-life battery evaluation
- Battery refurbishment
- Battery recycling
- Battery grading
- Stationary energy storage
- Battery diagnostics
- Predictive maintenance
- Smart Battery Management Systems

---

# About Zivita

**Zivita** is developing intelligent technologies for the sustainable reuse of retired lithium-ion batteries.

Our vision is to build an AI-powered battery intelligence platform capable of evaluating, grading, monitoring, and optimizing second-life batteries for energy storage applications.

This repository represents the initial research phase of that vision by establishing an end-to-end pipeline for battery data processing and State of Health estimation using machine learning.

---

# References

1. NASA Ames Prognostics Center of Excellence Battery Dataset

2. Saha, B., & Goebel, K. (2007). Battery Data Set.

3. Scikit-Learn Documentation

4. Python Scientific Computing Ecosystem

---

# License

This project is released under the MIT License.

---

# Contributors

**Zivita Research Team**

- Machine Learning
- Battery Analytics
- Second-Life Battery Research

---

## If you found this project useful

⭐ Star this repository

🤝 Contributions and discussions are welcome.

Together, let's accelerate the future of sustainable battery intelligence.
