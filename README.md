# Fraud-Insurance-Detection

![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

> A machine learning project for detecting fraudulent insurance claims using fairness-aware methods, ensemble learning, and advanced data preprocessing techniques.

---

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Data Overview](#data-overview)
- [EDA and Preprocessing](#eda-and-preprocessing)
- [Modeling Approach](#modeling-approach)
- [Fairness Analysis](#fairness-analysis)
- [Results](#results)
  
---

## Introduction

Insurance fraud detection is a critical task in minimizing financial losses and ensuring fairness in payouts. This project leverages machine learning to identify fraudulent auto insurance claims while incorporating fairness-aware techniques to mitigate biases and ensure equitable outcomes for sensitive groups.

---

## Problem Statement

Develop a supervised classification model to detect fraudulent claims in auto insurance datasets. Key challenges include class imbalance and fairness concerns arising from biased predictions across sensitive attributes.

---

## Data Overview

- **Dataset:** Historical insurance claims with numerical and categorical features.
- **Target Variable:** `fraud_reported` (binary classification).
- **Challenges:** Missing values, outliers, and class imbalance.

---

## EDA and Preprocessing

### Steps

1. **Missing Value Treatment:**
   - Imputed missing values using appropriate strategies for categorical and numerical data.

2. **Outlier Handling:**
   - Detected and capped extreme outliers in numerical features to ensure consistency.

3. **Feature Engineering:**
   - Extracted new features and removed irrelevant columns for improved prediction.

4. **Encoding & Scaling:**
   - Encoded categorical variables into numerical format and standardized numerical features.

5. **Class Imbalance Handling:**
   - Applied **SMOTE** to oversample the minority class and balance the dataset.

---

## Modeling Approach

### Algorithms Evaluated

- **Logistic Regression:** Baseline model for binary classification.
- **Decision Tree:** Simple yet interpretable model for capturing non-linear patterns.
- **Random Forest:** An ensemble method optimized using GridSearchCV and RandomizedSearchCV.
- **Naive Bayes:** Efficient for smaller datasets.
- **Support Vector Machine (SVM):** Evaluated for performance in high-dimensional spaces.
- **Ensemble Methods:**
  - **Voting Classifier:** Combined predictions from multiple base models.
  - **Stacking Classifier:** A meta-model combining strengths of individual models.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Fairness Metrics: Equalized Odds Difference

---

## Fairness Analysis

Fairness analysis was a key focus to ensure equitable predictions across sensitive groups like gender. Bias in predictions was reduced using techniques such as:

- **Threshold Optimization:** Adjusted predictions post-training to equalize selection rates.
- **Reweighing:** Pre-adjusted instance weights based on sensitive attributes.
- **Hybrid Approach:** Combined reweighing and threshold optimization, significantly improving fairness without major performance trade-offs.

---

## Results

### Summary of Model Performance

| Model                | Accuracy | Recall (Fraud) | Equalized Odds Diff |
|----------------------|----------|----------------|---------------------|
| Logistic Regression  | 71.5%   | 50.0%          | 0.179               |
| Decision Tree        | 74.2%   | 55.0%          | 0.145               |
| Random Forest (Tuned)| **83.0%**| **65.0%**      | **0.012**           |
| Naive Bayes          | 69.0%   | 45.0%          | 0.165               |
| Support Vector Machine| 72.5%  | 52.0%          | 0.140               |
| Stacking Classifier  | 82.5%   | 62.5%          | 0.015               |
| Voting Classifier    | 80.5%   | 60.0%          | 0.020               |

### Key Insights

- **Random Forest** with hyperparameter tuning achieved the best accuracy and fairness balance.
- **Ensemble Methods** (Stacking and Voting) improved overall model robustness.
- **Fairness Techniques** significantly reduced bias while maintaining strong predictive performance.
