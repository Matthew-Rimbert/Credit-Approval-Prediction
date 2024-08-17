# 💳 Credit Approval Prediction
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.2.4+-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.19.2+-orange.svg)
![SciKit-Learn](https://img.shields.io/badge/SciKit--Learn-0.23.2+-yellow.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11.1+-green.svg)

## 🎯 Project Overview

This project aims to predict credit approval using machine learning, specifically a logistic regression model. The analysis involves loading and preprocessing the credit data, visualizing data distributions, and training and evaluating the model.

## 📝 Purpose

- Predict credit approval based on various financial and personal factors.
- Analyze the influence of different factors on credit approval decisions.

## 📊 Scope of Work

1. **Load and Preprocess Data:** 
   - Load the credit dataset and preprocess it by handling missing values, encoding categorical variables, and scaling numerical features.
  
2. **Data Visualization:** 
   - Visualize the distributions of key variables such as income and debt.
   - Create correlation heatmaps to understand the relationships between features.
  
3. **Train and Evaluate a Logistic Regression Model:**
   - Train the model on the processed data and evaluate its performance using accuracy, confusion matrix, and classification metrics.

## 🚀 Methodology

### 1. Importing Necessary Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
```

# Load and preprocess the data
df = load_data('path_to_your_data.csv')
df = preprocess_data(df)

# Visualize the data
visualize_data(df)

# Train and evaluate the logistic regression model
y_test, y_pred, y_pred_proba = train_and_evaluate_model(df)

# Visualize the results
visualize_results(y_test, y_pred, y_pred_proba)

## 📈 Results
- **Model Accuracy:** The logistic regression model achieved an accuracy of `X%`.
- **Classification Report:** Detailed precision, recall, and F1-score metrics.

```plaintext
              precision    recall  f1-score   support

           0       X.XX      X.XX      X.XX        XX
           1       X.XX      X.XX      X.XX        XX

    accuracy                           X.XX       XXX
   macro avg       X.XX      X.XX      X.XX       XXX
weighted avg       X.XX      X.XX      X.XX       XXX
```
















