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
### Income Distribution
![Income Distribution]![Screenshot 2024-08-17 161520](https://github.com/user-attachments/assets/99cae952-8700-4d1f-81fb-a03d34254efb)

The distribution of applicant income shows a highly skewed pattern, with most applicants having low income. This skewness might influence the model's ability to generalize well across different income levels.

  
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

## 📊 Data Overview

### First 5 Rows:

| Index | Gender | Age   | Debt  | Married | BankCustomer | EducationLevel | Ethnicity | YearsEmployed | PriorDefault | Employed | CreditScore | DriversLicense | Citizen | ZipCode | Income | ApprovalStatus |
|-------|--------|-------|-------|---------|--------------|----------------|-----------|---------------|--------------|----------|-------------|----------------|---------|---------|--------|----------------|
| 0     | b      | 30.83 | 0.000 | u       | g            | ...            | ...       | ...           | ...          | ...      | ...         | f              | g       | 00202   | 0      | +              |
| 1     | a      | 58.67 | 4.460 | u       | g            | ...            | ...       | ...           | ...          | ...      | ...         | f              | g       | 00043   | 560    | +              |
| 2     | a      | 24.50 | 0.500 | u       | g            | ...            | ...       | ...           | ...          | ...      | ...         | f              | g       | 00280   | 824    | +              |
| 3     | b      | 27.83 | 1.540 | u       | g            | ...            | ...       | ...           | ...          | ...      | ...         | t              | g       | 00100   | 3      | +              |
| 4     | b      | 20.17 | 5.625 | u       | g            | ...            | ...       | ...           | ...          | ...      | ...         | f              | s       | 00120   | 0      | +              |

### Last 5 Rows:

| Index | Gender | Age   | Debt   | Married | BankCustomer | EducationLevel | Ethnicity | YearsEmployed | PriorDefault | Employed | CreditScore | DriversLicense | Citizen | ZipCode | Income | ApprovalStatus |
|-------|--------|-------|--------|---------|--------------|----------------|-----------|---------------|--------------|----------|-------------|----------------|---------|---------|--------|----------------|
| 685   | b      | 21.08 | 10.085 | y       | p            | ...            | ...       | ...           | ...          | ...      | ...         | f              | g       | 00260   | 0      | -              |
| 686   | a      | 22.67 | 0.750  | u       | g            | ...            | ...       | ...           | ...          | ...      | ...         | t              | g       | 00200   | 394    | -              |
| 687   | a      | 25.25 | 13.500 | y       | p            | ...            | ...       | ...           | ...          | ...      | ...         | t              | g       | 00200   | 1      | -              |
| 688   | b      | 17.92 | 0.205  | u       | g            | ...            | ...       | ...           | ...          | ...      | ...         | f              | g       | 00280   | 750    | -              |
| 689   | b      | 35.00 | 3.375  | u       | g            | ...            | ...       | ...           | ...          | ...      | ...         | t              | g       | 00000   | 0      | -              |

### Data Description:

| Statistic  | Gender | Age  | Debt  | Married | BankCustomer | EducationLevel | Ethnicity | YearsEmployed | PriorDefault | Employed | CreditScore | DriversLicense | Citizen | ZipCode | Income | ApprovalStatus |
|------------|--------|------|-------|---------|--------------|----------------|-----------|---------------|--------------|----------|-------------|----------------|---------|---------|--------|----------------|
| **count**  | 690    | 690  | 690   | 690     | 690          | ...            | ...       | 690           | 690          | 690      | 690         | 690            | 690     | 690     | 690    | 690            |
| **unique** | 3      | 350  | NaN   | 4       | 4            | ...            | ...       | NaN           | 2            | 2        | NaN         | 2              | 3       | 171     | NaN    | 2              |
| **top**    | b      | ?    | NaN   | u       | g            | ...            | ...       | NaN           | NaN          | NaN      | NaN         | f              | g       | 00000   | NaN    | -              |
| **freq**   | 468    | 12   | NaN   | 519     | 519          | ...            | ...       | NaN           | NaN          | NaN      | NaN         | 374            | 625     | 132     | NaN    | 383            |
| **mean**   | NaN    | NaN  | 4.759 | NaN     | NaN          | ...            | ...       | NaN           | NaN          | NaN      | NaN         | NaN            | NaN     | NaN     | 1017.4 | NaN            |
| **std**    | NaN    | NaN  | 4.978 | NaN     | NaN          | ...            | ...       | NaN           | NaN          | NaN      | NaN         | NaN            | NaN     | NaN     | 5210.1 | NaN            |
| **min**    | NaN    | NaN  | 0.000 | NaN     | NaN          | ...            | ...       | NaN           | NaN          | NaN      | NaN         | NaN            | NaN     | NaN     | 0      | NaN            |
| **25%**    | NaN    | NaN  | 1.000 | NaN     | NaN          | ...            | ...       | NaN           | NaN          | NaN      | NaN         | NaN            | NaN     | NaN     | 0      | NaN            |
| **50%**    | NaN    | NaN  | 2.750 | NaN     | NaN          | ...            | ...       | NaN           | NaN          | NaN      | NaN         | NaN            | NaN     | NaN     | 5      | NaN            |
| **75%**    | NaN    | NaN  | 7.208 | NaN     | NaN          | ...            | ...       | NaN           | NaN          | NaN      | NaN         | NaN            | NaN     | NaN     | 395.5  | NaN            |
| **max**    | NaN    | NaN  | 28.00 | NaN     | NaN          | ...            | ...       | NaN           | NaN          | NaN      | NaN         | NaN            | NaN     | NaN     | 100000 | NaN            |

### Data Info:

```plaintext
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 690 entries, 0 to 689
Data columns (total 16 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Gender          690 non-null    object 
 1   Age             690 non-null    object 
 2   Debt            690 non-null    float64
 3   Married         690 non-null    object 
 4   BankCustomer    690 non-null    object 
 5   EducationLevel  690 non-null    object 
 6   Ethnicity       690 non-null    object 
 7   YearsEmployed   690 non-null    float64
 8   PriorDefault    690 non-null    object 
 9   Employed        690 non-null    object 
 10  CreditScore     690 non-null    int64  
 11  DriversLicense  690 non-null    object 
 12  Citizen         690 non-null    object 
 13  ZipCode         690 non-null    object 
 14  Income          690 non-null    int64  
 15  ApprovalStatus  690 non-null    object 
dtypes: float64(2), int64(2), object(12)
memory usage: 86.4+ KB
```
## 🗒️ Conclusion

This project successfully demonstrates the use of logistic regression to predict credit approval based on financial and personal data. The model achieved an accuracy of 85.99%, with a strong AUC score of 93.52%. The insights gained from this model can help in making informed decisions for credit risk management.

Future enhancements could involve experimenting with more advanced machine learning models, incorporating additional features such as credit history length and credit utilization ratio, and integrating real-time data processing for live credit approval predictions.

















