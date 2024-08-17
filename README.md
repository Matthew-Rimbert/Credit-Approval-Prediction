# 💳 Credit Approval Prediction
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.2.4+-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.19.2+-orange.svg)
![SciKit-Learn](https://img.shields.io/badge/SciKit--Learn-0.23.2+-yellow.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11.1+-green.svg)

## 🎯 Project Overview

This project aims to predict credit approval using machine learning, specifically a logistic regression model. The analysis involves loading and preprocessing the credit data, visualizing data distributions, and training and evaluating the model.

## 📝 Purpose

- **Prediction**: Develop a machine learning model to predict credit approval based on applicant data.
- **Analysis**: Understand the influence of factors like income, debt, and credit history on credit decisions.
- **Application**: Provide a decision-support tool that improves the efficiency of credit approval processes.

## 📊 Scope of Work

1. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features to prepare the data for modeling.
2. **Data Visualization**: Analyze the distribution of key variables and relationships between features using visual tools like histograms and correlation heatmaps.
3. **Model Development**: Train and evaluate a logistic regression model using accuracy, precision, recall, and AUC as key performance metrics.
4. **Result Interpretation**: Assess model performance and interpret the significance of the findings in the context of credit approval.

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

## 📊 Data Visualization
### Income Distribution
![Screenshot 2024-08-17 161520](https://github.com/user-attachments/assets/99cae952-8700-4d1f-81fb-a03d34254efb)

The distribution of applicant income shows a highly skewed pattern, with most applicants having low income. This skewness might influence the model's ability to generalize well across different income levels.

### Debt Distribution
![Screenshot 2024-08-17 161534](https://github.com/user-attachments/assets/58dd2862-e63f-4d2a-9b92-c197123733cf)

The debt distribution among applicants is more varied, with a noticeable decrease in frequency as debt amounts increase. This distribution provides insights into how debt levels might impact credit approval.

### Pairplot for Numerical Features
![Screenshot 2024-08-17 161553](https://github.com/user-attachments/assets/ccec791e-8613-40ac-b6d7-ecee311dba3b)

The pairplot below shows the relationships between key numerical features such as debt, years employed, credit score, and income. The plot helps in identifying potential correlations and outliers.

### Correlation Heatmap
![Screenshot 2024-08-17 161607](https://github.com/user-attachments/assets/e8478d52-79a4-403a-8f9c-cbaddc4c1806)

The heatmap below visualizes the correlation between numerical features. The darker the color, the stronger the correlation. For instance, debt and years employed show a moderate correlation, which might influence the model's predictions.

## 📈 Results
### Confusion Matrix
![Screenshot 2024-08-17 161620](https://github.com/user-attachments/assets/50b87216-deba-4f87-aa48-2b3fc71b401f)
The confusion matrix below illustrates the model's performance in predicting approved versus rejected credit applications. The model shows strong performance with minimal false positives and false negatives.

### ROC Curve
![Screenshot 2024-08-17 161633](https://github.com/user-attachments/assets/a9fd672f-79fb-4e79-aca6-502a25184563)
The ROC curve shows the true positive rate against the false positive rate at various threshold settings. The AUC score of 0.94 indicates that the model has a high ability to distinguish between approved and rejected applications.

### Precision-Recall Curve
![Screenshot 2024-08-17 161647](https://github.com/user-attachments/assets/628d1c39-c946-4d6f-9bae-099b96664b54)
The precision-recall curve helps in understanding the trade-off between precision and recall for different thresholds. The curve shows that the model maintains high precision across a wide range of recall values, which is beneficial for imbalanced datasets like this one.

# 🗒️ Conclusion

This project successfully demonstrates the use of logistic regression to predict credit approval based on financial and personal data. The model achieved an accuracy of 85.99%, with a strong AUC score of 93.52%. The insights gained from this model can help in making informed decisions for credit risk management.

Future enhancements could involve experimenting with more advanced machine learning models, incorporating additional features such as credit history length and credit utilization ratio, and integrating real-time data processing for live credit approval predictions.





