# Phase 1 Experiment: Classification Approach Comparison

## Overview
This experiment compares single-stage (1v1v1) and two-tier classification approaches for chicken sound classification.

### Best Overall Configuration
- **Model**: Support Vector Machine (SVM)
- **Approach**: 1v1v1
- **Test Accuracy**: 0.8242
- **F1 Score**: 0.8231
- **Overfitting Gap**: 0.0400
- **Parameters**: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}

### Approach Analysis
| Approach | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |
|----------|--------------|--------------|--------------|------------------|
|     1v1v1
Name: 0, dtype: object | 0.7535 | 0.0799 | 0.8242 | 0.1074 |
|     Two-Tier
Name: 1, dtype: object | 0.7397 | 0.0924 | 0.8183 | 0.1151 |

### Model Analysis
| Model | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |
|-------|--------------|--------------|--------------|------------------|
|     AdaBoost
Name: 0, dtype: object | 0.6663 | 0.0053 | 0.6700 | 0.0120 |
|     Decision Tree
Name: 1, dtype: object | 0.7121 | 0.0218 | 0.7275 | 0.1371 |
|     Extra Trees
Name: 2, dtype: object | 0.8175 | 0.0071 | 0.8225 | 0.1824 |
|     Gradient Boosting
Name: 3, dtype: object | 0.7917 | 0.0047 | 0.7950 | 0.2083 |
|     K-Nearest Neighbors (KNN)
Name: 4, dtype: object | 0.8150 | 0.0000 | 0.8150 | 0.1850 |
|     Logistic Regression
Name: 5, dtype: object | 0.6317 | 0.0259 | 0.6500 | 0.0000 |
|     Naive Bayes
Name: 6, dtype: object | 0.6004 | 0.0336 | 0.6242 | -0.0302 |
|     Random Forest
Name: 7, dtype: object | 0.8067 | 0.0012 | 0.8075 | 0.1927 |
|     Support Vector Machine (SVM)
Name: 8, dtype: object | 0.8213 | 0.0041 | 0.8242 | 0.0427 |
|     XGBoost
Name: 9, dtype: object | 0.8033 | 0.0047 | 0.8067 | 0.1823 |

### Best Configuration Per Approach
#### 1v1v1
- **Model**: Support Vector Machine (SVM)
- **Test Accuracy**: 0.8242
- **F1 Score**: 0.8231
- **Overfitting Gap**: 0.0400
- **Parameters**: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}

#### Two-Tier
- **Model**: Support Vector Machine (SVM)
- **Test Accuracy**: 0.8183
- **F1 Score**: 0.8175
- **Overfitting Gap**: 0.0454
- **Parameters**: Tier 1: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}, Tier 2: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

### Statistical Comparison of Approaches
- **Support Vector Machine (SVM)**: 1v1v1 CV: 0.8122, Two-Tier CV: 0.8123, Difference: -0.0002
- **Extra Trees**: 1v1v1 CV: 0.8115, Two-Tier CV: 0.8018, Difference: 0.0097
- **XGBoost**: 1v1v1 CV: 0.7957, Two-Tier CV: 0.7870, Difference: 0.0087
- **Random Forest**: 1v1v1 CV: 0.7977, Two-Tier CV: 0.7967, Difference: 0.0010
- **Logistic Regression**: 1v1v1 CV: 0.6342, Two-Tier CV: 0.6178, Difference: 0.0163
- **Naive Bayes**: 1v1v1 CV: 0.5958, Two-Tier CV: 0.5583, Difference: 0.0375
- **K-Nearest Neighbors (KNN)**: 1v1v1 CV: 0.8093, Two-Tier CV: 0.8020, Difference: 0.0073
- **Decision Tree**: 1v1v1 CV: 0.7138, Two-Tier CV: 0.7070, Difference: 0.0068
- **Gradient Boosting**: 1v1v1 CV: 0.7928, Two-Tier CV: 0.7950, Difference: -0.0022
- **AdaBoost**: 1v1v1 CV: 0.6503, Two-Tier CV: 0.6542, Difference: -0.0038

Average CV Difference (1v1v1 - Two-Tier): 0.0081
Models favoring 1v1v1: 7/10
Models favoring Two-Tier: 3/10

## Conclusions

1. **Approach Impact**:     1v1v1
Name: 0, dtype: object approach provides the best overall performance (mean test accuracy: 0.7535).
2. **Model Performance**:     Support Vector Machine (SVM)
Name: 8, dtype: object shows the strongest overall performance (mean test accuracy: 0.8213).
3. **Overfitting Trends**:     1v1v1
Name: 0, dtype: object approach tends to reduce overfitting (mean gap: 0.1074).

## Recommendations for Phase 2
No clear winner between approaches (avg CV diff: 0.0081). Evaluate both with top models based on CV accuracy:
- Top models: Support Vector Machine (SVM), Extra Trees, K-Nearest Neighbors (KNN)
