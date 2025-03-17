# Phase 2 Experiment: Feature Enhancement and Model Optimization

## Overview
This experiment evaluates the impact of different feature sets and model optimizations on chicken sound classification performance, comparing single-stage (1v1v1) and two-tier classification approaches.

## Results Summary

### Best Overall Configuration
- **Model**: SVM
- **Feature Type**: mfcc_temporal
- **Approach**: Two-Tier
- **Test Accuracy**: 0.8325
- **F1 Score**: 0.8318
- **Overfitting Gap**: 0.0533
- **Parameters**: Tier 1: {'C': 10, 'class_weight': None, 'gamma': 'auto', 'kernel': 'rbf'}, Tier 2: {'C': 10, 'class_weight': None, 'gamma': 'auto', 'kernel': 'rbf'}

### Approach Analysis
| Approach | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |
|----------|--------------|--------------|--------------|------------------|
|     1v1v1
Name: 0, dtype: object | 0.7876 | 0.0402 | 0.8267 | 0.1301 |
|     Two-Tier
Name: 1, dtype: object | 0.7767 | 0.0424 | 0.8325 | 0.1544 |

### Feature Type Analysis
| Feature Type | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |
|-------------|--------------|--------------|--------------|------------------|
|     complete
Name: 0, dtype: object | 0.7815 | 0.0189 | 0.8100 | 0.1772 |
|     mfcc
Name: 1, dtype: object | 0.8069 | 0.0120 | 0.8183 | 0.1149 |
|     mfcc_derivatives
Name: 2, dtype: object | 0.7239 | 0.0284 | 0.7675 | 0.1635 |
|     mfcc_temporal
Name: 3, dtype: object | 0.8163 | 0.0120 | 0.8325 | 0.1134 |

### Model Analysis
| Model | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |
|-------|--------------|--------------|--------------|------------------|
|     Extra Trees
Name: 0, dtype: object | 0.7832 | 0.0336 | 0.8167 | 0.1442 |
|     KNN
Name: 1, dtype: object | 0.7695 | 0.0500 | 0.8150 | 0.1985 |
|     SVM
Name: 2, dtype: object | 0.7937 | 0.0388 | 0.8325 | 0.0840 |

### Feature Type and Approach Interaction
| Feature Type | 1v1v1 | Two-Tier | Difference |
|-------------|-------|----------|------------|
| complete | 0.7867 | 0.7764 | 0.0103 |
| mfcc | 0.8111 | 0.8028 | 0.0083 |
| mfcc_derivatives | 0.7333 | 0.7144 | 0.0189 |
| mfcc_temporal | 0.8194 | 0.8131 | 0.0064 |

### Best Configuration For Each Model and Approach
#### SVM
**1v1v1 Approach**
- Best Feature Type: mfcc_temporal
- Test Accuracy: 0.8267
- F1 Score: 0.8255
- Overfitting Gap: 0.0423
- Parameters: {'C': 1, 'class_weight': None, 'gamma': 0.1, 'kernel': 'rbf'}

**Two-Tier Approach**
- Best Feature Type: mfcc_temporal
- Test Accuracy: 0.8325
- F1 Score: 0.8318
- Overfitting Gap: 0.0533
- Parameters: Tier 1: {'C': 10, 'class_weight': None, 'gamma': 'auto', 'kernel': 'rbf'}, Tier 2: {'C': 10, 'class_weight': None, 'gamma': 'auto', 'kernel': 'rbf'}

#### Extra Trees
**1v1v1 Approach**
- Best Feature Type: mfcc_temporal
- Test Accuracy: 0.8167
- F1 Score: 0.8145
- Overfitting Gap: 0.1019
- Parameters: {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}

**Two-Tier Approach**
- Best Feature Type: mfcc_temporal
- Test Accuracy: 0.8008
- F1 Score: 0.8006
- Overfitting Gap: 0.1040
- Parameters: Tier 1: {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}, Tier 2: {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}

#### KNN
**1v1v1 Approach**
- Best Feature Type: mfcc
- Test Accuracy: 0.8150
- F1 Score: 0.8137
- Overfitting Gap: 0.1850
- Parameters: {'metric': 'euclidean', 'n_neighbors': 15, 'p': 1, 'weights': 'distance'}

**Two-Tier Approach**
- Best Feature Type: mfcc_temporal
- Test Accuracy: 0.8058
- F1 Score: 0.8048
- Overfitting Gap: 0.1942
- Parameters: Tier 1: {'metric': 'euclidean', 'n_neighbors': 15, 'p': 1, 'weights': 'distance'}, Tier 2: {'metric': 'manhattan', 'n_neighbors': 15, 'p': 1, 'weights': 'distance'}

## Conclusions

1. **Feature Impact**:     mfcc_temporal
Name: 3, dtype: object features provide the best overall performance across models and approaches.
2. **Model Performance**:     SVM
Name: 2, dtype: object shows the strongest overall performance across feature types and approaches.
3. **Approach Performance**:     1v1v1
Name: 0, dtype: object approach provides better performance overall (mean test accuracy: 0.7876).
4. **Overfitting Trends**:     SVM
Name: 2, dtype: object shows the least overfitting among models,     mfcc_temporal
Name: 3, dtype: object features tend to reduce overfitting, and the     1v1v1
Name: 0, dtype: object approach generally results in less overfitting.

## Recommendations for Phase 3

Based on the Phase 2 results, the following configurations are recommended for Phase 3:

1. **SVM** with **mfcc_temporal** features using the **Two-Tier** approach (Test Accuracy: 0.8325)
2. **SVM** with **mfcc_temporal** features using the **1v1v1** approach (Test Accuracy: 0.8267)
3. **SVM** with **mfcc** features using the **Two-Tier** approach (Test Accuracy: 0.8183)
