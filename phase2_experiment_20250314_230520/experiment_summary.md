# Phase 2 Experiment: Feature Enhancement and Model Optimization

## Overview
This experiment evaluates the impact of different feature sets and model optimizations on chicken sound classification performance.

## Results Summary

### Best Overall Configuration
- **Model**: SVM
- **Feature Type**: mfcc_temporal
- **Test Accuracy**: 0.8267
- **F1 Score**: 0.8255
- **Overfitting Gap**: 0.0423
- **Parameters**: {'C': 1, 'class_weight': None, 'gamma': 0.1, 'kernel': 'rbf'}

### Feature Type Analysis
| Feature Type | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |
|-------------|--------------|--------------|--------------|------------------|
|     complete
Name: 0, dtype: object | 0.7914 | 0.0320 | 0.8258 | 0.1888 |
|     mfcc
Name: 1, dtype: object | 0.8197 | 0.0055 | 0.8258 | 0.1400 |
|     mfcc_derivatives
Name: 2, dtype: object | 0.7475 | 0.0495 | 0.8000 | 0.1995 |
|     mfcc_temporal
Name: 3, dtype: object | 0.8236 | 0.0053 | 0.8267 | 0.1326 |

### Model Analysis
| Model | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |
|-------|--------------|--------------|--------------|------------------|
|     Extra Trees
Name: 0, dtype: object | 0.8196 | 0.0131 | 0.8267 | 0.1804 |
|     KNN
Name: 1, dtype: object | 0.7742 | 0.0546 | 0.8175 | 0.2258 |
|     SVM
Name: 2, dtype: object | 0.7929 | 0.0389 | 0.8267 | 0.0895 |

### Best Configuration For Each Model
#### SVM
- **Best Feature Type**: mfcc_temporal
- **Test Accuracy**: 0.8267
- **F1 Score**: 0.8255
- **Overfitting Gap**: 0.0423
- **Parameters**: {'C': 1, 'class_weight': None, 'gamma': 0.1, 'kernel': 'rbf'}

#### Extra Trees
- **Best Feature Type**: mfcc_temporal
- **Test Accuracy**: 0.8267
- **F1 Score**: 0.8255
- **Overfitting Gap**: 0.1731
- **Parameters**: {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}

#### KNN
- **Best Feature Type**: mfcc_temporal
- **Test Accuracy**: 0.8175
- **F1 Score**: 0.8168
- **Overfitting Gap**: 0.1825
- **Parameters**: {'metric': 'euclidean', 'n_neighbors': 5, 'p': 1, 'weights': 'distance'}

## Conclusions

1. **Feature Impact**:     mfcc_temporal
Name: 3, dtype: object features provide the best overall performance across models.
2. **Model Performance**:     Extra Trees
Name: 0, dtype: object shows the strongest overall performance across feature types.
3. **Overfitting Trends**:     SVM
Name: 2, dtype: object shows the least overfitting among models, while     mfcc_temporal
Name: 3, dtype: object features tend to reduce overfitting across all models.

