# Chicken Sound Classification - Phase 4 Final Report

## Best Model Summary

- **Model Type**: Extra_Trees
- **Feature Selection Method**: mutual_info
- **Test Accuracy**: 0.9226
- **F1 Score**: 0.9224
- **Overfitting Gap**: 0.0774

- **Number of Features**: 15

- **Model Parameters**: {'n_estimators': 350, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight': None}

## Model Comparison

| Model | Feature Selection | Test Accuracy | F1 Score | Overfitting Gap |
|-------|------------------|---------------|----------|----------------|
| Extra_Trees | mutual_info | 0.9226 | 0.9224 | 0.0774 |
| Extra_Trees | rfecv | 0.9193 | 0.9190 | 0.0807 |
| Extra_Trees | all_features | 0.9181 | 0.9179 | 0.0819 |
| Ensemble | SVM:all_features,ET:mutual_info | 0.9179 | 0.9177 | 0.0821 |
| SVM | all_features | 0.9107 | 0.9108 | 0.0618 |
| SVM | rfecv | 0.9106 | 0.9107 | 0.0627 |
| SVM | mutual_info | 0.8918 | 0.8917 | 0.0607 |
| Extra_Trees | pca | 0.8776 | 0.8775 | 0.1224 |
| SVM | pca | 0.7305 | 0.7286 | 0.0064 |

## Feature Analysis

Feature selection methods were applied to identify the most important features for chicken sound classification. See the feature_analysis directory for detailed results.

## Class-Specific Performance

The best model's performance varies across the three classes:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| healthy | 0.9257 | 0.8764 | 0.9004 | 2800 |
| sick | 0.9358 | 0.9475 | 0.9416 | 2800 |
| noise | 0.9070 | 0.9439 | 0.9251 | 2800 |

## Impact of Data Augmentation

Data augmentation techniques (time shifting, pitch shifting, and noise addition) were applied to increase the robustness of the model. The augmented dataset resulted in improved generalization capabilities as evidenced by the reduced overfitting gap.

