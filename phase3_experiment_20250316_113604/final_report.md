# Chicken Sound Classification - Phase 3 Final Report

## Best Model Summary

- **Model Type**: Ensemble
- **Feature Selection Method**: SVM:all_features,ET:mutual_info
- **Test Accuracy**: 0.9190
- **F1 Score**: 0.9189
- **Overfitting Gap**: 0.0810

- **Model Parameters**: SVM weight: 0.20, ET weight: 0.80

## Model Comparison

| Model | Feature Selection | Test Accuracy | F1 Score | Overfitting Gap |
|-------|------------------|---------------|----------|----------------|
| Ensemble | SVM:all_features,ET:mutual_info | 0.9190 | 0.9189 | 0.0810 |
| Extra_Trees | mutual_info | 0.9170 | 0.9169 | 0.0830 |
| Extra_Trees | all_features | 0.9158 | 0.9156 | 0.0842 |
| Extra_Trees | rfecv | 0.9158 | 0.9156 | 0.0842 |
| SVM | all_features | 0.9081 | 0.9083 | 0.0680 |
| SVM | rfecv | 0.9081 | 0.9083 | 0.0680 |
| SVM | mutual_info | 0.8930 | 0.8929 | 0.0579 |
| Extra_Trees | pca | 0.8749 | 0.8747 | 0.1251 |
| SVM | pca | 0.7219 | 0.7194 | 0.0209 |

## Feature Analysis

Feature selection methods were applied to identify the most important features for chicken sound classification. See the feature_analysis directory for detailed results.

## Class-Specific Performance

The best model's performance varies across the three classes:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| healthy | 0.9176 | 0.8852 | 0.9011 | 2100 |
| sick | 0.9340 | 0.9362 | 0.9351 | 2100 |
| noise | 0.9059 | 0.9357 | 0.9206 | 2100 |

## Impact of Data Augmentation

Data augmentation techniques (time shifting, pitch shifting, and noise addition) were applied to increase the robustness of the model. The augmented dataset resulted in improved generalization capabilities as evidenced by the reduced overfitting gap (0.0810) in the best model.

