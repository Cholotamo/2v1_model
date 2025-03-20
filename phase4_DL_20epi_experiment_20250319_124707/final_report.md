# Chicken Sound Classification - Phase 5 Final Report

## Overview

This experiment uses data augmentation and an enhanced ensemble with deep learning based hyperparameter tuning for traditional ML models.

## Models Evaluated

1. **SVM with mfcc_temporal features using Two-Tier approach**
2. **SVM with mfcc_temporal features using 1v1v1 approach**
3. **Extra Trees with mfcc_temporal features using 1v1v1 approach**
4. **Enhanced Ensemble** combining all three models with optimized weights

## Results Summary

| Model | Train Accuracy | Test Accuracy | F1 Score | Overfitting Gap |
|-------|----------------|---------------|----------|----------------|
| Enhanced_Ensemble | 0.9926 | 0.9400 | 0.9399 | 0.0526 |
| ET_1v1v1 | 1.0000 | 0.9199 | 0.9197 | 0.0801 |
| SVM_TwoTier | 0.9613 | 0.8950 | 0.8953 | 0.0663 |
| SVM_1v1v1 | 0.9402 | 0.8838 | 0.8838 | 0.0564 |

## Enhanced Ensemble

The ensemble uses the following weights:

- SVM Two-Tier: 0.22
- SVM 1v1v1: 0.22
- Extra Trees 1v1v1: 0.56

## Conclusions

The best performing model is **Enhanced_Ensemble** with a test accuracy of 0.9400 and F1 score of 0.9399.

### Key Findings

1. The **Two-Tier approach** outperforms the standard 1v1v1 approach (0.8950 vs 0.8838).

2. The **Enhanced Ensemble** successfully combines the models, achieving 0.9400 accuracy compared to 0.9199.

3. Data augmentation improved model robustness by exposing the models to varied sound patterns.

### Recommendations

1. Deploy the **Enhanced_Ensemble** model for production.
2. Continue data collection, especially for sick chicken sounds, to further improve performance.
3. Consider deployment constraints—if resources are limited, the SVM models remain a viable option.
