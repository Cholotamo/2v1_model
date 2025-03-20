# Chicken Sound Classification - Phase 4 Final Report

## Overview

This experiment uses data augmentation and an enhanced ensemble with Optuna tuning for hyperparameters and ensemble weights.

## Models Evaluated

1. **SVM with mfcc_temporal features using Two-Tier approach**
2. **SVM with mfcc_temporal features using 1v1v1 approach**
3. **Extra Trees with mfcc_temporal features using 1v1v1 approach**
4. **Enhanced Ensemble** combining all three models with optimized weights

## Results Summary

| Model | Train Accuracy | Test Accuracy | F1 Score | Overfitting Gap |
|-------|----------------|---------------|----------|----------------|
| Enhanced_Ensemble | 0.9953 | 0.9387 | 0.9386 | 0.0566 |
| ET_1v1v1 | 1.0000 | 0.9202 | 0.9200 | 0.0798 |
| SVM_1v1v1 | 0.9516 | 0.8990 | 0.8990 | 0.0526 |
| SVM_TwoTier | 0.9500 | 0.8989 | 0.8989 | 0.0510 |

## Enhanced Ensemble

The ensemble uses the following weights:

- SVM Two-Tier: 0.12
- SVM 1v1v1: 0.28
- Extra Trees 1v1v1: 0.60

## Conclusions

The best performing model is **Enhanced_Ensemble** with a test accuracy of 0.9387 and F1 score of 0.9386.

### Key Findings

1. The **1v1v1 approach** is competitive with the Two-Tier approach (0.8990 vs 0.8989).

2. The **Enhanced Ensemble** successfully combines the models, achieving 0.9387 accuracy compared to 0.9202.

3. Data augmentation improved model robustness by exposing the models to varied sound patterns.

### Recommendations

1. Deploy the **Enhanced_Ensemble** model for production.
2. Continue data collection, especially for sick chicken sounds, to further improve performance.
3. Consider deployment constraints—if resources are limited, the SVM models remain a viable option.
