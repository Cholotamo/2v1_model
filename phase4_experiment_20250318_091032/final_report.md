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
| Enhanced_Ensemble | 0.9898 | 0.9394 | 0.9392 | 0.0504 |
| ET_1v1v1 | 1.0000 | 0.9189 | 0.9187 | 0.0811 |
| SVM_1v1v1 | 0.9525 | 0.8999 | 0.8997 | 0.0526 |
| SVM_TwoTier | 0.9488 | 0.8976 | 0.8976 | 0.0512 |

## Enhanced Ensemble

The ensemble uses the following weights:

- SVM Two-Tier: 0.10
- SVM 1v1v1: 0.32
- Extra Trees 1v1v1: 0.58

## Conclusions

The best performing model is **Enhanced_Ensemble** with a test accuracy of 0.9394 and F1 score of 0.9392.

### Key Findings

1. The **1v1v1 approach** is competitive with the Two-Tier approach (0.8999 vs 0.8976).

2. The **Enhanced Ensemble** successfully combines the models, achieving 0.9394 accuracy compared to 0.9189.

3. Data augmentation improved model robustness by exposing the models to varied sound patterns.

### Recommendations

1. Deploy the **Enhanced_Ensemble** model for production.
2. Continue data collection, especially for sick chicken sounds, to further improve performance.
3. Consider deployment constraints—if resources are limited, the SVM models remain a viable option.
