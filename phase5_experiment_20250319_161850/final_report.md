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
| Enhanced_Ensemble | 0.9773 | 0.9305 | 0.9304 | 0.0468 |
| ET_1v1v1 | 1.0000 | 0.9195 | 0.9193 | 0.0805 |
| SVM_1v1v1 | 0.9563 | 0.8933 | 0.8934 | 0.0629 |
| SVM_TwoTier | 0.9432 | 0.8893 | 0.8895 | 0.0539 |

## Enhanced Ensemble

The ensemble uses the following weights:

- SVM Two-Tier: 0.20
- SVM 1v1v1: 0.37
- Extra Trees 1v1v1: 0.43

## Conclusions

The best performing model is **Enhanced_Ensemble** with a test accuracy of 0.9305 and F1 score of 0.9304.

### Key Findings

1. The **1v1v1 approach** is competitive with the Two-Tier approach (0.8933 vs 0.8893).

2. The **Enhanced Ensemble** successfully combines the models, achieving 0.9305 accuracy compared to 0.9195.

3. Data augmentation improved model robustness by exposing the models to varied sound patterns.

### Recommendations

1. Deploy the **Enhanced_Ensemble** model for production.
2. Continue data collection, especially for sick chicken sounds, to further improve performance.
3. Consider deployment constraints—if resources are limited, the SVM models remain a viable option.
