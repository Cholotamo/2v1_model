# Chicken Sound Classification - Phase 3 Final Report

## Overview

This experiment implements the top three model configurations from Phase 2, applying data augmentation and enhanced ensemble techniques.

## Models Evaluated

1. **SVM with mfcc_temporal features using Two-Tier approach**
2. **SVM with mfcc_temporal features using 1v1v1 approach**
3. **Extra Trees with mfcc_temporal features using 1v1v1 approach**
4. **Enhanced Ensemble** combining all three models with optimized weights

## Results Summary

| Model | Train Accuracy | Test Accuracy | F1 Score | Overfitting Gap |
|-------|---------------|---------------|----------|----------------|
| Enhanced_Ensemble | 0.9767 | 0.9324 | 0.9323 | 0.0443 |
| ET_1v1v1 | 1.0000 | 0.9179 | 0.9177 | 0.0821 |
| SVM_1v1v1 | 0.9503 | 0.9019 | 0.9018 | 0.0484 |
| SVM_TwoTier | 0.9482 | 0.8981 | 0.8981 | 0.0501 |

## Enhanced Ensemble

The enhanced ensemble combines all three models with the following weights:

- SVM Two-Tier: 0.20
- SVM 1v1v1: 0.40
- Extra Trees 1v1v1: 0.40

Cross-validation results: 0.9064 ± 0.0032

## Conclusions

The best performing model is **Enhanced_Ensemble** with a test accuracy of 0.9324 and F1 score of 0.9323.

### Key Findings

1. The **1v1v1 approach** is competitive with the Two-Tier approach (0.9019 vs 0.8981), suggesting that direct multi-class classification works well for this problem.

2. The **Enhanced Ensemble** approach successfully combines the strengths of all three models, achieving 0.9324 accuracy compared to 0.9179 for the best individual model.

3. **Data augmentation** techniques (time shifting, pitch shifting, and noise addition) increased the robustness of the models by exposing them to a wider variety of sound patterns.

### Recommendations

Based on the experimental results, we recommend:

1. Use the **Enhanced_Ensemble** model for production deployment, as it provides the best balance of accuracy and performance.

2. **Continue data collection**, particularly for sick chicken sounds, to further improve model robustness and address any remaining class imbalance.

3. **Consider deployment constraints** - if computational resources are limited, the SVM models might be preferred over the ensemble approach, while still maintaining high accuracy.

