# 2V1_MODEL

## Team Name
**Group 6** (Example—replace with your actual team name if needed)

## Team Members
1. **Muhammad Yusri Bin Abdullah** (2302950)  
2. **Soon Jun Hong Samuel** (2300489)  
3. **Tamo Cholo Rafael Tandoc** (2302944)  
4. **Julian Teo** (2303039)  
5. **Mok Ji Yong Jason** (2303089)  
6. **Ng Shu Yi** (2302940)

---

## Introduction
This repository contains a two-stage (2v1) chicken sound classification system. It first separates **chicken sounds** from **noise**, then further classifies chicken sounds as **healthy** or **sick**. By applying advanced data processing, feature extraction, and machine learning approaches, we aim to detect early signs of illness in poultry.

### Why “2v1”?
- **Stage 1**: Distinguish chicken vocalizations (both healthy and sick) from non-chicken noise.  
- **Stage 2**: Differentiate between **healthy** and **sick** chicken vocalizations.

Our experiments evolve over multiple phases, each adding new features or optimization methods (e.g., data augmentation, ensemble techniques, hyperparameter tuning with Optuna and deep reinforcement learning).

---

## Repository Structure

```
2V1_MODEL/
├── dataset/
│   ├── Healthy/            # Audio files of healthy chickens
│   ├── Sick/               # Audio files of sick chickens
│   └── None/               # Noise or non-chicken audio files
├── phase1_experiment_YYYYMMDD_HHMMSS/
│   └── ...                 # Output/results for Phase 1 experiments
├── phase2_experiment_YYYYMMDD_HHMMSS/
│   └── ...                 # Output/results for Phase 2 experiments
├── phase3_experiment_YYYYMMDD_HHMMSS/
│   └── ...                 # Output/results for Phase 3 experiments
├── phase4_DL_20epi_experiment_YYYYMMDD_HHMMSS/
│   └── ...                 # Example: deep learning or advanced tuning experiment
├── phase4_optuna_experiment_YYYYMMDD_HHMMSS/
│   └── ...                 # Example: hyperparameter tuning with Optuna
├── .gitignore
├── phase1_experiment.py
├── phase2_experiment.py
├── phase3_experiment.py
├── phase4_DL.py
├── phase4_optuna.py
└── README.md
```

### Key Scripts
- **phase1_experiment.py**: Initial approach, focusing on architectural comparison (e.g., single-stage vs. Two-Tier).  
- **phase2_experiment.py**: Feature-model co-optimization (e.g., exploring MFCC variants, zero-crossing rate, etc.).  
- **phase3_experiment.py**: Advanced system design with data augmentation, ensemble learning, and SMOTE for class balancing.  
- **phase4_DL.py**: Incorporates deep learning-based tuning or reinforcement learning for hyperparameter optimization.  
- **phase4_optuna.py**: Demonstrates hyperparameter tuning using Optuna.  
- **hdbscan_featureextraction.py**: Example script for HDBSCAN-based clustering or feature extraction (if applicable).  
- **hdbscan_test_notinuse.py**: A test script for HDBSCAN that is currently not in use.

Each **phaseX_experiment_YYYYMMDD_HHMMSS** directory typically contains:
- Logs, metrics, and plots generated during that phase’s experiment.  
- Saved models and checkpoints.

---

## How to Run

1. **Data Preparation**  
   - Ensure your audio files are placed under `dataset/Healthy`, `dataset/Sick`, and `dataset/None`.

2. **Experiment Scripts**  
   - Run each phase’s script in order, or focus on the phase you want to explore. For example:
     ```bash
     python phase1_experiment.py
     ```
   - Each script will generate its own results directory (e.g., `phase1_experiment_YYYYMMDD_HHMMSS/`).

3. **Checking Results**  
   - Look inside the generated results directory for logs, figures, and model accuracy metrics.

4. **Optional Tuning**  
   - To explore deep reinforcement learning or Optuna-based hyperparameter tuning, run:
     ```bash
     python phase4_DL.py
     python phase4_optuna.py
     ```
   - These scripts will generate specialized tuning results in their respective folders.

---

## Future Work
- **Further Ensemble Methods**: Investigate stacking or blending multiple classifiers for higher accuracy.  
- **Larger Dataset**: Expand the audio dataset, especially for sick chicken sounds, to improve model generalization.  
- **Real-Time Deployment**: Optimize model inference speed for on-site poultry monitoring systems.

---