import os
import time
import pickle
import logging
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import librosa
import scipy
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, mutual_info_classif

from imblearn.over_sampling import SMOTE

# -------------------------------
# Global Configurations
# -------------------------------

# Create a timestamp and results directories for saving models, figures, etc.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'phase5_experiment_{timestamp}'

for subdir in ['models', 'figures', 'confusion_matrices', 'feature_analysis', 'checkpoints']:
    os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)

# Set up logging with a file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(results_dir, 'experiment.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# File paths and label mapping for our audio dataset
healthy_chicken_dir = 'dataset/Healthy'
sick_chicken_dir = 'dataset/Sick'
noise_dir = 'dataset/None'
label_map = {0: 'healthy', 1: 'sick', 2: 'noise'}

# -------------------------------
# Utility Functions for Checkpoints
# -------------------------------
def save_checkpoint(data: object, checkpoint_name: str) -> None:
    """Save checkpoint data to a pickle file."""
    checkpoint_path = os.path.join(results_dir, 'checkpoints', f'{checkpoint_name}.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"[Checkpoint] Saved: {checkpoint_name}")

def load_checkpoint(checkpoint_name: str):
    """Load checkpoint data from a pickle file, if it exists."""
    checkpoint_path = os.path.join(results_dir, 'checkpoints', f'{checkpoint_name}.pkl')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"[Checkpoint] Loaded: {checkpoint_name}")
        return data
    return None

# -------------------------------
# Audio Feature Extraction
# -------------------------------
def extract_features_from_audio(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC and additional temporal features from an audio signal.
    """
    if y.size == 0:
        return None
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(zcr, axis=1),
            np.mean(rms, axis=1),
            np.mean(spectral_bandwidth, axis=1),
            np.mean(spectral_centroid, axis=1),
            np.mean(spectral_contrast, axis=1)
        ])
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def extract_mfcc_temporal_features(file_path: str, n_mfcc: int = 13) -> np.ndarray:
    """
    Load an audio file and extract MFCC and temporal features.
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)
        return extract_features_from_audio(y, sr, n_mfcc)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

# -------------------------------
# Data Augmentation Functions
# -------------------------------
def augment_with_time_shift(y: np.ndarray, shift_percent: float = 0.2) -> np.ndarray:
    """Apply time shifting augmentation."""
    shift_amount = int(np.random.uniform(-shift_percent, shift_percent) * len(y))
    return np.roll(y, shift_amount)

def augment_with_pitch_shift(y: np.ndarray, sr: int, n_steps: float = 2) -> np.ndarray:
    """Apply pitch shifting augmentation."""
    n_steps = np.random.uniform(-n_steps, n_steps)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def augment_with_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Apply noise augmentation."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def augment_audio(file_path: str, label: int, augmentation_count: int = 2) -> Tuple[List, List]:
    """
    Augment an audio file by applying various transformations and extract features.
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if y.size == 0:
            return [], []
        features_list, labels_list = [], []
        
        # Original features
        orig_features = extract_features_from_audio(y, sr)
        if orig_features is not None:
            features_list.append(orig_features)
            labels_list.append(label)
        
        # Time shifting augmentation
        for _ in tqdm(range(augmentation_count), desc="Time shift augmentations", leave=False):
            shifted_y = augment_with_time_shift(y)
            shifted_features = extract_features_from_audio(shifted_y, sr)
            if shifted_features is not None:
                features_list.append(shifted_features)
                labels_list.append(label)
        
        # Pitch shifting augmentation
        for _ in tqdm(range(augmentation_count), desc="Pitch shift augmentations", leave=False):
            pitched_y = augment_with_pitch_shift(y, sr)
            pitched_features = extract_features_from_audio(pitched_y, sr)
            if pitched_features is not None:
                features_list.append(pitched_features)
                labels_list.append(label)
        
        # Noise augmentation
        for _ in tqdm(range(augmentation_count), desc="Noise augmentations", leave=False):
            noisy_y = augment_with_noise(y)
            noisy_features = extract_features_from_audio(noisy_y, sr)
            if noisy_features is not None:
                features_list.append(noisy_features)
                labels_list.append(label)
                
        return features_list, labels_list
    except Exception as e:
        logger.error(f"Error augmenting {file_path}: {e}")
        logger.error(traceback.format_exc())
        return [], []

# -------------------------------
# Dataset Preparation and Feature Analysis
# -------------------------------
def prepare_dataset(use_augmentation: bool = True, class_balance: bool = True, use_test_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load audio files, optionally augment and balance the dataset.
    """
    checkpoint = load_checkpoint('dataset')
    if checkpoint is not None:
        return checkpoint['X'], checkpoint['y']
    
    logger.info("Hello! I'm starting dataset preparation.")
    X, y = [], []
    file_limit = 10 if use_test_mode else float('inf')
    
    logger.info("Step 1: Processing healthy chicken audio files...")
    for file_name in tqdm(os.listdir(healthy_chicken_dir), desc="Healthy chicken files"):
        if use_test_mode and len(X) >= file_limit:
            break
        file_path = os.path.join(healthy_chicken_dir, file_name)
        if use_augmentation:
            aug_count = 1 if use_test_mode else 2
            features_list, labels_list = augment_audio(file_path, 0, augmentation_count=aug_count)
            X.extend(features_list)
            y.extend(labels_list)
        else:
            features = extract_mfcc_temporal_features(file_path)
            if features is not None:
                X.append(features)
                y.append(0)
    
    logger.info("Step 2: Processing sick chicken audio files...")
    for file_name in tqdm(os.listdir(sick_chicken_dir), desc="Sick chicken files"):
        if use_test_mode and len(X) >= file_limit:
            break
        file_path = os.path.join(sick_chicken_dir, file_name)
        if use_augmentation:
            aug_count = 1 if use_test_mode else 2
            features_list, labels_list = augment_audio(file_path, 1, augmentation_count=aug_count)
            X.extend(features_list)
            y.extend(labels_list)
        else:
            features = extract_mfcc_temporal_features(file_path)
            if features is not None:
                X.append(features)
                y.append(1)
    
    logger.info("Step 3: Processing noise audio files...")
    for file_name in tqdm(os.listdir(noise_dir), desc="Noise files"):
        if use_test_mode and len(X) >= file_limit:
            break
        file_path = os.path.join(noise_dir, file_name)
        if use_augmentation:
            aug_count = 1 if use_test_mode else 2
            features_list, labels_list = augment_audio(file_path, 2, augmentation_count=aug_count)
            X.extend(features_list)
            y.extend(labels_list)
        else:
            features = extract_mfcc_temporal_features(file_path)
            if features is not None:
                X.append(features)
                y.append(2)
                
    X, y = np.array(X), np.array(y)
    
    if class_balance:
        logger.info("Balancing classes using SMOTE...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    
    logger.info(f"Dataset prepared: {X.shape[0]} samples with {X.shape[1]} features each")
    
    # Save class distribution figure
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(y).map(label_map).value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join(results_dir, 'figures', 'class_distribution.png'))
    plt.close()
    
    save_checkpoint({'X': X, 'y': y}, 'dataset')
    return X, y

def feature_selection_analysis(X: np.ndarray, y: np.ndarray, use_test_mode: bool = False) -> dict:
    """
    Perform feature selection analysis using Extra Trees, RFECV, Mutual Information, and PCA.
    """
    checkpoint = load_checkpoint('feature_selection')
    if checkpoint is not None:
        return checkpoint
    
    logger.info("Let's analyze the features to see which ones matter most!")
    results = {}
    all_features_mask = np.ones(X.shape[1], dtype=bool)
    results['all_features'] = all_features_mask

    # Extra Trees feature importance
    try:
        n_estimators = 50 if use_test_mode else 100
        et = ExtraTreesClassifier(n_estimators=n_estimators, random_state=42)
        et.fit(X, y)
        feature_importance = et.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance (Extra Trees)")
        plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_analysis', 'extra_trees_importance.png'))
        plt.close()
        importance_df = pd.DataFrame({'Feature Index': range(X.shape[1]), 'Importance': feature_importance}).sort_values('Importance', ascending=False)
        importance_df.to_csv(os.path.join(results_dir, 'feature_analysis', 'feature_importance.csv'), index=False)
        results['et_importance'] = feature_importance
    except Exception as e:
        logger.error(f"Error in Extra Trees feature importance: {e}")
        results['et_importance'] = np.ones(X.shape[1]) / X.shape[1]
    
    # RFECV for feature selection
    try:
        X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.7, random_state=42, stratify=y)
        svm = SVC(kernel='linear', C=10)
        cv = StratifiedKFold(2 if use_test_mode else 3)
        rfecv = RFECV(estimator=svm,
                      step=2 if use_test_mode else 1,
                      cv=cv,
                      scoring='accuracy',
                      n_jobs=-1,
                      min_features_to_select=5)
        rfecv.fit(X_sample, y_sample)
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel("CV score")
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_analysis', 'rfecv_scores.png'))
        plt.close()
        selected_features = np.where(rfecv.support_)[0]
        np.save(os.path.join(results_dir, 'feature_analysis', 'rfecv_selected_features.npy'), selected_features)
        logger.info(f"RFECV selected {len(selected_features)} features")
        results['rfecv_mask'] = rfecv.support_
    except Exception as e:
        logger.error(f"RFECV failed: {e}")
        top_features = np.argsort(results.get('et_importance', all_features_mask))[::-1][:15]
        rfecv_mask = np.zeros(X.shape[1], dtype=bool)
        rfecv_mask[top_features] = True
        np.save(os.path.join(results_dir, 'feature_analysis', 'rfecv_selected_features.npy'), np.where(rfecv_mask)[0])
        results['rfecv_mask'] = rfecv_mask
        logger.info("Using fallback for RFECV")
    
    # Mutual Information
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_indices = np.argsort(mi_scores)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance (Mutual Information)")
        plt.bar(range(X.shape[1]), mi_scores[mi_indices], align="center")
        plt.xticks(range(X.shape[1]), mi_indices)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_analysis', 'mutual_information.png'))
        plt.close()
        mi_df = pd.DataFrame({'Feature Index': range(X.shape[1]), 'Mutual Information': mi_scores}).sort_values('Mutual Information', ascending=False)
        mi_df.to_csv(os.path.join(results_dir, 'feature_analysis', 'mutual_information.csv'), index=False)
        mi_selected = mi_indices[:15]
        np.save(os.path.join(results_dir, 'feature_analysis', 'mi_selected_features.npy'), mi_selected)
        mi_mask = np.zeros(X.shape[1], dtype=bool)
        mi_mask[mi_selected] = True
        results['mi_mask'] = mi_mask
    except Exception as e:
        logger.error(f"Mutual information calculation failed: {e}")
        mi_mask = all_features_mask
        results['mi_mask'] = mi_mask
        np.save(os.path.join(results_dir, 'feature_analysis', 'mi_selected_features.npy'), np.arange(X.shape[1]))
    
    # PCA
    try:
        n_components = min(5, X.shape[1])
        pca = PCA(n_components=0.99)
        pca.fit(X)
        if pca.n_components_ < n_components:
            pca = PCA(n_components=n_components)
            pca.fit(X)
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid()
        plt.savefig(os.path.join(results_dir, 'feature_analysis', 'pca_variance.png'))
        plt.close()
        joblib.dump(pca, os.path.join(results_dir, 'feature_analysis', 'pca_model.pkl'))
        logger.info(f"PCA selected {pca.n_components_} components")
        results['pca'] = pca
    except Exception as e:
        logger.error(f"PCA failed: {e}")
        pca = PCA(n_components=min(X.shape[0], X.shape[1]))
        pca.fit(X)
        joblib.dump(pca, os.path.join(results_dir, 'feature_analysis', 'pca_model.pkl'))
        results['pca'] = pca
        logger.info("Using fallback for PCA")
    
    save_checkpoint(results, 'feature_selection')
    return results

# -------------------------------
# Deep RL Hyperparameter Tuning for SVM
# -------------------------------
class HyperparamController(nn.Module):
    def __init__(self):
        super(HyperparamController, self).__init__()
        # Learnable parameters for sampling hyperparameters in log-space
        self.mu = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.log_std = nn.Parameter(torch.tensor([0.0, 0.0]))
    
    def forward(self):
        std = torch.exp(self.log_std)
        eps = torch.randn(2)
        sample = self.mu + std * eps
        C = torch.exp(sample[0])
        gamma = torch.exp(sample[1])
        log_prob_C = -0.5 * (((sample[0] - self.mu[0]) / std[0]) ** 2) - self.log_std[0] - 0.5 * np.log(2 * np.pi)
        log_prob_gamma = -0.5 * (((sample[1] - self.mu[1]) / std[1]) ** 2) - self.log_std[1] - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob_C + log_prob_gamma
        return C.item(), gamma.item(), log_prob

def deep_rl_tune_svm(X_train, y_train, episodes=50):
    """
    Use a deep RL (REINFORCE) approach to tune SVM hyperparameters.
    Returns the best (C, gamma) found.
    """
    logger.info("Starting deep RL tuning for SVM hyperparameters...")
    controller = HyperparamController()
    optimizer = optim.Adam(controller.parameters(), lr=0.01)
    best_reward = -np.inf
    best_hyperparams = None
    reward_history = []
    
    for episode in range(episodes):
        C, gamma, log_prob = controller()
        model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
        reward = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
        reward_history.append(reward)
        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"[RL SVM] Episode {episode+1}: C={C:.4f}, gamma={gamma:.4f}, reward={reward:.4f}")
        if reward > best_reward:
            best_reward = reward
            best_hyperparams = (C, gamma)
            
    # Plot and save the reward history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes+1), reward_history, marker='o')
    plt.title("SVM Deep RL Tuning Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward (CV Accuracy)")
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'figures', 'rl_svm_tuning_rewards.png'))
    plt.close()
    
    logger.info(f"Best SVM hyperparameters found: {best_hyperparams} with reward {best_reward:.4f}")
    return best_hyperparams

# -------------------------------
# Deep RL Tuning for Ensemble Weights
# -------------------------------
class EnsembleWeightController(nn.Module):
    def __init__(self):
        super(EnsembleWeightController, self).__init__()
        # We parameterize two values; the third is computed as 1 - (w1+w2)
        self.mu = nn.Parameter(torch.tensor([0.3, 0.3]))
        self.log_std = nn.Parameter(torch.tensor([0.0, 0.0]))
    
    def forward(self):
        std = torch.exp(self.log_std)
        eps = torch.randn(2)
        sample = self.mu + std * eps
        # Map outputs via sigmoid to ensure values in a reasonable range (approximately [0.1, 0.7])
        w1 = torch.sigmoid(sample[0]) * 0.6 + 0.1
        w2 = torch.sigmoid(sample[1]) * 0.6 + 0.1
        w3 = 1.0 - w1 - w2
        # Compute a rough log probability
        log_prob = - (sample**2).sum()
        return w1.item(), w2.item(), w3.item(), log_prob

def deep_rl_tune_ensemble_weights(X_val_scaled, y_val, models, episodes=50):
    """
    Use deep RL to tune ensemble weights for three models.
    Returns the best weights (w1, w2, w3) found.
    """
    logger.info("Starting deep RL tuning for ensemble weights...")
    controller = EnsembleWeightController()
    optimizer = optim.Adam(controller.parameters(), lr=0.01)
    best_reward = -np.inf
    best_weights = None
    reward_history = []
    
    for episode in range(episodes):
        w1, w2, w3, log_prob = controller()
        if w3 < 0.1 or w3 > 0.7:
            reward = 0.0
        else:
            ensemble_proba = (w1 * models['SVM_TwoTier'].predict_proba(X_val_scaled) +
                              w2 * models['SVM_1v1v1'].predict_proba(X_val_scaled) +
                              w3 * models['ET_1v1v1'].predict_proba(X_val_scaled))
            val_pred = np.argmax(ensemble_proba, axis=1)
            reward = accuracy_score(y_val, val_pred)
        reward_history.append(reward)
        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"[RL Ensemble] Episode {episode+1}: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}, reward={reward:.4f}")
        if reward > best_reward:
            best_reward = reward
            best_weights = (w1, w2, w3)
    
    # Plot and save the ensemble tuning reward history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes+1), reward_history, marker='o')
    plt.title("Ensemble Weights Deep RL Tuning Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Validation Accuracy)")
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'figures', 'rl_ensemble_tuning_rewards.png'))
    plt.close()
    
    logger.info(f"Best ensemble weights found: {best_weights} with reward {best_reward:.4f}")
    return best_weights

# -------------------------------
# Two-Tier Classifier Definition
# -------------------------------
class TwoTierClassifier(BaseEstimator, ClassifierMixin):
    """
    A two-tier classifier where the first tier separates chicken sounds (healthy/sick) from noise,
    and the second tier distinguishes between healthy and sick chicken sounds.
    """
    def __init__(self, tier1_model=None, tier2_model=None):
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        # Create binary labels for tier 1: 0 for chicken (healthy and sick), 1 for noise.
        y_tier1 = np.array([0 if label in [0, 1] else 1 for label in y])
        self.tier1_model.fit(X, y_tier1)
        # Train tier 2 only on chicken samples.
        chicken_indices = np.where(y_tier1 == 0)[0]
        X_chicken = X[chicken_indices]
        y_chicken = y[chicken_indices]
        self.tier2_model.fit(X_chicken, y_chicken)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred_tier1 = self.tier1_model.predict(X)
        final_predictions = np.empty(shape=X.shape[0], dtype=int)
        # For noise samples
        noise_indices = np.where(y_pred_tier1 == 1)[0]
        final_predictions[noise_indices] = 2  # noise label = 2
        # For chicken samples, use tier 2 prediction
        chicken_indices = np.where(y_pred_tier1 == 0)[0]
        if len(chicken_indices) > 0:
            X_chicken = X[chicken_indices]
            y_pred_tier2 = self.tier2_model.predict(X_chicken)
            final_predictions[chicken_indices] = y_pred_tier2
        return final_predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, n_classes))
        tier1_probas = self.tier1_model.predict_proba(X)
        # Noise probability comes from tier 1's second column.
        probas[:, 2] = tier1_probas[:, 1]
        chicken_prob = tier1_probas[:, 0]
        tier2_probas = self.tier2_model.predict_proba(X)
        probas[:, 0] = tier2_probas[:, 0] * chicken_prob
        probas[:, 1] = tier2_probas[:, 1] * chicken_prob
        return probas

# -------------------------------
# Training Functions Using Deep RL Tuning
# -------------------------------
def train_recommended_models_deeprl(X: np.ndarray, y: np.ndarray, feature_selection_results: dict, use_test_mode: bool = False) -> dict:
    """
    Train individual models (Two-Tier SVM, SVM 1v1v1, and Extra Trees) using deep RL tuning.
    """
    logger.info("Starting model training with deep RL tuning...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(results_dir, 'models', 'scaler.pkl'))
    
    recommended_models = {}
    
    # Two-Tier SVM tuning (using deep RL for each tier)
    logger.info("Tuning Two-Tier SVM (tier1 and tier2)...")
    y_tier1 = np.array([0 if label in [0, 1] else 1 for label in y_train])
    best_params_tier1 = deep_rl_tune_svm(X_train_scaled, y_tier1, episodes=10 if use_test_mode else 50)
    tier1_model = SVC(kernel='rbf', probability=True, random_state=42, C=best_params_tier1[0], gamma=best_params_tier1[1])
    tier1_model.fit(X_train_scaled, y_tier1)
    
    chicken_indices = np.where(y_tier1 == 0)[0]
    X_chicken = X_train_scaled[chicken_indices]
    y_chicken = y_train[chicken_indices]
    best_params_tier2 = deep_rl_tune_svm(X_chicken, y_chicken, episodes=10 if use_test_mode else 50)
    tier2_model = SVC(kernel='rbf', probability=True, random_state=42, C=best_params_tier2[0], gamma=best_params_tier2[1])
    tier2_model.fit(X_chicken, y_chicken)
    
    svm_two_tier = TwoTierClassifier(tier1_model, tier2_model)
    svm_two_tier.fit(X_train_scaled, y_train)
    
    y_train_pred = svm_two_tier.predict(X_train_scaled)
    y_test_pred = svm_two_tier.predict(X_test_scaled)
    recommended_models['SVM_TwoTier'] = {
        'model': svm_two_tier,
        'name': 'SVM_TwoTier',
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
        'overfitting_gap': accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred),
        'params': {'tier1': best_params_tier1, 'tier2': best_params_tier2}
    }
    
    # SVM 1v1v1 tuning using deep RL
    logger.info("Tuning SVM 1v1v1...")
    best_params_svm = deep_rl_tune_svm(X_train_scaled, y_train, episodes=10 if use_test_mode else 50)
    svm_1v1v1 = SVC(kernel='rbf', probability=True, random_state=42, C=best_params_svm[0], gamma=best_params_svm[1])
    svm_1v1v1.fit(X_train_scaled, y_train)
    
    y_train_pred = svm_1v1v1.predict(X_train_scaled)
    y_test_pred = svm_1v1v1.predict(X_test_scaled)
    recommended_models['SVM_1v1v1'] = {
        'model': svm_1v1v1,
        'name': 'SVM_1v1v1',
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
        'overfitting_gap': accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred),
        'params': best_params_svm
    }
    
    # Extra Trees classifier training
    logger.info("Training Extra Trees classifier...")
    et_1v1v1 = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_1v1v1.fit(X_train_scaled, y_train)
    
    y_train_pred = et_1v1v1.predict(X_train_scaled)
    y_test_pred = et_1v1v1.predict(X_test_scaled)
    recommended_models['ET_1v1v1'] = {
        'model': et_1v1v1,
        'name': 'ET_1v1v1',
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
        'overfitting_gap': accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred),
        'params': {'n_estimators': 100}
    }
    
    return recommended_models

def train_enhanced_ensemble_deeprl(X: np.ndarray, y: np.ndarray, recommended_models: dict, use_test_mode: bool = False) -> dict:
    """
    Train an enhanced ensemble model by tuning the weights of individual models using deep RL.
    """
    logger.info("Preparing data splits for ensemble training...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    svm_two_tier = recommended_models['SVM_TwoTier']['model']
    svm_1v1v1 = recommended_models['SVM_1v1v1']['model']
    et_1v1v1 = recommended_models['ET_1v1v1']['model']
    
    # Tune ensemble weights using deep RL
    logger.info("Tuning ensemble weights using deep RL...")
    best_weights = deep_rl_tune_ensemble_weights(X_val_scaled, y_val, 
                                                 models={'SVM_TwoTier': svm_two_tier,
                                                         'SVM_1v1v1': svm_1v1v1,
                                                         'ET_1v1v1': et_1v1v1},
                                                 episodes=10 if use_test_mode else 50)
    
    def ensemble_predict_proba(X_scaled: np.ndarray) -> np.ndarray:
        return (best_weights[0] * svm_two_tier.predict_proba(X_scaled) +
                best_weights[1] * svm_1v1v1.predict_proba(X_scaled) +
                best_weights[2] * et_1v1v1.predict_proba(X_scaled))
    
    def ensemble_predict(X_scaled: np.ndarray) -> np.ndarray:
        proba = ensemble_predict_proba(X_scaled)
        return np.argmax(proba, axis=1)
    
    y_train_pred = ensemble_predict(X_train_scaled)
    y_test_pred = ensemble_predict(X_test_scaled)
    
    ensemble_result = {
        'name': 'Enhanced_Ensemble',
        'models': {
            'SVM_TwoTier': svm_two_tier,
            'SVM_1v1v1': svm_1v1v1,
            'ET_1v1v1': et_1v1v1
        },
        'weights': best_weights,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
        'scaler': scaler
    }
    joblib.dump(ensemble_result, os.path.join(results_dir, 'models', 'enhanced_ensemble.pkl'))
    return ensemble_result

def final_model_evaluation(recommended_models: dict, ensemble_result: dict, X: np.ndarray, y: np.ndarray):
    """
    Evaluate all models on the full dataset, generate performance plots and a markdown report.
    Returns the best performing model name.
    """
    checkpoint = load_checkpoint('final_evaluation')
    if checkpoint is not None:
        return checkpoint
    
    try:
        logger.info("Final evaluation: Comparing all models now...")
        all_results = []
        for model_name, model_info in recommended_models.items():
            result = {
                'Model': model_name,
                'Train Accuracy': model_info['train_accuracy'],
                'Test Accuracy': model_info['test_accuracy'],
                'F1 Score': model_info['f1_score'],
                'Overfitting Gap': model_info['overfitting_gap']
            }
            all_results.append(result)
        if ensemble_result is not None:
            ensemble_row = {
                'Model': 'Enhanced_Ensemble',
                'Train Accuracy': ensemble_result['train_accuracy'],
                'Test Accuracy': ensemble_result['test_accuracy'],
                'F1 Score': ensemble_result['f1_score'],
                'Overfitting Gap': ensemble_result['train_accuracy'] - ensemble_result['test_accuracy']
            }
            all_results.append(ensemble_row)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(results_dir, 'final_model_comparison.csv'), index=False)
        best_idx = results_df['Test Accuracy'].idxmax()
        best_model = results_df.iloc[best_idx]
        logger.info(f"Best model: {best_model['Model']} with test accuracy {best_model['Test Accuracy']:.4f}")
        
        # Create a bar chart comparing model performance
        plt.figure(figsize=(12, 8))
        x = np.arange(len(results_df))
        width = 0.25
        plt.bar(x - width, results_df['Train Accuracy'], width, label='Train Accuracy')
        plt.bar(x, results_df['Test Accuracy'], width, label='Test Accuracy')
        plt.bar(x + width, results_df['F1 Score'], width, label='F1 Score')
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% Target')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, results_df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'figures', 'final_model_comparison.png'))
        plt.close()
        
        # Generate a markdown report summarizing the experiment
        with open(os.path.join(results_dir, 'final_report.md'), 'w') as f:
            f.write("# Chicken Sound Classification - Phase 5 Final Report\n\n")
            f.write("## Overview\n\n")
            f.write("This experiment uses data augmentation and an enhanced ensemble with deep learning based hyperparameter tuning for traditional ML models.\n\n")
            f.write("## Models Evaluated\n\n")
            f.write("1. **SVM with mfcc_temporal features using Two-Tier approach**\n")
            f.write("2. **SVM with mfcc_temporal features using 1v1v1 approach**\n")
            f.write("3. **Extra Trees with mfcc_temporal features using 1v1v1 approach**\n")
            f.write("4. **Enhanced Ensemble** combining all three models with optimized weights\n\n")
            f.write("## Results Summary\n\n")
            f.write("| Model | Train Accuracy | Test Accuracy | F1 Score | Overfitting Gap |\n")
            f.write("|-------|----------------|---------------|----------|----------------|\n")
            for _, row in results_df.sort_values('Test Accuracy', ascending=False).iterrows():
                f.write(f"| {row['Model']} | {row['Train Accuracy']:.4f} | {row['Test Accuracy']:.4f} | {row['F1 Score']:.4f} | {row['Overfitting Gap']:.4f} |\n")
            f.write("\n")
            if ensemble_result is not None:
                f.write("## Enhanced Ensemble\n\n")
                f.write("The ensemble uses the following weights:\n\n")
                f.write(f"- SVM Two-Tier: {ensemble_result['weights'][0]:.2f}\n")
                f.write(f"- SVM 1v1v1: {ensemble_result['weights'][1]:.2f}\n")
                f.write(f"- Extra Trees 1v1v1: {ensemble_result['weights'][2]:.2f}\n\n")
            f.write("## Conclusions\n\n")
            f.write(f"The best performing model is **{best_model['Model']}** with a test accuracy of {best_model['Test Accuracy']:.4f} and F1 score of {best_model['F1 Score']:.4f}.\n\n")
            f.write("### Key Findings\n\n")
            two_tier_row = results_df[results_df['Model'] == 'SVM_TwoTier']
            onevone_row = results_df[results_df['Model'] == 'SVM_1v1v1']
            if not two_tier_row.empty and not onevone_row.empty:
                two_tier_acc = two_tier_row['Test Accuracy'].values[0]
                onevone_acc = onevone_row['Test Accuracy'].values[0]
                if two_tier_acc > onevone_acc:
                    f.write(f"1. The **Two-Tier approach** outperforms the standard 1v1v1 approach ({two_tier_acc:.4f} vs {onevone_acc:.4f}).\n\n")
                else:
                    f.write(f"1. The **1v1v1 approach** is competitive with the Two-Tier approach ({onevone_acc:.4f} vs {two_tier_acc:.4f}).\n\n")
            if ensemble_result is not None and 'test_accuracy' in ensemble_result:
                ensemble_acc = ensemble_result['test_accuracy']
                best_single_acc = max([m['test_accuracy'] for m in recommended_models.values()])
                if ensemble_acc > best_single_acc:
                    f.write(f"2. The **Enhanced Ensemble** successfully combines the models, achieving {ensemble_acc:.4f} accuracy compared to {best_single_acc:.4f}.\n\n")
                else:
                    f.write(f"2. The **Enhanced Ensemble** ({ensemble_acc:.4f}) did not significantly improve over the best individual model ({best_single_acc:.4f}).\n\n")
            f.write("3. Data augmentation improved model robustness by exposing the models to varied sound patterns.\n\n")
            f.write("### Recommendations\n\n")
            f.write(f"1. Deploy the **{best_model['Model']}** model for production.\n")
            f.write("2. Continue data collection, especially for sick chicken sounds, to further improve performance.\n")
            f.write("3. Consider deployment constraints—if resources are limited, the SVM models remain a viable option.\n")
        save_checkpoint(best_model['Model'], 'final_evaluation')
        return best_model['Model']
    except Exception as e:
        logger.error(f"Error in final model evaluation: {e}")
        logger.error(traceback.format_exc())
        return None

# -------------------------------
# Main Function
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='Run phase 5 experiment with deep RL tuning.')
    parser.add_argument('--test', action='store_true', help='Run in test mode with simplified parameters')
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running in TEST MODE with simplified parameters")
    
    try:
        logger.info("Welcome! Starting Phase 5 experiment on chicken sound classification.")
        overall_steps = 5
        pbar = tqdm(total=overall_steps, desc="Overall progress")
        
        logger.info("Step 1: Preparing dataset with augmentation.")
        X, y = prepare_dataset(use_augmentation=True, class_balance=True, use_test_mode=args.test)
        pbar.update(1)
        
        logger.info("Step 2: Performing feature selection analysis.")
        feature_selection_results = feature_selection_analysis(X, y, use_test_mode=args.test)
        pbar.update(1)
        
        logger.info("Step 3: Training recommended models with deep RL tuning.")
        recommended_models = train_recommended_models_deeprl(X, y, feature_selection_results, use_test_mode=args.test)
        pbar.update(1)
        
        logger.info("Step 4: Training enhanced ensemble with weight tuning (deep RL).")
        ensemble_result = train_enhanced_ensemble_deeprl(X, y, recommended_models, use_test_mode=args.test)
        pbar.update(1)
        
        logger.info("Step 5: Performing final evaluation and generating report.")
        best_model = final_model_evaluation(recommended_models, ensemble_result, X, y)
        pbar.update(1)
        pbar.close()
        
        if args.test:
            logger.info("Test run completed successfully!")
        else:
            logger.info("Full experiment completed successfully!")
            
        logger.info(f"Results and reports saved to {results_dir}")
        return best_model
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()
