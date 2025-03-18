import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.base import clone, BaseEstimator, ClassifierMixin
import librosa
import warnings
import scipy
from imblearn.over_sampling import SMOTE
import traceback
import pickle
from tqdm import tqdm
import itertools

# -------------------------------
# TwoTierClassifier definition (from Phase 2)
# -------------------------------
class TwoTierClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, tier1_model=None, tier2_model=None):
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        
    def fit(self, X, y):
        # Create binary labels for tier 1 (chicken vs noise)
        self.classes_ = np.unique(y)
        y_tier1 = np.array([0 if label in [0, 1] else 1 for label in y])
        self.tier1_model.fit(X, y_tier1)
        # Filter chicken samples (healthy and sick) for tier 2 training
        chicken_indices = np.where(y_tier1 == 0)[0]
        X_chicken = X[chicken_indices]
        y_chicken = y[chicken_indices]
        self.tier2_model.fit(X_chicken, y_chicken)
        return self
        
    def predict(self, X):
        y_pred_tier1 = self.tier1_model.predict(X)
        final_predictions = np.empty(shape=X.shape[0], dtype=int)
        # For samples predicted as noise in tier 1
        noise_indices = np.where(y_pred_tier1 == 1)[0]
        final_predictions[noise_indices] = 2  # noise=2
        # For samples predicted as chicken, use tier 2 model
        chicken_indices = np.where(y_pred_tier1 == 0)[0]
        if len(chicken_indices) > 0:
            X_chicken = X[chicken_indices]
            y_pred_tier2 = self.tier2_model.predict(X_chicken)
            final_predictions[chicken_indices] = y_pred_tier2
        return final_predictions
    
    def predict_proba(self, X):
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, n_classes))
        tier1_probas = self.tier1_model.predict_proba(X)
        # Set noise probability (class 2)
        probas[:, 2] = tier1_probas[:, 1]
        chicken_prob = tier1_probas[:, 0]
        tier2_probas = self.tier2_model.predict_proba(X)
        probas[:, 0] = tier2_probas[:, 0] * chicken_prob
        probas[:, 1] = tier2_probas[:, 1] * chicken_prob
        return probas

# -------------------------------
# General configurations and logging
# -------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Use phase4_experiment timestamp in results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'phase4_experiment_{timestamp}'

os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/models', exist_ok=True)
os.makedirs(f'{results_dir}/figures', exist_ok=True)
os.makedirs(f'{results_dir}/confusion_matrices', exist_ok=True)
os.makedirs(f'{results_dir}/feature_analysis', exist_ok=True)
os.makedirs(f'{results_dir}/checkpoints', exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{results_dir}/experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# File paths and label mapping
# -------------------------------
healthy_chicken_dir = 'dataset/Healthy'
sick_chicken_dir = 'dataset/Sick'
noise_dir = 'dataset/None'
label_map = {0: 'healthy', 1: 'sick', 2: 'noise'}

# -------------------------------
# Modified Checkpoint utility function
# -------------------------------
def save_checkpoint(data, checkpoint_name, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = results_dir
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints', f'{checkpoint_name}.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Checkpoint saved: {checkpoint_name} to {checkpoint_dir}")

def load_checkpoint(checkpoint_name, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = results_dir  # default to current results_dir
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints', f'{checkpoint_name}.pkl')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Checkpoint loaded: {checkpoint_name} from {checkpoint_dir}")
        return data
    return None

# -------------------------------
# Feature extraction and augmentation
# -------------------------------
def extract_mfcc_temporal_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if y.size == 0:
            return None
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
        logger.error(f"Error processing {file_path}: {e}")
        return None

def augment_with_time_shift(y, sr, shift_percent=0.2):
    shift_amount = int(np.random.uniform(-shift_percent, shift_percent) * len(y))
    return np.roll(y, shift_amount)

def augment_with_pitch_shift(y, sr, n_steps=2):
    n_steps = np.random.uniform(-n_steps, n_steps)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def augment_with_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def augment_audio(file_path, label, augmentation_count=2):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if y.size == 0:
            return [], []
        features_list = []
        labels_list = []
        # Original features
        orig_features = extract_mfcc_temporal_features(file_path)
        if orig_features is not None:
            features_list.append(orig_features)
            labels_list.append(label)
        # Time shifting
        for _ in tqdm(range(augmentation_count), desc="Time shift augmentations", leave=False):
            shifted_y = augment_with_time_shift(y, sr)
            temp_path = 'temp_augmented.wav'
            scipy.io.wavfile.write(temp_path, sr, shifted_y.astype(np.float32))
            shifted_features = extract_mfcc_temporal_features(temp_path)
            if shifted_features is not None:
                features_list.append(shifted_features)
                labels_list.append(label)
        # Pitch shifting
        for _ in tqdm(range(augmentation_count), desc="Pitch shift augmentations", leave=False):
            pitched_y = augment_with_pitch_shift(y, sr)
            temp_path = 'temp_augmented.wav'
            scipy.io.wavfile.write(temp_path, sr, pitched_y.astype(np.float32))
            pitched_features = extract_mfcc_temporal_features(temp_path)
            if pitched_features is not None:
                features_list.append(pitched_features)
                labels_list.append(label)
        # Adding noise
        for _ in tqdm(range(augmentation_count), desc="Noise augmentations", leave=False):
            noisy_y = augment_with_noise(y)
            temp_path = 'temp_augmented.wav'
            scipy.io.wavfile.write(temp_path, sr, noisy_y.astype(np.float32))
            noisy_features = extract_mfcc_temporal_features(temp_path)
            if noisy_features is not None:
                features_list.append(noisy_features)
                labels_list.append(label)
        if os.path.exists('temp_augmented.wav'):
            os.remove('temp_augmented.wav')
        return features_list, labels_list
    except Exception as e:
        logger.error(f"Error augmenting {file_path}: {e}")
        return [], []

# -------------------------------
# Prepare dataset (now with option to skip augmentation)
# -------------------------------
def prepare_dataset(use_augmentation=False, class_balance=True, use_test_mode=False, checkpoint_dir=None):
    checkpoint = load_checkpoint('dataset', checkpoint_dir)
    if checkpoint is not None:
        return checkpoint['X'], checkpoint['y']
    
    X, y = [], []
    file_limit = 10 if use_test_mode else float('inf')
    
    logger.info("Processing healthy chicken audio files...")
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
    logger.info("Processing sick chicken audio files...")
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
    logger.info("Processing noise audio files...")
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
                
    X = np.array(X)
    y = np.array(y)
    
    if class_balance:
        logger.info("Applying SMOTE for class balancing...")
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
    plt.savefig(f'{results_dir}/figures/class_distribution.png')
    plt.close()
    
    save_checkpoint({'X': X, 'y': y}, 'dataset', checkpoint_dir)
    return X, y

# -------------------------------
# Feature selection analysis (with external checkpoint support)
# -------------------------------
def feature_selection_analysis(X, y, use_test_mode=False, checkpoint_dir=None):
    checkpoint = load_checkpoint('feature_selection', checkpoint_dir)
    if checkpoint is not None:
        return checkpoint
    
    logger.info("Performing feature selection analysis...")
    feature_selection_results = {}
    
    # Extra Trees feature importance
    try:
        et = ExtraTreesClassifier(n_estimators=50 if use_test_mode else 100, random_state=42)
        et.fit(X, y)
        feature_importance = et.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance (Extra Trees)")
        plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        plt.savefig(f'{results_dir}/feature_analysis/extra_trees_importance.png')
        plt.close()
        importance_df = pd.DataFrame({'Feature Index': range(X.shape[1]), 'Importance': feature_importance}).sort_values('Importance', ascending=False)
        importance_df.to_csv(f'{results_dir}/feature_analysis/feature_importance.csv', index=False)
        feature_selection_results['et_importance'] = feature_importance
    except Exception as e:
        logger.error(f"Error in Extra Trees feature importance: {e}")
        logger.error(traceback.format_exc())
        feature_selection_results['et_importance'] = np.ones(X.shape[1]) / X.shape[1]
    
    all_features_mask = np.ones(X.shape[1], dtype=bool)
    feature_selection_results['all_features'] = all_features_mask
    
    # RFECV
    try:
        X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.7, random_state=42, stratify=y)
        svm = SVC(kernel='linear', C=10)
        rfecv = RFECV(estimator=svm,
                      step=2 if use_test_mode else 1,
                      cv=StratifiedKFold(2 if use_test_mode else 3),
                      scoring='accuracy',
                      n_jobs=-1,
                      min_features_to_select=5)
        rfecv.fit(X_sample, y_sample)
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel("CV score")
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.tight_layout()
        plt.savefig(f'{results_dir}/feature_analysis/rfecv_scores.png')
        plt.close()
        selected_features = np.where(rfecv.support_)[0]
        np.save(f'{results_dir}/feature_analysis/rfecv_selected_features.npy', selected_features)
        logger.info(f"RFECV selected {len(selected_features)} features")
        feature_selection_results['rfecv_mask'] = rfecv.support_
    except Exception as e:
        logger.error(f"RFECV failed: {e}")
        logger.error(traceback.format_exc())
        top_features = np.argsort(feature_selection_results.get('et_importance', all_features_mask))[::-1][:15]
        rfecv_mask = np.zeros(X.shape[1], dtype=bool)
        rfecv_mask[top_features] = True
        np.save(f'{results_dir}/feature_analysis/rfecv_selected_features.npy', np.where(rfecv_mask)[0])
        feature_selection_results['rfecv_mask'] = rfecv_mask
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
        plt.savefig(f'{results_dir}/feature_analysis/mutual_information.png')
        plt.close()
        mi_df = pd.DataFrame({'Feature Index': range(X.shape[1]), 'Mutual Information': mi_scores}).sort_values('Mutual Information', ascending=False)
        mi_df.to_csv(f'{results_dir}/feature_analysis/mutual_information.csv', index=False)
        mi_selected = mi_indices[:15]
        np.save(f'{results_dir}/feature_analysis/mi_selected_features.npy', mi_selected)
        mi_mask = np.zeros(X.shape[1], dtype=bool)
        mi_mask[mi_selected] = True
        feature_selection_results['mi_mask'] = mi_mask
    except Exception as e:
        logger.error(f"Mutual information calculation failed: {e}")
        logger.error(traceback.format_exc())
        mi_mask = all_features_mask
        feature_selection_results['mi_mask'] = mi_mask
        np.save(f'{results_dir}/feature_analysis/mi_selected_features.npy', np.arange(X.shape[1]))
    
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
        plt.savefig(f'{results_dir}/feature_analysis/pca_variance.png')
        plt.close()
        joblib.dump(pca, f'{results_dir}/feature_analysis/pca_model.pkl')
        logger.info(f"PCA selected {pca.n_components_} components")
        feature_selection_results['pca'] = pca
    except Exception as e:
        logger.error(f"PCA failed: {e}")
        logger.error(traceback.format_exc())
        pca = PCA(n_components=min(X.shape[0], X.shape[1]))
        pca.fit(X)
        joblib.dump(pca, f'{results_dir}/feature_analysis/pca_model.pkl')
        feature_selection_results['pca'] = pca
        logger.info("Using fallback for PCA")
    
    save_checkpoint(feature_selection_results, 'feature_selection', checkpoint_dir)
    return feature_selection_results

# -------------------------------
# Optuna objective functions for tuning
# -------------------------------
def objective_svm(trial, X_train, y_train):
    C = trial.suggest_float('C', 1, 20, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.05, 0.075, 0.1, 0.25])
    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    return score

def objective_svm_tier1(trial, X_train, y_tier1):
    C = trial.suggest_float('C', 1, 20, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.05, 0.075, 0.1, 0.25])
    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
    score = cross_val_score(model, X_train, y_tier1, cv=3, scoring='accuracy', n_jobs=-1).mean()
    return score

def objective_svm_tier2(trial, X_chicken, y_chicken):
    C = trial.suggest_float('C', 1, 20, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.05, 0.075, 0.1, 0.25])
    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
    score = cross_val_score(model, X_chicken, y_chicken, cv=3, scoring='accuracy', n_jobs=-1).mean()
    return score

def objective_et(trial, X_train, y_train):
    n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42
    )
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    return score

# -------------------------------
# Training functions using Optuna tuning
# -------------------------------
def train_recommended_models_optuna(X, y, feature_selection_results, use_test_mode=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f'{results_dir}/models/scaler.pkl')
    # In this tuning phase, we use all features (no selection)
    X_train_selected, X_test_selected = X_train_scaled, X_test_scaled
    recommended_models = {}
    
    # Two-Tier SVM
    y_tier1 = np.array([0 if label in [0, 1] else 1 for label in y_train])
    study_tier1 = optuna.create_study(direction='maximize')
    n_trials = 50 if not use_test_mode else 10
    study_tier1.optimize(lambda trial: objective_svm_tier1(trial, X_train_selected, y_tier1), n_trials=n_trials)
    best_params_tier1 = study_tier1.best_params
    tier1_model = SVC(kernel='rbf', probability=True, random_state=42, **best_params_tier1)
    tier1_model.fit(X_train_selected, y_tier1)
    
    chicken_indices = np.where(y_tier1 == 0)[0]
    X_chicken = X_train_selected[chicken_indices]
    y_chicken = y_train[chicken_indices]
    study_tier2 = optuna.create_study(direction='maximize')
    study_tier2.optimize(lambda trial: objective_svm_tier2(trial, X_chicken, y_chicken), n_trials=n_trials)
    best_params_tier2 = study_tier2.best_params
    tier2_model = SVC(kernel='rbf', probability=True, random_state=42, **best_params_tier2)
    tier2_model.fit(X_chicken, y_chicken)
    
    svm_two_tier = TwoTierClassifier(tier1_model, tier2_model)
    svm_two_tier.fit(X_train_selected, y_train)
    
    y_train_pred = svm_two_tier.predict(X_train_selected)
    y_test_pred = svm_two_tier.predict(X_test_selected)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    recommended_models['SVM_TwoTier'] = {
        'model': svm_two_tier,
        'name': 'SVM_TwoTier',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'overfitting_gap': train_acc - test_acc,
        'params': {'tier1': best_params_tier1, 'tier2': best_params_tier2}
    }
    
    # Save study_tier1 for visualization purposes
    recommended_models['study_tier1'] = study_tier1
    
    # SVM 1v1v1
    study_svm = optuna.create_study(direction='maximize')
    study_svm.optimize(lambda trial: objective_svm(trial, X_train_selected, y_train), n_trials=n_trials)
    best_params_svm = study_svm.best_params
    svm_1v1v1 = SVC(kernel='rbf', probability=True, random_state=42, **best_params_svm)
    svm_1v1v1.fit(X_train_selected, y_train)
    
    y_train_pred = svm_1v1v1.predict(X_train_selected)
    y_test_pred = svm_1v1v1.predict(X_test_selected)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    recommended_models['SVM_1v1v1'] = {
        'model': svm_1v1v1,
        'name': 'SVM_1v1v1',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'overfitting_gap': train_acc - test_acc,
        'params': best_params_svm
    }
    
    # Extra Trees 1v1v1
    study_et = optuna.create_study(direction='maximize')
    study_et.optimize(lambda trial: objective_et(trial, X_train_selected, y_train), n_trials=n_trials)
    best_params_et = study_et.best_params
    et_1v1v1 = ExtraTreesClassifier(random_state=42, **best_params_et)
    et_1v1v1.fit(X_train_selected, y_train)
    
    y_train_pred = et_1v1v1.predict(X_train_selected)
    y_test_pred = et_1v1v1.predict(X_test_selected)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    recommended_models['ET_1v1v1'] = {
        'model': et_1v1v1,
        'name': 'ET_1v1v1',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'overfitting_gap': train_acc - test_acc,
        'params': best_params_et
    }
    
    return recommended_models

def train_enhanced_ensemble_optuna(X, y, recommended_models, use_test_mode=False):
    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    svm_two_tier = recommended_models['SVM_TwoTier']['model']
    svm_1v1v1 = recommended_models['SVM_1v1v1']['model']
    et_1v1v1 = recommended_models['ET_1v1v1']['model']
    
    def objective_ensemble(trial):
        w1 = trial.suggest_float('w1', 0.1, 0.6)
        w2 = trial.suggest_float('w2', 0.1, 0.6)
        w3 = 1.0 - w1 - w2
        if w3 < 0.1 or w3 > 0.6:
            return 0.0
        ensemble_proba = (w1 * svm_two_tier.predict_proba(X_val_scaled) +
                          w2 * svm_1v1v1.predict_proba(X_val_scaled) +
                          w3 * et_1v1v1.predict_proba(X_val_scaled))
        val_pred = np.argmax(ensemble_proba, axis=1)
        return accuracy_score(y_val, val_pred)
    
    study_ensemble = optuna.create_study(direction='maximize')
    n_trials = 50 if not use_test_mode else 10
    study_ensemble.optimize(objective_ensemble, n_trials=n_trials)
    best_weights = study_ensemble.best_trial.params
    best_w1 = best_weights['w1']
    best_w2 = best_weights['w2']
    best_w3 = 1.0 - best_w1 - best_w2

    def ensemble_predict_proba(X_scaled):
        return (best_w1 * svm_two_tier.predict_proba(X_scaled) +
                best_w2 * svm_1v1v1.predict_proba(X_scaled) +
                best_w3 * et_1v1v1.predict_proba(X_scaled))
    def ensemble_predict(X_scaled):
        proba = ensemble_predict_proba(X_scaled)
        return np.argmax(proba, axis=1)
    
    # Compute training accuracy on the ensemble using training set
    y_train_pred = ensemble_predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    y_test_pred = ensemble_predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    ensemble_result = {
        'name': 'Enhanced_Ensemble',
        'models': {
            'SVM_TwoTier': svm_two_tier,
            'SVM_1v1v1': svm_1v1v1,
            'ET_1v1v1': et_1v1v1
        },
        'weights': (best_w1, best_w2, best_w3),
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'cv_accuracy': 0,  # dummy values if not computed
        'cv_std': 0,
        'scaler': scaler
    }
    joblib.dump(ensemble_result, f'{results_dir}/models/enhanced_ensemble.pkl')
    return ensemble_result

def final_model_evaluation(recommended_models, ensemble_result, X, y, checkpoint_dir=None):
    checkpoint = load_checkpoint('final_evaluation', checkpoint_dir)
    if checkpoint is not None:
        return checkpoint
    
    try:
        all_results = []
        for model_name, model_info in recommended_models.items():
            if model_name == 'study_tier1':
                continue
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
        results_df.to_csv(f'{results_dir}/final_model_comparison.csv', index=False)
        best_idx = results_df['Test Accuracy'].idxmax()
        best_model = results_df.iloc[best_idx]
        logger.info(f"Best model: {best_model['Model']} with test accuracy {best_model['Test Accuracy']:.4f}")
        
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
        plt.savefig(f'{results_dir}/figures/final_model_comparison.png')
        plt.close()
        
        with open(f'{results_dir}/final_report.md', 'w') as f:
            f.write("# Chicken Sound Classification - Phase 4 Final Report\n\n")
            f.write("## Overview\n\n")
            f.write("This experiment uses previously saved data (without re-augmentations) and an enhanced ensemble with Optuna tuning for hyperparameters and ensemble weights.\n\n")
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
                best_single_acc = max([m['test_accuracy'] for m in recommended_models.values() if isinstance(m, dict) and 'test_accuracy' in m])
                if ensemble_acc > best_single_acc:
                    f.write(f"2. The **Enhanced Ensemble** successfully combines the models, achieving {ensemble_acc:.4f} accuracy compared to {best_single_acc:.4f}.\n\n")
                else:
                    f.write(f"2. The **Enhanced Ensemble** ({ensemble_acc:.4f}) did not significantly improve over the best individual model ({best_single_acc:.4f}).\n\n")
            f.write("3. Skipping data augmentation allowed us to quickly tune and evaluate models using existing checkpoints.\n\n")
            f.write("### Recommendations\n\n")
            f.write(f"1. Deploy the **{best_model['Model']}** model for production.\n")
            f.write("2. Continue data collection, especially for sick chicken sounds, to further improve performance.\n")
            f.write("3. Consider deployment constraints—if resources are limited, the SVM models remain a viable option.\n")
        save_checkpoint(best_model['Model'], 'final_evaluation', checkpoint_dir)
        return best_model['Model']
    except Exception as e:
        logger.error(f"Error in final model evaluation: {e}")
        logger.error(traceback.format_exc())
        return None

# -------------------------------
# Additional visualization functions
# -------------------------------
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{results_dir}/confusion_matrices/confusion_matrix.png')
    plt.close()

# -------------------------------
# Main function with overall progress bar
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='Run phase 4 experiment with Optuna tuning.')
    parser.add_argument('--test', action='store_true', help='Run in test mode with simplified parameters')
    # New argument for external checkpoint directory
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory with existing checkpoints to load from')
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running in TEST MODE with simplified parameters")
    
    try:
        logger.info("Starting Phase 4 experiment")
        overall_steps = 5
        pbar = tqdm(total=overall_steps, desc="Overall progress")
        
        logger.info("Step 1: Preparing dataset without augmentation")
        X, y = prepare_dataset(use_augmentation=False, class_balance=True, use_test_mode=args.test, checkpoint_dir=args.checkpoint_dir)
        pbar.update(1)
        
        logger.info("Step 2: Performing feature selection analysis")
        feature_selection_results = feature_selection_analysis(X, y, use_test_mode=args.test, checkpoint_dir=args.checkpoint_dir)
        pbar.update(1)
        
        logger.info("Step 3: Training recommended models (Optuna tuning)")
        recommended_models = train_recommended_models_optuna(X, y, feature_selection_results, use_test_mode=args.test)
        pbar.update(1)
        
        logger.info("Step 4: Training enhanced ensemble with weight tuning (Optuna)")
        ensemble_result = train_enhanced_ensemble_optuna(X, y, recommended_models, use_test_mode=args.test)
        pbar.update(1)
        
        logger.info("Step 5: Performing final evaluation and generating report")
        best_model_name = final_model_evaluation(recommended_models, ensemble_result, X, y, checkpoint_dir=args.checkpoint_dir)
        pbar.update(1)
        pbar.close()
        
        # Additional Visualizations
        # Recompute test set for visualization
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = joblib.load(f'{results_dir}/models/scaler.pkl')
        X_test_scaled = scaler.transform(X_test)
        # Select the best model (from recommended_models)
        model = recommended_models[best_model_name]['model']
        y_test_pred = model.predict(X_test_scaled)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plot_confusion_matrix(cm, classes=[label_map[0], label_map[1], label_map[2]])
        
        # ROC Curve for class 1 (sick) - one-vs-all approach
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='best')
        plt.savefig(f'{results_dir}/figures/roc_curve.png')
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.savefig(f'{results_dir}/figures/precision_recall_curve.png')
        plt.close()
        
        # Distribution of a few sample features
        plt.figure(figsize=(12, 8))
        for i in range(5):  # adjust the number of features as needed
            plt.subplot(2, 3, i+1)
            sns.histplot(X[:, i], kde=True)
            plt.title(f'Feature {i}')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/figures/feature_distributions.png')
        plt.close()
        
        # Correlation heatmap
        corr_matrix = pd.DataFrame(X).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title('Feature Correlation Matrix')
        plt.savefig(f'{results_dir}/figures/feature_correlation_heatmap.png')
        plt.close()
        
        # Optuna visualizations (using study_tier1 stored in recommended_models)
        if 'study_tier1' in recommended_models:
            optuna.visualization.plot_optimization_history(recommended_models['study_tier1']).write_image(f'{results_dir}/figures/optuna_history_tier1.png')
            optuna.visualization.plot_param_importances(recommended_models['study_tier1']).write_image(f'{results_dir}/figures/optuna_param_importance_tier1.png')
        
        if args.test:
            logger.info("Test run completed successfully!")
        else:
            logger.info("Full experiment completed successfully!")
            
        logger.info(f"Results saved to {results_dir}")
        return best_model_name
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()
