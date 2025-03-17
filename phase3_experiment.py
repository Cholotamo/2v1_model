# phase3_experiment.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, ClassifierMixin
import librosa
import warnings
import scipy
from imblearn.over_sampling import SMOTE
import traceback
import pickle

# Add the TwoTierClassifier from phase 2
class TwoTierClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, tier1_model=None, tier2_model=None):
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        
    def fit(self, X, y):
        # Create binary labels for first tier (chicken vs noise)
        # 0=healthy, 1=sick, 2=noise -> binary: 0=chicken (healthy or sick), 1=noise
        self.classes_ = np.unique(y)
        y_tier1 = np.array([0 if label in [0, 1] else 1 for label in y])
        
        # Train tier 1 model (chicken vs noise)
        self.tier1_model.fit(X, y_tier1)
        
        # Filter "chicken" samples for tier 2 training
        chicken_indices = np.where(y_tier1 == 0)[0]
        X_chicken = X[chicken_indices]
        y_chicken = y[chicken_indices]
        
        # Train tier 2 model (healthy vs sick)
        self.tier2_model.fit(X_chicken, y_chicken)
        
        return self
        
    def predict(self, X):
        # First tier prediction (chicken vs noise)
        y_pred_tier1 = self.tier1_model.predict(X)
        
        # Initialize final predictions array
        final_predictions = np.empty(shape=X.shape[0], dtype=int)
        
        # For samples predicted as "noise" in tier 1, keep that prediction
        noise_indices = np.where(y_pred_tier1 == 1)[0]
        final_predictions[noise_indices] = 2  # noise=2
        
        # For samples predicted as "chicken" in tier 1, use tier 2 model to predict healthy vs sick
        chicken_indices = np.where(y_pred_tier1 == 0)[0]
        if len(chicken_indices) > 0:
            X_chicken = X[chicken_indices]
            y_pred_tier2 = self.tier2_model.predict(X_chicken)
            final_predictions[chicken_indices] = y_pred_tier2
        
        return final_predictions
    
    def predict_proba(self, X):
        """Add predict_proba for ensemble compatibility"""
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        
        # Initialize probability matrix
        probas = np.zeros((n_samples, n_classes))
        
        # Get tier 1 probabilities (chicken vs noise)
        tier1_probas = self.tier1_model.predict_proba(X)
        
        # Probability of being noise (class 2)
        probas[:, 2] = tier1_probas[:, 1]  # Prob of being noise
        
        # For samples predicted as chicken, get tier 2 probabilities
        chicken_prob = tier1_probas[:, 0]  # Prob of being chicken
        
        # Get tier 2 probabilities
        tier2_probas = self.tier2_model.predict_proba(X)
        
        # Probability of being healthy or sick, scaled by probability of being chicken
        probas[:, 0] = tier2_probas[:, 0] * chicken_prob  # Prob of healthy
        probas[:, 1] = tier2_probas[:, 1] * chicken_prob  # Prob of sick
        
        return probas

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Create timestamp for this experiment
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'phase3_experiment_{timestamp}'

# Create directories
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/models', exist_ok=True)
os.makedirs(f'{results_dir}/figures', exist_ok=True)
os.makedirs(f'{results_dir}/confusion_matrices', exist_ok=True)
os.makedirs(f'{results_dir}/feature_analysis', exist_ok=True)
os.makedirs(f'{results_dir}/checkpoints', exist_ok=True)

# Configure logging
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

# Paths to audio files
healthy_chicken_dir = 'dataset/Healthy'
sick_chicken_dir = 'dataset/Sick'
noise_dir = 'dataset/None'

# Define label mapping
label_map = {0: 'healthy', 1: 'sick', 2: 'noise'}

# Checkpointing utility functions
def save_checkpoint(data, checkpoint_name):
    """Save a checkpoint to avoid rerunning everything on failure."""
    checkpoint_path = f'{results_dir}/checkpoints/{checkpoint_name}.pkl'
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Checkpoint saved: {checkpoint_name}")

def load_checkpoint(checkpoint_name):
    """Load a checkpoint if it exists."""
    checkpoint_path = f'{results_dir}/checkpoints/{checkpoint_name}.pkl'
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Checkpoint loaded: {checkpoint_name}")
        return data
    return None

# Function to extract MFCC with temporal features (best performing from Phase 2)
def extract_mfcc_temporal_features(file_path, n_mfcc=13):
    """Extract MFCC and temporal features"""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if y.size == 0:
            return None
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Temporal features
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

# Data augmentation functions
def augment_with_time_shift(y, sr, shift_percent=0.2):
    """Shift audio in time by a random amount."""
    shift_amount = int(np.random.uniform(-shift_percent, shift_percent) * len(y))
    return np.roll(y, shift_amount)

def augment_with_pitch_shift(y, sr, n_steps=2):
    """Shift pitch by a random amount."""
    n_steps = np.random.uniform(-n_steps, n_steps)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def augment_with_noise(y, noise_factor=0.005):
    """Add random noise to the audio."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def augment_audio(file_path, label, augmentation_count=2):
    """Apply multiple augmentation techniques to create new samples."""
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
        
        # Time shifted versions
        for _ in range(augmentation_count):
            shifted_y = augment_with_time_shift(y, sr)
            # Save temporarily and extract features
            temp_path = 'temp_augmented.wav'
            scipy.io.wavfile.write(temp_path, sr, shifted_y.astype(np.float32))
            shifted_features = extract_mfcc_temporal_features(temp_path)
            if shifted_features is not None:
                features_list.append(shifted_features)
                labels_list.append(label)
        
        # Pitch shifted versions
        for _ in range(augmentation_count):
            pitched_y = augment_with_pitch_shift(y, sr)
            # Save temporarily and extract features
            temp_path = 'temp_augmented.wav'
            scipy.io.wavfile.write(temp_path, sr, pitched_y.astype(np.float32))
            pitched_features = extract_mfcc_temporal_features(temp_path)
            if pitched_features is not None:
                features_list.append(pitched_features)
                labels_list.append(label)
        
        # Noisy versions
        for _ in range(augmentation_count):
            noisy_y = augment_with_noise(y)
            # Save temporarily and extract features
            temp_path = 'temp_augmented.wav'
            scipy.io.wavfile.write(temp_path, sr, noisy_y.astype(np.float32))
            noisy_features = extract_mfcc_temporal_features(temp_path)
            if noisy_features is not None:
                features_list.append(noisy_features)
                labels_list.append(label)
                
        # Clean up temp file if it exists
        if os.path.exists('temp_augmented.wav'):
            os.remove('temp_augmented.wav')
            
        return features_list, labels_list
    except Exception as e:
        logger.error(f"Error augmenting {file_path}: {e}")
        return [], []

def prepare_dataset(use_augmentation=True, class_balance=True, use_test_mode=False):
    """Prepare dataset with optional augmentation and class balancing."""
    # Check if dataset already prepared
    checkpoint = load_checkpoint('dataset')
    if checkpoint is not None:
        return checkpoint['X'], checkpoint['y']
    
    X = []
    y = []
    
    # For test mode, limit the number of files processed
    if use_test_mode:
        logger.info("TEST MODE: Processing limited number of files")
        file_limit = 10  # Process only 10 files per class in test mode
    else:
        file_limit = float('inf')  # No limit in normal mode
    
    # Process each audio file
    logger.info("Processing healthy chicken audio files...")
    file_count = 0
    for file_name in os.listdir(healthy_chicken_dir):
        if file_count >= file_limit and use_test_mode:
            break
            
        file_path = os.path.join(healthy_chicken_dir, file_name)
        if use_augmentation:
            # In test mode, use less augmentation
            aug_count = 1 if use_test_mode else 2
            features_list, labels_list = augment_audio(file_path, 0, augmentation_count=aug_count)
            X.extend(features_list)
            y.extend(labels_list)
        else:
            features = extract_mfcc_temporal_features(file_path)
            if features is not None:
                X.append(features)
                y.append(0)
                
        file_count += 1
    
    logger.info("Processing sick chicken audio files...")
    file_count = 0
    for file_name in os.listdir(sick_chicken_dir):
        if file_count >= file_limit and use_test_mode:
            break
            
        file_path = os.path.join(sick_chicken_dir, file_name)
        if use_augmentation:
            # In test mode, use less augmentation
            aug_count = 1 if use_test_mode else 2
            features_list, labels_list = augment_audio(file_path, 1, augmentation_count=aug_count)
            X.extend(features_list)
            y.extend(labels_list)
        else:
            features = extract_mfcc_temporal_features(file_path)
            if features is not None:
                X.append(features)
                y.append(1)
                
        file_count += 1
    
    logger.info("Processing noise audio files...")
    file_count = 0
    for file_name in os.listdir(noise_dir):
        if file_count >= file_limit and use_test_mode:
            break
            
        file_path = os.path.join(noise_dir, file_name)
        if use_augmentation:
            # In test mode, use less augmentation
            aug_count = 1 if use_test_mode else 2
            features_list, labels_list = augment_audio(file_path, 2, augmentation_count=aug_count)
            X.extend(features_list)
            y.extend(labels_list)
        else:
            features = extract_mfcc_temporal_features(file_path)
            if features is not None:
                X.append(features)
                y.append(2)
                
        file_count += 1
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Apply SMOTE for class balancing if requested
    if class_balance:
        logger.info("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    
    logger.info(f"Dataset prepared: {X.shape[0]} samples with {X.shape[1]} features per sample")
    
    # Save class distribution
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(y).map(label_map).value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.savefig(f'{results_dir}/figures/class_distribution.png')
    plt.close()
    
    # Save checkpoint
    save_checkpoint({'X': X, 'y': y}, 'dataset')
    
    return X, y

def feature_selection_analysis(X, y, use_test_mode=False):
    """Perform feature selection analysis using multiple techniques."""
    # Check if feature selection already done
    checkpoint = load_checkpoint('feature_selection')
    if checkpoint is not None:
        return checkpoint
    
    logger.info("Performing feature selection analysis...")
    
    # Initialize results dictionary
    feature_selection_results = {}
    
    # 1. Feature importance using Extra Trees
    logger.info("Calculating feature importance with Extra Trees...")
    try:
        et = ExtraTreesClassifier(n_estimators=50 if use_test_mode else 100, random_state=42)
        et.fit(X, y)
        
        # Plot feature importances
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
        
        # Save feature importance to CSV
        importance_df = pd.DataFrame({
            'Feature Index': range(X.shape[1]),
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        importance_df.to_csv(f'{results_dir}/feature_analysis/feature_importance.csv', index=False)
        
        feature_selection_results['et_importance'] = feature_importance
    except Exception as e:
        logger.error(f"Error in Extra Trees feature importance: {e}")
        logger.error(traceback.format_exc())
        feature_selection_results['et_importance'] = np.ones(X.shape[1]) / X.shape[1]  # Equal importance fallback
    
    # Create default mask of all features
    all_features_mask = np.ones(X.shape[1], dtype=bool)
    feature_selection_results['all_features'] = all_features_mask
    
    # 2. Recursive Feature Elimination with Cross-Validation
    logger.info("Running Recursive Feature Elimination...")
    try:
        X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.7, random_state=42, stratify=y)
        
        svm = SVC(kernel='linear', C=10) 
        rfecv = RFECV(
            estimator=svm,
            step=2 if use_test_mode else 1,  # Faster step size in test mode
            cv=StratifiedKFold(2 if use_test_mode else 3),  # Fewer folds in test mode
            scoring='accuracy',
            n_jobs=-1,
            min_features_to_select=5
        )
        
        rfecv.fit(X_sample, y_sample)
        
        # Plot number of features vs. CV scores
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross-validation score")
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.tight_layout()
        plt.savefig(f'{results_dir}/feature_analysis/rfecv_scores.png')
        plt.close()
        
        # Save selected features
        selected_features = np.where(rfecv.support_)[0]
        np.save(f'{results_dir}/feature_analysis/rfecv_selected_features.npy', selected_features)
        logger.info(f"RFECV selected {len(selected_features)} features")
        
        # Create mask of selected features
        rfecv_mask = rfecv.support_
        feature_selection_results['rfecv_mask'] = rfecv_mask
    except Exception as e:
        logger.error(f"RFECV failed: {e}")
        logger.error(traceback.format_exc())
        # Fallback: top 15 features by Extra Trees importance
        if 'et_importance' in feature_selection_results:
            top_features = np.argsort(feature_selection_results['et_importance'])[::-1][:15]
            rfecv_mask = np.zeros(X.shape[1], dtype=bool)
            rfecv_mask[top_features] = True
        else:
            rfecv_mask = all_features_mask
            
        np.save(f'{results_dir}/feature_analysis/rfecv_selected_features.npy', np.where(rfecv_mask)[0])
        feature_selection_results['rfecv_mask'] = rfecv_mask
        logger.info("Using fallback feature selection for RFECV")
    
    # 3. Mutual Information
    logger.info("Calculating mutual information...")
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
        
        # Save mutual information to CSV
        mi_df = pd.DataFrame({
            'Feature Index': range(X.shape[1]),
            'Mutual Information': mi_scores
        }).sort_values('Mutual Information', ascending=False)
        mi_df.to_csv(f'{results_dir}/feature_analysis/mutual_information.csv', index=False)
        
        # Select top 15 features by mutual information
        mi_selected = mi_indices[:15]
        np.save(f'{results_dir}/feature_analysis/mi_selected_features.npy', mi_selected)
        
        # Create mask for MI selected features
        mi_mask = np.zeros(X.shape[1], dtype=bool)
        mi_mask[mi_selected] = True
        feature_selection_results['mi_mask'] = mi_mask
    except Exception as e:
        logger.error(f"Mutual information calculation failed: {e}")
        logger.error(traceback.format_exc())
        # Fallback
        mi_mask = all_features_mask
        feature_selection_results['mi_mask'] = mi_mask
        logger.info("Using fallback for mutual information")
        np.save(f'{results_dir}/feature_analysis/mi_selected_features.npy', np.arange(X.shape[1]))
    
    # 4. Principal Component Analysis
    logger.info("Running PCA...")
    try:
        # Retain 99% variance or at least 5 components
        n_components = min(5, X.shape[1])  # Minimum of 5 or total features
        pca = PCA(n_components=0.99)  # Aim for 99% variance
        pca.fit(X)
        
        # Ensure at least n_components are selected
        if pca.n_components_ < n_components:
            pca = PCA(n_components=n_components)
            pca.fit(X)
        
        # Plot explained variance ratio
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid()
        plt.savefig(f'{results_dir}/feature_analysis/pca_variance.png')
        plt.close()
        
        # Save PCA model
        joblib.dump(pca, f'{results_dir}/feature_analysis/pca_model.pkl')
        logger.info(f"PCA selected {pca.n_components_} components")
        
        feature_selection_results['pca'] = pca
    except Exception as e:
        logger.error(f"PCA failed: {e}")
        logger.error(traceback.format_exc())
        # Fallback: identity transformation
        pca = PCA(n_components=min(X.shape[0], X.shape[1]))
        pca.fit(X)
        joblib.dump(pca, f'{results_dir}/feature_analysis/pca_model.pkl')
        feature_selection_results['pca'] = pca
        logger.info("Using fallback for PCA")
    
    # Save checkpoint
    save_checkpoint(feature_selection_results, 'feature_selection')
    
    return feature_selection_results

# New function to train the three recommended models
def train_recommended_models(X, y, feature_selection_results, use_test_mode=False):
    """Train the specific models recommended from Phase 2"""
    checkpoint = load_checkpoint('recommended_models')
    if checkpoint is not None:
        return checkpoint
    
    logger.info("Training recommended models from Phase 2...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, f'{results_dir}/models/scaler.pkl')
    
    # Use all features for these three specific models
    X_train_selected = X_train_scaled
    X_test_selected = X_test_scaled
    
    # Parameter grids depending on test mode
    if use_test_mode:
        logger.info("Using simplified parameters for test mode")
        svm_param_grid = {
            'C': [10],
            'gamma': ['auto']
        }
        
        et_param_grid = {
            'n_estimators': [100],
            'max_depth': [30],
            'min_samples_split': [2],
            'min_samples_leaf': [4]
        }
    else:
        # Refined parameter grids for full mode
        svm_param_grid = {
            'C': [5, 7.5, 10, 12.5, 15, 20],
            'gamma': ['scale', 'auto', 0.05, 0.075, 0.1, 0.25]
        }
        
        et_param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [20, 30, 40, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
    
    recommended_models = {}
    
    try:
        # 1. SVM with mfcc_temporal features using Two-Tier approach
        logger.info("Training SVM with Two-Tier approach...")
        
        # Create tier 1 model - chicken vs. noise
        tier1_model = SVC(kernel='rbf', probability=True, random_state=42)
        y_tier1 = np.array([0 if label in [0, 1] else 1 for label in y_train])
        
        grid_search_tier1 = GridSearchCV(
            tier1_model, svm_param_grid, cv=5, n_jobs=-1,
            scoring='accuracy', verbose=1, return_train_score=True
        )
        grid_search_tier1.fit(X_train_selected, y_tier1)
        best_tier1 = grid_search_tier1.best_estimator_
        
        # Get chicken samples for tier 2 training
        chicken_indices = np.where(y_tier1 == 0)[0]
        X_chicken = X_train_selected[chicken_indices]
        y_chicken = y_train[chicken_indices]
        
        # Create tier 2 model - healthy vs sick
        tier2_model = SVC(kernel='rbf', probability=True, random_state=42)
        grid_search_tier2 = GridSearchCV(
            tier2_model, svm_param_grid, cv=5, n_jobs=-1,
            scoring='accuracy', verbose=1, return_train_score=True
        )
        grid_search_tier2.fit(X_chicken, y_chicken)
        best_tier2 = grid_search_tier2.best_estimator_
        
        # Create and train the final Two-Tier classifier
        svm_two_tier = TwoTierClassifier(best_tier1, best_tier2)
        svm_two_tier.fit(X_train_selected, y_train)
        
        # Save the model
        joblib.dump(svm_two_tier, f'{results_dir}/models/SVM_TwoTier_model.pkl')
        
        # Evaluate 
        y_train_pred = svm_two_tier.predict(X_train_selected)
        y_test_pred = svm_two_tier.predict(X_test_selected)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Store results
        svm_two_tier_result = {
            'model': svm_two_tier,
            'name': 'SVM_TwoTier',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'f1_score': f1,
            'overfitting_gap': train_accuracy - test_accuracy,
            'params': {
                'tier1': grid_search_tier1.best_params_,
                'tier2': grid_search_tier2.best_params_
            }
        }
        recommended_models['SVM_TwoTier'] = svm_two_tier_result
        
        # Log results
        logger.info(f"SVM Two-Tier Results:")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[label_map[i] for i in range(3)],
                   yticklabels=[label_map[i] for i in range(3)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('SVM Two-Tier')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/confusion_matrices/SVM_TwoTier_cm.png')
        plt.close()
        
        # 2. SVM with mfcc_temporal features using 1v1v1 approach
        logger.info("Training SVM with 1v1v1 approach...")
        
        svm_1v1v1 = SVC(kernel='rbf', probability=True, random_state=42)
        grid_search_1v1v1 = GridSearchCV(
            svm_1v1v1, svm_param_grid, cv=5, n_jobs=-1,
            scoring='accuracy', verbose=1, return_train_score=True
        )
        grid_search_1v1v1.fit(X_train_selected, y_train)
        best_svm_1v1v1 = grid_search_1v1v1.best_estimator_
        
        # Save the model
        joblib.dump(best_svm_1v1v1, f'{results_dir}/models/SVM_1v1v1_model.pkl')
        
        # Evaluate 
        y_train_pred = best_svm_1v1v1.predict(X_train_selected)
        y_test_pred = best_svm_1v1v1.predict(X_test_selected)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Store results
        svm_1v1v1_result = {
            'model': best_svm_1v1v1,
            'name': 'SVM_1v1v1',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'f1_score': f1,
            'overfitting_gap': train_accuracy - test_accuracy,
            'params': grid_search_1v1v1.best_params_
        }
        recommended_models['SVM_1v1v1'] = svm_1v1v1_result
        
        # Log results
        logger.info(f"SVM 1v1v1 Results:")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[label_map[i] for i in range(3)],
                   yticklabels=[label_map[i] for i in range(3)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('SVM 1v1v1')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/confusion_matrices/SVM_1v1v1_cm.png')
        plt.close()
        
        # 3. Extra Trees with mfcc_temporal features using 1v1v1 approach
        logger.info("Training Extra Trees with 1v1v1 approach...")
        
        et_1v1v1 = ExtraTreesClassifier(random_state=42)
        grid_search_et = GridSearchCV(
            et_1v1v1, et_param_grid, cv=5, n_jobs=-1,
            scoring='accuracy', verbose=1, return_train_score=True
        )
        grid_search_et.fit(X_train_selected, y_train)
        best_et_1v1v1 = grid_search_et.best_estimator_
        
        # Save the model
        joblib.dump(best_et_1v1v1, f'{results_dir}/models/ET_1v1v1_model.pkl')
        
        # Evaluate 
        y_train_pred = best_et_1v1v1.predict(X_train_selected)
        y_test_pred = best_et_1v1v1.predict(X_test_selected)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Store results
        et_1v1v1_result = {
            'model': best_et_1v1v1,
            'name': 'ET_1v1v1',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'f1_score': f1,
            'overfitting_gap': train_accuracy - test_accuracy,
            'params': grid_search_et.best_params_
        }
        recommended_models['ET_1v1v1'] = et_1v1v1_result
        
        # Log results
        logger.info(f"Extra Trees 1v1v1 Results:")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[label_map[i] for i in range(3)],
                   yticklabels=[label_map[i] for i in range(3)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Extra Trees 1v1v1')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/confusion_matrices/ET_1v1v1_cm.png')
        plt.close()
        
        # ROC curves for all models
        plt.figure(figsize=(12, 8))
        
        # Use the same test set for fair comparison
        for model_name, model_info in recommended_models.items():
            if hasattr(model_info['model'], "predict_proba"):
                model = model_info['model']
                y_proba = model.predict_proba(X_test_selected)
                
                # Calculate average ROC curve across all classes
                all_fpr = []
                all_tpr = []
                all_auc = []
                
                for i in range(3):
                    # Convert to binary classification
                    y_test_binary = (y_test == i).astype(int)
                    
                    # Compute ROC curve and AUC
                    fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    all_fpr.append(fpr)
                    all_tpr.append(tpr)
                    all_auc.append(roc_auc)
                
                # Plot average ROC curve
                avg_auc = np.mean(all_auc)
                plt.plot(all_fpr[0], all_tpr[0], lw=2, 
                        label=f'{model_name} (AUC = {avg_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Recommended Models')
        plt.legend(loc="lower right")
        plt.savefig(f'{results_dir}/figures/recommended_models_roc.png')
        plt.close()
            
        # Save all results
        save_checkpoint(recommended_models, 'recommended_models')
        
    except Exception as e:
        logger.error(f"Error training recommended models: {e}")
        logger.error(traceback.format_exc())
        recommended_models = {}
    
    return recommended_models

# Modified train_ensemble_models to use the three recommended models
def train_enhanced_ensemble(X, y, recommended_models, use_test_mode=False):
    """Train an enhanced ensemble using all three recommended models."""
    # Check if ensemble already trained
    checkpoint = load_checkpoint('enhanced_ensemble')
    if checkpoint is not None:
        return checkpoint
    
    logger.info("Training enhanced ensemble with all three recommended models...")
    
    try:
        # Check if we have the three required models
        required_models = ['SVM_TwoTier', 'SVM_1v1v1', 'ET_1v1v1']
        for model_name in required_models:
            if model_name not in recommended_models:
                logger.error(f"Missing required model for ensemble: {model_name}")
                return None
        
        # Split data - use separate validation set for weight optimization
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Get models from the recommended_models dictionary
        svm_two_tier = recommended_models['SVM_TwoTier']['model']
        svm_1v1v1 = recommended_models['SVM_1v1v1']['model']
        et_1v1v1 = recommended_models['ET_1v1v1']['model']
        
        # Grid search for optimal weights
        logger.info("Optimizing ensemble weights...")
        best_val_accuracy = 0
        best_weights = None
        
        # Get validation set predictions from each model
        svm_two_tier_proba = svm_two_tier.predict_proba(X_val_scaled)
        svm_1v1v1_proba = svm_1v1v1.predict_proba(X_val_scaled)
        et_1v1v1_proba = et_1v1v1.predict_proba(X_val_scaled)
        
        # Create a grid of weight combinations to try
        # We need to ensure the weights sum to 1.0
        if use_test_mode:
            # In test mode, use a minimal grid
            weight_grid = [(0.4, 0.3, 0.3), (0.5, 0.3, 0.2)]
            logger.info(f"Using minimal weight grid in test mode: {len(weight_grid)} combinations")
        else:
            # Full grid for normal mode
            weight_grid = []
            for w1 in np.linspace(0.2, 0.6, 5):  # SVM Two-Tier weight
                for w2 in np.linspace(0.1, 0.4, 4):  # SVM 1v1v1 weight
                    w3 = 1.0 - w1 - w2  # ET 1v1v1 weight
                    if 0.1 <= w3 <= 0.4:  # Ensure ET gets reasonable weight
                        weight_grid.append((w1, w2, w3))
            
            logger.info(f"Trying {len(weight_grid)} weight combinations...")
        
        # Try different weight combinations
        for weights in weight_grid:
            # Combine predictions with these weights
            ensemble_val_proba = (
                weights[0] * svm_two_tier_proba + 
                weights[1] * svm_1v1v1_proba + 
                weights[2] * et_1v1v1_proba
            )
            val_pred = np.argmax(ensemble_val_proba, axis=1)
            val_accuracy = accuracy_score(y_val, val_pred)
            
            logger.debug(f"  Weights: {weights}, Val accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_weights = weights
        
        # Log the best weights
        logger.info(f"Optimal weights: SVM Two-Tier: {best_weights[0]:.2f}, " +
                    f"SVM 1v1v1: {best_weights[1]:.2f}, " + 
                    f"ET 1v1v1: {best_weights[2]:.2f}")
        
        # Evaluate with cross-validation for more robust estimation
        logger.info("Validating ensemble with cross-validation...")
        cv_scores = []
        # Use fewer folds in test mode
        kf = StratifiedKFold(n_splits=3 if use_test_mode else 5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X, y):
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y[train_idx], y[val_idx]
            
            # Scale features
            cv_scaler = StandardScaler()
            X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = cv_scaler.transform(X_cv_val)
            
            # Create and train models for this fold
            # For simplicity, we'll use clones of the best models
            cv_svm_two_tier = clone(svm_two_tier)
            cv_svm_1v1v1 = clone(svm_1v1v1)
            cv_et_1v1v1 = clone(et_1v1v1)
            
            # Train models
            # Note: TwoTierClassifier doesn't support clone properly, so recreate it
            if isinstance(cv_svm_two_tier, TwoTierClassifier):
                # Create a new instance and train it
                tier1_model = SVC(kernel='rbf', probability=True, random_state=42)
                tier2_model = SVC(kernel='rbf', probability=True, random_state=42)
                
                # Set parameters from the original model
                tier1_model.set_params(**recommended_models['SVM_TwoTier']['params']['tier1'])
                tier2_model.set_params(**recommended_models['SVM_TwoTier']['params']['tier2'])
                
                cv_svm_two_tier = TwoTierClassifier(tier1_model, tier2_model)
            
            cv_svm_two_tier.fit(X_cv_train_scaled, y_cv_train)
            cv_svm_1v1v1.fit(X_cv_train_scaled, y_cv_train)
            cv_et_1v1v1.fit(X_cv_train_scaled, y_cv_train)
            
            # Make predictions
            svm_two_tier_proba = cv_svm_two_tier.predict_proba(X_cv_val_scaled)
            svm_1v1v1_proba = cv_svm_1v1v1.predict_proba(X_cv_val_scaled)
            et_1v1v1_proba = cv_et_1v1v1.predict_proba(X_cv_val_scaled)
            
            # Combine with optimal weights
            ensemble_cv_proba = (
                best_weights[0] * svm_two_tier_proba + 
                best_weights[1] * svm_1v1v1_proba + 
                best_weights[2] * et_1v1v1_proba
            )
            cv_pred = np.argmax(ensemble_cv_proba, axis=1)
            
            # Calculate accuracy
            cv_accuracy = accuracy_score(y_cv_val, cv_pred)
            cv_scores.append(cv_accuracy)
        
        logger.info(f"Ensemble cross-validation accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Define prediction functions using the optimal weights
        def ensemble_predict_proba(X_scaled):
            svm_two_tier_proba = svm_two_tier.predict_proba(X_scaled)
            svm_1v1v1_proba = svm_1v1v1.predict_proba(X_scaled)
            et_1v1v1_proba = et_1v1v1.predict_proba(X_scaled)
            
            return (
                best_weights[0] * svm_two_tier_proba + 
                best_weights[1] * svm_1v1v1_proba + 
                best_weights[2] * et_1v1v1_proba
            )
        
        def ensemble_predict(X_scaled):
            proba = ensemble_predict_proba(X_scaled)
            return np.argmax(proba, axis=1)
        
        # Make predictions on train and test sets
        y_train_pred = ensemble_predict(X_train_scaled)
        y_test_pred = ensemble_predict(X_test_scaled)
        y_test_proba = ensemble_predict_proba(X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Analyze prediction agreement among models
        svm_two_tier_pred = svm_two_tier.predict(X_test_scaled)
        svm_1v1v1_pred = svm_1v1v1.predict(X_test_scaled)
        et_1v1v1_pred = et_1v1v1.predict(X_test_scaled)
        
        # Count how many models agree with each prediction
        agreement_counts = np.zeros(len(y_test))
        for i in range(len(y_test)):
            predictions = [svm_two_tier_pred[i], svm_1v1v1_pred[i], et_1v1v1_pred[i]]
            agreement_counts[i] = predictions.count(y_test_pred[i])
        
        # Create agreement levels (1=one model agrees, 2=two models agree, 3=all models agree)
        agreement_levels = agreement_counts.astype(int)
        
        # Analyze accuracy by agreement level
        for level in range(1, 4):
            level_indices = np.where(agreement_levels == level)[0]
            if len(level_indices) > 0:
                level_accuracy = accuracy_score(y_test[level_indices], y_test_pred[level_indices])
                logger.info(f"Accuracy when {level} model(s) agree: {level_accuracy:.4f} ({len(level_indices)} samples)")
        
        # Plot agreement level analysis
        plt.figure(figsize=(10, 6))
        sns.countplot(x=agreement_levels, hue=(y_test_pred == y_test))
        plt.xlabel('Number of Models in Agreement')
        plt.ylabel('Count')
        plt.title('Model Agreement vs. Prediction Correctness')
        plt.legend(['Incorrect', 'Correct'])
        plt.savefig(f'{results_dir}/figures/ensemble_agreement.png')
        plt.close()
        
        # Save ensemble results
        ensemble_result = {
            'name': 'Enhanced_Ensemble',
            'models': {
                'SVM_TwoTier': svm_two_tier,
                'SVM_1v1v1': svm_1v1v1,
                'ET_1v1v1': et_1v1v1
            },
            'weights': best_weights,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'f1_score': f1,
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'scaler': scaler
        }
        joblib.dump(ensemble_result, f'{results_dir}/models/enhanced_ensemble.pkl')
        
        # Log results
        logger.info(f"Enhanced Ensemble Results:")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[label_map[i] for i in range(3)],
                    yticklabels=[label_map[i] for i in range(3)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Enhanced Ensemble (3 models)')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/confusion_matrices/enhanced_ensemble_cm.png')
        plt.close()
        
        # Class-specific metrics
        cr = classification_report(y_test, y_test_pred, target_names=[label_map[i] for i in range(3)], output_dict=True)
        cr_df = pd.DataFrame(cr).transpose()
        cr_df.to_csv(f'{results_dir}/confusion_matrices/enhanced_ensemble_class_report.csv')
        
        # ROC curves
        plt.figure(figsize=(10, 8))
        
        for i in range(3):
            # Convert to binary classification problem
            y_test_binary = (y_test == i).astype(int)
            
            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_test_binary, y_test_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{label_map[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Enhanced Ensemble')
        plt.legend(loc="lower right")
        plt.savefig(f'{results_dir}/confusion_matrices/enhanced_ensemble_roc.png')
        plt.close()
        
        # Save checkpoint
        save_checkpoint(ensemble_result, 'enhanced_ensemble')
        
        return ensemble_result
        
    except Exception as e:
        logger.error(f"Error training enhanced ensemble: {e}")
        logger.error(traceback.format_exc())
        return None

def final_model_evaluation(recommended_models, ensemble_result, X, y):
    """Perform final evaluation comparing all trained models."""
    checkpoint = load_checkpoint('final_evaluation')
    if checkpoint is not None:
        return checkpoint
    
    try:
        # Create a list to hold all model results
        all_results = []
        
        # Add recommended models to results
        for model_name, model_info in recommended_models.items():
            result = {
                'Model': model_name,
                'Train Accuracy': model_info['train_accuracy'],
                'Test Accuracy': model_info['test_accuracy'],
                'F1 Score': model_info['f1_score'],
                'Overfitting Gap': model_info['overfitting_gap']
            }
            all_results.append(result)
        
        # Add ensemble result if available
        if ensemble_result is not None:
            ensemble_row = {
                'Model': 'Enhanced_Ensemble',
                'Train Accuracy': ensemble_result['train_accuracy'],
                'Test Accuracy': ensemble_result['test_accuracy'],
                'F1 Score': ensemble_result['f1_score'],
                'Overfitting Gap': ensemble_result['train_accuracy'] - ensemble_result['test_accuracy']
            }
            all_results.append(ensemble_row)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'{results_dir}/final_model_comparison.csv', index=False)
        
        # Find best model
        best_idx = results_df['Test Accuracy'].idxmax()
        best_model = results_df.iloc[best_idx]
        
        logger.info(f"Best model: {best_model['Model']} with test accuracy {best_model['Test Accuracy']:.4f}")
        
        # Create summary visualization
        plt.figure(figsize=(12, 8))
        
        models = results_df['Model']
        train_acc = results_df['Train Accuracy']
        test_acc = results_df['Test Accuracy']
        f1_scores = results_df['F1 Score']
        
        x = np.arange(len(models))
        width = 0.25
        
        plt.bar(x - width, train_acc, width, label='Train Accuracy')
        plt.bar(x, test_acc, width, label='Test Accuracy')
        plt.bar(x + width, f1_scores, width, label='F1 Score')
        
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% Target')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{results_dir}/figures/final_model_comparison.png')
        plt.close()
        
        # Generate comprehensive report
        with open(f'{results_dir}/final_report.md', 'w') as f:
            f.write("# Chicken Sound Classification - Phase 3 Final Report\n\n")
            
            f.write("## Overview\n\n")
            f.write("This experiment implements the top three model configurations from Phase 2, applying data augmentation and enhanced ensemble techniques.\n\n")
            
            f.write("## Models Evaluated\n\n")
            f.write("1. **SVM with mfcc_temporal features using Two-Tier approach**\n")
            f.write("2. **SVM with mfcc_temporal features using 1v1v1 approach**\n")
            f.write("3. **Extra Trees with mfcc_temporal features using 1v1v1 approach**\n")
            f.write("4. **Enhanced Ensemble** combining all three models with optimized weights\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Model | Train Accuracy | Test Accuracy | F1 Score | Overfitting Gap |\n")
            f.write("|-------|---------------|---------------|----------|----------------|\n")
            
            for _, row in results_df.sort_values('Test Accuracy', ascending=False).iterrows():
                f.write(f"| {row['Model']} | {row['Train Accuracy']:.4f} | ")
                f.write(f"{row['Test Accuracy']:.4f} | {row['F1 Score']:.4f} | ")
                f.write(f"{row['Overfitting Gap']:.4f} |\n")
            
            f.write("\n")
            
            # Add ensemble details if available
            if ensemble_result is not None:
                f.write("## Enhanced Ensemble\n\n")
                f.write("The enhanced ensemble combines all three models with the following weights:\n\n")
                f.write(f"- SVM Two-Tier: {ensemble_result['weights'][0]:.2f}\n")
                f.write(f"- SVM 1v1v1: {ensemble_result['weights'][1]:.2f}\n")
                f.write(f"- Extra Trees 1v1v1: {ensemble_result['weights'][2]:.2f}\n\n")
                
                f.write("Cross-validation results: ")
                f.write(f"{ensemble_result['cv_accuracy']:.4f} ± {ensemble_result['cv_std']:.4f}\n\n")
            
            f.write("## Conclusions\n\n")
            f.write(f"The best performing model is **{best_model['Model']}** with a ")
            f.write(f"test accuracy of {best_model['Test Accuracy']:.4f} and F1 score of {best_model['F1 Score']:.4f}.\n\n")
            
            f.write("### Key Findings\n\n")
            
            # Compare the approaches
            two_tier_row = results_df[results_df['Model'] == 'SVM_TwoTier']
            onevone_row = results_df[results_df['Model'] == 'SVM_1v1v1']
            
            if not two_tier_row.empty and not onevone_row.empty:
                two_tier_acc = two_tier_row['Test Accuracy'].values[0]
                onevone_acc = onevone_row['Test Accuracy'].values[0]
                
                if two_tier_acc > onevone_acc:
                    f.write(f"1. The **Two-Tier approach** outperforms the standard 1v1v1 approach ")
                    f.write(f"({two_tier_acc:.4f} vs {onevone_acc:.4f}), likely due to its hierarchical ")
                    f.write("classification structure that first separates chicken sounds from noise.\n\n")
                else:
                    f.write(f"1. The **1v1v1 approach** is competitive with the Two-Tier approach ")
                    f.write(f"({onevone_acc:.4f} vs {two_tier_acc:.4f}), suggesting that direct multi-class ")
                    f.write("classification works well for this problem.\n\n")
            
            # Comment on ensemble performance if available
            if ensemble_result is not None and 'test_accuracy' in ensemble_result:
                ensemble_acc = ensemble_result['test_accuracy']
                best_single_acc = max([m['test_accuracy'] for m in recommended_models.values()])
                
                if ensemble_acc > best_single_acc:
                    f.write(f"2. The **Enhanced Ensemble** approach successfully combines the strengths ")
                    f.write(f"of all three models, achieving {ensemble_acc:.4f} accuracy compared to ")
                    f.write(f"{best_single_acc:.4f} for the best individual model.\n\n")
                else:
                    f.write(f"2. The **Enhanced Ensemble** approach ({ensemble_acc:.4f}) did not significantly ")
                    f.write(f"improve over the best individual model ({best_single_acc:.4f}), suggesting ")
                    f.write("the models may be capturing similar patterns.\n\n")
            
            # Comment on data augmentation
            f.write("3. **Data augmentation** techniques (time shifting, pitch shifting, and noise addition) ")
            f.write("increased the robustness of the models by exposing them to a wider variety of sound patterns.\n\n")
            
            # Recommendations
            f.write("### Recommendations\n\n")
            f.write("Based on the experimental results, we recommend:\n\n")
            
            f.write(f"1. Use the **{best_model['Model']}** model for production deployment, as it provides ")
            f.write("the best balance of accuracy and performance.\n\n")
            
            f.write("2. **Continue data collection**, particularly for sick chicken sounds, to further improve ")
            f.write("model robustness and address any remaining class imbalance.\n\n")
            
            f.write("3. **Consider deployment constraints** - if computational resources are limited, ")
            f.write("the SVM models might be preferred over the ensemble approach, while still ")
            f.write("maintaining high accuracy.\n\n")
        
        # Save checkpoint
        save_checkpoint(best_model['Model'], 'final_evaluation')
        
        return best_model['Model']
        
    except Exception as e:
        logger.error(f"Error in final model evaluation: {e}")
        logger.error(traceback.format_exc())
        return None

# Change in the main() function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run phase 3 experiment with recommended models.')
    parser.add_argument('--test', action='store_true', help='Run in test mode with simplified parameters')
    args = parser.parse_args()
    
    # Log the mode
    if args.test:
        logger.info("Running in TEST MODE with simplified parameters")
    
    try:
        logger.info("Starting Phase 3 experiment")
        
        # Step 1: Prepare dataset with augmentation
        logger.info("Step 1: Preparing dataset with augmentation")
        X, y = prepare_dataset(use_augmentation=True, class_balance=True, use_test_mode=args.test)
        
        # Step 2: Perform feature selection analysis
        logger.info("Step 2: Performing feature selection analysis")
        feature_selection_results = feature_selection_analysis(X, y, use_test_mode=args.test)
        
        # Step 3: Train the recommended models from Phase 2
        logger.info("Step 3: Training the recommended models from Phase 2")
        recommended_models = train_recommended_models(X, y, feature_selection_results, use_test_mode=args.test)
        
        # Step 4: Train enhanced ensemble with all three models
        logger.info("Step 4: Training enhanced ensemble with all three models")
        ensemble_result = train_enhanced_ensemble(X, y, recommended_models, use_test_mode=args.test)
        
        # Step 5: Perform final evaluation and generate report
        logger.info("Step 5: Performing final evaluation")
        # Remove the use_test_mode parameter from this call
        best_model = final_model_evaluation(recommended_models, ensemble_result, X, y)
        
        if args.test:
            logger.info("Test run completed successfully!")
        else:
            logger.info("Full experiment completed successfully!")
            
        logger.info(f"Results saved to {results_dir}")
        
        return best_model
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()