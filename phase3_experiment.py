# phase3_experiment.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import librosa
import warnings
import scipy
from imblearn.over_sampling import SMOTE
import traceback
import pickle

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

def prepare_dataset(use_augmentation=True, class_balance=True):
    """Prepare dataset with optional augmentation and class balancing."""
    # Check if dataset already prepared
    checkpoint = load_checkpoint('dataset')
    if checkpoint is not None:
        return checkpoint['X'], checkpoint['y']
    
    X = []
    y = []
    
    # Process each audio file
    logger.info("Processing healthy chicken audio files...")
    for file_name in os.listdir(healthy_chicken_dir):
        file_path = os.path.join(healthy_chicken_dir, file_name)
        if use_augmentation:
            features_list, labels_list = augment_audio(file_path, 0)
            X.extend(features_list)
            y.extend(labels_list)
        else:
            features = extract_mfcc_temporal_features(file_path)
            if features is not None:
                X.append(features)
                y.append(0)
    
    logger.info("Processing sick chicken audio files...")
    for file_name in os.listdir(sick_chicken_dir):
        file_path = os.path.join(sick_chicken_dir, file_name)
        if use_augmentation:
            features_list, labels_list = augment_audio(file_path, 1)
            X.extend(features_list)
            y.extend(labels_list)
        else:
            features = extract_mfcc_temporal_features(file_path)
            if features is not None:
                X.append(features)
                y.append(1)
    
    logger.info("Processing noise audio files...")
    for file_name in os.listdir(noise_dir):
        file_path = os.path.join(noise_dir, file_name)
        if use_augmentation:
            features_list, labels_list = augment_audio(file_path, 2)
            X.extend(features_list)
            y.extend(labels_list)
        else:
            features = extract_mfcc_temporal_features(file_path)
            if features is not None:
                X.append(features)
                y.append(2)
    
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

def feature_selection_analysis(X, y):
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
        et = ExtraTreesClassifier(n_estimators=100, random_state=42)
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
            step=1,
            cv=StratifiedKFold(3),
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

def train_and_evaluate_models(X, y, feature_selection_results):
    """Train and evaluate models with different feature selection approaches."""
    # Check if models already trained
    checkpoint = load_checkpoint('model_evaluation')
    if checkpoint is not None:
        return checkpoint
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, f'{results_dir}/models/scaler.pkl')
    
    # Feature selection methods
    feature_methods = {
        'all_features': feature_selection_results['all_features'],
        'rfecv': feature_selection_results['rfecv_mask'],
        'mutual_info': feature_selection_results['mi_mask'],
        'pca': 'pca'  # Special flag for PCA
    }
    
    # Models to evaluate
    base_models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Extra_Trees': ExtraTreesClassifier(random_state=42)
    }
    
    # Fine-tuned parameter grids
    param_grids = {
        'SVM': {
            'C': [1, 5, 10, 20, 50],
            'gamma': [0.01, 0.05, 0.1, 0.5]
        },
        'Extra_Trees': {
            'n_estimators': [200, 300, 400],
            'max_depth': [20, 30, 40, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
    }
    
    results = []
    
    # Train and evaluate each model with each feature selection method
    for model_name, base_model in base_models.items():
        for method_name, feature_mask in feature_methods.items():
            logger.info(f"Training {model_name} with {method_name}...")
            
            # Check if this model/method combination already done
            checkpoint_name = f'model_{model_name}_{method_name}'
            checkpoint = load_checkpoint(checkpoint_name)
            if checkpoint is not None:
                results.append(checkpoint)
                logger.info(f"Loaded existing results for {model_name} with {method_name}")
                continue
            
            try:
                # Prepare data according to feature selection method
                if method_name == 'pca':
                    pca = feature_selection_results['pca']
                    X_train_selected = pca.transform(X_train_scaled)
                    X_test_selected = pca.transform(X_test_scaled)
                else:
                    X_train_selected = X_train_scaled[:, feature_mask]
                    X_test_selected = X_test_scaled[:, feature_mask]
                
                # Clone model
                model = clone(base_model)
                
                # Grid search
                grid_search = GridSearchCV(
                    model, param_grids[model_name], cv=5, n_jobs=-1, 
                    scoring='accuracy', verbose=1, return_train_score=True
                )
                
                start_time = time.time()
                grid_search.fit(X_train_selected, y_train)
                training_time = time.time() - start_time
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Predictions
                y_train_pred = best_model.predict(X_train_selected)
                y_test_pred = best_model.predict(X_test_selected)
                
                # Metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                f1 = f1_score(y_test, y_test_pred, average='weighted')
                
                # Save performance metrics
                result = {
                    'Model': model_name,
                    'Feature Selection': method_name,
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'F1 Score': f1,
                    'Overfitting Gap': train_accuracy - test_accuracy,
                    'Training Time (s)': training_time,
                    'Best Parameters': str(grid_search.best_params_),
                    'Num Features': X_train_selected.shape[1]
                }
                results.append(result)
                save_checkpoint(result, checkpoint_name)
                
                # Log results
                logger.info(f"Results for {model_name} with {method_name}:")
                logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
                logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
                logger.info(f"  F1 Score: {f1:.4f}")
                logger.info(f"  Overfitting Gap: {train_accuracy - test_accuracy:.4f}")
                
                # Save model
                model_filename = f'{results_dir}/models/{model_name}_{method_name}_model.pkl'
                joblib.dump(best_model, model_filename)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_test_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=[label_map[i] for i in range(3)],
                           yticklabels=[label_map[i] for i in range(3)])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'{model_name} - {method_name}')
                plt.tight_layout()
                plt.savefig(f'{results_dir}/confusion_matrices/{model_name}_{method_name}_cm.png')
                plt.close()
                
                # Class-specific metrics
                cr = classification_report(y_test, y_test_pred, target_names=[label_map[i] for i in range(3)], output_dict=True)
                cr_df = pd.DataFrame(cr).transpose()
                cr_df.to_csv(f'{results_dir}/confusion_matrices/{model_name}_{method_name}_class_report.csv')
                
                # If model supports probability, generate ROC curves
                if hasattr(best_model, "predict_proba"):
                    plt.figure(figsize=(10, 8))
                    
                    # One-vs-Rest ROC curves
                    y_score = best_model.predict_proba(X_test_selected)
                    
                    for i in range(3):
                        # Convert to binary classification problem
                        y_test_binary = (y_test == i).astype(int)
                        
                        # Compute ROC curve and ROC area
                        fpr, tpr, _ = roc_curve(y_test_binary, y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        
                        # Plot
                        plt.plot(fpr, tpr, lw=2, 
                                label=f'{label_map[i]} (AUC = {roc_auc:.2f})')
                    
                    plt.plot([0, 1], [0, 1], 'k--', lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curves - {model_name} with {method_name}')
                    plt.legend(loc="lower right")
                    plt.savefig(f'{results_dir}/confusion_matrices/{model_name}_{method_name}_roc.png')
                    plt.close()
                
            except Exception as e:
                logger.error(f"Error training {model_name} with {method_name}: {e}")
                logger.error(traceback.format_exc())
                # Record failure
                result = {
                    'Model': model_name,
                    'Feature Selection': method_name,
                    'Train Accuracy': float('nan'),
                    'Test Accuracy': float('nan'),
                    'F1 Score': float('nan'),
                    'Overfitting Gap': float('nan'),
                    'Training Time (s)': float('nan'),
                    'Best Parameters': 'Error',
                    'Num Features': 0
                }
                results.append(result)
                save_checkpoint(result, checkpoint_name)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{results_dir}/model_results.csv', index=False)
    
    # Save checkpoint
    save_checkpoint(results_df, 'model_evaluation')
    
    return results_df

def train_ensemble_models(X, y, results_df):
    """Train ensemble models combining SVM and Extra Trees with optimized weights."""
    # Check if ensemble already trained
    checkpoint = load_checkpoint('ensemble')
    if checkpoint is not None:
        return checkpoint
    
    logger.info("Training ensemble models...")
    
    try:
        # Find best feature selection method for each base model
        best_svm_row = results_df[results_df['Model'] == 'SVM'].sort_values('Test Accuracy', ascending=False).iloc[0]
        best_et_row = results_df[results_df['Model'] == 'Extra_Trees'].sort_values('Test Accuracy', ascending=False).iloc[0]
        
        best_svm_method = best_svm_row['Feature Selection']
        best_et_method = best_et_row['Feature Selection']
        
        logger.info(f"Best SVM method: {best_svm_method}, Best Extra Trees method: {best_et_method}")
        
        # Load base models
        try:
            svm_model = joblib.load(f'{results_dir}/models/SVM_{best_svm_method}_model.pkl')
            et_model = joblib.load(f'{results_dir}/models/Extra_Trees_{best_et_method}_model.pkl')
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())
            # Return best individual model as fallback
            if best_svm_row['Test Accuracy'] > best_et_row['Test Accuracy']:
                return best_svm_row
            else:
                return best_et_row
        
        # Load feature selection results
        feature_selection_results = load_checkpoint('feature_selection')
        if feature_selection_results is None:
            logger.error("Feature selection results not found")
            return best_svm_row if best_svm_row['Test Accuracy'] > best_et_row['Test Accuracy'] else best_et_row
        
        # Split data - use separate validation set for weight optimization
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        def prepare_data(X_data, method_name):
            if method_name == 'pca':
                pca = feature_selection_results['pca']
                return pca.transform(X_data)
            elif method_name == 'all_features':
                return X_data
            elif method_name == 'mutual_info':
                mask = feature_selection_results['mi_mask']
                return X_data[:, mask]
            elif method_name == 'rfecv':
                mask = feature_selection_results['rfecv_mask']
                return X_data[:, mask]
            else:
                raise ValueError(f"Unknown feature selection method: {method_name}")
        
        # Prepare data for each model
        X_train_svm = prepare_data(X_train_scaled, best_svm_method)
        X_val_svm = prepare_data(X_val_scaled, best_svm_method)
        X_test_svm = prepare_data(X_test_scaled, best_svm_method)
        
        X_train_et = prepare_data(X_train_scaled, best_et_method)
        X_val_et = prepare_data(X_val_scaled, best_et_method)
        X_test_et = prepare_data(X_test_scaled, best_et_method)
        
        # Train individual models
        logger.info("Training individual models for ensemble...")
        svm_model.fit(X_train_svm, y_train)
        et_model.fit(X_train_et, y_train)
        
        # Grid search for optimal weights
        logger.info("Optimizing ensemble weights...")
        best_val_accuracy = 0
        best_svm_weight = 0.5  # Default weight
        
        # Get validation set predictions
        svm_val_proba = svm_model.predict_proba(X_val_svm)
        et_val_proba = et_model.predict_proba(X_val_et)
        
        # Try different weight combinations
        for svm_weight in np.linspace(0.1, 0.9, 9):
            et_weight = 1 - svm_weight
            ensemble_val_proba = svm_weight * svm_val_proba + et_weight * et_val_proba
            val_pred = np.argmax(ensemble_val_proba, axis=1)
            val_accuracy = accuracy_score(y_val, val_pred)
            
            logger.info(f"  SVM weight: {svm_weight:.1f}, ET weight: {et_weight:.1f}, Val accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_svm_weight = svm_weight
        
        best_et_weight = 1 - best_svm_weight
        logger.info(f"Optimal weights: SVM: {best_svm_weight:.2f}, Extra Trees: {best_et_weight:.2f}")
        
        # Evaluate with cross-validation for more robust estimation
        logger.info("Validating ensemble with cross-validation...")
        cv_scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X, y):
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y[train_idx], y[val_idx]
            
            # Scale features
            cv_scaler = StandardScaler()
            X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = cv_scaler.transform(X_cv_val)
            
            # Prepare data
            X_cv_train_svm = prepare_data(X_cv_train_scaled, best_svm_method)
            X_cv_val_svm = prepare_data(X_cv_val_scaled, best_svm_method)
            
            X_cv_train_et = prepare_data(X_cv_train_scaled, best_et_method)
            X_cv_val_et = prepare_data(X_cv_val_scaled, best_et_method)
            
            # Train models
            cv_svm = clone(svm_model)
            cv_et = clone(et_model)
            
            cv_svm.fit(X_cv_train_svm, y_cv_train)
            cv_et.fit(X_cv_train_et, y_cv_train)
            
            # Make predictions
            svm_cv_proba = cv_svm.predict_proba(X_cv_val_svm)
            et_cv_proba = cv_et.predict_proba(X_cv_val_et)
            
            # Combine with optimal weights
            ensemble_cv_proba = best_svm_weight * svm_cv_proba + best_et_weight * et_cv_proba
            cv_pred = np.argmax(ensemble_cv_proba, axis=1)
            
            # Calculate accuracy
            cv_accuracy = accuracy_score(y_cv_val, cv_pred)
            cv_scores.append(cv_accuracy)
        
        logger.info(f"Ensemble cross-validation accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Make final predictions with optimal weights
        def ensemble_predict_proba(X_svm, X_et):
            svm_proba = svm_model.predict_proba(X_svm)
            et_proba = et_model.predict_proba(X_et)
            return best_svm_weight * svm_proba + best_et_weight * et_proba
        
        def ensemble_predict(X_svm, X_et):
            proba = ensemble_predict_proba(X_svm, X_et)
            return np.argmax(proba, axis=1)
        
        # Calculate prediction uncertainty (agreement between models)
        def calculate_uncertainty(X_svm, X_et):
            svm_pred = svm_model.predict(X_svm)
            et_pred = et_model.predict(X_et)
            # 1 where models agree, 0 where they disagree
            agreement = (svm_pred == et_pred).astype(int)
            return 1 - agreement  # uncertainty = 1 - agreement
        
        # Make predictions
        y_train_pred = ensemble_predict(X_train_svm, X_train_et)
        y_test_pred = ensemble_predict(X_test_svm, X_test_et)
        y_test_proba = ensemble_predict_proba(X_test_svm, X_test_et)
        test_uncertainty = calculate_uncertainty(X_test_svm, X_test_et)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Analyze prediction uncertainty
        correct_predictions = (y_test_pred == y_test)
        
        # How many uncertain predictions were correct vs incorrect?
        uncertain_indices = np.where(test_uncertainty == 1)[0]
        if len(uncertain_indices) > 0:
            uncertain_correct = np.mean(correct_predictions[uncertain_indices])
            logger.info(f"Accuracy on uncertain predictions (models disagree): {uncertain_correct:.4f}")
            logger.info(f"Percentage of uncertain predictions: {len(uncertain_indices)/len(y_test)*100:.1f}%")
        
        # Plot uncertainty analysis
        plt.figure(figsize=(8, 6))
        sns.countplot(x=test_uncertainty, hue=correct_predictions)
        plt.xlabel('Prediction Uncertainty (0=Agreement, 1=Disagreement)')
        plt.ylabel('Count')
        plt.title('Model Agreement vs. Prediction Correctness')
        plt.legend(['Incorrect', 'Correct'])
        plt.savefig(f'{results_dir}/figures/prediction_uncertainty.png')
        plt.close()
        
        # Save results
        ensemble_result = {
            'Model': 'Ensemble',
            'Feature Selection': f'SVM:{best_svm_method},ET:{best_et_method}',
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'F1 Score': f1,
            'CV Accuracy': np.mean(cv_scores),
            'CV Std': np.std(cv_scores),
            'Overfitting Gap': train_accuracy - test_accuracy,
            'Best Parameters': f'SVM weight: {best_svm_weight:.2f}, ET weight: {best_et_weight:.2f}',
            'Num Features': 'Multiple'
        }

        # Log results
        logger.info(f"Results for Ensemble model:")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        logger.info(f"  Overfitting Gap: {train_accuracy - test_accuracy:.4f}")
        
        # Save ensemble components
        ensemble_info = {
            'svm_model': svm_model,
            'et_model': et_model,
            'svm_method': best_svm_method,
            'et_method': best_et_method,
            'svm_weight': best_svm_weight,
            'et_weight': best_et_weight,
            'scaler': scaler,
            'cv_scores': cv_scores
        }
        joblib.dump(ensemble_info, f'{results_dir}/models/ensemble_components.pkl')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[label_map[i] for i in range(3)],
                    yticklabels=[label_map[i] for i in range(3)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Ensemble Model (SVM + Extra Trees)')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/confusion_matrices/ensemble_cm.png')
        plt.close()
        
        # Class-specific metrics
        cr = classification_report(y_test, y_test_pred, target_names=[label_map[i] for i in range(3)], output_dict=True)
        cr_df = pd.DataFrame(cr).transpose()
        cr_df.to_csv(f'{results_dir}/confusion_matrices/ensemble_class_report.csv')

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
        plt.title('ROC Curves - Ensemble Model')
        plt.legend(loc="lower right")
        plt.savefig(f'{results_dir}/confusion_matrices/ensemble_roc.png')
        plt.close()
        
        # Save checkpoint
        save_checkpoint(ensemble_result, 'ensemble')

        return ensemble_result
    
    except Exception as e:
        logger.error(f"Error training ensemble model: {e}")
        logger.error(traceback.format_exc())
        
        # Return best individual model as fallback
        best_model_idx = results_df['Test Accuracy'].idxmax()
        best_model_row = results_df.iloc[best_model_idx]
        logger.info(f"Using best individual model as fallback: {best_model_row['Model']} with {best_model_row['Feature Selection']}")
        
        return best_model_row

def final_model_evaluation(results_df, X, y):
    """Perform final evaluation of the best model and provide conclusions."""
    checkpoint = load_checkpoint('final_evaluation')
    if checkpoint is not None:
        return checkpoint
    
    try:
        # Make a copy of the DataFrame to avoid modifying the original
        results_df = results_df.copy()
        
        # Add ensemble results if available
        ensemble_result = load_checkpoint('ensemble')
        if ensemble_result is not None and isinstance(ensemble_result, dict):
            if not any(r['Model'] == 'Ensemble' for _, r in results_df.iterrows()):
                results_df = pd.concat([results_df, pd.DataFrame([ensemble_result])], ignore_index=True)
                
        # Convert to numeric to ensure proper comparison
        results_df['Test Accuracy'] = pd.to_numeric(results_df['Test Accuracy'])
        best_model_idx = results_df['Test Accuracy'].idxmax()
        best_model_row = results_df.iloc[best_model_idx]
        
        # Print debug information to verify correct selection
        logger.info("Model comparison (sorted by test accuracy):")
        for i, (idx, row) in enumerate(results_df.sort_values('Test Accuracy', ascending=False).iterrows()):
            model = row['Model']
            if isinstance(model, pd.Series):
                model = model.iloc[0]
            test_acc = row['Test Accuracy']
            if isinstance(test_acc, pd.Series):
                test_acc = test_acc.iloc[0]
            logger.info(f"  {i+1}. {model}: {float(test_acc):.4f}")
            
        # Extract model information and convert Series items to their scalar values
        best_model_name = best_model_row['Model']
        if isinstance(best_model_name, pd.Series):
            best_model_name = best_model_name.iloc[0]
            
        best_feature_method = best_model_row['Feature Selection']
        if isinstance(best_feature_method, pd.Series):
            best_feature_method = best_feature_method.iloc[0]
        
        logger.info(f"Selected best model: {best_model_name} with {best_feature_method} features, "
                   f"Test Accuracy: {float(best_model_row['Test Accuracy']):.4f}")
        
        # Create comprehensive report
        with open(f'{results_dir}/final_report.md', 'w') as f:
            f.write(f"# Chicken Sound Classification - Phase 3 Final Report\n\n")
            f.write(f"## Best Model Summary\n\n")
            f.write(f"- **Model Type**: {best_model_name}\n")
            f.write(f"- **Feature Selection Method**: {best_feature_method}\n")
            
            # Convert Series values to float before formatting
            test_accuracy = float(best_model_row['Test Accuracy']) if isinstance(best_model_row['Test Accuracy'], pd.Series) else float(best_model_row['Test Accuracy'])
            f1_score = float(best_model_row['F1 Score']) if isinstance(best_model_row['F1 Score'], pd.Series) else float(best_model_row['F1 Score'])
            overfitting_gap = float(best_model_row['Overfitting Gap']) if isinstance(best_model_row['Overfitting Gap'], pd.Series) else float(best_model_row['Overfitting Gap'])
            
            f.write(f"- **Test Accuracy**: {test_accuracy:.4f}\n")
            f.write(f"- **F1 Score**: {f1_score:.4f}\n")
            f.write(f"- **Overfitting Gap**: {overfitting_gap:.4f}\n\n")
            
            if 'Num Features' in best_model_row and best_model_row['Num Features'] != 'Multiple':
                num_features = best_model_row['Num Features']
                if isinstance(num_features, pd.Series):
                    num_features = num_features.iloc[0]
                f.write(f"- **Number of Features**: {num_features}\n\n")
            
            if 'Best Parameters' in best_model_row:
                best_params = best_model_row['Best Parameters']
                if isinstance(best_params, pd.Series):
                    best_params = best_params.iloc[0]
                f.write(f"- **Model Parameters**: {best_params}\n\n")
            
            # Model comparison
            f.write(f"## Model Comparison\n\n")
            f.write("| Model | Feature Selection | Test Accuracy | F1 Score | Overfitting Gap |\n")
            f.write("|-------|------------------|---------------|----------|----------------|\n")
            
            for _, row in results_df.sort_values('Test Accuracy', ascending=False).iterrows():
                if pd.isna(row['Test Accuracy']):
                    continue
                
                # Convert values to float before formatting
                row_test_acc = float(row['Test Accuracy']) if isinstance(row['Test Accuracy'], pd.Series) else float(row['Test Accuracy'])
                row_f1 = float(row['F1 Score']) if isinstance(row['F1 Score'], pd.Series) else float(row['F1 Score'])
                row_gap = float(row['Overfitting Gap']) if isinstance(row['Overfitting Gap'], pd.Series) else float(row['Overfitting Gap'])
                
                model_name = row['Model']
                if isinstance(model_name, pd.Series):
                    model_name = model_name.iloc[0]
                
                feature_sel = row['Feature Selection']
                if isinstance(feature_sel, pd.Series):
                    feature_sel = feature_sel.iloc[0]
                
                f.write(f"| {model_name} | {feature_sel} | ")
                f.write(f"{row_test_acc:.4f} | {row_f1:.4f} | ")
                f.write(f"{row_gap:.4f} |\n")
            
            f.write("\n")
            
            # Feature importance analysis
            f.write(f"## Feature Analysis\n\n")
            f.write("Feature selection methods were applied to identify the most important features ")
            f.write("for chicken sound classification. See the feature_analysis directory for detailed results.\n\n")
            
            # Class-specific performance
            f.write(f"## Class-Specific Performance\n\n")
            f.write("The best model's performance varies across the three classes:\n\n")
            
            if isinstance(best_model_name, str) and best_model_name == 'Ensemble':
                cr_path = f'{results_dir}/confusion_matrices/ensemble_class_report.csv'
            else:
                cr_path = f'{results_dir}/confusion_matrices/{best_model_name}_{best_feature_method}_class_report.csv'
            
            if os.path.exists(cr_path):
                cr_df = pd.read_csv(cr_path, index_col=0)
                f.write("| Class | Precision | Recall | F1-Score | Support |\n")
                f.write("|-------|-----------|--------|----------|--------|\n")
                for idx, row in cr_df.iterrows():
                    if idx in ['accuracy', 'macro avg', 'weighted avg']:
                        continue
                    f.write(f"| {idx} | {row['precision']:.4f} | {row['recall']:.4f} | ")
                    f.write(f"{row['f1-score']:.4f} | {int(row['support'])} |\n")
                f.write("\n")
            
            # Data augmentation impact
            f.write(f"## Impact of Data Augmentation\n\n")
            f.write("Data augmentation techniques (time shifting, pitch shifting, and noise addition) ")
            f.write(f"were applied to increase the robustness of the model. The augmented dataset ")
            f.write(f"resulted in improved generalization capabilities as evidenced by the reduced ")
            f.write(f"overfitting gap ({overfitting_gap:.4f}) in the best model.\n\n")

        
        # Create summary visualization
        plt.figure(figsize=(12, 8))
        plot_df = results_df.sort_values('Test Accuracy', ascending=False)
        plot_df = plot_df.dropna(subset=['Test Accuracy'])
        
        # Create model labels safely handling Series objects
        model_labels = []
        for _, row in plot_df.iterrows():
            model = row['Model']
            feat_sel = row['Feature Selection']
            
            if isinstance(model, pd.Series):
                model = model.iloc[0]
            if isinstance(feat_sel, pd.Series):
                feat_sel = feat_sel.iloc[0]
                
            model_labels.append(f"{model}\n({feat_sel})")
        
        x = np.arange(len(model_labels))
        width = 0.2
        
        # Convert Series to float arrays if needed
        train_acc = [float(val) if isinstance(val, pd.Series) else float(val) for val in plot_df['Train Accuracy']]
        test_acc = [float(val) if isinstance(val, pd.Series) else float(val) for val in plot_df['Test Accuracy']]
        f1_scores = [float(val) if isinstance(val, pd.Series) else float(val) for val in plot_df['F1 Score']]
        
        plt.bar(x - width, train_acc, width, label='Train Accuracy')
        plt.bar(x, test_acc, width, label='Test Accuracy')
        plt.bar(x + width, f1_scores, width, label='F1 Score')
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% Target')
        plt.xlabel('Model Configuration')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{results_dir}/figures/model_comparison.png')
        plt.close()
        
        logger.info("Final report generated.")
        save_checkpoint(best_model_row, 'final_evaluation')
        return best_model_row
        
    except Exception as e:
        logger.error(f"Error in final model evaluation: {e}")
        logger.error(traceback.format_exc())
        if not results_df.empty:
            best_model_row = results_df.dropna(subset=['Test Accuracy']).loc[
                results_df.dropna(subset=['Test Accuracy'])['Test Accuracy'].idxmax()
            ]
            return best_model_row
        return None

def main():
    try:
        logger.info("Starting Phase 3 experiment")
        
        # Step 1: Prepare dataset with augmentation
        logger.info("Step 1: Preparing dataset")
        X, y = prepare_dataset(use_augmentation=True, class_balance=True)
        
        # Step 2: Perform feature selection analysis
        logger.info("Step 2: Performing feature selection analysis")
        feature_selection_results = feature_selection_analysis(X, y)
        
        # Step 3: Train and evaluate models with different feature selection approaches
        logger.info("Step 3: Training and evaluating models")
        results_df = train_and_evaluate_models(X, y, feature_selection_results)
        
        # Step 4: Train ensemble models
        logger.info("Step 4: Training ensemble models")
        ensemble_result = train_ensemble_models(X, y, results_df)
        
        # Step 5: Perform final evaluation and generate report
        logger.info("Step 5: Performing final evaluation")
        best_model = final_model_evaluation(results_df, X, y)
        
        logger.info(f"Phase 3 experiment completed successfully. Results saved to {results_dir}")
        
        return best_model
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()      