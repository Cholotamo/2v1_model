# phase2_experiment.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
import librosa
from datetime import datetime
import json
from sklearn.model_selection import KFold
import argparse


# ---- Create experiment directories ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'phase2_experiment_{timestamp}'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/confusion_matrices', exist_ok=True)
os.makedirs(f'{results_dir}/models', exist_ok=True)
os.makedirs(f'{results_dir}/figures', exist_ok=True)


# ---- Feature Extraction Functions ----
def extract_mfcc_features(file_path, n_mfcc=13):
    """Extract basic MFCC features"""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if y.size == 0:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_mfcc_with_derivatives(file_path, n_mfcc=13):
    """Extract MFCC features with 1st and 2nd derivatives"""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if y.size == 0:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(delta_mfcc, axis=1),
            np.mean(delta2_mfcc, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_mfcc_with_temporal(file_path, n_mfcc=13):
    """Extract MFCC features with temporal features"""
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
        print(f"Error processing {file_path}: {e}")
        return None

def extract_complete_features(file_path, n_mfcc=13):
    """Extract all features combined"""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if y.size == 0:
            return None
        
        # MFCC and derivatives
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Temporal features
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(delta_mfcc, axis=1),
            np.mean(delta2_mfcc, axis=1),
            np.mean(zcr, axis=1),
            np.mean(rms, axis=1),
            np.mean(spectral_bandwidth, axis=1),
            np.mean(spectral_centroid, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(chroma, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ---- Feature Extraction ----
def extract_features_for_dataset(feature_type):
    """Extract specified feature type from all audio files"""
    print(f"Extracting {feature_type} features...")
    
    # Paths to audio files
    healthy_chicken_dir = 'dataset/Healthy'
    sick_chicken_dir = 'dataset/Sick'
    noise_dir = 'dataset/None'
    
    # Select extraction function based on feature type
    if feature_type == 'mfcc':
        extract_fn = extract_mfcc_features
    elif feature_type == 'mfcc_derivatives':
        extract_fn = extract_mfcc_with_derivatives
    elif feature_type == 'mfcc_temporal':
        extract_fn = extract_mfcc_with_temporal
    elif feature_type == 'complete':
        extract_fn = extract_complete_features
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    X = []
    y = []
    
    # Process each directory
    for dir_path, label in [
        (healthy_chicken_dir, 0),  # healthy=0
        (sick_chicken_dir, 1),     # sick=1
        (noise_dir, 2)             # noise=2
    ]:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            features = extract_fn(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save features, labels, and scaler
    np.save(f'{results_dir}/{feature_type}_features.npy', X_scaled)
    np.save(f'{results_dir}/{feature_type}_labels.npy', y)
    joblib.dump(scaler, f'{results_dir}/{feature_type}_scaler.pkl')
    
    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} {feature_type} features per sample")
    return X_scaled, y


# Custom Two-Tier Classifier
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


# ---- Model Definitions (Testing, Simplified parameters) ----
def get_simplified_model_params(model_name):
    """Get simplified model and parameter grid with basic regularization"""
    if model_name == "SVM" or model_name == "Support Vector Machine (SVM)":
        model = SVC(probability=True)
        param_grid = {
            'kernel': ['rbf'],
            'C': [1],
            'gamma': ['scale']
        }
    
    elif model_name == "Extra Trees":
        model = ExtraTreesClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100],
            'max_depth': [None],
            'min_samples_split': [2],
            'min_samples_leaf': [5],  
            'max_features': ['sqrt']  
        }
    
    elif model_name == "KNN" or model_name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [15],       
            'weights': ['distance'],
            'metric': ['euclidean']
        }
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model, param_grid


def get_model_params(model_name):
    """Get model and parameter grid for each model with regularization to reduce overfitting"""
    if model_name == "SVM" or model_name == "Support Vector Machine (SVM)":
        model = SVC(probability=True)
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'class_weight': [None, 'balanced']
        }
    
    elif model_name == "Extra Trees":
        model = ExtraTreesClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [4, 8, 12],  
            'max_features': ['sqrt', 'log2'],  
            'bootstrap': [True, False]
        }
    
    elif model_name == "KNN" or model_name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [15, 21, 31, 51],  
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]  
        }
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model, param_grid


# ---- Checkpointing Functions ----
def load_checkpoint(checkpoint_file):
    """Load existing checkpoints from file"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoints, result, model_name, feature_type, approach, checkpoint_file):
    """Save a checkpoint for a completed model-feature-approach combination"""
    key = f"{model_name}_{feature_type}_{approach}"
    # Convert numpy types to native Python types for JSON serialization
    checkpoint_data = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in result.items()}
    checkpoints[key] = checkpoint_data
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoints, f, indent=4)


# ---- Experiment Execution ----
def run_experiment(use_simplified_params=True):
    """Run the full experiment with checkpointing"""
    # Create checkpoint file
    checkpoint_file = f'{results_dir}/experiment_checkpoints.json'
    checkpoints = load_checkpoint(checkpoint_file)
    
    # Feature types to extract and evaluate
    feature_types = ['mfcc', 'mfcc_derivatives', 'mfcc_temporal', 'complete']
    
    # Models to evaluate
    model_names = ["SVM", "Extra Trees", "KNN"]
    
    # Add approaches list
    approaches = ["1v1v1", "Two-Tier"]
    
    # Results storage - load any existing results from checkpoints
    all_results = [checkpoints[key] for key in checkpoints.keys()]
    
    # Extract features for each type
    feature_sets = {}
    for feature_type in feature_types:
        # Check if features are already extracted
        if os.path.exists(f'{results_dir}/{feature_type}_features.npy'):
            print(f"Loading existing {feature_type} features...")
            X = np.load(f'{results_dir}/{feature_type}_features.npy')
            y = np.load(f'{results_dir}/{feature_type}_labels.npy')
        else:
            X, y = extract_features_for_dataset(feature_type)
        
        feature_sets[feature_type] = (X, y)
    
    # For each feature type and model, run the experiment
    for feature_type, (X, y) in feature_sets.items():
        print(f"\n===== Evaluating {feature_type} features =====")
        
        # Apply consistent data splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        for model_name in model_names:
            for approach in approaches:
                # Check if this combination has already been evaluated
                checkpoint_key = f"{model_name}_{feature_type}_{approach}"
                if checkpoint_key in checkpoints:
                    print(f"Skipping {model_name} with {feature_type} features using {approach} approach (already completed)")
                    continue
                
                print(f"\nTraining {model_name} with {feature_type} features using {approach} approach...")
                
                # Get model and parameters - use simplified or full based on flag
                if use_simplified_params:
                    model, param_grid = get_simplified_model_params(model_name)
                    print("Using simplified parameter grid for testing")
                else:
                    model, param_grid = get_model_params(model_name)
                
                try:
                    if approach == "1v1v1":
                        # Direct multi-class classification
                        grid_search = GridSearchCV(
                            model, param_grid, cv=3, n_jobs=-1,  # Reduced CV folds for testing 
                            scoring='accuracy', verbose=1, return_train_score=True)
                        
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        best_params = str(grid_search.best_params_)
                        
                        # Evaluate
                        y_train_pred = best_model.predict(X_train)
                        y_test_pred = best_model.predict(X_test)
                        cv_scores = cross_val_score(best_model, X, y, cv=3)  # Reduced CV folds for testing
                        
                    else:  # Two-Tier approach
                        # For two-tier, train tier1 and tier2 models separately
                        tier1_model, _ = get_simplified_model_params(model_name) if use_simplified_params else get_model_params(model_name)
                        tier2_model, _ = get_simplified_model_params(model_name) if use_simplified_params else get_model_params(model_name)
                        
                        # Create binary labels for the first tier
                        y_tier1_train = np.array([0 if label in [0, 1] else 1 for label in y_train])
                        
                        # Grid search for tier 1
                        grid_search_tier1 = GridSearchCV(
                            tier1_model, param_grid, cv=3, n_jobs=-1,  # Reduced CV folds for testing
                            scoring='accuracy', verbose=1, return_train_score=True)
                        grid_search_tier1.fit(X_train, y_tier1_train)
                        best_tier1 = grid_search_tier1.best_estimator_
                        
                        # Get chicken samples for tier 2 training
                        y_pred_tier1 = best_tier1.predict(X_train)
                        chicken_indices = np.where(y_pred_tier1 == 0)[0]
                        X_chicken = X_train[chicken_indices]
                        y_chicken = y_train[chicken_indices]
                        
                        # Grid search for tier 2
                        grid_search_tier2 = GridSearchCV(
                            tier2_model, param_grid, cv=3, n_jobs=-1,  # Reduced CV folds for testing
                            scoring='accuracy', verbose=1, return_train_score=True)
                        grid_search_tier2.fit(X_chicken, y_chicken)
                        best_tier2 = grid_search_tier2.best_estimator_
                        
                        # Combine models
                        two_tier_classifier = TwoTierClassifier(best_tier1, best_tier2)
                        two_tier_classifier.fit(X_train, y_train)
                        best_model = two_tier_classifier
                        best_params = f"Tier 1: {grid_search_tier1.best_params_}, Tier 2: {grid_search_tier2.best_params_}"
                        
                        # Evaluate
                        y_train_pred = best_model.predict(X_train)
                        y_test_pred = best_model.predict(X_test)
                        
                        # Cross-validation for two-tier
                        from sklearn.model_selection import KFold
                        kf = KFold(n_splits=3, random_state=42, shuffle=True)  # Reduced CV folds for testing
                        cv_scores = []
                        for train_idx, test_idx in kf.split(X):
                            X_cv_train, X_cv_test = X[train_idx], X[test_idx]
                            y_cv_train, y_cv_test = y[train_idx], y[test_idx]
                            
                            cv_tier1 = clone(best_tier1)
                            cv_tier2 = clone(best_tier2)
                            cv_classifier = TwoTierClassifier(cv_tier1, cv_tier2)
                            cv_classifier.fit(X_cv_train, y_cv_train)
                            
                            y_cv_pred = cv_classifier.predict(X_cv_test)
                            cv_scores.append(accuracy_score(y_cv_test, y_cv_pred))
                    
                    # Calculate metrics
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    f1 = f1_score(y_test, y_test_pred, average='weighted')
                    
                    # Store results
                    result = {
                        'Feature Type': feature_type,
                        'Model': model_name,
                        'Approach': approach,
                        'Train Accuracy': train_accuracy,
                        'Test Accuracy': test_accuracy,
                        'F1 Score': f1,
                        'CV Mean': np.mean(cv_scores),
                        'CV Std': np.std(cv_scores),
                        'Overfitting Gap': train_accuracy - test_accuracy,
                        'Best Parameters': best_params
                    }
                    all_results.append(result)
                    
                    # Save model
                    joblib.dump(best_model, 
                            f'{results_dir}/models/{model_name}_{feature_type}_{approach}.pkl')
                    
                    # Plot confusion matrix
                    cm = confusion_matrix(y_test, y_test_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    plt.title(f'{model_name} - {feature_type} - {approach}')
                    plt.savefig(
                        f'{results_dir}/confusion_matrices/{model_name}_{feature_type}_{approach}_cm.png')
                    plt.close()
                    
                    # Print results
                    print(f"Best parameters: {best_params}")
                    print(f"Training accuracy: {train_accuracy:.4f}")
                    print(f"Test accuracy: {test_accuracy:.4f}")
                    print(f"F1 score: {f1:.4f}")
                    print(f"CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                    print(f"Overfitting gap: {train_accuracy - test_accuracy:.4f}")
                    
                    # Save checkpoint
                    save_checkpoint(checkpoints, result, model_name, feature_type, approach, checkpoint_file)
                    
                except Exception as e:
                    print(f"Error in {model_name} with {feature_type} features using {approach} approach: {e}")
                    print(f"Skipping this combination and continuing with others...")
                    continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(f'{results_dir}/experiment_results.csv', index=False)
    
    # Update visualization functions to include approach dimension
    generate_visualizations(results_df)
    
    # Generate summary report
    generate_summary_report(results_df)
    
    return results_df

# ---- Visualization Function ----
def generate_visualizations(results_df):
    """Generate visualizations that include the approach dimension"""
    
    # 1. Feature type comparison across models and approaches
    plt.figure(figsize=(15, 10))
    g = sns.catplot(
        x='Feature Type', y='Test Accuracy', hue='Model', col='Approach',
        data=results_df, kind='bar', height=6, aspect=1.2)
    g.set_titles("{col_name} Approach")
    g.fig.suptitle('Impact of Feature Types on Model Accuracy by Approach')
    g.fig.subplots_adjust(top=0.85)
    plt.ylim(0.5, 1.0)
    plt.savefig(f'{results_dir}/figures/feature_type_comparison_by_approach.png')
    plt.close()
    
    # 2. Approach comparison across models and feature types
    plt.figure(figsize=(15, 10))
    g = sns.catplot(
        x='Model', y='Test Accuracy', hue='Approach', col='Feature Type',
        data=results_df, kind='bar', height=6, aspect=1.2)
    g.set_titles("{col_name} Features")
    g.fig.suptitle('Approach Comparison by Feature Type')
    g.fig.subplots_adjust(top=0.85)
    plt.ylim(0.5, 1.0)
    plt.savefig(f'{results_dir}/figures/approach_comparison_by_feature.png')
    plt.close()
    
    # 3. Overfitting comparison by approach
    plt.figure(figsize=(15, 10))
    g = sns.catplot(
        x='Feature Type', y='Overfitting Gap', hue='Model', col='Approach',
        data=results_df, kind='bar', height=6, aspect=1.2)
    g.set_titles("{col_name} Approach")
    g.fig.suptitle('Overfitting Gap by Approach and Feature Type')
    g.fig.subplots_adjust(top=0.85)
    plt.savefig(f'{results_dir}/figures/overfitting_comparison_by_approach.png')
    plt.close()
    
    # 4. Model-specific comparisons across approaches and feature types
    for model in results_df['Model'].unique():
        model_df = results_df[results_df['Model'] == model]
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Feature Type', y='Test Accuracy', hue='Approach', data=model_df)
        plt.title(f'{model} Performance by Feature Type and Approach')
        plt.ylim(0.5, 1.0)
        plt.savefig(f'{results_dir}/figures/{model}_comparison.png')
        plt.close()


# ---- Summary Report Generation ----
def generate_summary_report(results_df):
    """Generate a detailed summary report"""
    with open(f'{results_dir}/experiment_summary.md', 'w') as f:
        f.write("# Phase 2 Experiment: Feature Enhancement and Model Optimization\n\n")
        
        f.write("## Overview\n")
        f.write("This experiment evaluates the impact of different feature sets and model optimizations ")
        f.write("on chicken sound classification performance, comparing single-stage (1v1v1) and ")
        f.write("two-tier classification approaches.\n\n")
        
        f.write("## Results Summary\n\n")
        
        # Best overall configuration
        best_row = results_df.loc[results_df['Test Accuracy'].idxmax()]
        f.write("### Best Overall Configuration\n")
        f.write(f"- **Model**: {best_row['Model']}\n")
        f.write(f"- **Feature Type**: {best_row['Feature Type']}\n")
        f.write(f"- **Approach**: {best_row['Approach']}\n")
        f.write(f"- **Test Accuracy**: {best_row['Test Accuracy']:.4f}\n")
        f.write(f"- **F1 Score**: {best_row['F1 Score']:.4f}\n")
        f.write(f"- **Overfitting Gap**: {best_row['Overfitting Gap']:.4f}\n")
        f.write(f"- **Parameters**: {best_row['Best Parameters']}\n\n")
        
        # Approach analysis
        f.write("### Approach Analysis\n")
        approach_analysis = results_df.groupby('Approach').agg({
            'Test Accuracy': ['mean', 'std', 'max'],
            'Overfitting Gap': 'mean'
        }).reset_index()
        
        f.write("| Approach | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |\n")
        f.write("|----------|--------------|--------------|--------------|------------------|\n")
        
        for _, row in approach_analysis.iterrows():
            f.write(f"| {row['Approach']} | {row[('Test Accuracy', 'mean')]:.4f} | ")
            f.write(f"{row[('Test Accuracy', 'std')]:.4f} | {row[('Test Accuracy', 'max')]:.4f} | ")
            f.write(f"{row[('Overfitting Gap', 'mean')]:.4f} |\n")
        
        f.write("\n")
        
        # Feature type analysis
        f.write("### Feature Type Analysis\n")
        feature_analysis = results_df.groupby('Feature Type').agg({
            'Test Accuracy': ['mean', 'std', 'max'],
            'Overfitting Gap': 'mean'
        }).reset_index()
        
        f.write("| Feature Type | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |\n")
        f.write("|-------------|--------------|--------------|--------------|------------------|\n")
        
        for _, row in feature_analysis.iterrows():
            f.write(f"| {row['Feature Type']} | {row[('Test Accuracy', 'mean')]:.4f} | ")
            f.write(f"{row[('Test Accuracy', 'std')]:.4f} | {row[('Test Accuracy', 'max')]:.4f} | ")
            f.write(f"{row[('Overfitting Gap', 'mean')]:.4f} |\n")
        
        f.write("\n")
        
        # Model analysis
        f.write("### Model Analysis\n")
        model_analysis = results_df.groupby('Model').agg({
            'Test Accuracy': ['mean', 'std', 'max'],
            'Overfitting Gap': 'mean'
        }).reset_index()
        
        f.write("| Model | Mean Accuracy | Std Accuracy | Max Accuracy | Mean Overfitting |\n")
        f.write("|-------|--------------|--------------|--------------|------------------|\n")
        
        for _, row in model_analysis.iterrows():
            f.write(f"| {row['Model']} | {row[('Test Accuracy', 'mean')]:.4f} | ")
            f.write(f"{row[('Test Accuracy', 'std')]:.4f} | {row[('Test Accuracy', 'max')]:.4f} | ")
            f.write(f"{row[('Overfitting Gap', 'mean')]:.4f} |\n")
        
        f.write("\n")
        
        # Feature-Approach interaction
        f.write("### Feature Type and Approach Interaction\n")
        feature_approach = results_df.groupby(['Feature Type', 'Approach']).agg({
            'Test Accuracy': 'mean'
        }).reset_index()
        
        pivot_table = feature_approach.pivot(index='Feature Type', columns='Approach', values='Test Accuracy')
        
        f.write("| Feature Type | 1v1v1 | Two-Tier | Difference |\n")
        f.write("|-------------|-------|----------|------------|\n")
        
        for feature in pivot_table.index:
            onev1 = pivot_table.loc[feature, '1v1v1']
            two_tier = pivot_table.loc[feature, 'Two-Tier']
            diff = onev1 - two_tier
            f.write(f"| {feature} | {onev1:.4f} | {two_tier:.4f} | {diff:.4f} |\n")
        
        f.write("\n")
        
        # Best configuration for each model and approach
        f.write("### Best Configuration For Each Model and Approach\n")
        for model in results_df['Model'].unique():
            f.write(f"#### {model}\n")
            
            for approach in results_df['Approach'].unique():
                subset = results_df[(results_df['Model'] == model) & (results_df['Approach'] == approach)]
                best_row = subset.loc[subset['Test Accuracy'].idxmax()]
                
                f.write(f"**{approach} Approach**\n")
                f.write(f"- Best Feature Type: {best_row['Feature Type']}\n")
                f.write(f"- Test Accuracy: {best_row['Test Accuracy']:.4f}\n")
                f.write(f"- F1 Score: {best_row['F1 Score']:.4f}\n")
                f.write(f"- Overfitting Gap: {best_row['Overfitting Gap']:.4f}\n")
                f.write(f"- Parameters: {best_row['Best Parameters']}\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Determine which feature type is best overall
        best_feature = feature_analysis.loc[feature_analysis[('Test Accuracy', 'mean')].idxmax(), 'Feature Type']
        f.write(f"1. **Feature Impact**: {best_feature} features provide the best overall performance across models and approaches.\n")
        
        # Determine which model performs best
        best_model = model_analysis.loc[model_analysis[('Test Accuracy', 'mean')].idxmax(), 'Model']
        f.write(f"2. **Model Performance**: {best_model} shows the strongest overall performance across feature types and approaches.\n")
        
        # Determine which approach performs best
        best_approach_idx = approach_analysis[('Test Accuracy', 'mean')].idxmax()
        best_approach = approach_analysis.iloc[best_approach_idx]['Approach']
        best_approach_accuracy = approach_analysis.iloc[best_approach_idx][('Test Accuracy', 'mean')]
        f.write(f"3. **Approach Performance**: {best_approach} approach provides better performance overall (mean test accuracy: {best_approach_accuracy:.4f}).\n")
        
        # Analyze overfitting trends
        min_overfitting_model = model_analysis.loc[model_analysis[('Overfitting Gap', 'mean')].idxmin(), 'Model']
        min_overfitting_feature = feature_analysis.loc[feature_analysis[('Overfitting Gap', 'mean')].idxmin(), 'Feature Type']
        min_overfitting_approach = approach_analysis.loc[approach_analysis[('Overfitting Gap', 'mean')].idxmin(), 'Approach']
        
        f.write(f"4. **Overfitting Trends**: {min_overfitting_model} shows the least overfitting among models, ")
        f.write(f"{min_overfitting_feature} features tend to reduce overfitting, and the {min_overfitting_approach} ")
        f.write(f"approach generally results in less overfitting.\n\n")
        
        # Recommendations for Phase 3
        f.write("## Recommendations for Phase 3\n\n")
        
        # Get top model-feature-approach combinations
        top_combinations = results_df.sort_values('Test Accuracy', ascending=False).head(3)
        
        f.write("Based on the Phase 2 results, the following configurations are recommended for Phase 3:\n\n")
        
        for i, (_, row) in enumerate(top_combinations.iterrows(), 1):
            f.write(f"{i}. **{row['Model']}** with **{row['Feature Type']}** features using the **{row['Approach']}** approach ")
            f.write(f"(Test Accuracy: {row['Test Accuracy']:.4f})\n")
            

# ---- Main Execution ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run chicken sound classification experiment.')
    parser.add_argument('--test', action='store_true', help='Run with simplified parameters for testing')
    args = parser.parse_args()
    
    if args.test:
        print("\n=== RUNNING IN TEST MODE WITH SIMPLIFIED PARAMETERS ===\n")
        results = run_experiment(use_simplified_params=True)
        print("\nTest run completed! Use without --test flag for full experiment.")
    else:
        print("\n=== RUNNING FULL EXPERIMENT ===\n")
        results = run_experiment(use_simplified_params=False)
        print("\nExperiment complete! Results saved to:", results_dir)