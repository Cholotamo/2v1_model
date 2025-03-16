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
from sklearn.base import clone
import librosa
from datetime import datetime

# Create experiment directories
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

# ---- Model Definitions ----

def get_model_params(model_name):
    """Get model and parameter grid for each model"""
    if model_name == "SVM":
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
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    
    elif model_name == "KNN":
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [5, 11, 21, 31, 51],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]  # Only for Minkowski metric
        }
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model, param_grid

# ---- Experiment Execution ----

def run_experiment():
    """Run the full experiment"""
    # Feature types to extract and evaluate
    feature_types = ['mfcc', 'mfcc_derivatives', 'mfcc_temporal', 'complete']
    
    # Models to evaluate
    model_names = ["SVM", "Extra Trees", "KNN"]
    
    # Results storage
    all_results = []
    
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
            print(f"\nTraining {model_name} with {feature_type} features...")
            
            # Get model and parameters
            model, param_grid = get_model_params(model_name)
            
            # Train with grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=5, n_jobs=-1, 
                scoring='accuracy', verbose=1, return_train_score=True)
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            cv_scores = cross_val_score(best_model, X, y, cv=5)
            
            # Store results
            result = {
                'Feature Type': feature_type,
                'Model': model_name,
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'F1 Score': f1,
                'CV Mean': np.mean(cv_scores),
                'CV Std': np.std(cv_scores),
                'Overfitting Gap': train_accuracy - test_accuracy,
                'Best Parameters': str(grid_search.best_params_)
            }
            all_results.append(result)
            
            # Save model
            joblib.dump(best_model, 
                      f'{results_dir}/models/{model_name}_{feature_type}.pkl')
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'{model_name} - {feature_type} Features')
            plt.savefig(
                f'{results_dir}/confusion_matrices/{model_name}_{feature_type}_cm.png')
            plt.close()
            
            # Print results
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Training accuracy: {train_accuracy:.4f}")
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"F1 score: {f1:.4f}")
            print(f"CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"Overfitting gap: {train_accuracy - test_accuracy:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(f'{results_dir}/experiment_results.csv', index=False)
    
    # Generate visualizations
    
    # 1. Feature type comparison across models
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Feature Type', y='Test Accuracy', hue='Model', data=results_df)
    plt.title('Impact of Feature Types on Model Accuracy')
    plt.ylim(0.5, 1.0)  # Adjust as needed
    plt.savefig(f'{results_dir}/figures/feature_type_comparison.png')
    plt.close()
    
    # 2. Overfitting comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Feature Type', y='Overfitting Gap', hue='Model', data=results_df)
    plt.title('Overfitting Gap by Feature Type and Model')
    plt.savefig(f'{results_dir}/figures/overfitting_comparison.png')
    plt.close()
    
    # 3. Detailed model performance for each feature type
    for feature_type in feature_types:
        subset = results_df[results_df['Feature Type'] == feature_type]
        plt.figure(figsize=(10, 6))
        
        # Plot train and test accuracy
        x = np.arange(len(subset))
        width = 0.35
        
        plt.bar(x - width/2, subset['Train Accuracy'], width, label='Train')
        plt.bar(x + width/2, subset['Test Accuracy'], width, label='Test')
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title(f'Performance with {feature_type} features')
        plt.xticks(x, subset['Model'])
        plt.legend()
        plt.ylim(0.5, 1.0)  # Adjust as needed
        
        plt.savefig(f'{results_dir}/figures/{feature_type}_performance.png')
        plt.close()
    
    # Generate summary report
    generate_summary_report(results_df)
    
    return results_df

def generate_summary_report(results_df):
    """Generate a detailed summary report"""
    with open(f'{results_dir}/experiment_summary.md', 'w') as f:
        f.write("# Phase 2 Experiment: Feature Enhancement and Model Optimization\n\n")
        
        f.write("## Overview\n")
        f.write("This experiment evaluates the impact of different feature sets and model optimizations ")
        f.write("on chicken sound classification performance.\n\n")
        
        f.write("## Results Summary\n\n")
        
        # Best overall configuration
        best_row = results_df.loc[results_df['Test Accuracy'].idxmax()]
        f.write("### Best Overall Configuration\n")
        f.write(f"- **Model**: {best_row['Model']}\n")
        f.write(f"- **Feature Type**: {best_row['Feature Type']}\n")
        f.write(f"- **Test Accuracy**: {best_row['Test Accuracy']:.4f}\n")
        f.write(f"- **F1 Score**: {best_row['F1 Score']:.4f}\n")
        f.write(f"- **Overfitting Gap**: {best_row['Overfitting Gap']:.4f}\n")
        f.write(f"- **Parameters**: {best_row['Best Parameters']}\n\n")
        
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
        
        # Best configuration for each model
        f.write("### Best Configuration For Each Model\n")
        for model in results_df['Model'].unique():
            model_rows = results_df[results_df['Model'] == model]
            best_model_row = model_rows.loc[model_rows['Test Accuracy'].idxmax()]
            
            f.write(f"#### {model}\n")
            f.write(f"- **Best Feature Type**: {best_model_row['Feature Type']}\n")
            f.write(f"- **Test Accuracy**: {best_model_row['Test Accuracy']:.4f}\n")
            f.write(f"- **F1 Score**: {best_model_row['F1 Score']:.4f}\n")
            f.write(f"- **Overfitting Gap**: {best_model_row['Overfitting Gap']:.4f}\n")
            f.write(f"- **Parameters**: {best_model_row['Best Parameters']}\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Determine which feature type is best overall
        best_feature = feature_analysis.loc[feature_analysis[('Test Accuracy', 'mean')].idxmax(), 'Feature Type']
        f.write(f"1. **Feature Impact**: {best_feature} features provide the best overall performance across models.\n")
        
        # Determine which model performs best
        best_model = model_analysis.loc[model_analysis[('Test Accuracy', 'mean')].idxmax(), 'Model']
        f.write(f"2. **Model Performance**: {best_model} shows the strongest overall performance across feature types.\n")
        
        # Analyze overfitting trends
        min_overfitting_model = model_analysis.loc[model_analysis[('Overfitting Gap', 'mean')].idxmin(), 'Model']
        min_overfitting_feature = feature_analysis.loc[feature_analysis[('Overfitting Gap', 'mean')].idxmin(), 'Feature Type']
        f.write(f"3. **Overfitting Trends**: {min_overfitting_model} shows the least overfitting among models, ")
        f.write(f"while {min_overfitting_feature} features tend to reduce overfitting across all models.\n\n")

if __name__ == "__main__":
    results = run_experiment()
    print("\nExperiment complete! Results saved to:", results_dir)