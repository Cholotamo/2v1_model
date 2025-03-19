# phase1_experiment.py

import joblib
import csv
import os  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import seaborn as sns
import librosa
from datetime import datetime
import json


# ---- Timestamped Directory Setup ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'phase1_experiment_{timestamp}'
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

def extract_features_for_dataset():
    """Extract MFCC features from all audio files"""
    print("Extracting MFCC features...")
    
    # Paths to audio files
    healthy_chicken_dir = 'dataset/Healthy'
    sick_chicken_dir = 'dataset/Sick'
    noise_dir = 'dataset/None'
    
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
            features = extract_mfcc_features(file_path)
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
    np.save(f'{results_dir}/mfcc_features.npy', X_scaled)
    np.save(f'{results_dir}/mfcc_labels.npy', y)
    joblib.dump(scaler, f'{results_dir}/mfcc_scaler.pkl')
    
    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features per sample")
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


# ---- Model Definitions ----
def get_model_params(model_name):
    """Get model and parameter grid for each model"""
    if model_name == "Support Vector Machine (SVM)":
        model = SVC(probability=True)
        param_grid = {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    elif model_name == "Extra Trees":
        model = ExtraTreesClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20, 30],
            'min_samples_split': [2, 5]
        }
    elif model_name == "XGBoost":
        model = XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20, 30],
            'min_samples_split': [2, 5]
        }
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    elif model_name == "Naive Bayes":
        model = GaussianNB()
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    elif model_name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model, param_grid


# ---- Checkpointing Functions ----
def load_checkpoints(checkpoint_file):
    """Load existing checkpoints from file"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoints, result, model_name, approach, checkpoint_file):
    """Save a checkpoint for a completed model-approach combination"""
    key = f"{model_name}_{approach}"
    # Convert numpy types to native Python types for JSON serialization
    checkpoint_data = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in result.items()}
    checkpoints[key] = checkpoint_data
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoints, f, indent=4)


# ---- Experiment Execution ----
def run_experiment():
    """Run the full experiment with checkpointing"""
    checkpoint_file = f'{results_dir}/checkpoints.json'
    
    if os.path.exists(f'{results_dir}/mfcc_features.npy'):
        print("Loading existing MFCC features...")
        X = np.load(f'{results_dir}/mfcc_features.npy')
        y = np.load(f'{results_dir}/mfcc_labels.npy')
    else:
        X, y = extract_features_for_dataset()
    

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    label_map = {0: 'healthy', 1: 'sick', 2: 'noise'}
    
    model_names = [
        "Support Vector Machine (SVM)",
        "Extra Trees",
        "XGBoost",
        "Random Forest",
        "Logistic Regression",
        "Naive Bayes",
        "K-Nearest Neighbors (KNN)",
        "Decision Tree",
        "Gradient Boosting",
        "AdaBoost"
    ]
    approaches = ["1v1v1", "Two-Tier"]
    all_results = []

    checkpoints = load_checkpoints(checkpoint_file)
    all_results = [checkpoints[key] for key in checkpoints.keys()]
    
    for approach in approaches:
        print(f"\n===== Evaluating {approach} Classification Approach =====")
        for model_name in model_names:
            checkpoint_key = f"{model_name}_{approach}"
            model_file = f'{results_dir}/models/{model_name.replace(" ", "_").lower()}_{approach}.pkl'
            
            if checkpoint_key in checkpoints:
                print(f"Skipping {model_name} with {approach} approach (already completed)")
                continue
            
            print(f"\nTraining {model_name} with {approach} approach...")
            model, param_grid = get_model_params(model_name)
            
            if approach == "1v1v1":
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                cv_scores = cross_val_score(best_model, X, y, cv=5)
                best_params = str(grid_search.best_params_)
            else:  # Two-Tier
                tier1_model, param_grid = get_model_params(model_name)
                tier2_model, _ = get_model_params(model_name)
                
                # Tune Tier 1 (chicken vs. noise)
                y_tier1_train = np.array([0 if label in [0, 1] else 1 for label in y_train])
                grid_search_tier1 = GridSearchCV(
                    tier1_model, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
                grid_search_tier1.fit(X_train, y_tier1_train)
                best_tier1 = grid_search_tier1.best_estimator_
                
                # Filter chicken samples for Tier 2
                y_pred_tier1 = best_tier1.predict(X_train)
                chicken_indices = np.where(y_pred_tier1 == 0)[0]
                X_chicken = X_train[chicken_indices]
                y_chicken = y_train[chicken_indices]
                
                # Tune Tier 2 (healthy vs. sick)
                grid_search_tier2 = GridSearchCV(
                    tier2_model, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
                grid_search_tier2.fit(X_chicken, y_chicken)
                best_tier2 = grid_search_tier2.best_estimator_
                
                # Special handling for XGBoost in Tier 2
                if model_name == "XGBoost":
                    best_tier2.set_params(objective='multi:softmax', num_class=2)
                
                # Combine in TwoTierClassifier
                classifier = TwoTierClassifier(best_tier1, best_tier2)
                classifier.fit(X_train, y_train)
                y_train_pred = classifier.predict(X_train)
                y_test_pred = classifier.predict(X_test)
                
                # CV for two-tier
                kf = KFold(n_splits=5, random_state=42, shuffle=True)
                cv_scores = []
                for train_idx, test_idx in kf.split(X):
                    X_cv_train, X_cv_test = X[train_idx], X[test_idx]
                    y_cv_train, y_cv_test = y[train_idx], y[test_idx]
                    cv_classifier = TwoTierClassifier(clone(best_tier1), clone(best_tier2))
                    cv_classifier.fit(X_cv_train, y_cv_train)
                    y_cv_pred = cv_classifier.predict(X_cv_test)
                    cv_scores.append(accuracy_score(y_cv_test, y_cv_pred))
                best_model = classifier
                best_params = f"Tier 1: {grid_search_tier1.best_params_}, Tier 2: {grid_search_tier2.best_params_}"
            
            # Metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            overfitting_gap = train_accuracy - test_accuracy
            
            # Store results
            result = {
                'Model': model_name,
                'Approach': approach,
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'F1 Score': f1,
                'CV Mean': np.mean(cv_scores),
                'CV Std': np.std(cv_scores),
                'Overfitting Gap': overfitting_gap,
                'Best Parameters': best_params
            }
            
            # Save model
            joblib.dump(best_model, model_file)
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[label_map[i] for i in range(3)],
                        yticklabels=[label_map[i] for i in range(3)])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'{model_name} - {approach}')
            plt.savefig(f'{results_dir}/confusion_matrices/{model_name.replace(" ", "_").lower()}_{approach}_cm.png')
            plt.close()
            
            # Print results
            print(f"Best parameters: {result['Best Parameters']}")
            print(f"Training accuracy: {train_accuracy:.4f}")
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"F1 score: {f1:.4f}")
            print(f"CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"Overfitting gap: {overfitting_gap:.4f}")
            
            # Save checkpoint
            save_checkpoint(checkpoints, result, model_name, approach, checkpoint_file)
            all_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{results_dir}/experiment_results.csv', index=False)
    
    # Generate visualizations
    visualize_results(results_df, label_map)
    
    # Generate summary report
    generate_summary_report(results_df)
    
    return results_df

# ---- Visualization Function ----
def visualize_results(results_df, label_map):
    """Generate rich visualizations"""
    print("Generating Approach Comparison Graph...")  # Print to verify code execution
    # 1. Approach comparison
    plt.figure(figsize=(16, 8))
    sns.barplot(x='Model', y='Test Accuracy', hue='Approach', data=results_df)
    plt.title('Test Accuracy Comparison: 1v1v1 vs Two-Tier Approach')
    plt.xticks(rotation=90)     
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7) 
    plt.savefig(f'{results_dir}/figures/approach_comparison.png', dpi=300)
    plt.close()

    print("Generating Overfitting Comparison Graph...")  # Print to verify code execution
    # 2. Overfitting comparison
    plt.figure(figsize=(16, 8))
    sns.barplot(x='Model', y='Overfitting Gap', hue='Approach', data=results_df)
    plt.title('Overfitting Gap by Model and Approach')
    plt.xticks(rotation=90)  
    plt.grid(True, linestyle='--', alpha=0.7)  
    plt.savefig(f'{results_dir}/figures/overfitting_comparison.png', dpi=300)
    plt.close()

    print("Generating Class Distribution Graph...")  # Print to verify code execution
    # 3. Class distribution (unchanged)
    X, y = np.load(f'{results_dir}/mfcc_features.npy'), np.load(f'{results_dir}/mfcc_labels.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    train_counts = pd.Series(y_train).map(label_map).value_counts()
    sns.barplot(x=train_counts.index, y=train_counts.values)
    plt.title('Training Set Class Distribution')
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.subplot(1, 2, 2)
    test_counts = pd.Series(y_test).map(label_map).value_counts()
    sns.barplot(x=test_counts.index, y=test_counts.values)
    plt.title('Test Set Class Distribution')
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/class_distribution.png', dpi=300)
    plt.close()


# ---- Summary Report Function ----
def generate_summary_report(results_df):
    """Generate a detailed summary report"""
    with open(f'{results_dir}/experiment_summary.md', 'w') as f:
        f.write("# Phase 1 Experiment: Classification Approach Comparison\n\n")
        f.write("## Overview\n")
        f.write("This experiment compares single-stage (1v1v1) and two-tier classification approaches for chicken sound classification.\n\n")
        
        # Best overall configuration
        best_row = results_df.loc[results_df['Test Accuracy'].idxmax()]
        f.write("### Best Overall Configuration\n")
        f.write(f"- **Model**: {best_row['Model']}\n")
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
        
        # Best configuration per approach
        f.write("### Best Configuration Per Approach\n")
        for approach in results_df['Approach'].unique():
            subset = results_df[results_df['Approach'] == approach]
            best_row = subset.loc[subset['Test Accuracy'].idxmax()]
            f.write(f"#### {approach}\n")
            f.write(f"- **Model**: {best_row['Model']}\n")
            f.write(f"- **Test Accuracy**: {best_row['Test Accuracy']:.4f}\n")
            f.write(f"- **F1 Score**: {best_row['F1 Score']:.4f}\n")
            f.write(f"- **Overfitting Gap**: {best_row['Overfitting Gap']:.4f}\n")
            f.write(f"- **Parameters**: {best_row['Best Parameters']}\n\n")
        
        # Statistical comparison of approaches
        f.write("### Statistical Comparison of Approaches\n")
        approach_comparison = []
        for model_name in results_df['Model'].unique():
            single_results = results_df[(results_df['Model'] == model_name) & (results_df['Approach'] == '1v1v1')]
            two_tier_results = results_df[(results_df['Model'] == model_name) & (results_df['Approach'] == 'Two-Tier')]
            if not single_results.empty and not two_tier_results.empty:
                single_cv = single_results['CV Mean'].iloc[0]
                two_tier_cv = two_tier_results['CV Mean'].iloc[0]
                diff = single_cv - two_tier_cv
                approach_comparison.append((model_name, single_cv, two_tier_cv, diff))
                f.write(f"- **{model_name}**: 1v1v1 CV: {single_cv:.4f}, Two-Tier CV: {two_tier_cv:.4f}, Difference: {diff:.4f}\n")
        avg_diff = sum(d for _, _, _, d in approach_comparison) / len(approach_comparison)
        f.write(f"\nAverage CV Difference (1v1v1 - Two-Tier): {avg_diff:.4f}\n")
        f.write(f"Models favoring 1v1v1: {sum(1 for d in approach_comparison if d[3] > 0)}/{len(approach_comparison)}\n")
        f.write(f"Models favoring Two-Tier: {sum(1 for d in approach_comparison if d[3] < 0)}/{len(approach_comparison)}\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        best_approach = approach_analysis.loc[approach_analysis[('Test Accuracy', 'mean')].idxmax(), 'Approach']
        f.write(f"1. **Approach Impact**: {best_approach} approach provides the best overall performance (mean test accuracy: {approach_analysis[('Test Accuracy', 'mean')].max():.4f}).\n")
        best_model = model_analysis.loc[model_analysis[('Test Accuracy', 'mean')].idxmax(), 'Model']
        f.write(f"2. **Model Performance**: {best_model} shows the strongest overall performance (mean test accuracy: {model_analysis[('Test Accuracy', 'mean')].max():.4f}).\n")
        min_overfitting_approach = approach_analysis.loc[approach_analysis[('Overfitting Gap', 'mean')].idxmin(), 'Approach']
        f.write(f"3. **Overfitting Trends**: {min_overfitting_approach} approach tends to reduce overfitting (mean gap: {approach_analysis[('Overfitting Gap', 'mean')].min():.4f}).\n\n")
        
        # Recommendations for Phase 2
        f.write("## Recommendations for Phase 2\n")
        top_models_cv = results_df.groupby('Model')['CV Mean'].mean().nlargest(3).index.tolist()
        if avg_diff > 0.01:
            f.write(f"The 1v1v1 approach outperforms Two-Tier (avg CV diff: {avg_diff:.4f}). Focus on 1v1v1 with top models based on CV accuracy:\n")
        elif avg_diff < -0.01:
            f.write(f"The Two-Tier approach outperforms 1v1v1 (avg CV diff: {avg_diff:.4f}). Focus on Two-Tier with top models based on CV accuracy:\n")
        else:
            f.write(f"No clear winner between approaches (avg CV diff: {avg_diff:.4f}). Evaluate both with top models based on CV accuracy:\n")
        f.write(f"- Top models: {', '.join(top_models_cv)}\n")

# ---- Main Execution ----
if __name__ == "__main__":
    results = run_experiment()
    print(f"\nPhase 1 evaluation complete. Results saved to: {results_dir}")



# # ------------------------------------------ deprecated -----------------------------------------
# # Define label mapping for readability in reports and plots
# label_map = {0: 'healthy', 1: 'sick', 2: 'noise'}

# # Load M1 features and labels
# X_m1 = np.load('M1_features.npy')
# y_m1 = np.load('M1_labels.npy')

# # Apply consistent data splitting
# X_train, X_test, y_train, y_test = train_test_split(X_m1, y_m1, test_size=0.2, random_state=42)

# # Create results directory structure
# if not os.path.exists('results'):
#     os.makedirs('results')
# if not os.path.exists('results/confusion_matrices'):
#     os.makedirs('results/confusion_matrices')
# if not os.path.exists('models'):
#     os.makedirs('models')

# # Visualize class distribution in train and test sets
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# train_counts = pd.Series(y_train).map(label_map).value_counts()
# sns.barplot(x=train_counts.index, y=train_counts.values)
# plt.title('Training Set Class Distribution')
# plt.ylabel('Count')
# plt.xlabel('Class')

# plt.subplot(1, 2, 2)
# test_counts = pd.Series(y_test).map(label_map).value_counts()
# sns.barplot(x=test_counts.index, y=test_counts.values)
# plt.title('Test Set Class Distribution')
# plt.ylabel('Count')
# plt.xlabel('Class')

# plt.tight_layout()
# plt.savefig('results/class_distribution.png')
# plt.close()

# # Results storage
# results = []
# best_models = {}

# # Part 1: Single-stage classifier evaluation (1v1v1)
# print("Evaluating Single-Stage (1v1v1) Classification Approach")
# for model_name, (model, param_grid) in models.items():
#     try:
#         print(f"\nTraining {model_name} for 1v1v1 classification")
        
#         # Clone the model to ensure a fresh instance
#         model_instance = clone(model)
        
#         # Perform grid search for hyperparameter tuning
#         grid_search = GridSearchCV(model_instance, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
#         grid_search.fit(X_train, y_train)
#         best_model = grid_search.best_estimator_
#         best_models[f"{model_name}_1v1v1"] = best_model
        
#         # Make predictions
#         y_train_pred = best_model.predict(X_train)
#         y_test_pred = best_model.predict(X_test)
        
#         # Calculate metrics
#         train_accuracy = accuracy_score(y_train, y_train_pred)
#         test_accuracy = accuracy_score(y_test, y_test_pred)
#         f1 = f1_score(y_test, y_test_pred, average='weighted')
        
#         # Confusion matrix
#         cm = confusion_matrix(y_test, y_test_pred)
        
#         # Cross-validation score
#         cv_scores = cross_val_score(best_model, X_m1, y_m1, cv=5)
        
#         print(f'Best parameters: {grid_search.best_params_}')
#         print(f'Training Accuracy: {train_accuracy:.3f}')
#         print(f'Test Accuracy: {test_accuracy:.3f}')
#         print(f'Weighted F1 Score: {f1:.3f}')
#         print(f'CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}')
#         print('Test Classification Report:')
#         print(classification_report(y_test, y_test_pred, target_names=[label_map[i] for i in range(3)]))
        
#         # Store results
#         results.append({
#             'Model': model_name,
#             'Approach': '1v1v1',
#             'Train Accuracy': train_accuracy,
#             'Test Accuracy': test_accuracy,
#             'F1 Score': f1,
#             'CV Mean': np.mean(cv_scores),
#             'CV Std': np.std(cv_scores),
#             'Accuracy Difference': abs(train_accuracy - test_accuracy)
#         })
        
#         # Save the trained model
#         joblib.dump(best_model, f'models/{model_name.replace(" ", "_").lower()}_1v1v1_model.pkl')
        
#         # Plot confusion matrix
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                     xticklabels=[label_map[i] for i in range(3)],
#                     yticklabels=[label_map[i] for i in range(3)])
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title(f'{model_name} - 1v1v1 Approach')
#         plt.tight_layout()
#         plt.savefig(f'results/confusion_matrices/{model_name.replace(" ", "_").lower()}_1v1v1_cm.png')
#         plt.close()
        
#     except Exception as e:
#         print(f"Error training {model_name} for 1v1v1: {e}")
#         continue

# # Part 2: Two-tier classifier evaluation
# print("\nEvaluating Two-Tier Classification Approach")
# for model_name, (model, param_grid) in models.items():
#     try:
#         print(f"\nTraining {model_name} for two-tier classification")
        
#         # Clone the model to create separate instances for each tier
#         tier1_model = clone(model)
#         tier2_model = clone(model)
        
#         # Create and train the two-tier classifier
#         two_tier_classifier = TwoTierClassifier(tier1_model, tier2_model)
#         two_tier_classifier.fit(X_train, y_train)
#         best_models[f"{model_name}_two_tier"] = two_tier_classifier
        
#         # Make predictions
#         y_train_pred = two_tier_classifier.predict(X_train)
#         y_test_pred = two_tier_classifier.predict(X_test)
        
#         # Calculate metrics
#         train_accuracy = accuracy_score(y_train, y_train_pred)
#         test_accuracy = accuracy_score(y_test, y_test_pred)
#         f1 = f1_score(y_test, y_test_pred, average='weighted')
        
#         # Confusion matrix
#         cm = confusion_matrix(y_test, y_test_pred)
        
#         # Cross-validation (need to use a simpler approach for custom classifier)
#         kf = KFold(n_splits=5, random_state=42, shuffle=True)
#         cv_scores = []
        
#         for train_idx, test_idx in kf.split(X_m1):
#             X_cv_train, X_cv_test = X_m1[train_idx], X_m1[test_idx]
#             y_cv_train, y_cv_test = y_m1[train_idx], y_m1[test_idx]
            
#             # Train two-tier classifier
#             cv_tier1 = clone(model)
#             cv_tier2 = clone(model)
#             cv_classifier = TwoTierClassifier(cv_tier1, cv_tier2)
#             cv_classifier.fit(X_cv_train, y_cv_train)
            
#             # Predict and calculate accuracy
#             y_cv_pred = cv_classifier.predict(X_cv_test)
#             cv_scores.append(accuracy_score(y_cv_test, y_cv_pred))
        
#         print(f'Training Accuracy: {train_accuracy:.3f}')
#         print(f'Test Accuracy: {test_accuracy:.3f}')
#         print(f'Weighted F1 Score: {f1:.3f}')
#         print(f'CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}')
#         print('Test Classification Report:')
#         print(classification_report(y_test, y_test_pred, target_names=[label_map[i] for i in range(3)]))
        
#         # Store results
#         results.append({
#             'Model': model_name,
#             'Approach': 'Two-Tier',
#             'Train Accuracy': train_accuracy,
#             'Test Accuracy': test_accuracy,
#             'F1 Score': f1,
#             'CV Mean': np.mean(cv_scores),
#             'CV Std': np.std(cv_scores),
#             'Accuracy Difference': abs(train_accuracy - test_accuracy)
#         })
        
#         # Save the trained model
#         joblib.dump(two_tier_classifier, f'models/{model_name.replace(" ", "_").lower()}_two_tier_model.pkl')
        
#         # Plot confusion matrix
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                     xticklabels=[label_map[i] for i in range(3)],
#                     yticklabels=[label_map[i] for i in range(3)])
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title(f'{model_name} - Two-Tier Approach')
#         plt.tight_layout()
#         plt.savefig(f'results/confusion_matrices/{model_name.replace(" ", "_").lower()}_two_tier_cm.png')
#         plt.close()
        
#     except Exception as e:
#         print(f"Error training {model_name} for two-tier: {e}")
#         continue

# # Part 3: Statistical comparison of approaches
# print("\nStatistical Comparison of Approaches")
# # Group results by model
# models_to_compare = list(models.keys())
# for model_name in models_to_compare:
#     single_results = [r for r in results if r['Model'] == model_name and r['Approach'] == '1v1v1']
#     two_tier_results = [r for r in results if r['Model'] == model_name and r['Approach'] == 'Two-Tier']
    
#     if single_results and two_tier_results:
#         single_cv = single_results[0]['CV Mean']
#         two_tier_cv = two_tier_results[0]['CV Mean']
        
#         print(f"\nModel: {model_name}")
#         print(f"1v1v1 CV Accuracy: {single_cv:.4f}")
#         print(f"Two-Tier CV Accuracy: {two_tier_cv:.4f}")
#         print(f"Difference: {abs(single_cv - two_tier_cv):.4f}")
        
#         if single_cv > two_tier_cv:
#             print(f"1v1v1 approach performs better for {model_name}")
#         else:
#             print(f"Two-tier approach performs better for {model_name}")

# # Save results to CSV
# with open('results/comparison_results.csv', 'w', newline='') as csvfile:
#     fieldnames = ['Model', 'Approach', 'Train Accuracy', 'Test Accuracy', 
#                  'F1 Score', 'CV Mean', 'CV Std', 'Accuracy Difference']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for row in results:
#         writer.writerow(row)

# # Create visualization of results
# plt.figure(figsize=(12, 8))
# df_results = pd.DataFrame(results)
# sns.barplot(x='Model', y='Test Accuracy', hue='Approach', data=df_results)
# plt.title('Test Accuracy Comparison: 1v1v1 vs Two-Tier Approach')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('results/approach_comparison.png')

# # Create a summary of 1v1v1 model performance
# df_1v1v1 = pd.DataFrame([r for r in results if r['Approach'] == '1v1v1'])
# df_1v1v1_sorted = df_1v1v1.sort_values(by='CV Mean', ascending=False)

# print("\nModel Ranking by Cross-Validation Accuracy (1v1v1 approach):")
# print(df_1v1v1_sorted[['Model', 'CV Mean', 'CV Std', 'Test Accuracy', 'F1 Score', 'Accuracy Difference']])

# # Generate a summary conclusion
# with open('results/phase1_conclusions.txt', 'w') as f:
#     f.write("Phase 1 Conclusions\n")
#     f.write("=================\n\n")
    
#     # Classification approach comparison
#     f.write("Classification Approach Comparison:\n")
#     approach_comparison = []
#     for model_name in models.keys():
#         single_results = [r for r in results if r['Model'] == model_name and r['Approach'] == '1v1v1']
#         two_tier_results = [r for r in results if r['Model'] == model_name and r['Approach'] == 'Two-Tier']
        
#         if single_results and two_tier_results:
#             single_cv = single_results[0]['CV Mean']
#             two_tier_cv = two_tier_results[0]['CV Mean']
#             diff = single_cv - two_tier_cv
#             approach_comparison.append((model_name, single_cv, two_tier_cv, diff))
    
#     # Calculate overall statistics
#     approach_diffs = [d for _, _, _, d in approach_comparison]
#     avg_diff = sum(approach_diffs) / len(approach_diffs)
#     better_approach = "1v1v1" if avg_diff > 0 else "Two-Tier"
    
#     f.write(f"Average difference (1v1v1 - Two-Tier): {avg_diff:.4f}\n")
#     f.write(f"Models where 1v1v1 performed better: {sum(1 for d in approach_diffs if d > 0)}/{len(approach_diffs)}\n")
#     f.write(f"Models where Two-Tier performed better: {sum(1 for d in approach_diffs if d < 0)}/{len(approach_diffs)}\n\n")
    
#     # Make definitive conclusion about approach
#     if avg_diff > 0.01:
#         f.write("CONCLUSION: The 1v1v1 approach consistently outperforms the two-tier approach across most models.\n")
#         f.write("For Phase 2, focus on 1v1v1 approach with top-performing models.\n\n")
#         primary_approach = "1v1v1"
#     elif avg_diff < -0.01:
#         f.write("CONCLUSION: The two-tier approach consistently outperforms the 1v1v1 approach across most models.\n")
#         f.write("For Phase 2, focus on two-tier approach with top-performing models.\n\n")
#         primary_approach = "Two-Tier"
#     else:
#         f.write("CONCLUSION: Both approaches perform similarly with no clear winner.\n")
#         f.write("For Phase 2, evaluate both approaches with top-performing models.\n\n")
#         primary_approach = "both"
    
#     # Top performing models for recommended approach
#     if primary_approach == "1v1v1" or primary_approach == "both":
#         f.write("Top Performing Models (1v1v1 approach):\n")
#         for i, (_, row) in enumerate(df_1v1v1_sorted.head(4).iterrows()):
#             f.write(f"{i+1}. {row['Model']}: CV Accuracy {row['CV Mean']:.4f} ± {row['CV Std']:.4f}, "
#                     f"Test Accuracy: {row['Test Accuracy']:.4f}, F1 Score: {row['F1 Score']:.4f}\n")
    
#     if primary_approach == "Two-Tier" or primary_approach == "both":
#         f.write("\nTop Performing Models (Two-Tier approach):\n")
#         df_two_tier = pd.DataFrame([r for r in results if r['Approach'] == 'Two-Tier'])
#         df_two_tier_sorted = df_two_tier.sort_values(by='CV Mean', ascending=False)
#         for i, (_, row) in enumerate(df_two_tier_sorted.head(4).iterrows()):
#             f.write(f"{i+1}. {row['Model']}: CV Accuracy {row['CV Mean']:.4f} ± {row['CV Std']:.4f}, "
#                     f"Test Accuracy: {row['Test Accuracy']:.4f}, F1 Score: {row['F1 Score']:.4f}\n")
    
#     # Final recommendations
#     f.write("\nFINAL RECOMMENDATIONS FOR PHASE 2:\n")
    
#     if primary_approach == "1v1v1":
#         top_models = df_1v1v1_sorted.head(3)['Model'].tolist()
#         f.write(f"Focus on 1v1v1 classification approach\n")
#     elif primary_approach == "Two-Tier":
#         top_models = df_two_tier_sorted.head(3)['Model'].tolist()
#         f.write(f"Focus on two-tier classification approach\n")
#     else:
#         top_1v1v1 = df_1v1v1_sorted.head(2)['Model'].tolist()
#         top_two_tier = df_two_tier_sorted.head(2)['Model'].tolist()
#         top_models = list(set(top_1v1v1 + top_two_tier))
#         f.write(f"Continue evaluating both classification approaches\n")

# print("\nPhase 1 evaluation complete. Results and conclusions saved to 'results' directory.")