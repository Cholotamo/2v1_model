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
from xgboost import XGBClassifier
import seaborn as sns

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

# Define models and their parameter grids
models = {
    "Support Vector Machine (SVM)": (SVC(kernel="rbf", probability=True), {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }),
    "Extra Trees": (ExtraTreesClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5]
    }),
    "XGBoost": (XGBClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }),
    "Random Forest": (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5]
    }),
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }),
    "Naive Bayes": (GaussianNB(), {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }),
    "K-Nearest Neighbors (KNN)": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),
    "AdaBoost": (AdaBoostClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    })
}

# Define label mapping for readability in reports and plots
label_map = {0: 'healthy', 1: 'sick', 2: 'noise'}

# Load M1 features and labels
X_m1 = np.load('M1_features.npy')
y_m1 = np.load('M1_labels.npy')

# Apply consistent data splitting
X_train, X_test, y_train, y_test = train_test_split(X_m1, y_m1, test_size=0.2, random_state=42)

# Create results directory structure
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/confusion_matrices'):
    os.makedirs('results/confusion_matrices')
if not os.path.exists('models'):
    os.makedirs('models')

# Visualize class distribution in train and test sets
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
plt.savefig('results/class_distribution.png')
plt.close()

# Results storage
results = []
best_models = {}

# Part 1: Single-stage classifier evaluation (1v1v1)
print("Evaluating Single-Stage (1v1v1) Classification Approach")
for model_name, (model, param_grid) in models.items():
    try:
        print(f"\nTraining {model_name} for 1v1v1 classification")
        
        # Clone the model to ensure a fresh instance
        model_instance = clone(model)
        
        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(model_instance, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_models[f"{model_name}_1v1v1"] = best_model
        
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_m1, y_m1, cv=5)
        
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Training Accuracy: {train_accuracy:.3f}')
        print(f'Test Accuracy: {test_accuracy:.3f}')
        print(f'Weighted F1 Score: {f1:.3f}')
        print(f'CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}')
        print('Test Classification Report:')
        print(classification_report(y_test, y_test_pred, target_names=[label_map[i] for i in range(3)]))
        
        # Store results
        results.append({
            'Model': model_name,
            'Approach': '1v1v1',
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'F1 Score': f1,
            'CV Mean': np.mean(cv_scores),
            'CV Std': np.std(cv_scores),
            'Accuracy Difference': abs(train_accuracy - test_accuracy)
        })
        
        # Save the trained model
        joblib.dump(best_model, f'models/{model_name.replace(" ", "_").lower()}_1v1v1_model.pkl')
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[label_map[i] for i in range(3)],
                    yticklabels=[label_map[i] for i in range(3)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} - 1v1v1 Approach')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrices/{model_name.replace(" ", "_").lower()}_1v1v1_cm.png')
        plt.close()
        
    except Exception as e:
        print(f"Error training {model_name} for 1v1v1: {e}")
        continue

# Part 2: Two-tier classifier evaluation
print("\nEvaluating Two-Tier Classification Approach")
for model_name, (model, param_grid) in models.items():
    try:
        print(f"\nTraining {model_name} for two-tier classification")
        
        # Clone the model to create separate instances for each tier
        tier1_model = clone(model)
        tier2_model = clone(model)
        
        # Create and train the two-tier classifier
        two_tier_classifier = TwoTierClassifier(tier1_model, tier2_model)
        two_tier_classifier.fit(X_train, y_train)
        best_models[f"{model_name}_two_tier"] = two_tier_classifier
        
        # Make predictions
        y_train_pred = two_tier_classifier.predict(X_train)
        y_test_pred = two_tier_classifier.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Cross-validation (need to use a simpler approach for custom classifier)
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        cv_scores = []
        
        for train_idx, test_idx in kf.split(X_m1):
            X_cv_train, X_cv_test = X_m1[train_idx], X_m1[test_idx]
            y_cv_train, y_cv_test = y_m1[train_idx], y_m1[test_idx]
            
            # Train two-tier classifier
            cv_tier1 = clone(model)
            cv_tier2 = clone(model)
            cv_classifier = TwoTierClassifier(cv_tier1, cv_tier2)
            cv_classifier.fit(X_cv_train, y_cv_train)
            
            # Predict and calculate accuracy
            y_cv_pred = cv_classifier.predict(X_cv_test)
            cv_scores.append(accuracy_score(y_cv_test, y_cv_pred))
        
        print(f'Training Accuracy: {train_accuracy:.3f}')
        print(f'Test Accuracy: {test_accuracy:.3f}')
        print(f'Weighted F1 Score: {f1:.3f}')
        print(f'CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}')
        print('Test Classification Report:')
        print(classification_report(y_test, y_test_pred, target_names=[label_map[i] for i in range(3)]))
        
        # Store results
        results.append({
            'Model': model_name,
            'Approach': 'Two-Tier',
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'F1 Score': f1,
            'CV Mean': np.mean(cv_scores),
            'CV Std': np.std(cv_scores),
            'Accuracy Difference': abs(train_accuracy - test_accuracy)
        })
        
        # Save the trained model
        joblib.dump(two_tier_classifier, f'models/{model_name.replace(" ", "_").lower()}_two_tier_model.pkl')
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[label_map[i] for i in range(3)],
                    yticklabels=[label_map[i] for i in range(3)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} - Two-Tier Approach')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrices/{model_name.replace(" ", "_").lower()}_two_tier_cm.png')
        plt.close()
        
    except Exception as e:
        print(f"Error training {model_name} for two-tier: {e}")
        continue

# Part 3: Statistical comparison of approaches
print("\nStatistical Comparison of Approaches")
# Group results by model
models_to_compare = list(models.keys())
for model_name in models_to_compare:
    single_results = [r for r in results if r['Model'] == model_name and r['Approach'] == '1v1v1']
    two_tier_results = [r for r in results if r['Model'] == model_name and r['Approach'] == 'Two-Tier']
    
    if single_results and two_tier_results:
        single_cv = single_results[0]['CV Mean']
        two_tier_cv = two_tier_results[0]['CV Mean']
        
        print(f"\nModel: {model_name}")
        print(f"1v1v1 CV Accuracy: {single_cv:.4f}")
        print(f"Two-Tier CV Accuracy: {two_tier_cv:.4f}")
        print(f"Difference: {abs(single_cv - two_tier_cv):.4f}")
        
        if single_cv > two_tier_cv:
            print(f"1v1v1 approach performs better for {model_name}")
        else:
            print(f"Two-tier approach performs better for {model_name}")

# Save results to CSV
with open('results/comparison_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Model', 'Approach', 'Train Accuracy', 'Test Accuracy', 
                 'F1 Score', 'CV Mean', 'CV Std', 'Accuracy Difference']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Create visualization of results
plt.figure(figsize=(12, 8))
df_results = pd.DataFrame(results)
sns.barplot(x='Model', y='Test Accuracy', hue='Approach', data=df_results)
plt.title('Test Accuracy Comparison: 1v1v1 vs Two-Tier Approach')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/approach_comparison.png')

# Create a summary of 1v1v1 model performance
df_1v1v1 = pd.DataFrame([r for r in results if r['Approach'] == '1v1v1'])
df_1v1v1_sorted = df_1v1v1.sort_values(by='CV Mean', ascending=False)

print("\nModel Ranking by Cross-Validation Accuracy (1v1v1 approach):")
print(df_1v1v1_sorted[['Model', 'CV Mean', 'CV Std', 'Test Accuracy', 'F1 Score', 'Accuracy Difference']])

# Generate a summary conclusion
with open('results/phase1_conclusions.txt', 'w') as f:
    f.write("Phase 1 Conclusions\n")
    f.write("=================\n\n")
    
    # Classification approach comparison
    f.write("Classification Approach Comparison:\n")
    approach_comparison = []
    for model_name in models.keys():
        single_results = [r for r in results if r['Model'] == model_name and r['Approach'] == '1v1v1']
        two_tier_results = [r for r in results if r['Model'] == model_name and r['Approach'] == 'Two-Tier']
        
        if single_results and two_tier_results:
            single_cv = single_results[0]['CV Mean']
            two_tier_cv = two_tier_results[0]['CV Mean']
            diff = single_cv - two_tier_cv
            approach_comparison.append((model_name, single_cv, two_tier_cv, diff))
    
    # Calculate overall statistics
    approach_diffs = [d for _, _, _, d in approach_comparison]
    avg_diff = sum(approach_diffs) / len(approach_diffs)
    better_approach = "1v1v1" if avg_diff > 0 else "Two-Tier"
    
    f.write(f"Average difference (1v1v1 - Two-Tier): {avg_diff:.4f}\n")
    f.write(f"Models where 1v1v1 performed better: {sum(1 for d in approach_diffs if d > 0)}/{len(approach_diffs)}\n")
    f.write(f"Models where Two-Tier performed better: {sum(1 for d in approach_diffs if d < 0)}/{len(approach_diffs)}\n\n")
    
    # Make definitive conclusion about approach
    if avg_diff > 0.01:
        f.write("CONCLUSION: The 1v1v1 approach consistently outperforms the two-tier approach across most models.\n")
        f.write("For Phase 2, focus on 1v1v1 approach with top-performing models.\n\n")
        primary_approach = "1v1v1"
    elif avg_diff < -0.01:
        f.write("CONCLUSION: The two-tier approach consistently outperforms the 1v1v1 approach across most models.\n")
        f.write("For Phase 2, focus on two-tier approach with top-performing models.\n\n")
        primary_approach = "Two-Tier"
    else:
        f.write("CONCLUSION: Both approaches perform similarly with no clear winner.\n")
        f.write("For Phase 2, evaluate both approaches with top-performing models.\n\n")
        primary_approach = "both"
    
    # Top performing models for recommended approach
    if primary_approach == "1v1v1" or primary_approach == "both":
        f.write("Top Performing Models (1v1v1 approach):\n")
        for i, (_, row) in enumerate(df_1v1v1_sorted.head(4).iterrows()):
            f.write(f"{i+1}. {row['Model']}: CV Accuracy {row['CV Mean']:.4f} ± {row['CV Std']:.4f}, "
                    f"Test Accuracy: {row['Test Accuracy']:.4f}, F1 Score: {row['F1 Score']:.4f}\n")
    
    if primary_approach == "Two-Tier" or primary_approach == "both":
        f.write("\nTop Performing Models (Two-Tier approach):\n")
        df_two_tier = pd.DataFrame([r for r in results if r['Approach'] == 'Two-Tier'])
        df_two_tier_sorted = df_two_tier.sort_values(by='CV Mean', ascending=False)
        for i, (_, row) in enumerate(df_two_tier_sorted.head(4).iterrows()):
            f.write(f"{i+1}. {row['Model']}: CV Accuracy {row['CV Mean']:.4f} ± {row['CV Std']:.4f}, "
                    f"Test Accuracy: {row['Test Accuracy']:.4f}, F1 Score: {row['F1 Score']:.4f}\n")
    
    # Final recommendations
    f.write("\nFINAL RECOMMENDATIONS FOR PHASE 2:\n")
    
    if primary_approach == "1v1v1":
        top_models = df_1v1v1_sorted.head(3)['Model'].tolist()
        f.write(f"Focus on 1v1v1 classification approach\n")
    elif primary_approach == "Two-Tier":
        top_models = df_two_tier_sorted.head(3)['Model'].tolist()
        f.write(f"Focus on two-tier classification approach\n")
    else:
        top_1v1v1 = df_1v1v1_sorted.head(2)['Model'].tolist()
        top_two_tier = df_two_tier_sorted.head(2)['Model'].tolist()
        top_models = list(set(top_1v1v1 + top_two_tier))
        f.write(f"Continue evaluating both classification approaches\n")

print("\nPhase 1 evaluation complete. Results and conclusions saved to 'results' directory.")