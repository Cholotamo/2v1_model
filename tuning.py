import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Define models and their parameter grids
models = {
    "Random Forest": (RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }),
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }),
    "Support Vector Machine (SVM)": (SVC(kernel="rbf"), {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }),
    "Naive Bayes": (GaussianNB(), {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }),
    "K-Nearest Neighbors (KNN)": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    })
}

# Load stage 1 features and labels
X_stage_1 = np.load('stage_1_features.npy')
y_stage_1 = np.load('stage_1_labels.npy')

# Encode stage 1 labels
y_stage_1 = np.where(y_stage_1 == 'chicken', 1, 0)

# Load stage 2 features and labels
X_stage_2 = np.load('stage_2_features.npy')
y_stage_2 = np.load('stage_2_labels.npy')

# Encode stage 2 labels
y_stage_2 = np.where(y_stage_2 == 'healthy', 2, 1)  # 2 for healthy, 1 for sick

# Perform grid search for each model and save the best models
for stage, (X, y) in enumerate([(X_stage_1, y_stage_1), (X_stage_2, y_stage_2)], start=1):
    for model_name, (model, params) in models.items():
        print(f"Performing grid search for {model_name} on stage {stage}")

        grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        # Save the best model
        joblib.dump(best_model, f'best_models/stage_{stage}_{model_name.replace(" ", "_").lower()}_best_model.pkl')