import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import clone
import joblib
import csv

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

# Load M1 features and labels
X_m1 = np.load('M1_features.npy')
y_m1 = np.load('M1_labels.npy')

# For M1, we want to train on all three classes: 'healthy', 'sick', and 'none'.
# Therefore, no label encoding is applied.

# List to store model names and their accuracies
model_accuracies = []

# Loop through each model for Method 1
for model_name, (model, param_grid) in models.items():
    print(f"Training {model_name}")

    # Clone the model to ensure a fresh instance for each iteration
    model_instance = clone(model)

    # Split M1 data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_m1, y_m1, test_size=0.2, random_state=42)

    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(model_instance, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

    # Save the trained model
    joblib.dump(best_model, f'models/{model_name.replace(" ", "_").lower()}_model.pkl')

    # Append the model name and accuracy to the list
    model_accuracies.append((model_name, accuracy))

# Write the model accuracies to a CSV file
with open('results/M1_model_accuracies.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model Name', 'Accuracy'])
    writer.writerows(model_accuracies)
