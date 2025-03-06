import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import clone
import joblib
import itertools
import csv

# Load best models for stage 1 and stage 2
best_models_stage_1 = {
    "Random Forest": joblib.load('best_models/stage_1_random_forest_best_model.pkl'),
    "Logistic Regression": joblib.load('best_models/stage_1_logistic_regression_best_model.pkl'),
    "Support Vector Machine (SVM)": joblib.load('best_models/stage_1_support_vector_machine_(svm)_best_model.pkl'),
    "Naive Bayes": joblib.load('best_models/stage_1_naive_bayes_best_model.pkl'),
    "K-Nearest Neighbors (KNN)": joblib.load('best_models/stage_1_k-nearest_neighbors_(knn)_best_model.pkl')
}

best_models_stage_2 = {
    "Random Forest": joblib.load('best_models/stage_2_random_forest_best_model.pkl'),
    "Logistic Regression": joblib.load('best_models/stage_2_logistic_regression_best_model.pkl'),
    "Support Vector Machine (SVM)": joblib.load('best_models/stage_2_support_vector_machine_(svm)_best_model.pkl'),
    "Naive Bayes": joblib.load('best_models/stage_2_naive_bayes_best_model.pkl'),
    "K-Nearest Neighbors (KNN)": joblib.load('best_models/stage_2_k-nearest_neighbors_(knn)_best_model.pkl')
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

# List to store model names and their accuracies
model_accuracies = []

# Iterate through all permutations of models for stage 1 and stage 2
for (stage_1_name, stage_1_model), (stage_2_name, stage_2_model) in itertools.product(best_models_stage_1.items(), best_models_stage_2.items()):
    print(f"Training {stage_1_name} for stage 1 and {stage_2_name} for stage 2")

    # Clone the models to ensure a fresh instance for each iteration
    stage_1_model = clone(stage_1_model)
    stage_2_model = clone(stage_2_model)

    # Split stage 1 data into training and testing sets
    X_train_stage_1, X_test_stage_1, y_train_stage_1, y_test_stage_1 = train_test_split(X_stage_1, y_stage_1, test_size=0.2, random_state=42)

    # Train the stage 1 model
    stage_1_model.fit(X_train_stage_1, y_train_stage_1)

    # Use the stage 1 model to predict labels for stage 2 data
    stage_1_predictions = stage_1_model.predict(X_stage_2)

    # Combine stage 1 predictions with stage 2 features
    X_combined_stage_2 = np.hstack((X_stage_2, stage_1_predictions.reshape(-1, 1)))

    # Add a column of zeros to X_stage_1 to match the shape of X_combined_stage_2
    X_stage_1_with_zeros = np.hstack((X_stage_1, np.zeros((X_stage_1.shape[0], 1))))

    # Combine stage 1 and stage 2 data
    X_combined = np.vstack((X_stage_1_with_zeros, X_combined_stage_2))
    y_combined = np.concatenate((y_stage_1, y_stage_2))

    # Split combined data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    # Train the stage 2 model
    stage_2_model.fit(X_train, y_train)

    # Make predictions
    y_pred = stage_2_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['noise', 'sick', 'healthy'])

    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

    # Save the trained stage 2 model with a filename that reflects both stage 1 and stage 2 model names
    joblib.dump(stage_2_model, f'models/{stage_1_name.replace(" ", "_").lower()}_to_{stage_2_name.replace(" ", "_").lower()}_model.pkl')

    # Append the model names and accuracy to the list
    model_accuracies.append((f'{stage_1_name} to {stage_2_name}', accuracy))

# Write the model accuracies to a CSV file
with open('results/model_accuracies.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model Name', 'Accuracy'])
    writer.writerows(model_accuracies)