import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import joblib

# Load the trained stage 1 model
model_stage_1 = joblib.load('models/stage_1_logistic_regression_model.pkl')

# Load stage 2 features and labels
X_stage_2 = np.load('stage_2_features.npy')
y_stage_2 = np.load('stage_2_labels.npy')

# Encode stage 2 labels
y_stage_2 = np.where(y_stage_2 == 'healthy', 2, 1)  # 2 for healthy, 1 for sick

# Use the stage 1 model to predict labels for stage 2 data, i.e. recognize chicken vs noise in stage 2 data
stage_1_predictions = model_stage_1.predict(X_stage_2)

# Combine stage 1 predictions with stage 2 features
X_combined_stage_2 = np.hstack((X_stage_2, stage_1_predictions.reshape(-1, 1)))

# Load stage 1 features and labels
X_stage_1 = np.load('stage_1_features.npy')
y_stage_1 = np.load('stage_1_labels.npy')

# Encode stage 1 labels
y_stage_1 = np.where(y_stage_1 == 'chicken', 1, 0)

# Add a column of zeros to X_stage_1 to match the shape of X_combined_stage_2
X_stage_1 = np.hstack((X_stage_1, np.zeros((X_stage_1.shape[0], 1))))

# Combine stage 1 and stage 2 data
X_combined = np.vstack((X_stage_1, X_combined_stage_2))
y_combined = np.concatenate((y_stage_1, y_stage_2))

# Split combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model for stage 2 using OneVsRestClassifier
model_stage_2 = OneVsRestClassifier(LogisticRegression(random_state=42))
model_stage_2.fit(X_train, y_train)

# Make predictions
y_pred = model_stage_2.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['noise', 'sick', 'healthy'])

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Save the trained stage 2 model
joblib.dump(model_stage_2, 'models/stage_2_logistic_regression_model.pkl')