import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load features and labels
X_stage_1 = np.load('stage_1_features.npy')
y_stage_1 = np.load('stage_1_labels.npy')

# Encode labels
y_stage_1 = np.where(y_stage_1 == 'chicken', 1, 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_stage_1, y_stage_1, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['noise', 'chicken'])

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Save the trained model
import joblib
joblib.dump(model, 'models/stage_1_logistic_regression_model.pkl')