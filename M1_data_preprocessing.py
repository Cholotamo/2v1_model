import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

# Function to extract MFCC features
def extract_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=16000)  # Load audio at 16kHz
        if y.size == 0:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)  # Take mean over time dimension
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Paths to your audio files
healthy_chicken_dir = 'dataset/Healthy'
sick_chicken_dir = 'dataset/Sick'
noise_dir = 'dataset/None'

# Extract features and labels for stage 1 classification: healthy vs sick vs noise
X_stage_1 = []
y_stage_1 = []

for file_name in os.listdir(healthy_chicken_dir):
    file_path = os.path.join(healthy_chicken_dir, file_name)
    features = extract_features(file_path)
    if features is not None:
        X_stage_1.append(features)
        y_stage_1.append(0)  # healthy=0

for file_name in os.listdir(sick_chicken_dir):
    file_path = os.path.join(sick_chicken_dir, file_name)
    features = extract_features(file_path)
    if features is not None:
        X_stage_1.append(features)
        y_stage_1.append(1)  # sick=1

for file_name in os.listdir(noise_dir):
    file_path = os.path.join(noise_dir, file_name)
    features = extract_features(file_path)
    if features is not None:
        X_stage_1.append(features)
        y_stage_1.append(2)  # noise=2

# Convert to numpy arrays
X_stage_1 = np.array(X_stage_1)
y_stage_1 = np.array(y_stage_1)

# Normalize the features
scaler_1 = StandardScaler()
X_stage_1 = scaler_1.fit_transform(X_stage_1)

# Save features and labels for stage 1 classification: chicken vs noise
np.save('M1_features.npy', X_stage_1)
np.save('M1_labels.npy', y_stage_1)

print(f"Features shape: {X_stage_1.shape}")
print(f"Labels shape: {y_stage_1.shape}")