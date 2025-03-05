import os
import numpy as np
import librosa

def load_wav_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            y, sr = librosa.load(filepath, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs = np.mean(mfccs.T, axis=0)  # Take the mean of the MFCCs
            data.append(mfccs)
    return np.array(data)

def save_npy(data, filename):
    np.save(filename, data)

def main():
    dataset_dir = 'dataset'
    categories = ['healthy', 'sick', 'none']
    
    combined_data = []
    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        category_data = load_wav_files(category_dir)
        save_npy(category_data, f'{category}.npy')
        if category != 'none':
            combined_data.extend(category_data)
    
    combined_data = np.array(combined_data)
    save_npy(combined_data, 'chicken.npy')

if __name__ == '__main__':
    main()