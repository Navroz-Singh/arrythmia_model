import wfdb
import numpy as np
import os
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import pickle

CLASSES = ['Normal', 'LBBB', 'RBBB', 'PVC', 'APB']
DATASET_PATH = 'dataset/'
WINDOW_SIZE = 180

def process_data():
    print("Starting data processing...")

    record_names = sorted(list(set([f.split('.')[0] for f in os.listdir(DATASET_PATH) if f.endswith('.dat')])))
    
    all_signals = []
    all_labels = []

    for record_name in record_names:
        try:
            record_path = os.path.join(DATASET_PATH, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            signal = record.p_signal[:, 0]
            r_peaks = annotation.sample
            symbols = annotation.symbol

            kernel_size = int(0.6 * record.fs)

            if kernel_size % 2 == 0:
                kernel_size += 1
            
            baseline = medfilt(signal, kernel_size=kernel_size)
            filtered_signal = signal - baseline

            for i, peak in enumerate(r_peaks):
                symbol = symbols[i]
                if symbol in CLASSES:
                    start, end = peak - (WINDOW_SIZE // 2), peak + (WINDOW_SIZE // 2)
                    if start >= 0 and end < len(filtered_signal):
                        heartbeat = filtered_signal[start:end]
                        all_signals.append(heartbeat)
                        all_labels.append(CLASSES.index(symbol))
        except Exception as e:
            print(f"Could not process record {record_name}. Error: {e}")

    X = np.array(all_signals)
    y = np.array(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rus = RandomUnderSampler(random_state=42)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train_reshaped, y_train)
    X_train_balanced = X_train_balanced.reshape(-1, WINDOW_SIZE)
    print("Class distribution after balancing:", np.bincount(y_train_balanced))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler has been saved to 'models/scaler.pkl'")

    X_train_final = np.expand_dims(X_train_scaled, axis=-1)
    X_test_final = np.expand_dims(X_test_scaled, axis=-1)

    np.save('X_train.npy', X_train_final)
    np.save('y_train.npy', y_train_balanced)
    np.save('X_test.npy', X_test_final)
    np.save('y_test.npy', y_test)

    print("Preprocessing complete. Data saved as .npy files.")
    print(f"X_train shape: {X_train_final.shape}, y_train shape: {y_train_balanced.shape}")
    print(f"X_test shape: {X_test_final.shape}, y_test shape: {y_test.shape}")

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    process_data()