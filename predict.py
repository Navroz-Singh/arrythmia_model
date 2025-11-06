import numpy as np
import tensorflow as tf
import pickle
import wfdb
from scipy.signal import medfilt

MODEL_PATH = 'models/best_model_advanced.h5'
SCALER_PATH = 'models/scaler.pkl'
CLASSES = ['Normal', 'LBBB', 'RBBB', 'PVC', 'APB']
WINDOW_SIZE = 180

model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

def predict_single_heartbeat(record_name, peak_index):
    """
    Loads a single heartbeat from a record, preprocesses it, and predicts its class.
    """
    print(f"Predicting for record '{record_name}' at R-peak index {peak_index}...")
    
    record_path = f'dataset/{record_name}'
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]
    
    kernel_size = int(0.6 * record.fs)

    if kernel_size % 2 == 0:
        kernel_size += 1
        
    baseline = medfilt(signal, kernel_size=kernel_size)
    filtered_signal = signal - baseline

    start, end = peak_index - (WINDOW_SIZE // 2), peak_index + (WINDOW_SIZE // 2)
    if start < 0 or end >= len(filtered_signal):
        print("Error: Peak is too close to the signal boundary.")
        return

    heartbeat = filtered_signal[start:end]
    
    heartbeat_scaled = scaler.transform(heartbeat.reshape(1, -1))
    
    heartbeat_final = np.expand_dims(heartbeat_scaled, axis=-1)

    prediction_probs = model.predict(heartbeat_final)
    predicted_class_index = np.argmax(prediction_probs)
    predicted_class_name = CLASSES[predicted_class_index]
    confidence = prediction_probs[0][predicted_class_index] * 100
    
    print(f"\n--> Prediction: {predicted_class_name} ({confidence:.2f}% confidence)")

if __name__ == '__main__':
    example_record = '107'
    example_r_peak_location = 7303
    
    predict_single_heartbeat(example_record, example_r_peak_location)

    example_r_peak_location_normal = 4426
    predict_single_heartbeat(example_record, example_r_peak_location_normal)