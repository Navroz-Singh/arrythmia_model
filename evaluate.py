import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = 'models/best_model_advanced.h5'
HISTORY_PATH = 'models/training_history.csv'
CLASSES = ['Normal', 'LBBB', 'RBBB', 'PVC', 'APB']

print("Loading test data and model...")
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
model = tf.keras.models.load_model(MODEL_PATH)
history_df = pd.read_csv(HISTORY_PATH)
print("Loaded successfully.")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)

def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

print("\nDisplaying training history plots...")
plot_history(history_df)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()