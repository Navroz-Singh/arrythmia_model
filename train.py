import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from model import create_advanced_model
import datetime


print("Loading preprocessed data...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
print("Data loaded.")

input_shape = (X_train.shape[1], 1)
num_classes = len(np.unique(y_train))
EPOCHS = 100
BATCH_SIZE = 64

model = create_advanced_model(input_shape, num_classes)
model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    ModelCheckpoint(
        filepath='models/best_model_advanced.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    ),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)


history_df = pd.DataFrame(history.history)
history_df.to_csv('models/training_history.csv', index=False)
print("\nTraining finished and history saved.")
print("The best model has been saved to 'models/best_model_advanced.h5'")
print(f"To view TensorBoard logs, run: tensorboard --logdir {log_dir}")