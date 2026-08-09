"""
train_model.py (Located in training/)
Loads collected CSVs, calculates normalization stats from Class 0, trains the MLP, 
and exports artifacts to the inference/ directory.
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)
tf.random.set_seed(42)

# Directory routing based on your tree
BASE_DIR = os.path.dirname(__file__)
INFERENCE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'inference')

# 1. Load Data
try:
    data_0 = np.loadtxt(os.path.join(BASE_DIR, 'dataset_class_0.csv'), delimiter=',')
    data_1 = np.loadtxt(os.path.join(BASE_DIR, 'dataset_class_1.csv'), delimiter=',')
    data_2 = np.loadtxt(os.path.join(BASE_DIR, 'dataset_class_2.csv'), delimiter=',')
except OSError as e:
    raise FileNotFoundError("Missing CSV files. Run generate_dataset.py for all 3 classes first.") from e

dataset = np.vstack([data_0, data_1, data_2])
X = dataset[:, :6].astype(np.float32)
y = dataset[:, 6].astype(np.int32)

# 2. Extract Normalization Stats (From Class 0 only)
X_normal = data_0[:, :6]
mean = np.mean(X_normal, axis=0)
std = np.std(X_normal, axis=0)
std = np.where(std == 0, 1e-7, std)

# Save directly to inference folder
np.save(os.path.join(INFERENCE_DIR, 'training_stats.npy'), np.array([mean, std]))

# 3. Normalize & Split
X_norm = (X - mean) / std
X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.2, random_state=42, stratify=y)

# 4. Build & Train MLP (32 -> 16)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(6,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1)

# 5. Evaluate Rubric Threshold
val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
if val_acc < 0.88:
    print("[WARNING] Accuracy < 88%. Do not proceed. Re-collect data.")
else:
    print("[SUCCESS] Accuracy > 88%. Exporting model...")

# 6. Export to Inference Directory
converter = tf.lite.TFLiteConverter.from_keras_model(model)
with open(os.path.join(INFERENCE_DIR, 'anomaly_model.tflite'), 'wb') as f:
    f.write(converter.convert())
print(f"Successfully exported anomaly_model.tflite to {INFERENCE_DIR}")