import numpy as np
import tensorflow as tf
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'inference', 'anomaly_model.tflite')
STATS_PATH = os.path.join(BASE_DIR, 'inference', 'training_stats.npy')
TRAIN_DIR = os.path.join(BASE_DIR, 'training')

# --- Load Data & Stats ---
data_0 = np.loadtxt(os.path.join(TRAIN_DIR, 'dataset_class_0.csv'), delimiter=',')
data_1 = np.loadtxt(os.path.join(TRAIN_DIR, 'dataset_class_1.csv'), delimiter=',')
data_2 = np.loadtxt(os.path.join(TRAIN_DIR, 'dataset_class_2.csv'), delimiter=',')

dataset = np.vstack([data_0, data_1, data_2])
X_raw = dataset[:, :6].astype(np.float32)
y_true = dataset[:, 6].astype(np.int32)

stats = np.load(STATS_PATH)
mean, std = stats[0], stats[1]

# --- Prepare the 3 Test Conditions ---
# Condition 1: Correct baseline
X_baseline = (X_raw - mean) / std

# Condition 2: Shifted by +3 Sigma
X_plus_3 = (X_raw - (mean + (3 * std))) / std

# Condition 3: Shifted by -3 Sigma
X_minus_3 = (X_raw - (mean - (3 * std))) / std

# --- Evaluation Function ---
def evaluate_tflite(X_test, condition_name):
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    
    correct = 0
    for i in range(len(X_test)):
        interpreter.set_tensor(inp_det['index'], [X_test[i]])
        interpreter.invoke()
        pred = np.argmax(interpreter.get_tensor(out_det['index'])[0])
        if pred == y_true[i]:
            correct += 1
            
    accuracy = (correct / len(X_test)) * 100
    print(f"{condition_name}: {accuracy:.2f}%")

# --- Run the Experiment ---
print("--- 2.3 Normalisation Experiment Results ---")
evaluate_tflite(X_baseline.astype(np.float32), "1. Correct training_stats.npy")
evaluate_tflite(X_plus_3.astype(np.float32),   "2. Stats shifted by +3σ      ")
evaluate_tflite(X_minus_3.astype(np.float32),  "3. Stats shifted by -3σ      ")
print("--------------------------------------------")