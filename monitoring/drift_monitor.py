"""
Task E1 — PSI Drift Monitoring
Population Stability Index (PSI) monitoring on model's output confidence score distribution.

Generates reference distribution from 300 Normal-class samples (simulating clean cold-chain operation).
"""

import paho.mqtt.client as mqtt
import json
import time
import threading
import numpy as np
import os
from collections import deque
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- Configuration ---
REFERENCE_PATH = 'reference_dist.json'
BROKER = os.environ.get('MQTT_BROKER', 'localhost')
TOPIC = "logibridge/trucks/+/inference"
MODEL_PATH = os.environ.get('MODEL_PATH', 'anomaly_model.tflite')

# PSI Bins: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0]
PSI_BINS = [0.0, 0.25, 0.50, 0.75, 1.0]

# Rolling window for PSI calculation
confidence_window = deque(maxlen=100)
lock = threading.Lock()


def generate_normal_samples(n_samples=300):
    """
    Generate 300 Normal-class samples matching the simulator's 'none' mode.
    Normal operation: temp ~4°C, vib_rms ~0.45g
    """
    np.random.seed(42)  # Reproducible reference
    samples = []
    
    for _ in range(n_samples):
        # Feature vector: [temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurtosis]
        temp_mean = np.random.normal(4.0, 0.3)      # Normal: ~4°C
        temp_std = np.random.normal(0.1, 0.02)
        temp_roc = np.random.normal(0.0, 0.05)      # Near zero rate of change
        vib_rms = np.random.normal(0.45, 0.05)       # Normal: ~0.45g
        vib_peak = np.random.normal(0.5, 0.08)
        vib_kurtosis = np.random.normal(-0.5, 0.2)  # Normal kurtosis
        
        samples.append([temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurtosis])
    
    return np.array(samples, dtype=np.float32)


def compute_reference_distribution(model_path, n_samples=300, output_file=REFERENCE_PATH):
    """
    Run inference on 300 Normal-class samples to create reference distribution.
    """
    print(f"[INIT] Creating reference distribution from {n_samples} Normal-class samples...")
    
    # Load model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    
    # Load normalization stats
    stats_path = os.path.join(os.path.dirname(model_path), 'training_stats.npy')
    stats = np.load(stats_path)
    norm_mean, norm_std = stats[0], stats[1]
    
    # Generate Normal-class samples
    samples = generate_normal_samples(n_samples)
    
    # Run inference and collect confidence scores
    confidence_scores = []
    for features in samples:
        x_norm = ((features - norm_mean) / norm_std).astype(np.float32)
        interpreter.set_tensor(inp_det['index'], [x_norm])
        interpreter.invoke()
        probs = interpreter.get_tensor(out_det['index'])[0]
        confidence = float(np.max(probs))
        confidence_scores.append(confidence)
    
    # Compute distribution across 4 bins
    hist, _ = np.histogram(confidence_scores, bins=PSI_BINS)
    ref_dist = (hist / n_samples).tolist()
    
    # Save reference distribution
    reference_data = {
        "n_samples": n_samples,
        "bins": ["[0,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.0)"],
        "distribution": ref_dist,
        "description": "Reference distribution from 300 Normal-class inferences (clean cold-chain operation)"
    }
    
    with open(output_file, 'w') as f:
        json.dump(reference_data, f, indent=2)
    
    print(f"[INIT] Reference distribution saved to {output_file}")
    print(f"[INIT] Bin distribution: {[f'{p:.3f}' for p in ref_dist]}")
    
    return np.array(ref_dist)


def calculate_psi(expected, actual, epsilon=1e-6):
    """
    Compute Population Stability Index (PSI).
    PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
    """
    expected_pct = np.clip(expected, epsilon, None)
    actual_pct = np.clip(actual, epsilon, None)
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return np.sum(psi_values)


# --- Load or generate reference distribution ---
if not os.path.exists(REFERENCE_PATH):
    print(f"[INIT] Reference distribution not found at {REFERENCE_PATH}")
    print(f"[INIT] Generating from model: {MODEL_PATH}")
    expected_dist = compute_reference_distribution(MODEL_PATH)
else:
    with open(REFERENCE_PATH, 'r') as f:
        ref_data = json.load(f)
    expected_dist = np.array(ref_data['distribution'])
    print(f"[INIT] Loaded reference distribution: {[f'{p:.3f}' for p in expected_dist]}")


def on_message(client, userdata, msg):
    """MQTT message handler - receives inference results."""
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        with lock:
            confidence_window.append(payload['confidence'])
    except Exception as e:
        print(f"[ERROR] Failed to process message: {e}")


def psi_monitor_loop():
    """Background thread: calculates PSI every 60 seconds."""
    while True:
        time.sleep(60)  # Calculate every 60 seconds
        
        with lock:
            if len(confidence_window) < 100:
                print(f"[MONITOR] Buffering... ({len(confidence_window)}/100 inferences)")
                continue
            
            # Compute distribution across 4 bins
            hist, _ = np.histogram(list(confidence_window), bins=PSI_BINS)
            actual_dist = hist / 100.0
            
            psi = calculate_psi(expected_dist, actual_dist)
            
            if psi > 0.25:
                print(f"\n*** [LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f} (Threshold > 0.25) ***\n")
            else:
                print(f"[MONITOR] Current PSI={psi:.3f} (Healthy)")


def main():
    """Main entry point - starts MQTT listener and PSI monitor."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="logiedge-monitor")
    client.on_message = on_message
    
    print(f"Connecting to MQTT Broker at {BROKER}:1883...")
    client.connect(BROKER, 1883, keepalive=60)
    client.subscribe(TOPIC, qos=1)
    
    print(f"Subscribed to topic: {TOPIC}")
    print("Starting PSI Drift Monitor...")
    print(f"Reference distribution: {[f'{p:.3f}' for p in expected_dist]}")
    print("-" * 50)
    
    # Start PSI monitor in background thread
    threading.Thread(target=psi_monitor_loop, daemon=True).start()
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopping PSI Drift Monitor...")
        client.disconnect()


if __name__ == "__main__":
    main()
