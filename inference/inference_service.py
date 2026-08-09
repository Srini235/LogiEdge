"""
inference_service.py (Located in inference/)
Part of the LogiEdge Inference Pipeline.
Subscribes to feature vectors, normalizes, runs TFLite inference, and handles alerts and DB logging.
"""

import paho.mqtt.client as mqtt
import json
import time
import sqlite3
import numpy as np
import os
from collections import deque


# --- TFLite Engine Import ---
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    TFLITE_AVAILABLE = True

# --- Path Resolution & Directory Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'anomaly_model.tflite')
STATS_PATH = os.path.join(BASE_DIR, 'training_stats.npy')

# Dynamically routes DB logs into your data_pipeline/ folder
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), 'data_pipeline', 'edge_data.db')

BROKER = os.environ.get('MQTT_BROKER', 'localhost')
PORT = 1883
TRUCK_ID = 'truck_LE_01'
FEATURE_TOPIC = f'logiedge/{TRUCK_ID}/features'

ALERT_THRESHOLD = 3
CLASSES = ['Normal', 'Warning', 'Critical']

# 1. Read the model path from the Docker environment variable
MODEL_PATH = os.environ.get('MODEL_PATH', 'anomaly_model.tflite')


interpreter = Interpreter(model_path=MODEL_PATH)


# --- Initialize Normalization Stats ---
if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"[CRITICAL ERROR] Missing {STATS_PATH}. Cannot boot inference engine without normalization bounds. Run train_model.py first.")

stats = np.load(STATS_PATH)
norm_mean, norm_std = stats[0], stats[1]

stats = np.load(STATS_PATH)
norm_mean, norm_std = stats[0], stats[1]

# --- Initialize TFLite Engine ---
if os.path.exists(MODEL_PATH):
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    print(f"TFLite Engine Initialized: {MODEL_PATH}")
else:
    print(f"[WARNING] {MODEL_PATH} not found. Defaulting to algorithmic execution.")
    interpreter = None

# --- SQLite Setup ---
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('PRAGMA journal_mode=WAL')  # Concurrent read/write performance
cursor.execute('''
    CREATE TABLE IF NOT EXISTS inference_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        truck_id TEXT NOT NULL,
        temp_mean REAL,
        temp_roc REAL,
        vib_rms REAL,
        class_id INTEGER NOT NULL,
        class_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        alert_fired INTEGER NOT NULL DEFAULT 0
    )
''')
conn.commit()

# --- Tracking State ---
consecutive_critical = deque(maxlen=ALERT_THRESHOLD)
total_inferences = 0

def on_message(client, userdata, msg):
    global total_inferences
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        features = np.array(payload['features'], dtype=np.float32)
        timestamp = payload['timestamp']
        
        # 1. Apply Normalization
        # EXPERIMENT FLAG: Swap with (features - (norm_mean + (3 * norm_std))) / norm_std for 3-sigma offset test
        x_norm = ((features - norm_mean) / norm_std).astype(np.float32)
        
        # 2. Run Inference
        if interpreter:
            interpreter.set_tensor(inp_det['index'], [x_norm])
            interpreter.invoke()
            probs = interpreter.get_tensor(out_det['index'])[0]
            class_id = int(np.argmax(probs))
            confidence = float(np.max(probs))
        else:
            # Algorithmic backup logic (Normal: MeanTemp near 4C, VibRMS near 0.45g)
            temp_mean, _, _, vib_rms, _, _ = features
            if temp_mean > 8.0 or vib_rms > 1.0:
                class_id = 2
            elif temp_mean > 5.5 or vib_rms > 0.7:
                class_id = 1
            else:
                class_id = 0
            confidence = 0.95

        label = CLASSES[class_id]
        total_inferences += 1
        
        # 3. Process Sequential Alerts
        consecutive_critical.append(class_id == 2)
        alert_fired = len(consecutive_critical) == ALERT_THRESHOLD and all(consecutive_critical)
        
        # 4. Insert Entry into SQLite Database
        cursor.execute(
            '''INSERT INTO inference_log 
            (timestamp, truck_id, temp_mean, temp_roc, vib_rms, class_id, class_label, confidence, alert_fired) 
            VALUES (?,?,?,?,?,?,?,?,?)''',
            (timestamp, TRUCK_ID, float(features[0]), float(features[2]), float(features[3]), class_id, label, confidence, int(alert_fired))
        )
        conn.commit()
        
        # 5. Display Console Logs
        prefix = {0: '[OK]   ', 1: '[WARN] ', 2: '[CRIT] '}[class_id]
        print(f"{prefix} μTemp={features[0]:5.1f}°C  RoC={features[2]:5.2f}°C/m  vibRMS={features[3]:4.2f}g "
              f"→ {label:8s} conf={confidence:.2f}  inf={total_inferences}")

        if alert_fired:
            print(f"\n  *** SUSTAINED ALERT FIRED ***")
            print(f"  {ALERT_THRESHOLD} consecutive Critical readings on {TRUCK_ID}.")
            print(f"  DRIVER NOTIFICATION: Pull over safely and check cargo integrity.\n")

        # 6. Publish inference result to the requested topic (Task D2 Rubric)
        publish_topic = f"logibridge/trucks/{TRUCK_ID}/inference"
        outbound_payload = json.dumps({
            "timestamp": timestamp,
            "prediction": class_id,
            "confidence": confidence
        })
        client.publish(publish_topic, outbound_payload, qos=1)

    except Exception as e:
        print(f"[INFERENCE ERROR] Pipeline failure: {e}")

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id='logiedge-inference')
    client.on_message = on_message
    
    print(f"Connecting Inference Engine to Broker at {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, keepalive=60)
    client.subscribe(FEATURE_TOPIC, qos=1)
    
    print(f"Subscribed to {FEATURE_TOPIC}. Monitoring stream...")
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\nSubscriber stopped. Total inferences compiled: {total_inferences}")
    finally:
        conn.close()
        client.disconnect()

if __name__ == "__main__":
    main()