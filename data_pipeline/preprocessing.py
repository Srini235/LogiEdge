"""
Task C2: Edge Inference Node
Subscribes to LogiEdge MQTT topics. Buffers data into a 30s window (10s step).
Pipeline: 5-sample Moving Average -> 6-Feature Extraction -> Normalization -> TFLite -> SQLite.
"""

import paho.mqtt.client as mqtt
import json
import time
import sqlite3
import threading
import numpy as np
from collections import deque
import os
from pathlib import Path

# --- TFLite Import (with Fallback) ---
try:
    from tflite_runtime.interpreter import Interpreter
    print('Using tflite_runtime')
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    print('Using tensorflow.lite')

# configuration
broker = 'localhost'
port = 1883
truck_id = 'truck_LE_01'

# path configuration
BASE_DIR = Path(__file__).resolve().parent

CONF_PATH = BASE_DIR / "dev.conf"

# Topics
subscriber_topics = [
    (f'logiedge/{truck_id}/temperature', 1),
    (f'logiedge/{truck_id}/vibration', 1)
]

publisher_topics = f'logiedge/{truck_id}/features'

window_size_time = 30
step_size_time = 10

buffer_lock = threading.Lock()
temperature_buffer = []
vibration_buffer = []

def moving_average(data, window=5):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode = 'valid')

def calculate_kurtosis(data):
    if len(data) < 2:
        return 0.0
    mean = np.mean(data)
    m_power4 = np.mean((data - mean)**4)
    m_power2 = np.mean((data-mean)**2)
    if m_power2 == 0:
        return 0.0
    return (m_power4 / (m_power2**2)) - 3

def on_message(client, userdata, msg):
    """
    Asynchronous MQTT message handler to ingest raw sensor readings.
    """
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        topic = msg.topic

        with buffer_lock:
            if 'temperature' in topic:
                temperature_buffer.append((payload['timestamp'], payload['value']))
            elif 'vibration' in topic:
                vibration_buffer.append((payload['timestamp'], payload['value']))

    except Exception as e:
        print(f"failed to parse message: {e}")

def process_and_publish_features(client):
    global temperature_buffer, vibration_buffer

    current_time = time.time()
    cutoff_time = current_time - window_size_time

    with buffer_lock:
        # 1. Slide window outside 30 s boundary
        temperature_buffer = [(t, v) for t, v in temperature_buffer if t >= cutoff_time]
        vibration_buffer = [(t,v) for t, v in vibration_buffer if t >= cutoff_time]

        if len(temperature_buffer) < 5 or len(vibration_buffer) < 5:
            print(f"[{time.strftime('%H:%M:%S')}] Buffering data... Temp count: {len(temperature_buffer)}, Vib count: {len(vibration_buffer)}")
            return
        
        t_times = [t for t, v in temperature_buffer]
        t_vals = [v for t, v in temperature_buffer]
        v_vals = [v for t, v in vibration_buffer]

    # 2. Filtering: apply 5 sample moving avg
    t_smooth = moving_average(t_vals, 5)
    v_smooth = moving_average(v_vals, 5)

# 3. Feature extraction
    temp_mean = np.mean(t_smooth)
    temp_std = np.std(t_smooth)  # <--- Corrected variable name and input
    
    # Rate of Change calculation in degrees C per minute
    duration_mins = (t_times[-1] - t_times[0]) / 60.0
    temp_roc = (t_smooth[-1] - t_smooth[0]) / duration_mins if duration_mins > 0 else 0.0
    
    vib_rms = np.sqrt(np.mean(np.square(v_smooth)))
    vib_peak = np.max(v_smooth)
    vib_kurtosis = calculate_kurtosis(v_smooth)
    
    # 4. Construct Feature Vector (6-value array)
    feature_vector = [
        float(temp_mean), 
        float(temp_std), 
        float(temp_roc), 
        float(vib_rms), 
        float(vib_peak), 
        float(vib_kurtosis)        
    ]
# 5. Publish features to the downstream MQTT broker topic
    payload = json.dumps({
        'truck_id': truck_id,
        'timestamp': current_time,
        'features': feature_vector
    })
    client.publish(publisher_topics, payload, qos=1)
    print(f"[{time.strftime('%H:%M:%S')}] Features Published -> MeanTemp: {temp_mean:.2f}°C, VibRMS: {vib_rms:.2f}g")

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id='logiedge-preprocessor')
    client.on_message = on_message
    
    print(f"Connecting Ingestion Pipeline to Broker at {broker}:{port}...")
    client.connect(broker, port, keepalive=60)
    client.subscribe(subscriber_topics)
    client.loop_start()
    
    print("Preprocessing Microservice Running. Running window processing...")
    
    try:
        while True:
            time.sleep(step_size_time)
            process_and_publish_features(client)
    except KeyboardInterrupt:
        print("\nStopping Preprocessing Microservice.")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()