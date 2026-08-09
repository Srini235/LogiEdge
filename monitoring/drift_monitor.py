"""
Task E1 — PSI Drift Monitoring
Population Stability Index (PSI) monitoring on model's output confidence score distribution.

Loads the central reference distribution generated during training to calculate real-time drift.
"""

import paho.mqtt.client as mqtt
import json
import time
import threading
import numpy as np
import os
from collections import deque
import logging
import sys
from pathlib import Path

# 1. Initialize this at the very top of your script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    # Explicitly routing to stderr guarantees zero buffering on any Linux OS
    handlers=[logging.StreamHandler(sys.stderr)] 
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
REFERENCE_PATH = BASE_DIR / 'reference_dist.json'
BROKER = os.environ.get('MQTT_BROKER', 'localhost')
TOPIC = "logiedge/trucks/+/inference"

# PSI Bins: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0]
PSI_BINS = [0.0, 0.25, 0.50, 0.75, 1.0]

# Rolling window for PSI calculation
confidence_window = deque(maxlen=5)
lock = threading.Lock()


def calculate_psi(expected, actual, epsilon=1e-6):
    """
    Compute Population Stability Index (PSI).
    PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
    """
    expected_pct = np.clip(expected, epsilon, None)
    actual_pct = np.clip(actual, epsilon, None)
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return np.sum(psi_values)


# --- Load reference distribution ---
if not os.path.exists(REFERENCE_PATH):
    logging.critical(f"FATAL: Deployment artifact missing! {REFERENCE_PATH} not found.")
    logging.critical("Cannot monitor drift without a central baseline. Exiting.")
    sys.exit(1)
else:
    with open(REFERENCE_PATH, 'r') as f:
        ref_data = json.load(f)
    expected_dist = np.array(ref_data['distribution'])
    logging.info(f"[INIT] Loaded production reference distribution: {[f'{p:.3f}' for p in expected_dist]}")


def on_message(client, userdata, msg):
    """MQTT message handler - receives inference results."""
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        with lock:
            confidence_window.append(payload['confidence'])
    except Exception as e:
        logging.error(f"[ERROR] Failed to process message: {e}")


def psi_monitor_loop():
    """Background thread: calculates PSI every 60 seconds."""
    while True:
        time.sleep(60)  # Calculate every 60 seconds
        
        with lock:
            if len(confidence_window) < 5:
                logging.info(f"[MONITOR] Buffering... ({len(confidence_window)}/5 inferences)")
                continue
            
            # Compute distribution across 4 bins
            hist, _ = np.histogram(list(confidence_window), bins=PSI_BINS)
            actual_dist = hist / 5.0
            
            psi = calculate_psi(expected_dist, actual_dist)
            
            if psi > 0.25:
                logging.warning(f"*** [LOGIEDGE DRIFT ALERT] PSI={psi:.3f} (Threshold > 0.25) ***")
            else:
                logging.info(f"Current PSI={psi:.3f} (Healthy)")


def main():
    """Main entry point - starts MQTT listener and PSI monitor."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="logiedge-monitor")
    client.on_message = on_message
    
    logging.info(f"Connecting to MQTT Broker at {BROKER}:1883...")
    client.connect(BROKER, 1883, keepalive=60)
    client.subscribe(TOPIC, qos=1)
    
    logging.info(f"Subscribed to topic: {TOPIC}")
    logging.info("Starting PSI Drift Monitor...")
    logging.info(f"Reference distribution: {[f'{p:.3f}' for p in expected_dist]}")
    logging.info("-" * 50)
    
    # Start PSI monitor in background thread
    threading.Thread(target=psi_monitor_loop, daemon=True).start()
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logging.info("\nStopping PSI Drift Monitor...")
        client.disconnect()


if __name__ == "__main__":
    main()