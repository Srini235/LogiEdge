"""
Task E1 — PSI Real-Time Live Dashboard
Subscribes to logibridge/trucks/+/inference, computes rolling PSI scores,
and visualizes PSI drift trends + confidence distribution in real-time.
"""

from collections import deque
import json
import os
import sys
import threading
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt

# --- Path Resolution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATH = os.path.join(BASE_DIR, 'reference_dist.json')

BROKER = os.environ.get('MQTT_BROKER', 'localhost')
PORT = 1883
TOPIC = "logibridge/trucks/+/inference"

PSI_BINS = [0.0, 0.25, 0.50, 0.75, 1.0]
MAX_HISTORY = 30  # Keep last 30 PSI computations on screen

# rolling confidence scores and calculated PSI history storage
confidence_window = deque(maxlen=100)
lock = threading.Lock()

psi_timestamps = deque(maxlen=MAX_HISTORY)
psi_values = deque(maxlen=MAX_HISTORY)
current_actual_dist = np.zeros(4)

# Load expected reference distribution
if os.path.exists(REFERENCE_PATH):
    with open(REFERENCE_PATH, 'r') as f:
        ref_data = json.load(f)
    expected_dist = np.array(ref_data['distribution'])
    print(
        "[INIT] Reference distribution loaded:"
        f" {[f'{p:.3f}' for p in expected_dist]}"
    )
else:
    print(
        f"[ERROR] {REFERENCE_PATH} not found! Run drift_monitor.py first to"
        " generate reference."
    )
    sys.exit(1)


def calculate_psi(expected, actual, epsilon=1e-6):
    """Compute Population Stability Index (PSI)."""
    expected_pct = np.clip(expected, epsilon, None)
    actual_pct = np.clip(actual, epsilon, None)
    psi_vals = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return np.sum(psi_vals)


# --- mqtt func ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to Broker! Subscribed to {TOPIC}")
        client.subscribe(TOPIC, qos=1)


# --- mqtt func ---
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        with lock:
            confidence_window.append(payload['confidence'])
    except Exception as e:
        print(f"[ERROR] Failed parsing MQTT payload: {e}")


time_start = time.time()


# --- bg PSI Calculator ---
def psi_calculator_loop():
    """Calculates PSI every 5 seconds for responsive UI updates."""
    global current_actual_dist
    while True:
        time.sleep(5)
        with lock:
            if len(confidence_window) < 10:  # Need at least 10 samples to render
                continue

            # Compute actual distribution
            hist, _ = np.histogram(list(confidence_window), bins=PSI_BINS)
            current_actual_dist = hist / float(len(confidence_window))

            psi = calculate_psi(expected_dist, current_actual_dist)
            elapsed = round(time.time() - time_start, 1)

            psi_timestamps.append(elapsed)
            psi_values.append(psi)


#  mqtt client
client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2, client_id="logiedge-psi-plotter"
)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, keepalive=60)
client.loop_start()

# Start bg calculator thread
threading.Thread(target=psi_calculator_loop, daemon=True).start()

# -- Real-Time Matplotlib Dashboard --
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.canvas.manager.set_window_title(
    'LogiBridge - Real-Time PSI Drift Dashboard'
)

# Plot 1: PSI Trend Line
(line_psi,) = ax1.plot(
    [], [], 'o-', color='#1f77b4', linewidth=2, label='Current PSI Score'
)
ax1.axhline(
    y=0.1, color='orange', linestyle='--', linewidth=1.5, label='Slight Drift (0.1)'
)
ax1.axhline(
    y=0.25,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Significant Drift Alert (>0.25)',
)
ax1.set_ylabel('PSI Score')
ax1.set_title(
    'Population Stability Index (PSI) Trend', fontsize=12, fontweight='bold'
)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left')

# Plot 2: Distribution Comparison Bar Chart
bin_labels = ['[0-0.25)', '[0.25-0.50)', '[0.50-0.75)', '[0.75-1.0]']
x = np.arange(len(bin_labels))
width = 0.35

rects1 = ax2.bar(
    x - width / 2,
    expected_dist,
    width,
    label='Reference (Clean)',
    color='#2ca02c',
    alpha=0.8,
)
rects2 = ax2.bar(
    x + width / 2,
    [0, 0, 0, 0],
    width,
    label='Live Actual Window',
    color='#ff7f0e',
    alpha=0.8,
)

ax2.set_xlabel('Confidence Bins')
ax2.set_ylabel('Proportion')
ax2.set_title('Reference vs Live Confidence Score Distribution')
ax2.set_xticks(x)
ax2.set_xticklabels(bin_labels)
ax2.set_ylim(0, 1.0)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left')


def update_plot(frame):
    """Refreshes the dynamic trend line and histogram bars."""
    if not psi_timestamps:
        return line_psi, rects2

    # 1. Update PSI Trend
    ts_list = list(psi_timestamps)
    psi_list = list(psi_values)

    line_psi.set_data(ts_list, psi_list)
    ax1.set_xlim(
        max(0, ts_list[-1] - 300), max(60, ts_list[-1] + 10)
    )  # 5 min view
    ax1.set_ylim(0, max(0.35, max(psi_list) + 0.05))

    # Alert visual trigger: flash background light red if PSI > 0.25
    if psi_list[-1] > 0.25:
        ax1.set_facecolor('#ffe6e6')
    elif psi_list[-1] > 0.1:
        ax1.set_facecolor('#fff5e6')
    else:
        ax1.set_facecolor('#ffffff')

    # 2. Update Distribution Bars
    for rect, val in zip(rects2, current_actual_dist):
        rect.set_height(val)

    return line_psi, rects2


# Animate every 1000 ms
ani = animation.FuncAnimation(
    fig, update_plot, interval=1000, cache_frame_data=False
)

try:
    plt.tight_layout()
    plt.show()
finally:
    client.loop_stop()
    client.disconnect()
