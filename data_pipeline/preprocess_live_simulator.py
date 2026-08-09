import json
import time
from collections import deque
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

# --- config ---
BROKER = 'localhost'
PORT = 1883
TRUCK_ID = 'truck_LE_01'
TOPIC = f'logiedge/{TRUCK_ID}/features'
MAX_POINTS = 30  # Keep last 30 feature window updates (~5 minutes of data)

# time-series data storage
time_start = None
times = deque(maxlen=MAX_POINTS)

# 6 Feature Buffers
f_temp_mean = deque(maxlen=MAX_POINTS)
f_temp_std = deque(maxlen=MAX_POINTS)
f_temp_roc = deque(maxlen=MAX_POINTS)
f_vib_rms = deque(maxlen=MAX_POINTS)
f_vib_peak = deque(maxlen=MAX_POINTS)
f_vib_kurtosis = deque(maxlen=MAX_POINTS)


# --- mqtt callbacks ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to Broker! Subscribed to: {TOPIC}")
        client.subscribe(TOPIC, qos=1)
    else:
        print(f"Failed to connect, return code {rc}")


# --- mqtt callbacks ---
def on_message(client, userdata, msg):
    global time_start
    if time_start is None:
        time_start = time.time()

    elapsed_time = round(time.time() - time_start, 1)

    try:
        data = json.loads(msg.payload.decode('utf-8'))
        features = data.get('features', [])

        if len(features) == 6:
            times.append(elapsed_time)

            # Unpack 6-feature vector
            f_temp_mean.append(features[0])
            f_temp_std.append(features[1])
            f_temp_roc.append(features[2])
            f_vib_rms.append(features[3])
            f_vib_peak.append(features[4])
            f_vib_kurtosis.append(features[5])

            print(
                f"[{time.strftime('%H:%M:%S')}] Received Features -> Mean Temp:"
                f" {features[0]:.2f}°C, Vib RMS: {features[3]:.2f}g"
            )
    except Exception as e:
        print(f"Error parsing feature message: {e}")


# --- subscriber ---
client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2, client_id='edge-features-plotter'
)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, keepalive=60)
client.loop_start()

# --- setup dashboard figure (2x3 Grid) ---
fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
fig.canvas.manager.set_window_title(
    'LogiEdge - Real-Time Windowed Features (Task C2)'
)

# Unpack axis objects
((ax_temp_mean, ax_vib_rms), (ax_temp_std, ax_vib_peak),
 (ax_temp_roc, ax_vib_kurt)) = axes

# Setup Lines
(line_t_mean,) = ax_temp_mean.plot(
    [], [], 'o-', color='#d95f02', label='Mean Temp (°C)')
(line_t_std,) = ax_temp_std.plot([], [], 's-',
                                 color='#e6ab02', label='Temp Std Dev (°C)')
(line_t_roc,) = ax_temp_roc.plot([], [], '^--',
                                 color='#e7298a', label='Temp RoC (°C/min)')

(line_v_rms,) = ax_vib_rms.plot([], [], 'o-', color='#7570b3', label='Vib RMS (g)')
(line_v_peak,) = ax_vib_peak.plot(
    [], [], 's-', color='#66a61e', label='Vib Peak (g)')
(line_v_kurt,) = ax_vib_kurt.plot(
    [], [], 'd--', color='#1b9e77', label='Vib Kurtosis')

# Formatting Labels & Grids
ax_temp_mean.set_ylabel('Mean Temp (°C)')
ax_temp_std.set_ylabel('Std Dev (°C)')
ax_temp_roc.set_ylabel('RoC (°C/min)')

ax_vib_rms.set_ylabel('RMS (g)')
ax_vib_peak.set_ylabel('Peak (g)')
ax_vib_kurt.set_ylabel('Kurtosis')

ax_temp_roc.set_xlabel('Elapsed Time (Seconds)')
ax_vib_kurt.set_xlabel('Elapsed Time (Seconds)')

for ax in axes.flat:
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')

fig.suptitle('Edge Inference Pipeline: 6-Feature Stream (30s Window / 10s Step)',
             fontsize=14, fontweight='bold')


def update_plot(frame):
    """Refreshes feature subplots on every animation tick."""
    if not times:
        return (line_t_mean, line_t_std, line_t_roc, line_v_rms, line_v_peak, line_v_kurt)

    t_list = list(times)

    # Update Temperature Features
    line_t_mean.set_data(t_list, list(f_temp_mean))
    line_t_std.set_data(t_list, list(f_temp_std))
    line_t_roc.set_data(t_list, list(f_temp_roc))

    # Update Vibration Features
    line_v_rms.set_data(t_list, list(f_vib_rms))
    line_v_peak.set_data(t_list, list(f_vib_peak))
    line_v_kurt.set_data(t_list, list(f_vib_kurtosis))

    # Rescale subplots dynamically
    for ax in axes.flat:
        ax.relim()
        ax.autoscale_view()

    return (
        line_t_mean,
        line_t_std,
        line_t_roc,
        line_v_rms,
        line_v_peak,
        line_v_kurt,
    )


# animate every 1000ms
ani = animation.FuncAnimation(
    fig, update_plot, interval=1000, cache_frame_data=False
)

try:
    plt.tight_layout()
    plt.show()
finally:
    client.loop_stop()
    client.disconnect()
