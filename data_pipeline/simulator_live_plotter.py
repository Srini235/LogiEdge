import json
import time
from collections import deque
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

# --- config ---
BROKER = 'localhost'
PORT = 1883
MAX_POINTS = 60  # window of 60 seconds

# time-series data storage
time_start = None
temp_times, temp_values = deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS)
vib_times, vib_values = deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS)
door_times, door_values = deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS)


# --- mqtt Callback ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe("logiedge/+/+")
    else:
        print(f"Failed to connect, return code {rc}")


# --- mqtt Callback ---
def on_message(client, userdata, msg):
    global time_start
    if time_start is None:
        time_start = time.time()

    elapsed_time = round(time.time() - time_start, 1)

    try:
        data = json.loads(msg.payload.decode('utf-8'))
        sensor = data.get('sensor')
        value = data.get('value')

        if sensor == 'temperature':
            temp_times.append(elapsed_time)
            temp_values.append(value)
        elif sensor == 'vibration_rms':
            vib_times.append(elapsed_time)
            vib_values.append(value)
        elif sensor == 'door_event':
            door_times.append(elapsed_time)
            # Map state text to discrete values (1 for open, 0 for close)
            door_values.append(1 if value == "open" else 0)
    except Exception as e:
        print(f"Error parsing payload: {e}")


# --- mqtt Client ---
client = mqtt.Client(client_id='live-plot-subscriber')
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, keepalive=60)
client.loop_start()

# --- multi panel plot ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.canvas.manager.set_window_title('LogiEdge Multi-Sensor Live Dashboard')

# high-performance updates
(line_temp,) = ax1.plot([], [], 'o-', color='#d95f02', label='Temperature (°C)')
(line_vib,) = ax2.plot([], [], 's--', color='#7570b3', label='Vibration RMS (g)')
(line_door,) = ax3.step([], [], where='post',
                        color='#2ca02c', linewidth=2, label='Door State')

# Panel 1: temperature
ax1.set_ylabel('Temp (°C)')
ax1.set_title('Real-time Cold Chain Sensor Telemetry')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left')

# Panel 2: vibration
ax2.set_ylabel('Vibration (g)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left')

# Panel 3: door state
ax3.set_xlabel('Elapsed Time (Seconds)')
ax3.set_ylabel('Door State')
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['CLOSED', 'OPEN'])
ax3.set_ylim(-0.2, 1.2)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper left')


def update_plot(frame):
    """Refreshes all three sensor plots dynamically."""
    # 1. update Temperature
    if temp_times:
        line_temp.set_data(list(temp_times), list(temp_values))
        ax1.set_xlim(
            max(0, temp_times[-1] - MAX_POINTS), max(MAX_POINTS, temp_times[-1]))
        ax1.set_ylim(min(temp_values) - 0.5, max(temp_values) + 0.5)

    # 2. update Vibration
    if vib_times:
        line_vib.set_data(list(vib_times), list(vib_values))
        ax2.set_ylim(min(vib_values) - 0.1, max(vib_values) + 0.1)

    # 3. update Door State
    if door_times:
        line_door.set_data(list(door_times), list(door_values))

    return line_temp, line_vib, line_door


# animate every 500ms
ani = animation.FuncAnimation(
    fig, update_plot, interval=500, cache_frame_data=False)

try:
    plt.tight_layout()
    plt.show()
finally:
    client.loop_stop()
    client.disconnect()
