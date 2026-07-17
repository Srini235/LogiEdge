import time, random, json, argparse, threading
import numpy as np
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

client = mqtt.Client(CallbackAPIVersion.VERSION2)
client.connect("localhost", 1883)

def temp_stream(anomaly):
    setpoint = 4.0
    drift = 0.0
    t = 0
    while True:
        if anomaly in ("temp_drift", "combined"):
            drift += 0.08  # linear drift per reading
        temp = np.random.normal(setpoint + drift, 0.3)
        client.publish("logibridge/trucks/T1/sensor/temperature",
                        json.dumps({"ts": time.time(), "value": temp}))
        time.sleep(1.0)  # 1 Hz

def vibration_stream(anomaly):
    while True:
        if anomaly in ("vibration", "combined"):
            v = np.random.normal(1.2, 0.15)  # bearing wear
        else:
            v = np.random.normal(0.45, 0.05)  # normal compressor RMS
        client.publish("logibridge/trucks/T1/sensor/vibration",
                        json.dumps({"ts": time.time(), "value": v}))
        time.sleep(2.0)  # 0.5 Hz

def door_stream():
    state = "CLOSE"
    while True:
        time.sleep(random.uniform(30, 300))
        state = "OPEN" if state == "CLOSE" else "CLOSE"
        client.publish("logibridge/trucks/T1/sensor/door",
                        json.dumps({"ts": time.time(), "event": state}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly", choices=["none","temp_drift","vibration","combined"], default="none")
    args = parser.parse_args()

    threading.Thread(target=temp_stream, args=(args.anomaly,), daemon=True).start()
    threading.Thread(target=vibration_stream, args=(args.anomaly,), daemon=True).start()
    threading.Thread(target=door_stream, daemon=True).start()
    while True: time.sleep(1)