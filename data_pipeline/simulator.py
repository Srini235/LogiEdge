'''
File name: simulator.py
Function: Does the job of a sensor simulator via Python.
This generates realistic cold-chain truck sensor data with three streams:
temperature, vibration_rms, door_event
Topics:
logiedge/
'''

# imports
import paho.mqtt.client as mqtt
import json, time, random, math
import argparse
from pathlib import Path

# Configuration parameters
BASE_DIR = Path(__file__).resolve().parent

# System parameters
broker = 'localhost'
port = 1883
truck_id = 'truck_LE_01'
interval = 1.0

topics = {
    'temperature':f'logiedge/{truck_id}/temperature',
    'vibration':f'logiedge/{truck_id}/vibration',
    'door_state':f'logiedge/{truck_id}/door'
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="LogiEdge Sensor Simulator")
    parser.add_argument(
        '--anomaly', 
        type=str, 
        choices=['none', 'temp_drift', 'vibration', 'combined'],
        default='none',
        help="Select the anomaly mode to simulate."
    )
    return parser.parse_args()

def publish_data(client, topic, sensor, value, unit):
    """Formats and publishes the JSON payload via MQTT."""
    payload = json.dumps({
        'truck_id': truck_id,
        'sensor': sensor,
        'value': round(value, 4) if isinstance(value, float) else value,
        'unit': unit,
        'timestamp': time.time()
    })
    client.publish(topic, payload, qos=1)
    
    # Console output for verification
    if isinstance(value, float):
        print(f"[{sensor:14s}] {value:6.2f} {unit}")
    else:
        print(f"[{sensor:14s}] {value} {unit}")
    
def main():
    args = parse_arguments()
    anomaly_mode = args.anomaly
    print(f"Starting simulation in mode {anomaly_mode}")

    # Setup MQTT Client
    client = mqtt.Client(client_id=f'simulator-{truck_id}')
    print(f"Connecting to MQTT Broker at {broker}:{port}")
    client.connect(broker, port, keepalive=60)
    client.loop_start()

    tick = 0
    door_state = "close"

    try:
        while True:
            loop_start_time = time.time()

            # 1. Temperature calculation (1 Hz)
            # Normal: (temp = 4.0, vib = 0.3)
            # Anomaly increases 0.08 per tick linear drift
            base_temp = 4.0
            if anomaly_mode in ['temp_drift', 'combined']:
                base_temp += (0.08 * tick)

            current_temp = random.gauss(base_temp, 0.3)
            publish_data(client, topics['temperature'], 'temperature', current_temp, 'C')                   

            # 2. Vibration Calculation (0.5 Hz - every 2nd tick)
            if tick % 2 == 0:
                # Normal: (0.45, 0.05). Anomaly, step to (1.2, 0.15)
                if anomaly_mode in ['vibration', 'combined']:
                    current_vib = random.gauss(1.2, 0.15)
                else:
                    current_vib = random.gauss(0.45, 0.05)
            
                publish_data(client, topics['vibration'], 'vibration_rms', current_vib, 'g')

            # 3. Door event (discrete)
            # 2% chance per second
            if random.random() < 0.02:
                door_state = "open" if door_state == "close" else "close"
                publish_data(client, topics['door_state'], 'door_event', door_state, 'state')
            
            tick += 1

            # Precise sleep to maintain 1 Hz despite execution time
            time.sleep(interval - ((time.time() - loop_start_time) % interval))
    
    except KeyboardInterrupt:
        print(f"\nSimulation stopped manually after {tick} seconds.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Disconnected from MQTT broker.")

if __name__ == "__main__":
    main()