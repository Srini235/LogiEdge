"""
generate_dataset.py
Listens to the MQTT features topic and saves the 6-value vectors to CSVs.
"""
import paho.mqtt.client as mqtt
import argparse, json, time, csv, os

BROKER = 'localhost'
PORT = 1883
TRUCK_ID = 'truck_LE_01'
FEATURE_TOPIC = f'logiedge/{TRUCK_ID}/features'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=int, required=True, choices=[0,1,2])
    parser.add_argument('--minutes', type=int, required=True)
    return parser.parse_args()

def on_message(client, userdata, msg):
    try:
        features = json.loads(msg.payload.decode('utf-8'))['features']
        with open(userdata['filename'], 'a', newline='') as f:
            csv.writer(f).writerow(features + [userdata['label']])
        userdata['count'] += 1
        print(f"Captured window {userdata['count']} for Class {userdata['label']}...", end='\r')
    except Exception as e:
        print(f"Error capturing row: {e}")

def main():
    args = parse_args()
    filename = os.path.join(os.path.dirname(__file__), f'dataset_class_{args.label}.csv')
    
    if os.path.exists(filename): 
        os.remove(filename)
        
    userdata = {'label': args.label, 'filename': filename, 'count': 0}
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata=userdata)
    client.on_message = on_message
    
    client.connect(BROKER, PORT, keepalive=60)
    client.subscribe(FEATURE_TOPIC)
    
    print(f"Collecting Class {args.label} for {args.minutes} minutes...")
    client.loop_start()
    time.sleep(args.minutes * 60)
    client.loop_stop()
    client.disconnect()
    
    print(f"\nCollection complete. Saved {userdata['count']} samples to {filename}")

if __name__ == "__main__":
    main()