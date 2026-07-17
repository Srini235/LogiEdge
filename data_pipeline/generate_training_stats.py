"""
LogiEdge - Generate training_stats.npy (Task C2)

Collects ~N minutes of CLEAN (Normal-class) feature vectors from the live
MQTT stream and freezes their mean/std into training_stats.npy.

This must be run ONCE, offline, against a simulator running with
`--anomaly none`. The resulting stats file is then loaded (never
recomputed) by preprocessing.py at runtime.

Usage:
    # Terminal 1:
    python3 simulator.py --anomaly none

    # Terminal 2:
    python3 generate_training_stats.py --duration-minutes 10
"""

import argparse
import json
import time
import threading

import numpy as np
import paho.mqtt.client as mqtt

from preprocessing import SlidingWindowExtractor, Normaliser
from paho.mqtt.enums import CallbackAPIVersion


class StatsCollector:
    def __init__(self, truck_id: str, broker_host: str, broker_port: int,
                 duration_seconds: int, out_path: str):
        self.truck_id = truck_id
        self.duration_seconds = duration_seconds
        self.out_path = out_path

        self.extractor = SlidingWindowExtractor(window_seconds=30, step_seconds=10)
        self.feature_vectors = []

        self._start_time = None
        self._lock = threading.Lock()
        self._done = threading.Event()

        self.client = mqtt.Client(CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(broker_host, broker_port)

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        base = f"logibridge/trucks/{self.truck_id}/sensor"
        client.subscribe(f"{base}/temperature")
        client.subscribe(f"{base}/vibration")
        print(f"[stats-collector] subscribed under {base}/#")
        print(f"[stats-collector] collecting for {self.duration_seconds}s "
              f"(make sure simulator is running with --anomaly none)")
        self._start_time = time.time()

    def _on_message(self, client, userdata, msg, properties=None):
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        ts = payload.get("ts", time.time())

        with self._lock:
            if msg.topic.endswith("/temperature"):
                self.extractor.add_temperature(ts, payload["value"])
            elif msg.topic.endswith("/vibration"):
                self.extractor.add_vibration(ts, payload["value"])
            else:
                return

            now = time.time()
            if self.extractor.ready_to_emit(now):
                feats = self.extractor.extract(now)
                if feats is not None:
                    self.feature_vectors.append(feats)
                    print(f"[stats-collector] collected window "
                          f"#{len(self.feature_vectors)}: "
                          f"{np.round(feats, 3).tolist()}")

        if self._start_time and (time.time() - self._start_time) >= self.duration_seconds:
            self._done.set()
            client.disconnect()

    def run(self):
        self.client.loop_start()
        # block until duration elapses or we disconnect
        self._done.wait(timeout=self.duration_seconds + 30)
        self.client.loop_stop()
        self._finalise()

    def _finalise(self):
        if len(self.feature_vectors) < 5:
            raise RuntimeError(
                f"Only collected {len(self.feature_vectors)} feature windows - "
                f"too few to compute reliable stats. Check that the simulator "
                f"is running with --anomaly none and publishing to the "
                f"correct truck_id/broker."
            )

        feature_matrix = np.array(self.feature_vectors)
        mean, std = Normaliser.fit_and_save(feature_matrix, self.out_path)

        print(f"\n[stats-collector] Saved {self.out_path}")
        print(f"[stats-collector] Based on {len(self.feature_vectors)} clean windows")
        print(f"[stats-collector] mean: {np.round(mean, 4).tolist()}")
        print(f"[stats-collector] std:  {np.round(std, 4).tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training_stats.npy from clean Normal-class MQTT data"
    )
    parser.add_argument("--truck-id", default="T1")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--duration-minutes", type=float, default=10.0,
                         help="Spec requires 10 minutes of clean data")
    parser.add_argument("--out-path", default="training_stats.npy")
    args = parser.parse_args()

    collector = StatsCollector(
        truck_id=args.truck_id,
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        duration_seconds=int(args.duration_minutes * 60),
        out_path=args.out_path,
    )
    collector.run()


if __name__ == "__main__":
    main()