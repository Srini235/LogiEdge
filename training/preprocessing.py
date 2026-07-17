"""
LogiEdge - Preprocessing Pipeline (Task C2)

Pipeline stages, in order:
  1. Filtering        - 5-sample moving average on temperature and vibration
  2. Feature extraction - 30s sliding window, 10s step -> 6-value feature vector
  3. Normalisation    - load precomputed training_stats.npy, never recompute live
  4. (experiment hook) - run with stats shifted by 3-sigma, compare accuracy

Subscribes to:
  logibridge/trucks/{truck_id}/sensor/temperature
  logibridge/trucks/{truck_id}/sensor/vibration
  logibridge/trucks/{truck_id}/sensor/door

Publishes extracted (and normalised) feature vectors to:
  logibridge/trucks/{truck_id}/features
"""

import argparse
import json
import time
import threading
from collections import deque

import numpy as np
from scipy.stats import kurtosis
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion


# --------------------------------------------------------------------------
# Stage 1: Filtering - simple moving average
# --------------------------------------------------------------------------
class MovingAverage:
    """Fixed-window moving average filter."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buf = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self.buf.append(value)
        return float(sum(self.buf) / len(self.buf))


# --------------------------------------------------------------------------
# Stage 2: Sliding-window feature extraction
# --------------------------------------------------------------------------
class SlidingWindowExtractor:
    """
    Maintains timestamped filtered samples and extracts a 6-value
    feature vector every `step_seconds`, over the trailing `window_seconds`.

    Feature order (fixed - must match training):
      [temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurtosis]
    """

    def __init__(self, window_seconds: int = 30, step_seconds: int = 10):
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds

        # each entry: (timestamp, filtered_value)
        self.temp_samples = deque()
        self.vib_samples = deque()

        self.temp_filter = MovingAverage(5)
        self.vib_filter = MovingAverage(5)

        self._last_emit = None

    def add_temperature(self, ts: float, raw_value: float):
        filtered = self.temp_filter.update(raw_value)
        self.temp_samples.append((ts, filtered))
        self._trim(self.temp_samples, ts)

    def add_vibration(self, ts: float, raw_value: float):
        filtered = self.vib_filter.update(raw_value)
        self.vib_samples.append((ts, filtered))
        self._trim(self.vib_samples, ts)

    def _trim(self, buf: deque, now_ts: float):
        # drop samples older than the window
        cutoff = now_ts - self.window_seconds
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def ready_to_emit(self, now_ts: float) -> bool:
        if self._last_emit is None:
            # need at least one full window before first emission
            if self.temp_samples and (now_ts - self.temp_samples[0][0]) >= self.window_seconds:
                return True
            return False
        return (now_ts - self._last_emit) >= self.step_seconds

    def extract(self, now_ts: float):
        """
        Returns a 6-value np.ndarray, or None if there isn't enough data yet.
        """
        if len(self.temp_samples) < 2 or len(self.vib_samples) < 2:
            return None

        temp_arr = np.array([v for _, v in self.temp_samples], dtype=np.float64)
        vib_arr = np.array([v for _, v in self.vib_samples], dtype=np.float64)

        temp_mean = temp_arr.mean()
        temp_std = temp_arr.std()

        # rate of change in degC/min, over the actual elapsed span of the window
        span_seconds = self.temp_samples[-1][0] - self.temp_samples[0][0]
        if span_seconds <= 0:
            temp_roc = 0.0
        else:
            span_minutes = span_seconds / 60.0
            temp_roc = (temp_arr[-1] - temp_arr[0]) / span_minutes

        vib_rms = float(np.sqrt(np.mean(vib_arr ** 2)))
        vib_peak = float(np.max(np.abs(vib_arr)))
        # kurtosis needs a few points to be meaningful; guard against tiny windows
        vib_kurt = float(kurtosis(vib_arr)) if len(vib_arr) >= 4 else 0.0

        self._last_emit = now_ts

        return np.array(
            [temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurt],
            dtype=np.float64,
        )


# --------------------------------------------------------------------------
# Stage 3: Normalisation
# --------------------------------------------------------------------------
class Normaliser:
    """
    Loads mean/std from training_stats.npy and applies z-score normalisation.
    NEVER recomputes stats from live data - stats are frozen at training time.
    """

    def __init__(self, stats_path: str = "training_stats.npy", shift_sigma: float = 0.0):
        stats = np.load(stats_path, allow_pickle=True).item()
        self.mean = np.array(stats["mean"], dtype=np.float64)
        self.std = np.array(stats["std"], dtype=np.float64)

        if shift_sigma != 0.0:
            # Task C2 mandatory experiment: shift stats by N sigma and observe
            # the downstream accuracy impact. shift_sigma=3.0 reproduces the
            # required 3-sigma shift experiment.
            self.mean = self.mean + shift_sigma * self.std

        # avoid division by zero for any near-constant feature
        self.std = np.where(self.std < 1e-8, 1e-8, self.std)

    def transform(self, feature_vector: np.ndarray) -> np.ndarray:
        return (feature_vector - self.mean) / self.std

    @staticmethod
    def fit_and_save(feature_matrix: np.ndarray, out_path: str = "training_stats.npy"):
        """
        One-time helper: compute mean/std from N minutes of CLEAN Normal-class
        feature vectors and persist them. Run this once, offline, then never
        recompute at runtime.
        """
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0)
        np.save(out_path, {"mean": mean, "std": std})
        return mean, std


# --------------------------------------------------------------------------
# MQTT wiring
# --------------------------------------------------------------------------
FEATURE_NAMES = [
    "temp_mean", "temp_std", "temp_roc",
    "vib_rms", "vib_peak", "vib_kurtosis",
]


def _make_mqtt_client():
    """
    paho-mqtt v2.x requires an explicit callback_api_version and warns/
    errors without it; v1.x doesn't have that parameter at all. This
    picks the right constructor call based on whatever is installed,
    so the code works unmodified on either major version.
    """
    if hasattr(mqtt, "CallbackAPIVersion"):
        return mqtt.Client(CallbackAPIVersion.VERSION2)
    return mqtt.Client()


class PreprocessingPipeline:
    def __init__(self, truck_id: str, broker_host: str, broker_port: int,
                 stats_path: str, shift_sigma: float):
        self.truck_id = truck_id
        self.extractor = SlidingWindowExtractor(window_seconds=30, step_seconds=10)
        self.normaliser = Normaliser(stats_path=stats_path, shift_sigma=shift_sigma)

        self.client = _make_mqtt_client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(broker_host, broker_port)

        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _on_connect(self, client, userdata, flags, rc, *args):
        # *args absorbs the extra `properties` argument passed by
        # paho-mqtt v2.x's callback API (MQTT v5 support); harmless
        # no-op under v1.x, which doesn't pass it at all.
        base = f"logibridge/trucks/{self.truck_id}/sensor"
        client.subscribe(f"{base}/temperature")
        client.subscribe(f"{base}/vibration")
        client.subscribe(f"{base}/door")
        print(f"[preprocessing] subscribed under {base}/#")

    def _on_message(self, client, userdata, msg, *args):
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
            elif msg.topic.endswith("/door"):
                # door events pass through untouched - not part of the 6-value
                # feature vector per the spec, but logged for chain-of-custody
                print(f"[preprocessing] door event: {payload.get('event')} at {ts}")
                return

            now = time.time()
            if self.extractor.ready_to_emit(now):
                features = self.extractor.extract(now)
                if features is not None:
                    self._publish_features(features)

    def _publish_features(self, raw_features: np.ndarray):
        normalised = self.normaliser.transform(raw_features)

        out_topic = f"logibridge/trucks/{self.truck_id}/features"
        payload = {
            "ts": time.time(),
            "raw": dict(zip(FEATURE_NAMES, raw_features.tolist())),
            "normalised": normalised.tolist(),
        }
        self.client.publish(out_topic, json.dumps(payload))
        print(f"[preprocessing] emitted features -> {out_topic}: "
              f"{np.round(normalised, 3).tolist()}")

    def run_forever(self):
        self.client.loop_forever()


def main():
    parser = argparse.ArgumentParser(description="LogiEdge preprocessing pipeline")
    parser.add_argument("--truck-id", default="T1")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--stats-path", default="training_stats.npy")
    parser.add_argument(
        "--shift-sigma", type=float, default=0.0,
        help="Set to 3.0 to run the mandatory 3-sigma stats-shift experiment"
    )
    args = parser.parse_args()

    pipeline = PreprocessingPipeline(
        truck_id=args.truck_id,
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        stats_path=args.stats_path,
        shift_sigma=args.shift_sigma,
    )
    pipeline.run_forever()


if __name__ == "__main__":
    main()