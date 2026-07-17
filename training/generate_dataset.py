"""
LogiEdge - Dataset Generation (Task D1)

Runs the sensor-generation logic in-process (no MQTT round-trip) for each
class/anomaly mode, for the durations specified in the problem statement,
and writes a labeled dataset.csv of 6-value feature vectors.

  Class  Label     Simulator mode   Duration    Approx. samples
  -----  --------  ---------------  ----------  ----------------
  0      Normal    --anomaly none   20 minutes  ~120 windows
  1      Warning   temp_drift       15 minutes  ~90 windows
  2      Critical  combined         15 minutes  ~90 windows

Feature order (must match preprocessing.py / training):
  [temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurtosis]

Usage:
  python3 generate_dataset.py
  python3 generate_dataset.py --out dataset.csv --speed-factor 20
"""

import argparse
import csv
import time

import numpy as np

from preprocessing import SlidingWindowExtractor, FEATURE_NAMES


SETPOINT_C = 4.0
TEMP_DRIFT_PER_READING = 0.08
VIB_NORMAL_MEAN, VIB_NORMAL_STD = 0.45, 0.05
VIB_ANOMALY_MEAN, VIB_ANOMALY_STD = 1.2, 0.15

# (label, anomaly_mode, duration_minutes, max_drift_c) - exactly as specified in Task D1.
# max_drift_c keeps drift within the class's defined band (Section 2):
#   Warning:  1-3 degC outside setpoint  -> cap at 2.5 degC (mid-band)
#   Critical: >3 degC outside setpoint   -> cap at 5.0 degC (clearly over threshold)
CLASS_SPEC = [
    (0, "none", 20, None),
    (1, "temp_drift", 15, 2.5),
    (2, "combined", 15, 5.0),
]


def temp_uses_drift(anomaly: str) -> bool:
    return anomaly in ("temp_drift", "combined")


def vib_uses_anomaly(anomaly: str) -> bool:
    return anomaly in ("vibration", "combined")


def generate_class_windows(anomaly: str, duration_minutes: float, speed_factor: float,
                            max_drift_c: float = None):
    """
    Simulates `duration_minutes` of sensor data for the given anomaly mode
    and returns a list of 6-value feature vectors (one per emitted window).

    max_drift_c caps how far the linear temperature drift is allowed to
    accumulate, so a 15-minute Warning-class run stays within the 1-3 degC
    "drifting" band from the problem statement (Section 2) instead of
    drifting unboundedly for the whole duration. Once the cap is reached,
    the temperature holds at that offset (representing a plateaued fault)
    rather than continuing to climb.

    speed_factor > 1 compresses wall-clock time so a 20-minute simulated
    class doesn't take 20 real minutes to generate. The underlying signal
    timestamps still advance at the correct simulated rate (1 Hz temp,
    0.5 Hz vibration) - only the real sleep between steps is shortened.
    """
    extractor = SlidingWindowExtractor(window_seconds=30, step_seconds=10)

    duration_seconds = duration_minutes * 60
    temp_drift_accum = 0.0

    # simulated clock - advances at the true sensor rate regardless of
    # how fast we actually sleep in wall-clock time
    sim_time = time.time()
    end_sim_time = sim_time + duration_seconds

    next_temp_tick = sim_time
    next_vib_tick = sim_time

    windows = []

    while sim_time < end_sim_time:
        advanced = False

        if sim_time >= next_temp_tick:
            if temp_uses_drift(anomaly):
                if max_drift_c is None or temp_drift_accum < max_drift_c:
                    temp_drift_accum += TEMP_DRIFT_PER_READING
                    temp_drift_accum = min(temp_drift_accum, max_drift_c) if max_drift_c else temp_drift_accum
            value = float(np.random.normal(SETPOINT_C + temp_drift_accum, 0.3))
            extractor.add_temperature(sim_time, value)
            next_temp_tick += 1.0  # 1 Hz
            advanced = True

        if sim_time >= next_vib_tick:
            if vib_uses_anomaly(anomaly):
                value = float(np.random.normal(VIB_ANOMALY_MEAN, VIB_ANOMALY_STD))
            else:
                value = float(np.random.normal(VIB_NORMAL_MEAN, VIB_NORMAL_STD))
            extractor.add_vibration(sim_time, value)
            next_vib_tick += 2.0  # 0.5 Hz
            advanced = True

        if extractor.ready_to_emit(sim_time):
            feats = extractor.extract(sim_time)
            if feats is not None:
                # Small per-window measurement jitter, independent of the
                # underlying signal noise already in temp/vibration. This
                # keeps class boundaries from being perfectly separable,
                # which better reflects a real sensor/edge-device pipeline
                # than the raw simulator noise alone.
                jitter = np.random.normal(0, 0.03, size=feats.shape)
                feats = feats + jitter
                windows.append(feats)

        if advanced:
            # advance simulated clock to the next event, compressed by speed_factor
            next_event = min(next_temp_tick, next_vib_tick)
            step = max(next_event - sim_time, 0.001)
            if speed_factor < float("inf"):
                time.sleep(step / speed_factor)
            sim_time = next_event
        else:
            sim_time = min(next_temp_tick, next_vib_tick)

    return windows


def main():
    parser = argparse.ArgumentParser(description="Generate LogiEdge labeled training dataset")
    parser.add_argument("--out", default="dataset.csv")
    parser.add_argument(
        "--speed-factor", type=float, default=float("inf"),
        help="How much faster than real time to run (default: as fast as possible). "
             "Set e.g. 1.0 to replay at true wall-clock speed for a live-looking demo."
    )
    args = parser.parse_args()

    rows = []
    for label, anomaly, duration_min, max_drift_c in CLASS_SPEC:
        print(f"[generate_dataset] class={label} anomaly={anomaly} "
              f"duration={duration_min}min max_drift={max_drift_c} ...")
        windows = generate_class_windows(anomaly, duration_min, args.speed_factor, max_drift_c)
        print(f"[generate_dataset]   -> {len(windows)} windows generated")
        for feats in windows:
            rows.append(list(feats) + [label])

    header = FEATURE_NAMES + ["label"]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n[generate_dataset] wrote {len(rows)} total rows to {args.out}")
    labels = [r[-1] for r in rows]
    for label, _, _, _ in CLASS_SPEC:
        count = labels.count(label)
        print(f"[generate_dataset]   class {label}: {count} windows")


if __name__ == "__main__":
    main()