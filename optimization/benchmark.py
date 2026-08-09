"""
Task F2 — Five-Metric Benchmarking
Benchmark all three model variants:
- M1: FP32 Baseline
- M2: PTQ INT8
- M3: Structured Pruning + PTQ INT8

Metrics measured:
1. Mean inference latency (ms) — 200 runs after 10 warm-up
2. p95 inference latency (ms)
3. Model file size (KB)
4. Classification accuracy on validation set (%)
5. Energy per inference (mJ)
"""

import time
import os
import sys
import numpy as np
from pathlib import Path

# Add paths
OPT_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = OPT_DIR / '..' / 'inference'
TRAINING_DIR = OPT_DIR / '..' / 'training'

# Import tflite properly
try:
    import tflite_runtime.interpreter as tflite_runtime
    Interpreter = tflite_runtime.Interpreter
    print('Using tflite_runtime')
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    print('Using tensorflow.lite')

import psutil


def load_data():
    """Load validation data for accuracy testing."""
    # Load training data
    data_0 = np.loadtxt(os.path.join(TRAINING_DIR, 'dataset_class_0.csv'), delimiter=',')
    data_1 = np.loadtxt(os.path.join(TRAINING_DIR, 'dataset_class_1.csv'), delimiter=',')
    data_2 = np.loadtxt(os.path.join(TRAINING_DIR, 'dataset_class_2.csv'), delimiter=',')
    
    dataset = np.vstack([data_0, data_1, data_2])
    X = dataset[:, :6].astype(np.float32)
    y = dataset[:, 6].astype(np.int32)
    
    # Normalize
    stats = np.load(os.path.join(INFERENCE_DIR, 'training_stats.npy'))
    mean, std = stats[0], stats[1]
    X_norm = (X - mean) / std
    
    # Take last 200 samples for testing
    return X_norm[-200:], y[-200:]


def benchmark_model(model_path, X_val, y_val):
    """
    Benchmark a single TFLite model.
    Returns dictionary with 5 metrics.
    """
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}
    
    # File size
    file_size_kb = os.path.getsize(model_path) / 1024
    
    # Load model
    try:
        interpreter = Interpreter(model_path=model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}
    
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']
    
    # Get input shape and type
    input_shape = interpreter.get_input_details()[0]['shape']
    input_dtype = interpreter.get_input_details()[0]['dtype']
    
    # CPU power estimation (Pi 5 TDP ~ 12W, laptop ~15-28W)
    TDP_WATTS = 15.0  # Laptop TDP estimate
    
    # 10 warm-up runs
    for i in range(10):
        if input_dtype == np.int8:
            # INT8 model - need quantization
            dummy_input = (X_val[0].reshape(input_shape) * 127).astype(np.int8)
        else:
            dummy_input = X_val[0].reshape(input_shape).astype(np.float32)
        interpreter.set_tensor(input_idx, dummy_input)
        interpreter.invoke()
    
    # 200 benchmark runs
    n_runs = min(200, len(X_val))
    latencies = []
    correct = 0
    
    # Get process for CPU monitoring
    process = psutil.Process(os.getpid())
    
    for i in range(n_runs):
        # Reset CPU measurement
        process.cpu_percent()
        
        # Run inference
        start = time.perf_counter()
        
        # Handle INT8 vs FP32 models
        if input_dtype == np.int8:
            input_data = (X_val[i].reshape(input_shape) * 127).astype(np.int8)
        else:
            input_data = X_val[i].reshape(input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_idx, input_data)
        interpreter.invoke()
        
        end = time.perf_counter()
        
        # Latency in ms
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        
        # Get prediction for accuracy
        output = interpreter.get_tensor(output_idx)
        pred = np.argmax(output[0])
        if pred == y_val[i]:
            correct += 1
        
        # Small sleep to allow CPU measurement
        time.sleep(0.001)
    
    # Calculate metrics
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    accuracy = (correct / n_runs) * 100
    
    # Energy calculation: E = P × t
    cpu_util = process.cpu_percent() / 100.0
    power_watts = max(2.5, TDP_WATTS * cpu_util)  # Minimum 2.5W for the SoC
    energy_mj = power_watts * (mean_latency / 1000)  # mJ = W × s
    
    return {
        "file_size_kb": round(file_size_kb, 2),
        "mean_latency_ms": round(mean_latency, 4),
        "p95_latency_ms": round(p95_latency, 4),
        "accuracy_percent": round(accuracy, 2),
        "energy_mj": round(energy_mj, 4)
    }


def main():
    print("=" * 70)
    print("Five-Metric Benchmark Suite - LogiEdge Model Variants")
    print("=" * 70)
    
    # Define model paths
    models = {
        "M1 (FP32 Baseline)": os.path.join(INFERENCE_DIR, "anomaly_model.tflite"),
        "M2 (PTQ INT8)": os.path.join(OPT_DIR, "anomaly_model_int8.tflite"),
        "M3 (Pruned + INT8)": os.path.join(OPT_DIR, "anomaly_model_pruned_int8.tflite")
    }
    
    # Load validation data
    print("\nLoading validation data...")
    X_val, y_val = load_data()
    print(f"Validation samples: {len(X_val)}")
    
    # Benchmark each model
    results = {}
    for name, path in models.items():
        print(f"\n{'='*50}")
        print(f"Benchmarking: {name}")
        print(f"Path: {path}")
        print(f"{'='*50}")
        
        result = benchmark_model(path, X_val, y_val)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            results[name] = None
        else:
            print(f"  File Size:     {result['file_size_kb']:.2f} KB")
            print(f"  Mean Latency:  {result['mean_latency_ms']:.4f} ms")
            print(f"  p95 Latency:   {result['p95_latency_ms']:.4f} ms")
            print(f"  Accuracy:      {result['accuracy_percent']:.2f} %")
            print(f"  Energy/Inf:    {result['energy_mj']:.4f} mJ")
            results[name] = result
    
    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Size(KB)':<10} {'Mean(ms)':<10} {'p95(ms)':<10} {'Acc(%)':<10} {'Energy(mJ)':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        if result:
            print(f"{name:<25} {result['file_size_kb']:<10} {result['mean_latency_ms']:<10.4f} "
                  f"{result['p95_latency_ms']:<10.4f} {result['accuracy_percent']:<10.2f} {result['energy_mj']:<12.4f}")
        else:
            print(f"{name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
    
    print("=" * 70)
    
    # Find best model for deployment
    if results.get("M2 (PTQ INT8)") and results.get("M1 (FP32 Baseline)"):
        m2 = results["M2 (PTQ INT8)"]
        m1 = results["M1 (FP32 Baseline)"]
        
        print("\nAnalysis:")
        print(f"  - INT8 reduces size by {m1['file_size_kb'] - m2['file_size_kb']:.2f} KB ({((m1['file_size_kb'] - m2['file_size_kb'])/m1['file_size_kb'])*100:.1f}%)")
        print(f"  - INT8 speedup: {m1['mean_latency_ms']/m2['mean_latency_ms']:.2f}x faster")
        print(f"  - Energy savings: {(1 - m2['energy_mj']/m1['energy_mj'])*100:.1f}% per inference")


if __name__ == "__main__":
    main()
