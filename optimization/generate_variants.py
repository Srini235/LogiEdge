"""
Generate Model Variants for Benchmarking (Task F1 & F2)
- M1: FP32 Baseline (already exists as anomaly_model.tflite)
- M2: PTQ INT8 (Post-Training Quantization)
- M3: Structured Pruning + PTQ INT8
"""

import numpy as np
import tensorflow as tf
import os
import shutil

# Paths - use absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_DIR = os.path.join(BASE_DIR, 'inference')
OPT_DIR = os.path.join(BASE_DIR, 'optimization')
TRAINING_DIR = os.path.join(BASE_DIR, 'training')

MODEL_FP32 = os.path.join(INFERENCE_DIR, 'anomaly_model.tflite')
MODEL_INT8 = os.path.join(OPT_DIR, 'anomaly_model_int8.tflite')
MODEL_PRUNED_INT8 = os.path.join(OPT_DIR, 'anomaly_model_pruned_int8.tflite')


def load_data():
    """Load training data."""
    data_0 = np.loadtxt(os.path.join(TRAINING_DIR, 'dataset_class_0.csv'), delimiter=',')
    data_1 = np.loadtxt(os.path.join(TRAINING_DIR, 'dataset_class_1.csv'), delimiter=',')
    data_2 = np.loadtxt(os.path.join(TRAINING_DIR, 'dataset_class_2.csv'), delimiter=',')
    
    dataset = np.vstack([data_0, data_1, data_2])
    X = dataset[:, :6].astype(np.float32)
    y = dataset[:, 6].astype(np.int32)
    
    # Load stats
    stats = np.load(os.path.join(INFERENCE_DIR, 'training_stats.npy'))
    mean, std = stats[0], stats[1]
    
    # Normalize
    X_norm = (X - mean) / std
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val, X_norm


def build_model():
    """Build the standard MLP model."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(6,)),
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def convert_to_tflite(model, save_path, quantize=False, representative_data=None, do_prune=False):
    """Convert Keras model to TFLite with optional quantization."""
    if do_prune:
        try:
            import tensorflow_model_optimization as tfmot
            model = tfmot.sparsity.keras.strip_pruning(model)
        except:
            pass
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize and representative_data is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_dataset():
            for data in representative_data:
                yield [data.reshape(1, 6).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    return save_path


def main():
    print("=" * 60)
    print("Generating Model Variants for Benchmarking")
    print("=" * 60)
    
    # Load data
    print("\nLoading training data...")
    X_train, X_val, y_train, y_val, X_all = load_data()
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Representative dataset for PTQ (200 samples)
    np.random.seed(42)
    rep_indices = np.random.choice(len(X_all), min(200, len(X_all)), replace=False)
    rep_data = X_all[rep_indices]
    
    # === M1: FP32 Baseline ===
    print("\n[M1] Training FP32 Baseline...")
    model_fp32 = build_model()
    model_fp32.fit(X_train, y_train, validation_data=(X_val, y_val), 
                   epochs=50, batch_size=16, verbose=1)
    
    # Evaluate
    _, acc = model_fp32.evaluate(X_val, y_val, verbose=0)
    print(f"[M1] Validation Accuracy: {acc*100:.2f}%")
    
    # Convert to TFLite (FP32)
    convert_to_tflite(model_fp32, MODEL_FP32, quantize=False)
    size_m1 = os.path.getsize(MODEL_FP32) / 1024
    print(f"[M1] Saved: {MODEL_FP32} ({size_m1:.2f} KB)")
    
    # === M2: PTQ INT8 ===
    print("\n[M2] Converting to INT8 (PTQ)...")
    convert_to_tflite(model_fp32, MODEL_INT8, quantize=True, representative_data=rep_data)
    size_m2 = os.path.getsize(MODEL_INT8) / 1024
    print(f"[M2] Saved: {MODEL_INT8} ({size_m2:.2f} KB)")
    
    # === M3: Structured Pruning + PTQ INT8 ===
    print("\n[M3] Training with Pruning + INT8...")
    try:
        import tensorflow_model_optimization as tfmot
        
        # Pruning schedule: 35% sparsity
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.35,
                begin_step=0,
                end_step=len(X_train) // 16 * 30
            )
        }
        
        model_pruned = tfmot.sparsity.keras.prune_low_magnitude(build_model(), **pruning_params)
        model_pruned.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        model_pruned.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=30, batch_size=16, callbacks=callbacks, verbose=1)
        
        # Strip pruning and convert
        convert_to_tflite(model_pruned, MODEL_PRUNED_INT8, quantize=True, representative_data=rep_data, do_prune=True)
        
    except ImportError:
        print("[M3] tfmot not available, using M2 INT8 model as M3")
        shutil.copy(MODEL_INT8, MODEL_PRUNED_INT8)
    
    size_m3 = os.path.getsize(MODEL_PRUNED_INT8) / 1024
    print(f"[M3] Saved: {MODEL_PRUNED_INT8} ({size_m3:.2f} KB)")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  M1 (FP32):        {size_m1:.2f} KB")
    print(f"  M2 (PTQ INT8):   {size_m2:.2f} KB")
    print(f"  M3 (Pruned+INT8): {size_m3:.2f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
