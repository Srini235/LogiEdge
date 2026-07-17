"""
LogiEdge - Model Training (Task D1)

Trains a 2-hidden-layer MLP (32, 16 units, ReLU) to classify cold-chain
cargo state into 3 classes: Normal (0), Warning (1), Critical (2).

Hard requirement from the spec: validation accuracy (20% held-out split)
must exceed 88%, or the model is not fit for pharmaceutical cold-chain
monitoring and grading is capped for this component.

Usage:
    python3 train_model.py --dataset dataset.csv --out-dir models/logiedge_fp32
"""

import argparse
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLUMNS = [
    "temp_mean", "temp_std", "temp_roc",
    "vib_rms", "vib_peak", "vib_kurtosis",
]
VAL_ACCURACY_THRESHOLD = 0.88


def load_dataset(path: str):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].values.astype("float32")
    y = df["label"].values.astype("int32")
    return X, y


def build_model(input_dim: int, num_classes: int = 3) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LogiEdge classification model")
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--out-dir", default="models/logiedge_fp32")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    X, y = load_dataset(args.dataset)
    print(f"[train] loaded {len(X)} samples, class distribution: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, stratify=y, random_state=args.seed
    )
    print(f"[train] train={len(X_train)} val={len(X_val)}")

    # class_weight guards against the mild class imbalance (Normal has more
    # windows than Warning/Critical per the spec's suggested durations)
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weight = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    print(f"[train] class_weight: {class_weight}")

    model = build_model(input_dim=X.shape[1])
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=16,
        class_weight=class_weight,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[train] Final validation accuracy: {val_acc:.4f}")

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\n[train] Classification report (validation set):")
    print(classification_report(y_val, y_pred, target_names=["Normal", "Warning", "Critical"]))
    print("[train] Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, y_pred))

    # Critical-class recall is separately load-bearing later (Task F3 gate,
    # >95%), so surface it clearly now even though this is the FP32 baseline.
    critical_recall = classification_report(
        y_val, y_pred, target_names=["Normal", "Warning", "Critical"], output_dict=True
    )["Critical"]["recall"]
    print(f"[train] Critical-class recall: {critical_recall:.4f} "
          f"(F3 deployment gate requires >0.95 on the RECOMMENDED variant, "
          f"not necessarily this FP32 baseline)")

    if val_acc <= VAL_ACCURACY_THRESHOLD:
        print(
            f"\n[train] FAILED: validation accuracy {val_acc:.4f} does not exceed "
            f"the required {VAL_ACCURACY_THRESHOLD:.2f} threshold.\n"
            f"[train] Per the spec, revisit feature extraction or architecture "
            f"before proceeding. Common causes: window/step size too coarse, "
            f"insufficient class separation, or too few epochs.",
            file=sys.stderr,
        )
        sys.exit(1)

    import os
    os.makedirs(os.path.dirname(args.out_dir) or ".", exist_ok=True)

    keras_path = args.out_dir.rstrip("/") + ".keras"
    model.save(keras_path)
    model.export(args.out_dir)  # SavedModel format, required as input to TFLiteConverter
    print(f"\n[train] PASSED threshold.")
    print(f"[train] Keras model saved to {keras_path}")
    print(f"[train] SavedModel exported to {args.out_dir} (use this for TFLite conversion)")


if __name__ == "__main__":
    main()