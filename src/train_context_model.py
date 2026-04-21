"""
train_context_model.py
Phase 3 — Train a TensorFlow classifier on extracted YOLO features.
Input : dataset/processed/synthetic_dataset.csv
Output: models/context_classifier/context_model.keras
        results/metrics/classification_report.txt
        results/metrics/confusion_matrix.png
        results/metrics/training_history.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import joblib
import json

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(BASE_DIR, "dataset", "processed", "synthetic_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "context_classifier")
METRIC_DIR = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

# ─── Feature columns (all numeric columns except label) ───────────────────────
FEATURE_COLS = [
    "total_objects", "productive_count", "distracting_count", "neutral_count",
    "productive_ratio", "distracting_ratio", "focused_score",
    "avg_object_area", "total_area_coverage", "avg_confidence",
    "clutter_score", "has_laptop", "has_phone", "has_book", "has_person",
]
LABEL_COL = "label"


# ─── 1. Load & validate ────────────────────────────────────────────────────────
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"[data] Loaded {len(df)} rows | columns: {list(df.columns)}")

    # Accept both string labels ("Focused"/"Distracted") and numeric (1/0)
    if df[LABEL_COL].dtype == object:
        label_map = {"Focused": 1, "Distracted": 0,
                     "focused": 1, "distracted": 0}
        df[LABEL_COL] = df[LABEL_COL].map(label_map)

    missing = df[LABEL_COL].isna().sum()
    if missing:
        print(f"[warn] Dropping {missing} rows with unmapped labels")
        df = df.dropna(subset=[LABEL_COL])

    # Keep only known feature cols that exist in the CSV
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"[data] Using {len(available)}/{len(FEATURE_COLS)} feature columns")

    X = df[available].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)

    class_counts = dict(zip(*np.unique(y, return_counts=True, sorted=True)))
    print(f"[data] Class distribution: {class_counts}")
    return X, y, available


# ─── 2. Split & scale ─────────────────────────────────────────────────────────
def prepare_splits(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Save scaler so predict_context.py can reuse it
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[prep] Scaler saved → {scaler_path}")
    print(f"[prep] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─── 3. Build model ───────────────────────────────────────────────────────────
def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="workspace_context_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()
    return model


# ─── 4. Train ─────────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_auc", save_best_only=True, mode="max"
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─── 5. Evaluate ──────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, feature_cols):
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(
        y_test, y_pred,
        target_names=["Distracted", "Focused"]
    )
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(report)

    # Save text report
    report_path = os.path.join(METRIC_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Workspace Context Classifier — Evaluation\n")
        f.write("="*50 + "\n\n")
        f.write(report)
    print(f"[eval] Report saved → {report_path}")

    # Save metrics as JSON (useful for README badges)
    from sklearn.metrics import accuracy_score, roc_auc_score
    metrics_dict = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_auc":      float(roc_auc_score(y_test, y_prob)),
        "n_test":        int(len(y_test)),
    }
    json_path = os.path.join(METRIC_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[eval] Metrics JSON → {json_path}")

    return y_pred, y_prob


# ─── 6. Plot helpers ──────────────────────────────────────────────────────────
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history.history["loss"],     label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # AUC
    axes[1].plot(history.history["auc"],     label="Train AUC")
    axes[1].plot(history.history["val_auc"], label="Val AUC")
    axes[1].set_title("AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(METRIC_DIR, "training_history.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[plot] Training history → {out}")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Distracted", "Focused"],
        yticklabels=["Distracted", "Focused"],
        ax=ax,
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    out = os.path.join(METRIC_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[plot] Confusion matrix → {out}")


# ─── 7. Save feature list ─────────────────────────────────────────────────────
def save_feature_list(feature_cols):
    path = os.path.join(MODEL_DIR, "feature_cols.json")
    with open(path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"[save] Feature list → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n🔵 Loading data...")
    X, y, feature_cols = load_data(CSV_PATH)
    save_feature_list(feature_cols)

    print("\n🔵 Splitting & scaling...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(X, y)

    print("\n🔵 Building model...")
    model = build_model(input_dim=X_train.shape[1])

    print("\n🔵 Training...")
    history = train_model(model, X_train, y_train, X_val, y_val)

    print("\n🔵 Evaluating on test set...")
    y_pred, y_prob = evaluate_model(model, X_test, y_test, feature_cols)

    print("\n🔵 Saving plots...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)

    print("\n🔵 Saving final model...")
    model_path = os.path.join(MODEL_DIR, "context_model.keras")
    model.save(model_path)
    print(f"[save] Model → {model_path}")
    print("\n✅ Phase 3 complete!")


if __name__ == "__main__":
    main()
