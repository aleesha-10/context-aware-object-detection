"""
predict_context.py
Phase 3 — Run the full pipeline on a single image:
  YOLO detection → feature extraction → context classification → print result

Usage:
    python src/predict_context.py --image path/to/desk.jpg
    python src/predict_context.py --image path/to/desk.jpg --save
"""

import os
import sys
import argparse
import json
import numpy as np
import joblib
import cv2
import tensorflow as tf
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(BASE_DIR, "models", "context_classifier")
YOLO_DIR      = os.path.join(BASE_DIR, "models", "yolo")
OUTPUT_DIR    = os.path.join(BASE_DIR, "results", "sample_outputs")
MODEL_PATH    = os.path.join(MODEL_DIR, "context_model.keras")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_cols.json")
YOLO_WEIGHTS  = os.path.join(YOLO_DIR,  "yolov8n.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Label constants ──────────────────────────────────────────────────────────
PRODUCTIVE_CLASSES   = {"laptop", "book", "keyboard", "mouse", "monitor"}
DISTRACTING_CLASSES  = {"cell phone", "remote", "tv", "game pad"}
NEUTRAL_CLASSES      = {"cup", "bottle", "chair", "person"}

COLOR_PRODUCTIVE  = (0, 200, 100)   # green
COLOR_DISTRACTING = (0, 80,  220)   # red-ish (BGR)
COLOR_NEUTRAL     = (180, 180, 0)   # cyan


# ─── Feature extraction (mirrors extract_features.py) ────────────────────────
def extract_features_from_detections(results, img_shape) -> dict:
    H, W = img_shape[:2]
    img_area = H * W

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label  = r.names[cls_id].lower()
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            detections.append({"label": label, "conf": conf, "area": area})

    total = len(detections)
    prod_det  = [d for d in detections if d["label"] in PRODUCTIVE_CLASSES]
    dist_det  = [d for d in detections if d["label"] in DISTRACTING_CLASSES]
    neut_det  = [d for d in detections if d["label"] in NEUTRAL_CLASSES]

    productive_count   = len(prod_det)
    distracting_count  = len(dist_det)
    neutral_count      = len(neut_det)
    productive_ratio   = productive_count  / max(total, 1)
    distracting_ratio  = distracting_count / max(total, 1)
    focused_score      = productive_ratio - distracting_ratio

    all_areas = [d["area"] for d in detections] or [0]
    total_area = sum(all_areas)
    avg_area   = total_area / max(total, 1)
    total_coverage = total_area / img_area

    all_confs = [d["conf"] for d in detections] or [0]
    avg_conf  = float(np.mean(all_confs))

    clutter_score = min(total / 10.0, 1.0)

    labels_set = {d["label"] for d in detections}

    return {
        "total_objects":     total,
        "productive_count":  productive_count,
        "distracting_count": distracting_count,
        "neutral_count":     neutral_count,
        "productive_ratio":  productive_ratio,
        "distracting_ratio": distracting_ratio,
        "focused_score":     focused_score,
        "avg_object_area":   avg_area,
        "total_area_coverage": total_coverage,
        "avg_confidence":    avg_conf,
        "clutter_score":     clutter_score,
        "has_laptop":        int("laptop"      in labels_set),
        "has_phone":         int("cell phone"  in labels_set),
        "has_book":          int("book"        in labels_set),
        "has_person":        int("person"      in labels_set),
        "detections":        detections,
    }


# ─── Load model assets ────────────────────────────────────────────────────────
def load_assets():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[error] Model not found at {MODEL_PATH}. Run train_context_model.py first.")
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
    yolo = YOLO(YOLO_WEIGHTS)
    print("[load] All assets loaded.")
    return model, scaler, feature_cols, yolo


# ─── Classify ─────────────────────────────────────────────────────────────────
def classify_workspace(feats: dict, model, scaler, feature_cols) -> tuple:
    row = np.array([[feats.get(c, 0) for c in feature_cols]], dtype=np.float32)
    row_scaled = scaler.transform(row)
    prob = float(model.predict(row_scaled, verbose=0)[0][0])
    label = "Focused" if prob >= 0.5 else "Distracted"
    return label, prob


# ─── Visualise ────────────────────────────────────────────────────────────────
def draw_results(img, results, label: str, prob: float) -> np.ndarray:
    vis = img.copy()
    H, W = vis.shape[:2]

    # Draw detection boxes
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name   = r.names[cls_id].lower()
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name in PRODUCTIVE_CLASSES:
                color = COLOR_PRODUCTIVE
            elif name in DISTRACTING_CLASSES:
                color = COLOR_DISTRACTING
            else:
                color = COLOR_NEUTRAL

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            tag = f"{name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Overlay banner
    banner_h = 60
    overlay = vis.copy()
    banner_color = (30, 160, 60) if label == "Focused" else (30, 40, 200)
    cv2.rectangle(overlay, (0, 0), (W, banner_h), banner_color, -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

    score_pct = int(prob * 100) if label == "Focused" else int((1 - prob) * 100)
    text = f"Workspace: {label}  ({score_pct}% confident)"
    cv2.putText(vis, text, (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend bottom-right
    legend = [
        ("Productive", COLOR_PRODUCTIVE),
        ("Distracting", COLOR_DISTRACTING),
        ("Neutral",    COLOR_NEUTRAL),
    ]
    for i, (lname, lcol) in enumerate(legend):
        lx, ly = W - 160, H - 20 - i * 22
        cv2.rectangle(vis, (lx, ly - 12), (lx + 16, ly), lcol, -1)
        cv2.putText(vis, lname, (lx + 22, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Predict workspace context from an image")
    parser.add_argument("--image",  required=True,       help="Path to input image")
    parser.add_argument("--conf",   type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--save",   action="store_true", help="Save annotated image to results/")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"[error] Image not found: {args.image}")

    model, scaler, feature_cols, yolo = load_assets()

    img = cv2.imread(args.image)
    print(f"[run] Running YOLO on {args.image} ...")
    results = yolo(img, conf=args.conf, verbose=False)

    feats = extract_features_from_detections(results, img.shape)
    label, prob = classify_workspace(feats, model, scaler, feature_cols)

    print("\n" + "="*45)
    print(f"  Workspace State : {label}")
    focused_pct    = f"{prob*100:.1f}%"
    distracted_pct = f"{(1-prob)*100:.1f}%"
    print(f"  Focused prob    : {focused_pct}")
    print(f"  Distracted prob : {distracted_pct}")
    print(f"  Objects found   : {feats['total_objects']}")
    print(f"    Productive     : {feats['productive_count']}")
    print(f"    Distracting    : {feats['distracting_count']}")
    print("="*45 + "\n")

    vis = draw_results(img, results, label, prob)

    if args.save:
        import time
        fname = f"prediction_{int(time.time())}.jpg"
        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, vis)
        print(f"[save] Annotated image → {out_path}")
    else:
        cv2.imshow("Workspace Monitor", vis)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
