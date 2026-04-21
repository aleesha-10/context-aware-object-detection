"""
webcam_demo.py
Phase 4 — Real-time webcam pipeline:
  OpenCV capture → YOLO detection → feature extraction → context classification → overlay

Controls:
  Q / ESC  quit
  S        save current frame to results/sample_outputs/
  P        pause / resume

Usage:
    python src/webcam_demo.py
    python src/webcam_demo.py --source 0        # webcam index
    python src/webcam_demo.py --source image.jpg  # run on a static image
    python src/webcam_demo.py --source video.mp4  # run on a video file
"""

import os
import sys
import json
import time
import argparse
import collections
import numpy as np
import cv2
import joblib
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

# ─── Object taxonomy ──────────────────────────────────────────────────────────
PRODUCTIVE_CLASSES  = {"laptop", "book", "keyboard", "mouse", "monitor"}
DISTRACTING_CLASSES = {"cell phone", "remote", "tv", "game pad"}
NEUTRAL_CLASSES     = {"cup", "bottle", "chair", "person"}

COLOR_P = (0, 200, 100)   # green  (BGR)
COLOR_D = (30,  40, 220)  # red
COLOR_N = (180, 180,  0)  # cyan

# ─── Smoothing: average last N predictions ────────────────────────────────────
SMOOTHING_WINDOW = 8


def load_assets():
    for path, label in [
        (MODEL_PATH,    "Context model"),
        (SCALER_PATH,   "Scaler"),
        (FEATURES_PATH, "Feature list"),
    ]:
        if not os.path.exists(path):
            sys.exit(f"[error] {label} not found: {path}\n"
                     "        Run train_context_model.py first.")

    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
    yolo = YOLO(YOLO_WEIGHTS)
    print("[load] All assets ready.")
    return model, scaler, feature_cols, yolo


def extract_features(results, img_shape) -> dict:
    H, W = img_shape[:2]
    img_area = H * W
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name   = r.names[cls_id].lower()
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({"label": name, "conf": conf,
                                "area": (x2 - x1) * (y2 - y1)})

    total   = len(detections)
    prod    = [d for d in detections if d["label"] in PRODUCTIVE_CLASSES]
    dist    = [d for d in detections if d["label"] in DISTRACTING_CLASSES]
    neut    = [d for d in detections if d["label"] in NEUTRAL_CLASSES]
    p_cnt   = len(prod);  d_cnt = len(dist);  n_cnt = len(neut)
    p_ratio = p_cnt / max(total, 1)
    d_ratio = d_cnt / max(total, 1)
    areas   = [d["area"] for d in detections] or [0]
    confs   = [d["conf"] for d in detections] or [0]
    labels_set = {d["label"] for d in detections}

    return {
        "total_objects":       total,
        "productive_count":    p_cnt,
        "distracting_count":   d_cnt,
        "neutral_count":       n_cnt,
        "productive_ratio":    p_ratio,
        "distracting_ratio":   d_ratio,
        "focused_score":       p_ratio - d_ratio,
        "avg_object_area":     np.mean(areas),
        "total_area_coverage": sum(areas) / img_area,
        "avg_confidence":      float(np.mean(confs)),
        "clutter_score":       min(total / 10.0, 1.0),
        "has_laptop":  int("laptop"     in labels_set),
        "has_phone":   int("cell phone" in labels_set),
        "has_book":    int("book"       in labels_set),
        "has_person":  int("person"     in labels_set),
        "detections":  detections,
    }


def classify(feats, model, scaler, feature_cols):
    row = np.array([[feats.get(c, 0) for c in feature_cols]], dtype=np.float32)
    prob = float(model.predict(scaler.transform(row), verbose=0)[0][0])
    return prob   # probability of Focused


def draw_frame(frame, results, smooth_prob: float, fps: float,
               feats: dict) -> np.ndarray:
    vis = frame.copy()
    H, W = vis.shape[:2]

    # Detection boxes
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name   = r.names[cls_id].lower()
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (COLOR_P if name in PRODUCTIVE_CLASSES
                     else COLOR_D if name in DISTRACTING_CLASSES
                     else COLOR_N)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            tag = f"{name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, tag, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Top banner ──────────────────────────────────────────────────────────
    state  = "Focused" if smooth_prob >= 0.5 else "Distracted"
    score  = smooth_prob if state == "Focused" else (1 - smooth_prob)
    banner_color = (30, 130, 40) if state == "Focused" else (30, 40, 180)
    ov = vis.copy()
    cv2.rectangle(ov, (0, 0), (W, 55), banner_color, -1)
    cv2.addWeighted(ov, 0.65, vis, 0.35, 0, vis)

    cv2.putText(vis, f"{state}  {score:.0%}",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Productivity bar (right side of banner)
    bar_w = min(W // 3, 220)
    bar_x = W - bar_w - 12
    bar_y = 16;  bar_h = 20
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(smooth_prob * bar_w)
    bar_fill_col = (0, 200, 80) if smooth_prob >= 0.5 else (40, 40, 200)
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_fill_col, -1)
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    cv2.putText(vis, f"Focus {smooth_prob:.0%}",
                (bar_x, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Stats sidebar (bottom-left) ──────────────────────────────────────────
    lines = [
        f"FPS: {fps:.1f}",
        f"Objects: {feats['total_objects']}",
        f"  Productive:  {feats['productive_count']}",
        f"  Distracting: {feats['distracting_count']}",
        f"  Neutral:     {feats['neutral_count']}",
        f"Clutter: {feats['clutter_score']:.2f}",
    ]
    y0 = H - len(lines) * 20 - 10
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Controls hint (bottom-right) ────────────────────────────────────────
    cv2.putText(vis, "Q=quit  S=save  P=pause",
                (W - 210, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    return vis


def run(source, conf_thresh: float):
    model, scaler, feature_cols, yolo = load_assets()
    prob_buffer = collections.deque([0.5] * SMOOTHING_WINDOW, maxlen=SMOOTHING_WINDOW)

    # Open source
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        sys.exit(f"[error] Cannot open source: {source}")

    paused = False
    prev_time = time.time()
    frame_count = 0
    last_feats  = {"total_objects": 0, "productive_count": 0,
                   "distracting_count": 0, "neutral_count": 0,
                   "clutter_score": 0}
    last_results = []
    last_smooth  = 0.5

    print("[demo] Running — press Q to quit, S to save, P to pause")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[demo] End of stream.")
                break

            # Inference every frame (YOLOv8n is fast enough on CPU at 640px)
            results = yolo(frame, conf=conf_thresh, verbose=False)
            feats   = extract_features(results, frame.shape)
            prob    = classify(feats, model, scaler, feature_cols)
            prob_buffer.append(prob)
            smooth_prob  = float(np.mean(prob_buffer))
            last_feats   = feats
            last_results = results
            last_smooth  = smooth_prob

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            frame_count += 1

            vis = draw_frame(frame, results, smooth_prob, fps, feats)
        else:
            vis = draw_frame(frame, last_results, last_smooth, 0, last_feats)

        cv2.imshow("Smart Workspace Monitor", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # Q or ESC
            break
        elif key == ord("s"):
            fname = os.path.join(OUTPUT_DIR, f"frame_{int(time.time())}.jpg")
            cv2.imwrite(fname, vis)
            print(f"[save] {fname}")
        elif key == ord("p"):
            paused = not paused
            print(f"[demo] {'Paused' if paused else 'Resumed'}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] Processed {frame_count} frames.")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real-time workspace monitor demo")
    parser.add_argument("--source", default="0",
                        help="Webcam index (0), image path, or video path")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="YOLO confidence threshold (default 0.35)")
    args = parser.parse_args()

    # Convert numeric string → int for webcam
    source = int(args.source) if args.source.isdigit() else args.source
    run(source, args.conf)


if __name__ == "__main__":
    main()
