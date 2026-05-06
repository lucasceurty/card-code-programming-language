"""
CardCode CV Detection — YOLOv8
Detects and classifies playing cards using your locally trained model.

Expected model file:
    cardcode/yolov8_cards.pt

To train it:
    yolo train model=yolov8n.pt \
               data=/path/to/cardcode/dataset/data.yaml \
               epochs=3 imgsz=416 batch=32 name=playing_cards device=mps

Then copy the result:
    cp runs/detect/playing_cards/weights/best.pt cardcode/yolov8_cards.pt
"""

import os
import cv2
import numpy as np
from pathlib import Path
from classifier import card_to_token

# ── Model path ────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "yolov8m_cards.pt"

# ── Label parsing ─────────────────────────────────────────────────────────────
#
# Roboflow playing-card datasets use one of these label formats:
#
#   Compact      "AH"  "10S"  "QD"  "JC"
#   Spaced       "Ace Hearts"  "10 Spades"  "Queen Diamonds"
#   With 'of'    "Ace of Hearts"  "10 of Spades"
#   Underscored  "ace_hearts"  "ten_spades"
#
# All are normalised to ("A","H"), ("10","S") etc. to match classifier.py.

_RANK_MAP = {
    "ACE": "A", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5",
    "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9", "TEN": "10",
    "JACK": "J", "QUEEN": "Q", "KING": "K",
    "A": "A", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
    "J": "J", "Q": "Q", "K": "K",
}

_SUIT_MAP = {
    "HEARTS": "H", "HEART": "H", "H": "H",
    "SPADES": "S", "SPADE": "S", "S": "S",
    "DIAMONDS": "D", "DIAMOND": "D", "D": "D",
    "CLUBS": "C", "CLUB": "C", "C": "C",
}


def parse_label(raw: str) -> tuple[str | None, str | None]:
    """
    Convert any model label string into (rank, suit) matching
    classifier.py's CARD_MAP keys. Returns (None, None) if unparseable.
    """
    label = raw.upper().strip()
    label = label.replace("_", " ").replace("-", " ").replace(" OF ", " ")
    parts = label.split()

    if len(parts) == 1:
        token = parts[0]
        if token.startswith("10") and len(token) == 3:   # "10H", "10S" etc.
            return _RANK_MAP.get("10"), _SUIT_MAP.get(token[2])
        if len(token) == 2:                               # "AH", "QD" etc.
            return _RANK_MAP.get(token[0]), _SUIT_MAP.get(token[1])
        return None, None

    if len(parts) == 2:                                   # "ACE HEARTS" etc.
        return _RANK_MAP.get(parts[0]), _SUIT_MAP.get(parts[1])

    return _RANK_MAP.get(parts[0]), _SUIT_MAP.get(parts[-1])  # fallback


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model():
    if not MODEL_PATH.exists():
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  No model found at: {str(MODEL_PATH):<41}║
║                                                              ║
║  Train one (≈15 min on Mac M-series):                        ║
║                                                              ║
║  1. Download dataset (YOLOv8 format, free):                  ║
║     roboflow.com/augmented-startups/playing-cards-ow27d      ║
║                                                              ║
║  2. Edit dataset/data.yaml — set path: to absolute folder    ║
║                                                              ║
║  3. Train:                                                   ║
║     yolo train model=yolov8n.pt                              ║
║          data=/full/path/to/cardcode/dataset/data.yaml       ║
║          epochs=3 imgsz=416 batch=32 device=mps              ║
║                                                              ║
║  4. Copy result:                                             ║
║     cp runs/detect/train/weights/best.pt                     ║
║        /full/path/to/cardcode/yolov8_cards.pt                ║
║                                                              ║
║  Card detection is disabled until the model is present.      ║
╚══════════════════════════════════════════════════════════════╝
        """)
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        print(f"✓ YOLO model loaded — {len(model.names)} classes")
        print(f"  Sample labels: {list(model.names.values())[:8]}")
        return model
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return None


yolo_model = _load_model()


# ── Detection pipeline ────────────────────────────────────────────────────────

def _letterbox(frame, target=640):
    """
    Resize frame to target×target with letterboxing (black padding),
    preserving aspect ratio. Returns (resized_frame, scale, pad_x, pad_y)
    so bounding boxes can be mapped back to original coordinates.
    """
    h, w = frame.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def detect_tokens_in_frame(frame) -> tuple[list[str], list[dict]]:
    """
    Every box YOLO draws becomes one token, sorted left to right by center x.
    No deduplication, no merging — what YOLO sees is what you get.
    """
    tokens: list[str] = []
    card_info: list[dict] = []

    if yolo_model is None:
        return tokens, card_info

    target_size = 640
    lb_frame, scale, pad_x, pad_y = _letterbox(frame, target_size)

    try:
        results = yolo_model.predict(
            source=lb_frame,
            conf=0.15,
            imgsz=target_size,
            verbose=False,
        )
    except Exception as e:
        print(f"YOLO inference error: {e}")
        return tokens, card_info

    detections = []

    for box in results[0].boxes:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        raw_label  = yolo_model.names[class_id]

        # Map coords back to original frame
        lx1, ly1, lx2, ly2 = map(int, box.xyxy[0])
        x1 = int((lx1 - pad_x) / scale)
        y1 = int((ly1 - pad_y) / scale)
        x2 = int((lx2 - pad_x) / scale)
        y2 = int((ly2 - pad_y) / scale)

        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)

        w  = x2 - x1
        h  = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2

        rank, suit = parse_label(raw_label)
        token = card_to_token(rank, suit) if (rank and suit) else None

        detections.append({
            "cx": cx, "cy": cy,
            "x": x1, "y": y1, "w": w, "h": h,
            "rank":      rank      or "?",
            "suit":      suit      or "?",
            "token":     token     or "?",
            "raw_label": raw_label,
            "conf":      confidence,
            "label":     f"{rank}{suit}" if (rank and suit) else raw_label,
        })

    # Sort right to left (reverse=True) because the camera mirrors reality —
    # what appears on the right of the frame is physically on the left
    detections.sort(key=lambda c: c["cx"], reverse=True)

    for d in detections:
        if d["token"] != "?":
            tokens.append(d["token"])
        card_info.append(d)

    return tokens, card_info


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_overlays(frame, card_info: list[dict]) -> np.ndarray:
    """
    Draws bounding boxes and labels onto the frame for the live preview.

    Green  = valid CardCode card (token recognised)
    Orange = card detected but not in the CardCode mapping
             (e.g. King of Hearts — has no command assigned)
    """
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, card in enumerate(card_info):
        x, y, w, h = card["x"], card["y"], card["w"], card["h"]
        token = card["token"]
        conf  = card.get("conf", 0.0)

        color = (0, 220, 120) if token != "?" else (0, 165, 255)

        # Bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # Label text
        display = (
            f"{card['label']} {conf:.0%}"
            if token != "?"
            else f"{card['label']} (unmapped)"
        )

        scale = 0.55
        (tw, th), _ = cv2.getTextSize(display, font, scale, 1)

        # Filled rect behind text
        cv2.rectangle(overlay, (x, y - th - 10), (x + tw + 8, y), color, -1)

        # Black shadow + white text
        cv2.putText(overlay, display, (x + 4, y - 6), font, scale, (0, 0, 0), 2)
        cv2.putText(overlay, display, (x + 4, y - 6), font, scale, (255, 255, 255), 1)

        # Left-to-right order number in bottom-right corner of box
        cv2.putText(
            overlay, str(i + 1),
            (x + w - 18, y + h - 6),
            font, 0.45, color, 1,
        )

    return overlay


# ── Compatibility stub ────────────────────────────────────────────────────────

def load_templates(*args, **kwargs):
    """Not needed with YOLO — kept for app.py compatibility."""
    pass