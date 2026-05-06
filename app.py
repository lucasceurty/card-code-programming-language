"""
CardCode Flask Server
Serves the web UI and handles:
- /stream        : MJPEG camera feed with card overlays
- /capture       : Snap current frame, extract tokens, add to program
- /finish        : Snap final frame, transpile full program, run it, save .star file
- /reset         : Clear current program
- /program       : Get current token lines
"""

import os
import sys
import time
import threading
import base64
import json
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory

# Add cardcode dir to path
sys.path.insert(0, os.path.dirname(__file__))
from detector import detect_tokens_in_frame, draw_overlays, load_templates
from transpiler import tokens_to_python, run_program, TranspileError, tokens_from_line_text
from classifier import describe_token, token_to_card_name, token_to_shorthand

app = Flask(__name__, static_folder="static")

# ── Global state ─────────────────────────────────────────────────────────────

latest_frame = None
frame_lock   = threading.Lock()

program_lines = []      # list of lists: [[token, token, ...], [token, ...], ...]
program_lock  = threading.Lock()

PROGRAMS_DIR = os.path.join(os.path.dirname(__file__), "programs")
os.makedirs(PROGRAMS_DIR, exist_ok=True)


# ── Camera thread ─────────────────────────────────────────────────────────────

def camera_thread():
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        try:
            _, card_info = detect_tokens_in_frame(frame)
            annotated = draw_overlays(frame, card_info)
        except Exception:
            annotated = frame

        with frame_lock:
            latest_frame = annotated.copy()

        time.sleep(0.033)  # ~30fps


def get_latest_frame():
    with frame_lock:
        if latest_frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera initializing...",
                       (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 100), 2)
            return placeholder
        return latest_frame.copy()


def generate_mjpeg():
    """Generator for MJPEG stream."""
    while True:
        frame = get_latest_frame()
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )
        time.sleep(0.033)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/stream")
def stream():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture", methods=["POST"])
def capture():
    """Snap the current frame, detect cards, add tokens as a new program line."""
    frame = get_latest_frame()

    try:
        tokens, card_info = detect_tokens_in_frame(frame)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    if not tokens:
        return jsonify({
            "ok": False,
            "error": "No cards detected. Make sure cards are clearly visible."
        })

    with program_lock:
        program_lines.append(tokens)
        line_number = len(program_lines)
        all_lines   = list(program_lines)

    # Full card details for the sidebar — one entry per detected card
    # including unmapped ones so the user can see what was seen
    cards = [
        {
            "label":  c["label"],
            "token":  c["token"],
            "card":   token_to_card_name(c["token"]) if c["token"] != "?" else c["label"],
            "desc":   describe_token(c["token"])      if c["token"] != "?" else "Unmapped — no CardCode command",
            "conf":   round(c.get("conf", 0.0), 2),
            "mapped": c["token"] != "?",
        }
        for c in card_info
    ]

    # Flat token-only list (mapped cards only, for the program log)
    token_details = [
        {"token": t, "card": token_to_card_name(t), "desc": describe_token(t)}
        for t in tokens
    ]

    shorthand = " ".join(token_to_shorthand(t) for t in tokens)

    return jsonify({
        "ok":          True,
        "line_number": line_number,
        "tokens":      token_details,
        "cards":       cards,           # full per-card detail including unmapped
        "shorthand":   shorthand,       # editable .card-style line
        "total_lines": len(all_lines),
    })


@app.route("/finish", methods=["POST"])
def finish():
    """
    Transpile, run, save .card and .py files.

    JSON body (preferred): {\"lines\": [\"10H 7H\", \"3S JH\", ...]}
    Each string is card shorthand one row (same rules as generated .card files).

    Omit body (legacy): same as before — append one snapshot from camera, then run.
    """
    payload = request.get_json(force=False, silent=True) or {}
    lines_payload = payload.get("lines")

    if lines_payload is not None:
        if not isinstance(lines_payload, list) or any(
            not isinstance(s, str) for s in lines_payload
        ):
            return (
                jsonify(
                    {"ok": False, "error": 'Request body must include "lines": [string, ...].'}
                ),
                400,
            )

        row_strings = [str(s).strip() for s in lines_payload if str(s).strip()]
        parsed_rows: list[list[str]] = []
        for idx, raw in enumerate(row_strings, start=1):
            toks = tokens_from_line_text(raw)
            if not toks:
                return jsonify(
                    {
                        "ok": False,
                        "error": f'Line {idx} decoded to zero cards.\nEnsure shorthand like "10H 7H 3S".',
                    }
                )
            parsed_rows.append(toks)

        with program_lock:
            program_lines.clear()
            program_lines.extend(parsed_rows)
            all_lines = list(program_lines)
    else:
        frame = get_latest_frame()

        try:
            tokens, _ = detect_tokens_in_frame(frame)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        with program_lock:
            if tokens:
                program_lines.append(tokens)
            all_lines = list(program_lines)

    if not all_lines:
        return jsonify({"ok": False, "error": "No program to run."})

    flat_tokens = [t for line in all_lines for t in line]

    # Transpile
    try:
        python_source = tokens_to_python(flat_tokens)
    except TranspileError as e:
        return jsonify({"ok": False, "error": f"Transpile error: {e}"})

    # Run
    output, error = run_program(python_source)

    # Save .card file — rank+suit shorthand (e.g. 10H 3S), re-runnable by the transpiler
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    card_filename = f"program_{timestamp}.card"
    card_filepath = os.path.join(PROGRAMS_DIR, card_filename)

    with open(card_filepath, "w") as f:
        f.write(build_card_file(all_lines, python_source, output, error))

    # Save generated Python alongside it
    py_filename = f"program_{timestamp}.py"
    py_filepath = os.path.join(PROGRAMS_DIR, py_filename)
    with open(py_filepath, "w") as f:
        f.write(python_source)

    return jsonify({
        "ok":          True,
        "python_source": python_source,
        "output":      output,
        "error":       error,
        "card_file":   card_filename,
        "py_file":     py_filename,
        "token_count": len(flat_tokens),
        "line_count":  len(all_lines)
    })


@app.route("/reset", methods=["POST"])
def reset():
    with program_lock:
        program_lines.clear()
    return jsonify({"ok": True})


@app.route("/program/manual_line", methods=["POST"])
def add_manual_program_line():
    """
    Append an empty program row so the UI and server agree on ordering:
    typed lines occupy a slot before the next camera capture adds another row.
    """
    with program_lock:
        program_lines.append([])
        n = len(program_lines)
    return jsonify({"ok": True, "line_number": n, "total_lines": n})


@app.route("/program/line/<int:line_idx>", methods=["DELETE"])
def delete_program_line(line_idx: int):
    """Remove one row from server state so it stays aligned when lines are deleted in the UI."""
    with program_lock:
        if line_idx < 0 or line_idx >= len(program_lines):
            return jsonify({"ok": False, "error": "Invalid line index."}), 400
        program_lines.pop(line_idx)
        remaining = len(program_lines)
    return jsonify({"ok": True, "total_lines": remaining})


@app.route("/program", methods=["GET"])
def get_program():
    with program_lock:
        all_lines = list(program_lines)

    flat_tokens = [t for line in all_lines for t in line]

    token_lines = []
    for i, line in enumerate(all_lines):
        token_lines.append({
            "line": i + 1,
            "tokens": [
                {"token": t, "card": token_to_card_name(t), "desc": describe_token(t)}
                for t in line
            ],
            "shorthand": " ".join(token_to_shorthand(t) for t in line),
        })

    try:
        python_preview = tokens_to_python(flat_tokens)
    except Exception as e:
        python_preview = f"# Error: {e}"

    return jsonify({
        "lines": token_lines,
        "python_preview": python_preview,
        "total_tokens": len(flat_tokens)
    })


@app.route("/programs/<filename>")
def download_program(filename):
    return send_from_directory(PROGRAMS_DIR, filename)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_card_file(all_lines, python_source, output, error):
    """
    Build the .card file content.

    Format:
        - Program lines are plain whitespace-separated card shorthand
          (rank + suit letter, e.g. 10H, QH, 3S), one line per row of cards.
          The transpiler reads these lines and maps them back to logic tokens.
        - Comments (lines starting with #) are ignored by the transpiler.
        - Generated Python and output are appended as comments so the file
          is both human-readable and fully re-runnable.

    Example:
        10H 7H 3S JH
        10H 10H 3S 5H AS JH
        ...
    """
    lines = []

    # Header comment
    lines.append(f"# CardCode Program")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append(f"# Lines: {len(all_lines)}  Tokens: {sum(len(l) for l in all_lines)}")
    lines.append("")

    # ── Card shorthand — the actual .card program ────────────────────────────
    for line in all_lines:
        lines.append(" ".join(token_to_shorthand(t) for t in line))

    # ── Generated Python (as comments) ──────────────────────────────────────
    lines.append("")
    lines.append("# ── Generated Python ───────────────────────────────────")
    for py_line in python_source.splitlines():
        lines.append(f"# {py_line}")

    # ── Output (as comments) ─────────────────────────────────────────────────
    lines.append("")
    lines.append("# ── Output ─────────────────────────────────────────────")
    if error:
        lines.append(f"# ERROR: {error}")
    else:
        for out_line in (output or "(no output)").splitlines():
            lines.append(f"# {out_line}")

    return "\n".join(lines)


# ── Start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load templates if available
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    if os.path.exists(template_dir):
        load_templates(template_dir)
        print("Card templates loaded.")
    else:
        print("No templates directory found — using color-only suit detection.")

    # Start camera thread
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    print("Camera thread started.")

    print("CardCode server running at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)