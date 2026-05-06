# CardCode

CardCode is a card-based programming language with a webcam UI.
It detects playing cards as instructions, lets you edit each detected line,
transpiles the result to Python, runs it, and saves both `.card` and `.py` outputs.

## Features

- Live webcam stream with card overlays.
- Capture cards one line at a time.
- Edit each line before execution using compact shorthand (`10H 7H 3S JH`).
- Add manual typed lines without camera capture.
- Run from UI (`End & Run`) or from CLI (`python run.py file.card`).
- Save generated artifacts in `programs/`.

## Quick Start

```bash
cd cardcode
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:5001`

## How the UI Workflow Works

1. Press `Capture Line` to detect a row of cards.
2. Review and edit shorthand in the Program panel (left side).
3. Optional: press `+ Typed line` to insert a manual line slot.
4. Repeat until the program is complete.
5. Press `End & Run`.

`End & Run` uses the current edited lines from the UI (not raw camera history),
transpiles to Python, executes, and shows output/error in the Output tab.

## Card Language Mapping

### Hearts (numbers)

- `AH` -> `PUSH_1`
- `2H` .. `10H` -> `PUSH_2` .. `PUSH_10`
- `QH` -> `PUSH_0`

### Spades (arithmetic / stack)

- `AS` -> `ADD`
- `2S` -> `SUB`
- `3S` -> `MUL`
- `4S` -> `MOD`
- `5S` -> `EQ`
- `6S` -> `AND`
- `7S` -> `NOT`
- `8S` -> `DUP`
- `9S` -> `SWAP`
- `10S` -> `POP`

### Diamonds (control flow)

- `AD` -> `FOR`
- `2D` -> `LOAD_CTR`
- `3D` -> `IF`
- `4D` -> `ELSE`
- `5D` -> `END`

### Clubs (registers)

- `AC` -> `STORE_A`
- `2C` -> `STORE_B`
- `3C` -> `LOAD_A`
- `4C` -> `LOAD_B`
- `5C` -> `STORE_STR_A`
- `6C` -> `STORE_STR_B`
- `7C` -> `LOAD_STR_A`
- `8C` -> `LOAD_STR_B`

### Face card operations

- `JH` -> `CHAR`
- `JS` -> `CONCAT`
- `JD` -> `NUM_TO_STR`
- `JC` -> `PRINT_STR`
- `QC` -> `PRINT_NUM`

## `.card` Format

Program lines are saved as shorthand card notation, e.g.:

```text
10H 7H 3S JH
10H 10H 3S 5H AS JH
```

Saved `.card` files also append generated Python and output as `#` comments.
The parser ignores comments and supports shorthand and legacy raw token lines.

## Run a `.card` File from CLI

```bash
python run.py programs/program_YYYYMMDD_HHMMSS.card
```

## Project Structure

```text
cardcode/
├── app.py            # Flask server + API routes + save/run flow
├── detector.py       # OpenCV / YOLO model detection pipeline
├── classifier.py     # Card <-> token mapping and labels
├── transpiler.py     # Tokens -> Python + .card parsing helpers
├── run.py            # CLI runner for .card files
├── static/
│   └── index.html    # Single-page web UI
├── programs/         # Saved .card and generated .py outputs
└── templates/        # Optional template images for detection assist
```

## Optional Templates

If you provide card corner templates in `templates/`, detection can improve in
challenging lighting. If not present, the app still runs with model-based detection.
