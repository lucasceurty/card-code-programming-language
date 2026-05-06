"""
CardCode Card Classifier
Maps (rank, suit) tuples from CV detection to CardCode tokens.
"""

# Full card → token mapping
CARD_MAP = {
    # ── Hearts: Integer Literals ────────────────────────────────────────────
    ("A",  "H"): "PUSH_1",
    ("2",  "H"): "PUSH_2",
    ("3",  "H"): "PUSH_3",
    ("4",  "H"): "PUSH_4",
    ("5",  "H"): "PUSH_5",
    ("6",  "H"): "PUSH_6",
    ("7",  "H"): "PUSH_7",
    ("8",  "H"): "PUSH_8",
    ("9",  "H"): "PUSH_9",
    ("10", "H"): "PUSH_10",
    ("Q",  "H"): "PUSH_0",   # Queen of Hearts = zero

    # ── Spades: Arithmetic + Stack ──────────────────────────────────────────
    ("A",  "S"): "ADD",
    ("2",  "S"): "SUB",
    ("3",  "S"): "MUL",
    ("4",  "S"): "MOD",
    ("5",  "S"): "EQ",
    ("6",  "S"): "AND",
    ("7",  "S"): "NOT",
    ("8",  "S"): "DUP",
    ("9",  "S"): "SWAP",
    ("10", "S"): "POP",

    # ── Diamonds: Control Flow ──────────────────────────────────────────────
    ("A",  "D"): "FOR",
    ("2",  "D"): "LOAD_CTR",
    ("3",  "D"): "IF",
    ("4",  "D"): "ELSE",
    ("5",  "D"): "END",

    # ── Clubs: Registers ────────────────────────────────────────────────────
    ("A",  "C"): "STORE_A",
    ("2",  "C"): "STORE_B",
    ("3",  "C"): "LOAD_A",
    ("4",  "C"): "LOAD_B",
    ("5",  "C"): "STORE_STR_A",
    ("6",  "C"): "STORE_STR_B",
    ("7",  "C"): "LOAD_STR_A",
    ("8",  "C"): "LOAD_STR_B",

    # ── Face Cards: String Ops + Output ─────────────────────────────────────
    ("J",  "H"): "CHAR",
    ("J",  "S"): "CONCAT",
    ("J",  "D"): "NUM_TO_STR",
    ("J",  "C"): "PRINT_STR",
    ("Q",  "C"): "PRINT_NUM",
}

# Human-readable labels for the UI
CARD_LABELS = {v: k for k, v in CARD_MAP.items()}

TOKEN_DESCRIPTIONS = {
    "PUSH_0":      "Push 0",
    "PUSH_1":      "Push 1",
    "PUSH_2":      "Push 2",
    "PUSH_3":      "Push 3",
    "PUSH_4":      "Push 4",
    "PUSH_5":      "Push 5",
    "PUSH_6":      "Push 6",
    "PUSH_7":      "Push 7",
    "PUSH_8":      "Push 8",
    "PUSH_9":      "Push 9",
    "PUSH_10":     "Push 10",
    "ADD":         "Add (+)",
    "SUB":         "Subtract (−)",
    "MUL":         "Multiply (×)",
    "MOD":         "Modulo (%)",
    "EQ":          "Equal (==)",
    "AND":         "Logical AND",
    "NOT":         "Logical NOT",
    "DUP":         "Duplicate",
    "SWAP":        "Swap",
    "POP":         "Pop",
    "STORE_A":     "Store → Reg A",
    "STORE_B":     "Store → Reg B",
    "LOAD_A":      "Load Reg A",
    "LOAD_B":      "Load Reg B",
    "STORE_STR_A": "Store → Str A",
    "STORE_STR_B": "Store → Str B",
    "LOAD_STR_A":  "Load Str A",
    "LOAD_STR_B":  "Load Str B",
    "CHAR":        "Int → Char",
    "CONCAT":      "Concat strings",
    "NUM_TO_STR":  "Int → String",
    "PRINT_STR":   "Print string",
    "PRINT_NUM":   "Print number",
    "FOR":         "For loop",
    "LOAD_CTR":    "Load counter",
    "IF":          "If condition",
    "ELSE":        "Else branch",
    "END":         "End block",
}

SUIT_NAMES = {"H": "Hearts", "S": "Spades", "D": "Diamonds", "C": "Clubs"}
SUIT_SYMBOLS = {"H": "♥", "S": "♠", "D": "♦", "C": "♣"}
RANK_NAMES = {
    "A": "Ace", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
    "J": "Jack", "Q": "Queen", "K": "King"
}


def card_to_token(rank: str, suit: str) -> str | None:
    """Convert a (rank, suit) pair to a CardCode token. Returns None if unmapped."""
    return CARD_MAP.get((rank.upper(), suit.upper()[0]))


def token_to_shorthand(token: str) -> str:
    """Compact notation for .card files, e.g. PUSH_10 -> 10H, MUL -> 3S."""
    key = CARD_LABELS.get(token)
    if not key:
        return token
    rank, suit = key
    return f"{rank}{suit}"


def token_to_card_name(token: str) -> str:
    """Get human-readable card name for a token."""
    key = CARD_LABELS.get(token)
    if not key:
        return token
    rank, suit = key
    return f"{RANK_NAMES.get(rank, rank)} of {SUIT_NAMES.get(suit, suit)} {SUIT_SYMBOLS.get(suit, '')}"


def describe_token(token: str) -> str:
    return TOKEN_DESCRIPTIONS.get(token, token)
