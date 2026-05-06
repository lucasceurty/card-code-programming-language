"""
CardCode Transpiler
Converts a flat list of CardCode tokens into valid Python source code.
Indentation is managed entirely by the transpiler — card layout spacing is irrelevant.
"""

import classifier

class TranspileError(Exception):
    pass


class CardCodeTranspiler:
    def __init__(self):
        self.expr_stack = []
        self.output_lines = []
        self.indent = 0
        self.block_stack = []  # tracks FOR / IF / ELSE for proper END handling

    def emit(self, line):
        self.output_lines.append("    " * self.indent + line)

    def push(self, val):
        self.expr_stack.append(str(val))

    def pop(self):
        if not self.expr_stack:
            raise TranspileError("Stack underflow — not enough values on the stack.")
        return self.expr_stack.pop()

    def peek(self):
        if not self.expr_stack:
            raise TranspileError("Stack underflow — tried to peek at empty stack.")
        return self.expr_stack[-1]

    def transpile(self, tokens):
        """Take a flat list of token strings, return Python source code."""
        self.expr_stack = []
        self.output_lines = []
        self.indent = 0
        self.block_stack = []

        for token in tokens:
            self.handle(token)

        if self.block_stack:
            raise TranspileError(
                f"Unclosed blocks: {self.block_stack}. Missing END card(s)."
            )

        return "\n".join(self.output_lines)

    def handle(self, token):
        match token:

            # ── Integer Literals (Hearts A–10) ──────────────────────────────
            case "PUSH_1":  self.push(1)
            case "PUSH_2":  self.push(2)
            case "PUSH_3":  self.push(3)
            case "PUSH_4":  self.push(4)
            case "PUSH_5":  self.push(5)
            case "PUSH_6":  self.push(6)
            case "PUSH_7":  self.push(7)
            case "PUSH_8":  self.push(8)
            case "PUSH_9":  self.push(9)
            case "PUSH_10": self.push(10)
            case "PUSH_0":  self.push(0)   # Q♥

            # ── Arithmetic (Spades A–4) ──────────────────────────────────────
            case "ADD":
                b, a = self.pop(), self.pop()
                self.push(f"({a} + {b})")
            case "SUB":
                b, a = self.pop(), self.pop()
                self.push(f"({a} - {b})")
            case "MUL":
                b, a = self.pop(), self.pop()
                self.push(f"({a} * {b})")
            case "MOD":
                b, a = self.pop(), self.pop()
                self.push(f"({a} % {b})")

            # ── Logic (Spades 5–7) ───────────────────────────────────────────
            case "EQ":
                b, a = self.pop(), self.pop()
                self.push(f"({a} == {b})")
            case "AND":
                b, a = self.pop(), self.pop()
                self.push(f"({a} and {b})")
            case "NOT":
                a = self.pop()
                self.push(f"(not {a})")

            # ── Stack Ops (Spades 8–10) ──────────────────────────────────────
            case "DUP":
                self.push(self.peek())
            case "SWAP":
                a, b = self.pop(), self.pop()
                self.push(a)
                self.push(b)
            case "POP":
                self.pop()

            # ── Integer Registers (Clubs A–4) ────────────────────────────────
            case "STORE_A":
                expr = self.pop()
                self.emit(f"_reg_a = {expr}")
                self.push("_reg_a")
            case "STORE_B":
                expr = self.pop()
                self.emit(f"_reg_b = {expr}")
                self.push("_reg_b")
            case "LOAD_A":
                self.push("_reg_a")
            case "LOAD_B":
                self.push("_reg_b")

            # ── String Registers (Clubs 5–8) ─────────────────────────────────
            case "STORE_STR_A":
                expr = self.pop()
                self.emit(f"_str_a = {expr}")
            case "STORE_STR_B":
                expr = self.pop()
                self.emit(f"_str_b = {expr}")
            case "LOAD_STR_A":
                self.push("_str_a")
            case "LOAD_STR_B":
                self.push("_str_b")

            # ── String Ops (Jack cards) ──────────────────────────────────────
            case "CHAR":
                a = self.pop()
                self.push(f"chr({a})")
            case "CONCAT":
                b, a = self.pop(), self.pop()
                self.push(f"({a} + {b})")
            case "NUM_TO_STR":
                a = self.pop()
                self.push(f"str({a})")

            # ── Output ───────────────────────────────────────────────────────
            case "PRINT_STR":
                expr = self.pop()
                self.emit(f"print({expr})")
            case "PRINT_NUM":
                expr = self.pop()
                self.emit(f"print({expr})")

            # ── Control Flow (Diamonds) ──────────────────────────────────────
            case "FOR":
                limit = self.pop()
                self.emit(f"for _ctr in range(1, ({limit}) + 1):")
                self.indent += 1
                self.block_stack.append("FOR")

            case "LOAD_CTR":
                self.push("_ctr")

            case "IF":
                cond = self.pop()
                self.emit(f"if {cond}:")
                self.indent += 1
                self.block_stack.append("IF")

            case "ELSE":
                if not self.block_stack or self.block_stack[-1] not in ("IF",):
                    raise TranspileError("ELSE without matching IF.")
                self.indent -= 1
                self.emit("else:")
                self.indent += 1
                self.block_stack.pop()
                self.block_stack.append("ELSE")

            case "END":
                if not self.block_stack:
                    raise TranspileError("END without an open block.")
                self.indent -= 1
                self.block_stack.pop()

            case _:
                raise TranspileError(f"Unknown token: '{token}'")


def tokens_to_python(tokens: list[str]) -> str:
    """Public entry point. Returns Python source as a string."""
    t = CardCodeTranspiler()
    return t.transpile(tokens)


def run_program(python_source: str) -> tuple[str, str]:
    """
    Execute generated Python. Returns (stdout_output, error_message).
    stdout is captured; errors are caught and returned as strings.
    """
    import io
    import sys

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout

    try:
        sys.stdout = stdout_capture
        exec(python_source, {})
        return stdout_capture.getvalue(), ""
    except Exception as e:
        return "", str(e)
    finally:
        sys.stdout = old_stdout


def tokens_from_line_text(fragment: str) -> list[str]:
    """
    One program row: whitespace-separated shorthand (10H 7H 3S) or PUSH_* tokens,
    trailing # comments stripped. Matches .card parsing rules per line fragment.
    """
    tokens_out: list[str] = []
    clean_line = fragment.split("#")[0].strip()
    if not clean_line or clean_line.startswith("═"):
        return tokens_out

    parts = clean_line.replace("|", " ").split()
    for p in parts:
        if p.upper() == "LINE" or p.isdigit():
            continue

        suit = p[-1].upper()
        rank = p[:-1].upper()

        tok = classifier.card_to_token(rank, suit) if rank else None
        if tok:
            tokens_out.append(tok)
        elif p.upper() in classifier.CARD_LABELS:
            tokens_out.append(p.upper())
        # Else: silently skip invalid fragments (parity with load_card_file)
    return tokens_out


def load_card_file(path: str) -> list[str]:
    """
    Parses a .card file and converts shorthand card names
    into logic tokens using the classifier (all non-comment lines flattened).
    """
    tokens = []
    with open(path, "r") as f:
        for line in f:
            tokens.extend(tokens_from_line_text(line))
    return tokens


def run_card_file(path: str) -> tuple[str, str]:
    """
    Load a .card file, transpile it, and run it.
    Returns (output, error) same as run_program.
    Usage:
        output, error = run_card_file("programs/program_20250505_143022.card")
    """
    tokens = load_card_file(path)
    python_source = tokens_to_python(tokens)
    return run_program(python_source)