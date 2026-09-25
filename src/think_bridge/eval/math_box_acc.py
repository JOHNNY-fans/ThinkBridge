"""Boxed-answer extraction and mathematical equivalence judging."""

from __future__ import annotations

import re


_BOXED_RE = re.compile(r"\\boxed\s*\{")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_boxed_final(text: str) -> str | None:
    """Extract the last balanced ``\\boxed{...}`` expression."""
    if not text:
        return None
    last = None
    for m in _BOXED_RE.finditer(text):
        i = m.end()
        depth, j = 1, m.end()
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[i : j - 1]
    return last.strip() if last is not None else None


def split_after_think(text: str) -> str:
    """Return the visible suffix after the final thinking boundary."""
    if not text:
        return ""
    if "</think>" in text:
        _, _, tail = text.rpartition("</think>")
        return tail
    return text


def last_nonempty_line(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def extract_pred_str(raw: str) -> str:
    """Extract a boxed answer or the final non-empty visible line."""
    if not raw:
        return ""
    tail = split_after_think(raw)
    box = extract_boxed_final(tail)
    if box is not None:
        return box
    line = last_nonempty_line(tail)
    if line:
        return line
    box_full = extract_boxed_final(raw)
    if box_full is not None:
        return box_full
    return last_nonempty_line(raw)


def _normalize_numeric(s: str) -> str:
    """Normalize basic numeric text for the dependency-free fallback."""
    if s is None:
        return ""
    t = str(s).strip().strip("$").replace(",", "").replace(" ", "")
    t = t.rstrip(".。,;:")
    return t.lower()


def _fallback_match(gold: str, pred: str) -> bool:
    """Compare final numeric values, then fall back to normalized equality."""
    g, p = _normalize_numeric(gold), _normalize_numeric(pred)
    if not g or not p:
        return False
    try:
        gm = _NUM_RE.findall(g)
        pm = _NUM_RE.findall(p)
        if gm and pm:
            return abs(float(gm[-1]) - float(pm[-1])) < 1e-6
    except Exception:  # noqa: BLE001
        pass
    return g == p


def _verify_with_math_verify(
    gold: str, pred: str, *, diagnostics: dict | None = None
) -> bool | None:
    """Use math-verify when installed and report absence with ``None``."""
    try:
        from math_verify import parse as mv_parse, verify as mv_verify
    except Exception:  # noqa: BLE001
        return None
    if diagnostics is not None:
        from think_bridge.eval.judge_diagnostics import verify_with_diagnostics

        return verify_with_diagnostics(
            gold, pred, parse=mv_parse, verify=mv_verify, diagnostics=diagnostics
        )
    try:
        gp = mv_parse(f"\\boxed{{{gold}}}")
        pp = mv_parse(f"\\boxed{{{pred}}}")
        return bool(mv_verify(gp, pp))
    except Exception:  # noqa: BLE001
        return False


def judge_math(gold: str, pred: str, *, diagnostics: dict | None = None) -> bool:
    """Judge mathematical equivalence using the formal or fallback path."""
    if gold is None or pred is None:
        return False
    g, p = str(gold).strip(), str(pred).strip()
    if not g or not p:
        return False
    v = _verify_with_math_verify(g, p, diagnostics=diagnostics)
    if v is not None:
        return v
    return _fallback_match(g, p)


def math_box_match(
    prediction: str, reference: str, *, diagnostics: dict | None = None
) -> bool:
    """Extract a model answer and compare it with the math reference."""
    pred = extract_pred_str(prediction)
    return judge_math(reference, pred, diagnostics=diagnostics)
