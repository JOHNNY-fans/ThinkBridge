"""Shared answer extraction and exact-match helpers."""

from __future__ import annotations

import re
import string


_STRIP_PUNCT = "".join(c for c in string.punctuation if c not in ".-/")
_WS_RE = re.compile(r"\s+")
_INTRO_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)?\s*",
    flags=re.IGNORECASE,
)
_LEAD_RE = re.compile(r"^[\s=:]+")


def _strip_final_answer_cue(text: str) -> str:
    """Remove a leading natural-language final-answer cue."""
    if not text:
        return ""
    return re.sub(
        r"^(?:final\s+answer|answer)\s*(?:is|:)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )


def _extract_last_boxed(text: str) -> str | None:
    """Extract the final balanced boxed or fbox expression."""
    if not text:
        return None
    hits: list[str] = []
    i = 0
    while i < len(text):
        cmd = None
        if text.startswith("\\boxed", i):
            cmd = "\\boxed"
        elif text.startswith("\\fbox", i):
            cmd = "\\fbox"
        if cmd is None:
            i += 1
            continue
        j = i + len(cmd)
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text):
            break
        if text[j] != "{":
            m = re.match(r"([^\s,.;]+)", text[j:])
            if m:
                hits.append(m.group(1).strip())
                i = j + len(m.group(1))
                continue
            i = j + 1
            continue
        depth = 0
        start = j + 1
        k = j
        while k < len(text):
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    hits.append(text[start:k].strip())
                    i = k + 1
                    break
            k += 1
        else:
            break
    return hits[-1] if hits else None


def normalize_answer(s: str) -> str:
    """Normalize an answer for strict textual comparison."""
    if s is None:
        return ""
    s = _strip_final_answer_cue(s).strip().lower()
    s = _LEAD_RE.sub("", s)
    s = _INTRO_RE.sub("", s)
    s = s.replace(",", "")
    s = s.translate(str.maketrans("", "", _STRIP_PUNCT))
    s = _WS_RE.sub(" ", s).strip()
    return s


def extract_prediction(decoded: str, close_str: str = "</think>") -> str:
    """Extract the first visible answer after the thinking boundary."""
    if not decoded:
        return ""
    if close_str in decoded:
        _, _, tail = decoded.partition(close_str)
    else:
        tail = decoded
    tail = _LEAD_RE.sub("", tail)
    tail = _strip_final_answer_cue(tail)
    lines = tail.splitlines()
    if lines:
        tail = lines[0]
    tail = tail.strip()
    boxed = _extract_last_boxed(tail)
    if boxed is not None:
        tail = boxed.strip()
    tail = tail.rstrip(".,;")
    return tail


def answer_match(prediction: str, reference: str, close_str: str = "</think>") -> bool:
    """Compare a visible prediction with a normalized textual reference."""
    pred = normalize_answer(extract_prediction(prediction, close_str=close_str))
    ref = normalize_answer(reference)
    if not ref:
        return False
    return pred == ref


def judge_answer(
    prediction: str,
    reference: str,
    task_type: str | None,
    *,
    diagnostics: dict | None = None,
) -> bool:
    """Dispatch to the shared math or textual answer judge."""

    from think_bridge.eval.math_box_acc import math_box_match

    tt = (task_type or "native_qa").lower()
    return (
        math_box_match(prediction, reference, diagnostics=diagnostics)
        if tt == "math"
        else answer_match(prediction, reference)
    )
