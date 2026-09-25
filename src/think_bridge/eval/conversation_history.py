"""Dependency-free MathChat loading and selection of original generated history."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from think_bridge.eval.math_box_acc import extract_boxed_final


HISTORY_SOURCES = ("response_only",)
DEFAULT_HISTORY_SOURCES = HISTORY_SOURCES

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class ReplyParts:
    """Generated fields, with a rewrapped visible boxed answer or ``None``.

    ``issues`` is serializable diagnostic metadata: ``missing_final_answer``
    means the final balanced visible box is absent or empty; ``unclosed_think``
    means generation ended in thinking mode. Neither permits an answer fallback.
    """

    reasoning: str
    visible_response: str
    final_answer: str | None
    issues: tuple[str, ...] = ()


def make_reply_parts(reasoning: str, visible_response: str) -> ReplyParts:
    """Use explicitly separated, unwrapped generated fields without trimming.

    The caller owns the separation of these fields. Only ``visible_response``
    is searched for an answer; neither reasoning nor reference data is used.
    """
    if not isinstance(reasoning, str) or not isinstance(visible_response, str):
        raise TypeError("reasoning and visible_response must be strings")
    boxed = extract_boxed_final(visible_response) or None
    return ReplyParts(
        reasoning=reasoning,
        visible_response=visible_response,
        final_answer=None if boxed is None else "\\boxed{" + boxed + "}",
        issues=("missing_final_answer",) if boxed is None else (),
    )


def split_reply(text: str, *, starts_in_thinking: bool = False) -> ReplyParts:
    """Split one Qwen-style reply, preserving text inside/after the boundary.

    With no tags, text is visible unless ``starts_in_thinking=True`` declares
    that the prompt already supplied ``<think>``. That flag also admits a
    closing tag without an opening tag in the generated continuation. A
    complete explicit opening tag is accepted with either flag value.

    Unclosed thinking retains all following text as reasoning, leaves the
    visible response empty, and records ``unclosed_think``. A stray closing
    tag, repeated/nested tags, or non-whitespace before an explicit opening
    tag raises ``ValueError`` rather than guessing field boundaries. Leading
    whitespace outside an explicit opening tag is framing and is discarded;
    text inside the tags and the entire visible suffix remain unchanged.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(starts_in_thinking, bool):
        raise TypeError("starts_in_thinking must be a boolean")
    opens, closes = text.count(_THINK_OPEN), text.count(_THINK_CLOSE)
    if opens > 1 or closes > 1:
        raise ValueError("repeated or nested think tags are not supported")

    if opens:
        prefix, _, body = text.partition(_THINK_OPEN)
        if prefix.strip():
            raise ValueError("unexpected text before <think>")
    elif starts_in_thinking:
        body = text
    elif closes:
        raise ValueError(
            "unmatched </think>; set starts_in_thinking for a prompt opener"
        )
    else:
        return make_reply_parts("", text)

    if closes:
        reasoning, _, visible = body.partition(_THINK_CLOSE)
        return make_reply_parts(reasoning, visible)
    parts = make_reply_parts(body, "")
    return ReplyParts(
        parts.reasoning,
        parts.visible_response,
        parts.final_answer,
        issues=("unclosed_think", *parts.issues),
    )


def history_content(parts: ReplyParts, source: str) -> str:
    """Retain the complete generated response verbatim."""
    if source != "response_only":
        raise ValueError("history source must be response_only")
    return parts.visible_response


def _read_rows(path: Path) -> list:
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            rows = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL in {path} at line {line_number}: {exc.msg}"
                    ) from exc
        else:
            try:
                rows = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} at line {exc.lineno}: {exc.msg}"
                ) from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"conversation dataset {path} must contain a nonempty list of rows"
        )
    return rows


def _nonempty_text(value: object, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be a nonempty string")
    return value


def load_conversations(
    path: str | Path, maximum: int | None = None
) -> list[list[dict]]:
    """Validate and load flat JSON/JSONL records into complete three-turn groups.

    Groups follow the first appearance of each conversation; turns are ordered
    by ``turn_index`` (exactly 0, 1, 2). All rows and groups are validated before
    applying ``maximum`` as a nonnegative conversation count; zero selects no
    conversations. Truncation never masks invalid records later in the file.

    Required text fields are ``id``, ``conversation_id``, ``question``, and
    ``reference_answer`` (or raw MathChat's ``answer``). Both answer keys may
    coexist only if they agree exactly. Text is preserved without trimming.
    ``task_type`` is preserved, defaults from ``type`` or to ``math``, and
    ``row_index`` is preserved if supplied, otherwise it is the zero-based
    source record index (excluding blank JSONL lines). IDs are globally unique;
    turn indices are unique within each conversation. ``is_final_turn`` must
    be a boolean and true only on turn 2. No history or model input is built.
    """
    if maximum is not None and (type(maximum) is not int or maximum < 0):
        raise ValueError(
            "maximum must be a nonnegative integer conversation count or None"
        )
    source = Path(path)
    groups: dict[str, list[dict]] = {}
    ids: set[str] = set()
    turns: set[tuple[str, int]] = set()

    for index, row in enumerate(_read_rows(source)):
        location = f"{source} row {index}"
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected a JSON object")
        record_id = _nonempty_text(row.get("id"), "id", location)
        conversation_id = _nonempty_text(
            row.get("conversation_id"), "conversation_id", location
        )
        question = _nonempty_text(row.get("question"), "question", location)
        reference = _nonempty_text(
            row.get("reference_answer")
            if "reference_answer" in row
            else row.get("answer"),
            "reference_answer/answer",
            location,
        )
        if "reference_answer" in row and "answer" in row and row["answer"] != reference:
            raise ValueError(f"{location}: conflicting answer and reference_answer")
        task_type = _nonempty_text(
            row.get("task_type", row.get("type", "math")), "task_type", location
        )
        turn_index = row.get("turn_index")
        if type(turn_index) is not int or turn_index < 0:
            raise ValueError(f"{location}: turn_index must be a nonnegative integer")
        final = row.get("is_final_turn")
        if type(final) is not bool:
            raise ValueError(f"{location}: is_final_turn must be a boolean")
        row_index = row.get("row_index", index)
        if type(row_index) is not int or row_index < 0:
            raise ValueError(f"{location}: row_index must be a nonnegative integer")
        if record_id in ids:
            raise ValueError(f"{location}: duplicate id {record_id!r}")
        turn_key = (conversation_id, turn_index)
        if turn_key in turns:
            raise ValueError(f"{location}: duplicate conversation/turn {turn_key!r}")
        ids.add(record_id)
        turns.add(turn_key)
        groups.setdefault(conversation_id, []).append(
            {
                "id": record_id,
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "is_final_turn": final,
                "question": question,
                "reference_answer": reference,
                "task_type": task_type,
                "row_index": row_index,
            }
        )

    for conversation_id, rows in groups.items():
        rows.sort(key=lambda row: row["turn_index"])
        if [row["turn_index"] for row in rows] != [0, 1, 2]:
            raise ValueError(
                f"{source} conversation {conversation_id!r}: expected exactly three turns 0, 1, 2"
            )
        if [row["is_final_turn"] for row in rows] != [False, False, True]:
            raise ValueError(
                f"{source} conversation {conversation_id!r}: is_final_turn must be true only on turn 2"
            )

    conversations = list(groups.values())
    return conversations if maximum is None else conversations[:maximum]


__all__ = [
    "DEFAULT_HISTORY_SOURCES",
    "HISTORY_SOURCES",
    "ReplyParts",
    "split_reply",
    "make_reply_parts",
    "history_content",
    "load_conversations",
]


def tokenizer_identity(tokenizer):
    payload = {
        "vocab": tokenizer.get_vocab(),
        "chat_template": tokenizer.chat_template,
        "special_tokens": tokenizer.special_tokens_map,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
