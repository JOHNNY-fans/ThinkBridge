"""Prompt construction shared by Stage 0 collection and model execution."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


THINK_OPEN_TEXT = "<think>\n"
THINK_BOUNDARY_TEXT = "\n</think>\n\n"


def build_thinking_prompt(
    tokenizer,
    question: str | None = None,
    *,
    task_type: str = "math",
    messages: Sequence[Mapping[str, Any]] | None = None,
    preserve_history_thinking: bool = False,
) -> str:
    """Exact R/F/D question prefix shared by compilation and deployment."""
    return (
        build_prompt(
            tokenizer,
            question,
            task_type=task_type,
            think=True,
            messages=messages,
            preserve_history_thinking=preserve_history_thinking,
        )
        + THINK_OPEN_TEXT
    )


@dataclass(frozen=True)
class PromptContract:
    """Minimal immutable state that determines Stage 0 prompt rendering."""

    task_type: str
    max_position_embeddings: int

    def __post_init__(self) -> None:
        if self.task_type not in {"math", "native_qa"}:
            raise ValueError("task_type must be 'math' or 'native_qa'")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")


DEFAULT_THINK_INSTRUCTION = (
    "Please reason step by step inside <think>...</think>, "
    "then on a new line after </think>, output ONLY the final answer with no extra words."
)
DEFAULT_NOTHINK_INSTRUCTION = "Output ONLY the final answer with no extra words."


MATH_THINK_INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}."
)

MATH_NOTHINK_INSTRUCTION = "Output the final answer within \\boxed{}."


def instruction_for_task(task_type: str | None, think: bool = True) -> str:
    """Return the output instruction for one task and thinking mode."""
    if str(task_type or "").lower() == "math":
        return MATH_THINK_INSTRUCTION if think else MATH_NOTHINK_INSTRUCTION
    return DEFAULT_THINK_INSTRUCTION if think else DEFAULT_NOTHINK_INSTRUCTION


def get_think_instruction() -> str:
    """Prompt-level output contract for Qwen-style thinking models.

    Keep this as a prompt instruction rather than a forced bridge suffix so
    direct/oracle/pred/zero modes all face the same task protocol. Set
    TB_THINK_INSTRUCTION=0/false/no to disable, or provide custom text via the
    environment variable.
    """
    raw = os.environ.get("TB_THINK_INSTRUCTION", DEFAULT_THINK_INSTRUCTION)
    if str(raw).lower() in {"0", "false", "no", "none", ""}:
        return ""
    return str(raw).strip()


def build_user_prompt(
    question: str,
    think_instruction: str | None = None,
    task_type: str | None = None,
    think: bool = True,
) -> str:
    """Build the user content for a question and output contract."""
    q = str(question or "")
    if task_type is not None and str(task_type).lower() == "math":
        return f"{q} {instruction_for_task('math', think)}"
    if think_instruction is not None:
        instr = str(think_instruction or "").strip()
    elif task_type is not None:
        instr = instruction_for_task(task_type, think)
    else:
        instr = get_think_instruction()
    body = f"Question:\n{q}"
    return f"{instr}\n\n{body}" if instr else body


def build_chat_messages(
    question: str | None = None,
    *,
    messages: Sequence[Mapping[str, Any]] | None = None,
    think_instruction: str | None = None,
    task_type: str | None = None,
    think: bool = True,
) -> list[dict[str, str]]:
    """Validate and augment a single-turn or multi-turn chat history."""
    if not messages:
        q = str(question or "")
        if not q.strip():
            raise ValueError("prompt requires a question or messages")
        return [
            {
                "role": "user",
                "content": build_user_prompt(
                    q,
                    think_instruction=think_instruction,
                    task_type=task_type,
                    think=think,
                ),
            }
        ]

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        role = str(message.get("role", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}].role is unsupported: {role!r}")
        content = message.get("content")
        if content is None:
            raise ValueError(f"messages[{index}] requires content")
        normalized.append({"role": role, "content": str(content)})

    if normalized[-1]["role"] != "user":
        raise ValueError("multi-turn messages must end at the current user turn")
    if not normalized[-1]["content"].strip():
        raise ValueError("the final user message must not be empty")
    history_start = 1 if normalized[0]["role"] == "system" else 0
    if any(message["role"] == "system" for message in normalized[history_start:]):
        raise ValueError("a system message is allowed only at the start")
    dialogue = normalized[history_start:]
    if not dialogue or any(
        message["role"] != ("user" if index % 2 == 0 else "assistant")
        for index, message in enumerate(dialogue)
    ):
        raise ValueError("multi-turn messages must alternate user and assistant")

    if think_instruction is not None:
        instruction = str(think_instruction or "").strip()
    elif task_type is not None:
        instruction = instruction_for_task(task_type, think)
    else:
        instruction = get_think_instruction()
    if instruction:
        separator = " " if str(task_type or "").lower() == "math" else "\n\n"
        normalized[-1]["content"] = (
            f"{normalized[-1]['content'].rstrip()}{separator}{instruction}"
        )
    return normalized


def _apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str | None:
    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None


def _render_qwen_chatml_preserving_thinking(
    messages: Sequence[Mapping[str, str]],
) -> str:
    """Render Qwen ChatML without its history-only ``</think>`` stripping.

    Qwen3's official no-tools template rewrites historical assistant messages
    to the text after ``</think>``. Multi-turn raw-history evaluation needs
    the same ChatML envelopes while retaining the full assistant content
    verbatim, including D-produced CoT.
    """
    rendered = [
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    ]
    rendered.append("<|im_start|>assistant\n")
    return "".join(rendered)


def _plain_chat_fallback(messages: Sequence[Mapping[str, str]]) -> str:
    parts = [f"{str(m['role']).capitalize()}: {m['content']}" for m in messages]
    parts.append("Assistant: ")
    return "\n".join(parts)


def build_prompt(
    tokenizer,
    question: str | None = None,
    use_chat_template: bool = True,
    think_instruction: str | None = None,
    task_type: str | None = None,
    think: bool = True,
    *,
    messages: Sequence[Mapping[str, Any]] | None = None,
    preserve_history_thinking: bool = False,
) -> str:
    """Render a generation-ready prompt without changing message semantics."""
    chat_messages = build_chat_messages(
        question,
        messages=messages,
        think_instruction=think_instruction,
        task_type=task_type,
        think=think,
    )
    if use_chat_template and preserve_history_thinking:
        rendered = _render_qwen_chatml_preserving_thinking(chat_messages)
    else:
        rendered = (
            _apply_chat_template(tokenizer, chat_messages)
            if use_chat_template
            else None
        )
    return rendered if rendered is not None else _plain_chat_fallback(chat_messages)
