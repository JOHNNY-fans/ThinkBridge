"""Readable previews of existing validation generations (no model execution)."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from think_bridge.training.progress import write_progress


def answer_examples(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    *,
    count: int = 2,
) -> list[dict[str, Any]]:
    sources = {str(row["record_id"]): row for row in rows}
    examples = []
    # Independent of correctness/score; stable across validation steps and ranks.
    for result in sorted(paired, key=lambda row: str(row["record_id"]))[:count]:
        source = sources[str(result["record_id"])]
        ids = result["generated_token_ids"]["true_z"]
        examples.append(
            dict(
                record_id=str(result["record_id"]),
                population=result["population"],
                prompt_text=tokenizer.decode(
                    source["prompt_ids"], skip_special_tokens=True
                ),
                reference_answer=source["reference_answer"],
                generated_answer_text=tokenizer.decode(ids, skip_special_tokens=True),
                correct=bool(result["correct"]["true_z"]),
                generated_token_length=len(ids),
                terminated_with_eos=bool(ids and ids[-1] == tokenizer.eos_token_id),
            )
        )
    return examples


def _text(label: str, value: Any) -> None:
    text = str(value).replace("\r", "\\r")
    if len(text) > 2400:
        text = text[:1800] + "\n…[终端预览截短；完整内容见报告]…\n" + text[-600:]
    write_progress(label + ":")
    for line in text.splitlines() or ["（空）"]:
        write_progress("  " + line)


def print_evaluation_examples(report: Mapping[str, Any], *, report_path: Path) -> None:
    """Call once on rank zero after report publication; compatible with tqdm."""
    answers = report.get("qualitative_answer_examples", [])
    step = report.get("step")
    for sample in answers:
        write_progress(
            f"[Route1 示例 | step={step} | id={sample['record_id']} | "
            f"人口={sample['population']} | 正确={sample['correct']} | "
            f"tokens={sample['generated_token_length']} | EOS={sample['terminated_with_eos']}]"
        )
        _text("题目", sample["prompt_text"])
        _text("参考答案", sample["reference_answer"])
        _text("true-z 回答", sample["generated_answer_text"])
    if answers:
        write_progress(f"完整验证报告: {report_path}")
