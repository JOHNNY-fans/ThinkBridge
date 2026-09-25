"""Single-turn, multi-turn and HF timing benchmarks."""

from __future__ import annotations
import argparse, logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from think_bridge.eval.answer_match import judge_answer
from importlib.metadata import version, PackageNotFoundError


def score_prediction_rows(
    rows: Iterable[Mapping[str, Any]], *, task_type: str = "math"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score complete predictions through the shared formal answer judge."""

    if not rows:
        raise ValueError("benchmark prediction rows are empty")
    try:
        math_verify_version = version("math-verify")
    except PackageNotFoundError:
        math_verify_version = None
    from think_bridge.eval.judge_diagnostics import JUDGE_PROTOCOL

    scored: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        prediction = str(raw.get("prediction") or "")
        reference = str(raw.get("reference_answer") or raw.get("answer") or "")
        if not reference.strip():
            raise ValueError(
                f"benchmark prediction row {index} lacks a reference answer"
            )
        diagnostics = dict(
            protocol=JUDGE_PROTOCOL,
            status="ok",
            attempts=0,
            timeout_events=0,
            math_verify_version=math_verify_version,
        )
        correct = bool(
            judge_answer(prediction, reference, task_type, diagnostics=diagnostics)
        )
        if diagnostics["status"] == "timeout":
            logging.getLogger(__name__).warning(
                "[math-judge] row_index=%s parsing/comparison timed out after 10s/15s attempts; counted incorrect",
                raw.get("row_index", index),
            )
        scored.append(
            {**dict(raw), "correct": correct, "judge_diagnostics": diagnostics}
        )
    return summarize_scored_prediction_rows(scored, task_type=task_type), scored


def summarize_scored_prediction_rows(
    scored: Sequence[Mapping[str, Any]], *, task_type: str = "math"
) -> dict[str, Any]:
    """Aggregate persisted judgments without repeating symbolic comparisons."""
    from think_bridge.eval.judge_diagnostics import JUDGE_PROTOCOL

    if not scored:
        raise ValueError("benchmark prediction rows are empty")
    for row in scored:
        detail = row.get("judge_diagnostics", {})
        if (
            type(row.get("correct")) is not bool
            or detail.get("protocol") != JUDGE_PROTOCOL
            or detail.get("status") not in {"ok", "timeout"}
            or (detail.get("status") == "timeout" and row["correct"])
        ):
            raise ValueError("invalid persisted benchmark judgment")
    versions = {row["judge_diagnostics"].get("math_verify_version") for row in scored}
    if len(versions) != 1:
        raise ValueError("HF workers used different math-verify versions")
    correct_count = sum(bool(row["correct"]) for row in scored)
    summary = {
        "task_type": task_type,
        "judge": "think_bridge.eval.answer_match.judge_answer",
        "math_verify_version": versions.pop(),
        "judge_protocol": JUDGE_PROTOCOL,
        "judge_timeout_policy": "10s-retry-on-timeout-15s-then-incorrect",
        "judge_retry_count": sum(
            row["judge_diagnostics"]["attempts"] > 1 for row in scored
        ),
        "judge_timeout_count": sum(
            row["judge_diagnostics"]["status"] == "timeout" for row in scored
        ),
        "judge_timeout_event_count": sum(
            row["judge_diagnostics"]["timeout_events"] for row in scored
        ),
        "correct": correct_count,
        "total": len(scored),
        "accuracy": correct_count / len(scored),
    }
    return summary


def build_parser():
    parser = argparse.ArgumentParser(prog="think-bridge benchmark", allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--mode", choices=("single-turn", "multi-turn"), default="single-turn"
    )
    parser.add_argument(
        "--measurement", choices=("accuracy", "ttft"), default="accuracy"
    )
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    for name in ("reasoner_checkpoint", "output_dir"):
        parser.add_argument(
            "--" + name.replace("_", "-"), dest=name, type=Path, required=True
        )
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    for name, default in [
        ("batch_size", None),
        ("reasoner_batch_size", None),
        ("max_new_tokens", 2048),
        ("warmup_samples", 2),
        ("seed", 42),
    ]:
        parser.add_argument(
            "--" + name.replace("_", "-"), dest=name, type=int, default=default
        )
    parser.add_argument(
        "--hf-workers",
        type=int,
        default=1,
        help="Independent HF replicas on visible GPUs; multi-turn only",
    )
    parser.add_argument("--inter-turn-delay-seconds", type=float, default=0.0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--local-device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size is None:
        args.batch_size = 1 if args.measurement == "ttft" else 16
    if args.reasoner_batch_size is None:
        args.reasoner_batch_size = (
            1 if args.mode == "multi-turn" or args.measurement == "ttft" else 64
        )
    if args.mode == "multi-turn" and args.reasoner_batch_size != 1:
        parser.error(
            "multi-turn uses singleton R; --batch-size controls parallel F requests"
        )
    if args.hf_workers < 1 or (
        args.hf_workers > 1 and (args.mode != "multi-turn" or args.backend != "hf")
    ):
        parser.error("multiple HF replicas require multi-turn HF")
    import math

    if (
        not math.isfinite(args.inter_turn_delay_seconds)
        or args.inter_turn_delay_seconds < 0
    ):
        parser.error("inter-turn delay must be finite and nonnegative")
    if args.backend == "vllm" and (
        args.mode != "single-turn" or args.measurement != "accuracy"
    ):
        parser.error(
            "vLLM supports single-turn answer accuracy; use HF for multi-turn/TTFT"
        )
    if args.measurement == "ttft" and args.reasoner_batch_size != 1:
        parser.error(
            "TTFT uses singleton R; --batch-size controls concurrent answer requests"
        )
    if min(args.batch_size, args.reasoner_batch_size, args.max_new_tokens) < 1:
        parser.error("batch sizes and generation budgets must be positive")
    if args.max_new_tokens > 2048 or args.warmup_samples < 0:
        parser.error("answer budget must not exceed 2048; warmups must be nonnegative")
    if not 0 < args.vllm_gpu_memory_utilization < 1:
        parser.error("vLLM memory utilization must be between zero and one")
    args.reasoner_eval_group_size = args.reasoner_batch_size
    args.attn_implementation = "sdpa"
    args.temperature = 0.0
    args.top_p = 1.0
    args.reader_context = "full"
    if args.measurement != "ttft" or args.mode != "multi-turn":
        args.inter_turn_delay_seconds = None
    args.max_samples = args.max_conversations = None
    args.vllm_data_parallel_size = args.vllm_tensor_parallel_size = 1
    args.vllm_max_num_seqs = args.batch_size
    args.vllm_request_timeout_seconds = 1800.0
    if args.hf_workers > 1:
        from think_bridge.eval.hf_parallel import run_parallel

        return run_parallel(args)
    if args.mode == "multi-turn":
        from think_bridge.eval.multiturn import run_multiturn_benchmark as run
    else:
        from think_bridge.eval.benchmark import run_benchmark as run
    import torch

    with torch.inference_mode():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
