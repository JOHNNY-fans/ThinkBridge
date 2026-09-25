"""Evaluate selected checkpoints on an independent QA dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="think-bridge eval")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    for name in ("reasoner_checkpoint", "output_dir"):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            "--" + name,
            dest=name,
            type=Path,
            required=True,
        )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--behavior",
        type=Path,
        help="Optional complete, prompt-bound native/direct behavior manifest",
    )
    parser.add_argument(
        "--controls",
        default="true",
        help="Comma-separated true,wrong,zero; true is always required",
    )
    parser.add_argument(
        "--control-protocol",
        "--control_protocol",
        choices=("standard", "strict-identities", "text-only"),
        default="standard",
        help="standard excludes matching prompts and any supplied problem/semantic IDs; strict-identities additionally requires these IDs",
    )
    parser.add_argument(
        "--reasoner-batch-size",
        dest="reasoner_eval_group_size",
        type=int,
        default=64,
        help="Maximum R inference batch; independent of answer decoding",
    )
    parser.add_argument("--wrong-k", "--wrong_k", type=int, default=8)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=2048)
    parser.add_argument("--local-device", "--local_device", default="auto")
    parser.add_argument("--local-files-only", "--local_files_only", action="store_true")
    parser.add_argument("--no-progress", "--no_progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.controls = tuple(value.strip() for value in args.controls.split(","))
    if (
        not args.controls
        or len(set(args.controls)) != len(args.controls)
        or "true" not in args.controls
        or set(args.controls) - {"true", "wrong", "zero"}
    ):
        parser.error(
            "controls must be distinct members of true,wrong,zero including true"
        )
    if args.reasoner_eval_group_size < 1:
        parser.error("reasoner-batch-size must be positive")
    if args.wrong_k < 1:
        parser.error("wrong-k must be positive")
    if not 1 <= args.max_new_tokens <= 2048:
        parser.error("answer budget must be within 1..2048")
    args.attn_implementation = "sdpa"
    from think_bridge.eval.release_evaluation import evaluate

    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
