"""Produce the immutable five-artifact input set for R training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from think_bridge.data.dataset import load_data_file


def validate_sources(train: Path, validation: Path) -> tuple[int, int]:
    """Require explicit disjoint splits; never choose or rewrite a split here."""
    populations = []
    for path in (train, validation):
        rows = load_data_file(path)
        if not rows:
            raise ValueError(f"Empty source: {path}")
        keys = set()
        identifiers = set()
        for row in rows:
            raw_question, raw_answer = row.get("question"), row.get("answer")
            if (
                not isinstance(raw_question, str)
                or not raw_question.strip()
                or not isinstance(raw_answer, (str, int, float))
                or isinstance(raw_answer, bool)
                or not str(raw_answer).strip()
                or (isinstance(raw_answer, float) and not math.isfinite(raw_answer))
            ):
                raise ValueError(
                    f"Every source row needs nonempty question/answer: {path}"
                )
            question = raw_question.strip()
            if row.get("messages"):
                raise ValueError(
                    "Stage0 training sources are single-turn question/answer rows"
                )
            key = " ".join(question.split())
            if key in keys:
                raise ValueError(f"Duplicate source question: {path}")
            keys.add(key)
            identifier = str(row.get("id", "")).strip()
            if identifier and identifier in identifiers:
                raise ValueError(f"Duplicate source id: {path}: {identifier}")
            if identifier:
                identifiers.add(identifier)
        populations.append(keys)
    if populations[0] & populations[1]:
        raise ValueError(
            "Training and validation contain the same question; provide disjoint splits"
        )
    return len(populations[0]), len(populations[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Explicit training question/answer JSON or JSONL",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
        help="Nonempty disjoint held-out question/answer rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing directories are refused",
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="Native rollout backend; behavior is exact HF",
    )
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_prompts", type=int, default=16)
    parser.add_argument(
        "--no-progress", "--no_progress", dest="no_progress", action="store_true"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.n <= 0 or args.batch_prompts <= 0 or args.seed < 0:
        raise ValueError("n/batch_prompts must be positive; seed must be nonnegative")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be nonnegative and top_p in (0, 1]")
    validate_sources(args.train, args.validation)
    # mkdir without exist_ok is the atomic no-clobber boundary for this run.
    args.output.mkdir(parents=True, exist_ok=False)
    validation_rows = load_data_file(args.validation)
    (args.output / "validation.json").write_text(
        json.dumps(validation_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress = ["--no_progress"] if args.no_progress else []
    common = ["--model", args.model, "--seed", str(args.seed), *progress]
    native = args.output / "native.json"
    commands = [
        [
            "sample_fnative_vllm",
            "--input",
            str(args.train),
            "--output",
            str(native),
            "--frontier_out",
            str(args.output / "native_audit.json"),
            "--backend",
            args.backend,
            "--n",
            str(args.n),
            "--temperature",
            str(args.temperature),
            "--top_p",
            str(args.top_p),
            "--max_new_tokens",
            "8192",
            "--max_keep",
            "-1",
            "--batch_prompts",
            str(args.batch_prompts),
        ],
        [
            "precompute_behavior_manifest",
            "--input",
            str(native),
            "--output",
            str(args.output / "behavior.json"),
            "--direct_raw_output",
            str(args.output / "direct.json"),
            "--backend",
            "hf",
            "--native_label_source",
            "stage0",
            "--generation_batch_size",
            str(args.batch_prompts),
        ],
        [
            "precompute_behavior_manifest",
            "--input",
            str(args.output / "validation.json"),
            "--output",
            str(args.output / "validation_behavior.json"),
            "--backend",
            "hf",
            "--native_label_source",
            "generate",
            "--generation_batch_size",
            str(args.batch_prompts),
        ],
    ]
    for index, (module, *arguments) in enumerate(commands, 1):
        print(f"[Stage0 {index}/3] {module}", flush=True)
        subprocess.run(
            [sys.executable, "-m", f"think_bridge.cli.{module}", *arguments, *common],
            check=True,
        )
    print(f"Stage0 complete: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
