"""Collect immutable paired native rollouts from a frozen executor."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def _iter_raw(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSON array or JSONL stream."""
    from think_bridge.data.dataset import load_data_file

    yield from load_data_file(path)


def _build_prompt_str(tok, record: dict[str, Any]) -> str:
    """Render the canonical thinking prompt for one source record."""
    from think_bridge.data.templates import build_thinking_prompt

    return build_thinking_prompt(
        tok,
        str(record.get("question", "")).strip(),
        task_type="math",
        messages=record.get("messages"),
    )


def _split_think(full: str) -> tuple[str, str]:
    """Split generated reasoning from the visible answer suffix."""
    cot = full
    tail = full
    if "</think>" in full:
        head, _, tail = full.rpartition("</think>")
        cot = head
    if "<think>" in cot:
        cot = cot.split("<think>", 1)[1]
    return cot.strip(), tail.strip()


@dataclass(frozen=True)
class Stage0Completion:
    """One real backend completion plus the metadata needed by Stage1 preflight."""

    text: str
    token_ids: tuple[int, ...]
    finish_reason: str

    @property
    def generated_token_count(self) -> int:
        return len(self.token_ids)

    @property
    def think_closed(self) -> bool:
        return "</think>" in self.text


def _configure_vllm_worker_multiprocessing() -> None:
    """Use CUDA-safe process creation for vLLM's internal EngineCore.

    vLLM's library default may be ``fork``, which cannot re-initialize CUDA in
    a forked EngineCore once the parent has touched CUDA.  This module has a
    proper ``__main__`` guard, so ``spawn`` is the safe deterministic contract
    for both single-process and outer-DP runs.
    """

    env_name = "VLLM_WORKER_MULTIPROC_METHOD"
    previous = os.environ.get(env_name, "").strip().lower()
    if previous and previous != "spawn":
        print(
            f"[fnative][WARN] overriding {env_name}={previous!r}; "
            "Stage 0 vLLM with CUDA requires 'spawn'",
            flush=True,
        )
    os.environ[env_name] = "spawn"


def _load_vllm_classes():
    """Load the explicitly requested vLLM backend without silent fallback."""
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "--backend vllm was requested but vLLM could not be imported; "
            "install/fix vLLM or explicitly select --backend hf"
        ) from exc
    return LLM, SamplingParams


def _extract_vllm_completions(
    outs: list[Any],
    n: int,
    *,
    expected_prompts: int,
) -> list[list[Stage0Completion]]:
    """Validate and extract a complete prompt-major vLLM response.

    Missing prompts/completions, token ids, or finish reasons are backend
    contract failures. They must never be padded with duplicated or empty text,
    because that silently changes Stage0 trajectory counts.
    """

    if len(outs) != int(expected_prompts):
        raise RuntimeError(
            "vLLM completion batch incomplete: "
            f"prompts={len(outs)} expected={expected_prompts}")
    groups: list[list[Stage0Completion]] = []
    for prompt_index, request_output in enumerate(outs):
        outputs = list(getattr(request_output, "outputs", ()) or ())
        if len(outputs) != int(n):
            raise RuntimeError(
                "vLLM completion group incomplete: "
                f"prompt_index={prompt_index} completions={len(outputs)} expected={n}")
        completion_indices = [getattr(output, "index", None) for output in outputs]
        if all(index is not None for index in completion_indices):
            normalized_indices = [int(index) for index in completion_indices]
            if sorted(normalized_indices) != list(range(int(n))):
                raise RuntimeError(
                    "vLLM completion indices invalid: "
                    f"prompt_index={prompt_index} indices={normalized_indices}")
            outputs = [
                output for _, output in sorted(
                    zip(normalized_indices, outputs), key=lambda pair: pair[0])
            ]

        group: list[Stage0Completion] = []
        for completion_index, output in enumerate(outputs):
            token_ids = getattr(output, "token_ids", None)
            finish_reason = getattr(output, "finish_reason", None)
            if token_ids is None or finish_reason is None or not str(finish_reason).strip():
                raise RuntimeError(
                    "vLLM completion metadata incomplete: "
                    f"prompt_index={prompt_index} completion_index={completion_index} "
                    f"has_token_ids={token_ids is not None} finish_reason={finish_reason!r}")
            group.append(Stage0Completion(
                text=str(getattr(output, "text", "")),
                token_ids=tuple(int(token_id) for token_id in token_ids),
                finish_reason=str(finish_reason).strip().lower(),
            ))
        groups.append(group)
    return groups


def _generate_vllm_chunk_complete(
    llm,
    prompts: list[str],
    sampling_params,
    *,
    n: int,
    max_attempts: int = 2,
) -> list[list[Stage0Completion]]:
    """Generate one chunk, retrying the whole chunk once on incomplete output."""

    attempts = max(int(max_attempts), 1)
    last_error: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        outs = llm.generate(prompts, sampling_params, use_tqdm=False)
        try:
            return _extract_vllm_completions(
                list(outs), n, expected_prompts=len(prompts))
        except RuntimeError as exc:
            last_error = exc
            if attempt < attempts:
                print(
                    "[fnative][WARN] vLLM returned an incomplete completion; "
                    f"retrying the entire chunk ({attempt}/{attempts}): {exc}",
                    flush=True,
                )
    raise RuntimeError(
        f"vLLM completion contract failed after {attempts} attempts: {last_error}")


def _completion_from_hf_tokens(
    tok,
    token_ids: list[int],
    *,
    max_new_tokens: int,
) -> Stage0Completion:
    """Build the same completion contract for the HF fallback backend."""

    ids = [int(token_id) for token_id in token_ids]
    eos_token_id = getattr(tok, "eos_token_id", None)
    pad_token_id = getattr(tok, "pad_token_id", None)
    hit_eos = eos_token_id is not None and int(eos_token_id) in ids
    if hit_eos:
        ids = ids[:ids.index(int(eos_token_id)) + 1]
        finish_reason = "stop"
    else:
        if pad_token_id is not None and pad_token_id != eos_token_id:
            while ids and ids[-1] == int(pad_token_id):
                ids.pop()
        finish_reason = "length" if len(ids) >= int(max_new_tokens) else "unknown"
    return Stage0Completion(
        text=str(tok.decode(ids, skip_special_tokens=True)),
        token_ids=tuple(ids),
        finish_reason=finish_reason,
    )


def _bucket_of(n_correct: int, n_sampled: int) -> str:
    if n_correct == 0:
        return "none_correct"
    if n_correct >= n_sampled:
        return "all_correct"
    return "mixed"


def _assign_stable_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve explicit IDs and assign deterministic IDs when absent."""
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for idx, raw in enumerate(records):
        rec = dict(raw)
        rid = rec.get("id")
        if rid is None or str(rid).strip() == "":
            rid = f"sample_{idx:06d}"
            rec["id"] = rid
        if rid in seen:
            raise ValueError(f"[fnative] duplicate input id {rid!r} at row {idx}")
        seen.add(rid)
        out.append(rec)
    return out


# ──────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────
def _run_sampling_rank(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    rank: int,
    out_raw: str | Path,
    out_frontier: str | Path,
    is_dp_child: bool = False,
) -> dict[str, int]:
    """Collect and validate the rollout shard owned by one data-parallel rank."""
    tag = f"[fnative:rank{rank}]" if is_dp_child else "[fnative]"
    rank_seed = int(args.seed)
    random.seed(rank_seed)
    try:
        import numpy as np
        np.random.seed(rank_seed)
    except Exception:  # noqa: BLE001
        pass
    # vLLM receives the same seed explicitly in both ``LLM`` and
    # ``SamplingParams`` below.  Touching ``torch.cuda`` before ``LLM`` is both
    # redundant and unsafe if an incompatible vLLM release ever ignores the
    # spawn contract.  The HF backend still owns and seeds its torch RNG here.
    if str(args.backend) == "hf":
        try:
            import torch

            torch.manual_seed(rank_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rank_seed)
        except Exception:  # noqa: BLE001
            pass
    N = max(1, int(args.n))

    max_keep = int(args.max_keep)
    keep_all = max_keep < 0
    n_total = len(records)
    print(f"{tag} rank_prompts={n_total} model={args.model} samples={N} "
          f"T={args.temperature} top_p={args.top_p} max_new={args.max_new_tokens}", flush=True)




    _tqdm_pos = rank if is_dp_child else 0
    _tqdm_desc_prefix = f"rank{rank}|" if is_dp_child else ""
    try:
        if bool(args.no_progress):
            raise ImportError("progress disabled by --no-progress")
        from tqdm import tqdm as _real_tqdm

        def tqdm(x, **k):  # type: ignore

            if is_dp_child:
                k.setdefault("position", _tqdm_pos)
                user_desc = k.get("desc", "")
                k["desc"] = f"{_tqdm_desc_prefix}{user_desc}" if user_desc else _tqdm_desc_prefix[:-1]
            return _real_tqdm(x, **k)
    except ImportError:
        def tqdm(x, **k):  # type: ignore
            return x

    from think_bridge.eval.answer_match import judge_answer


    backend = str(args.backend)
    vllm_classes = _load_vllm_classes() if backend == "vllm" else None


    items: list[tuple[dict[str, Any], str]] = []  # (raw, gold)
    prompts: list[str] = []


    if vllm_classes is not None:
        LLM, SamplingParams = vllm_classes


        tp_for_rank = 1 if is_dp_child else int(args.tensor_parallel_size)
        llm = LLM(
            model=args.model,
            tensor_parallel_size=tp_for_rank,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            trust_remote_code=args.trust_remote_code,
            max_model_len=None,
            seed=rank_seed,
        )
        tok = llm.get_tokenizer()
        for r in records:
            gold = str(r.get("answer", "")).strip()
            if not gold:
                continue
            items.append((r, gold))
            prompts.append(_build_prompt_str(tok, r))

        sampling_params = SamplingParams(
            n=N,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(args.max_new_tokens),
            seed=rank_seed,
        )


        chunk = max(1, int(args.batch_prompts))
        gen_completions: list[list[Stage0Completion]] = []
        for g0 in tqdm(range(0, len(prompts), chunk), desc="fnative-vllm", unit="blk"):
            prompt_chunk = prompts[g0:g0 + chunk]
            gen_completions.extend(_generate_vllm_chunk_complete(
                llm,
                prompt_chunk,
                sampling_params,
                n=N,
                max_attempts=2,
            ))
    else:

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        for r in records:
            gold = str(r.get("answer", "")).strip()
            if not gold:
                continue
            items.append((r, gold))
            prompts.append(_build_prompt_str(tok, r))

        gen_completions = []
        bp = max(1, int(args.batch_prompts))
        for g0 in tqdm(range(0, len(prompts), bp), desc="fnative-hf", unit="blk"):
            batch_prompts = prompts[g0:g0 + bp]
            enc = tok(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=N,
                    pad_token_id=tok.pad_token_id,
                )
            plen = enc["input_ids"].shape[1]
            cont = gen[:, plen:]  # [bp*N, T]
            for bi in range(len(batch_prompts)):
                gen_completions.append([
                    _completion_from_hf_tokens(
                        tok,
                        cont[bi * N + k].detach().cpu().tolist(),
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    for k in range(N)
                ])


    from think_bridge.model.executor_identity import model_source_identity
    frozen_identity = model_source_identity(args.model, local_files_only=True, no_progress=args.no_progress)
    out_records: list[dict[str, Any]] = []
    bucket_counts = {"all_correct": 0, "mixed": 0, "none_correct": 0}
    frontier_ids: list[Any] = []
    n_sampled_total = 0
    n_correct_total = 0
    n_incomplete_total = 0

    if len(gen_completions) != len(items):
        raise RuntimeError(
            "Stage0 backend prompt/completion mapping incomplete: "
            f"groups={len(gen_completions)} prompts={len(items)}")

    for (r, gold), completions in zip(items, gen_completions):
        rec_id = r.get("id", f"sample_{len(out_records):06d}")


        # completion, generation_complete)
        rollouts: list[
            tuple[int, str, str, bool, bool, Stage0Completion, bool]
        ] = []
        n_correct = 0
        for ri, completion in enumerate(completions):
            full = completion.text
            answer_correct = bool(judge_answer(full, gold, "math"))
            cot, tail = _split_think(full)
            generation_complete = bool(
                completion.finish_reason == "stop"
                and completion.think_closed
                and cot
                and tail
            )
            # ``correct`` is the Stage1-eligible correctness bit.  Preserve raw
            # mathematical correctness separately for diagnosing malformed or
            # length-truncated generations.
            ok = bool(answer_correct and generation_complete)
            n_incomplete_total += int(not generation_complete)
            if ok:
                n_correct += 1


            if keep_all:
                rollouts.append((
                    ri, cot, tail, ok, answer_correct, completion, generation_complete))
            elif ok and generation_complete:
                rollouts.append((
                    ri, cot, tail, ok, answer_correct, completion, generation_complete))
        n_sampled_total += len(completions)
        n_correct_total += n_correct
        bucket = _bucket_of(n_correct, len(completions))
        bucket_counts[bucket] += 1
        if bucket == "none_correct":
            frontier_ids.append(rec_id)


        if not keep_all and len(rollouts) > max_keep:
            rollouts.sort(key=lambda x: len(x[1]), reverse=True)
            rollouts = rollouts[:max_keep]

        for (
            ri, cot, tail, ok, answer_correct, completion, generation_complete
        ) in rollouts:

            carried = {
                key: value for key, value in r.items()
                if key not in {
                    "rollout_idx", "cot", "self_answer", "correct", "answer_correct",
                    "source", "bucket",
                    "n_sampled", "n_correct", "schema_version",
                    "stage0_generation_contract_version",
                    "finish_reason", "generated_token_count", "think_closed",
                    "generation_complete", "generation_backend", "generation_model",
                    # Read old rows permissively, but never propagate their
                    # persisted executor digest into a new Stage0 artifact.
                    "generation_executor_artifact_sha256",
                    "generation_seed", "generation_temperature", "generation_top_p",
                    "generation_max_new_tokens",
                }
            }
            out_records.append({
                **carried,
                "id": rec_id,
                "rollout_idx": ri,
                "question": r.get("question", ""),
                "cot": cot,
                "answer": gold,
                "self_answer": tail,
                "correct": ok,
                "answer_correct": answer_correct,
                "source": "fnative",
                "schema_version": 1,
                "generation_backend": backend,
                "generation_model": str(args.model),
                "generation_executor_artifact_sha256": frozen_identity["sha256"],
                "generation_executor_identity_algorithm": frozen_identity["algorithm"],
                "generation_seed": rank_seed,
                "generation_temperature": float(args.temperature),
                "generation_top_p": float(args.top_p),
                "generation_max_new_tokens": int(args.max_new_tokens),
                "finish_reason": completion.finish_reason,
                "generated_token_count": completion.generated_token_count,
                "think_closed": completion.think_closed,
                "generation_complete": generation_complete,
                "bucket": bucket,
                "n_sampled": len(completions),
                "n_correct": n_correct,
            })


    out_path = Path(out_raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    frontier_path = Path(out_frontier)
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frontier_path, "w", encoding="utf-8") as f:
        json.dump({"bucket": "none_correct", "ids": frontier_ids, "n": len(frontier_ids),
                   "generation_backend": backend,
                   "generation_model": str(args.model),
                   "generation_seed": rank_seed,
                   "generation_temperature": float(args.temperature),
                   "generation_top_p": float(args.top_p),
                   "generation_max_new_tokens": int(args.max_new_tokens),
                   "n_rollout": N,
                   "n_sampled_total": n_sampled_total,
                   "n_correct_total": n_correct_total,
                   "n_incomplete_total": n_incomplete_total,
                   "bucket_counts": bucket_counts},
                  f, ensure_ascii=False, indent=2)

    rate = n_correct_total / max(n_sampled_total, 1)
    print(f"{tag} complete records={len(out_records)} output={out_path}", flush=True)
    print(f"{tag} buckets all_correct={bucket_counts['all_correct']} "
          f"mixed={bucket_counts['mixed']} none_correct={bucket_counts['none_correct']} "
          f"prompts={sum(bucket_counts.values())}", flush=True)
    print(f"{tag} sample_accuracy={rate:.3f} "
          f"({n_correct_total}/{n_sampled_total})", flush=True)
    print(f"{tag} incomplete_generations={n_incomplete_total}/{n_sampled_total}", flush=True)
    print(f"{tag} frontier(none_correct ids)={len(frontier_ids)} -> {frontier_path}", flush=True)
    if not out_records:
        print(f"{tag}[WARN] no correct rollouts; check model, prompt, and sampling", flush=True)

    return {
        "n_records": len(out_records),
        "n_sampled_total": n_sampled_total,
        "n_correct_total": n_correct_total,
        "n_incomplete_total": n_incomplete_total,
        "bucket_all": bucket_counts["all_correct"],
        "bucket_mixed": bucket_counts["mixed"],
        "bucket_none": bucket_counts["none_correct"],
        "frontier_n": len(frontier_ids),
    }


# ──────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────
def _shard_indices(n_total: int, world_size: int) -> list[tuple[int, int]]:
    """Partition record indices into contiguous balanced rank shards."""
    base = n_total // world_size
    rem = n_total % world_size
    shards: list[tuple[int, int]] = []
    start = 0
    for r in range(world_size):
        size = base + (1 if r < rem else 0)
        shards.append((start, start + size))
        start += size
    return shards


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sample paired native rollouts from frozen F")
    ap.add_argument("--input", required=True,
                    help="Input question/answer JSON or JSONL")
    ap.add_argument("--output", required=True, help="Output paired-rollout JSON")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Frozen model path")
    ap.add_argument("--n", type=int, default=8, help="Rollouts per prompt")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=8192,
                    help="Maximum generated reasoning-plus-answer tokens")
    ap.add_argument("--max_prompts", type=int, default=-1, help="Maximum prompts; -1 means all")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                    help="Generation backend; no implicit fallback")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_keep", type=int, default=-1,
                    help="Rows retained per prompt; -1 keeps every rollout")
    ap.add_argument("--frontier_out", default="data/stage0/audit/frontier.json",
                    help="Output path for prompts with no correct rollout")
    ap.add_argument("--batch_prompts", type=int, default=64,
                    help="Distinct prompts per HF generation batch")
    ap.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress output",
    )
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--dp_world_size", type=int, default=0,
                    help="Data-parallel worker count; zero disables sharding")
    ap.add_argument("--dp", action="store_true", default=False,
                    help="Use one data-parallel worker per visible GPU")
    return ap


def main(argv: list[str] | None = None) -> None:

    args = build_parser().parse_args(argv)
    if args.max_keep != -1:
        raise ValueError("Stage0 requires every paired rollout; max_keep must be -1")
    if args.n <= 0 or args.max_new_tokens <= 0 or args.batch_prompts <= 0:
        raise ValueError("rollout count and token/batch budgets must be positive")
    if not 0 < args.top_p <= 1 or args.temperature < 0:
        raise ValueError("temperature must be nonnegative and top_p in (0, 1]")
    outputs = [Path(args.output), Path(args.frontier_out)]
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("native and audit outputs must be distinct")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Immutable Stage0 output already exists: {path}; use a new output path")

    if args.backend == "vllm":


        _configure_vllm_worker_multiprocessing()
        print(
            "[fnative] vLLM worker multiprocessing=spawn",
            flush=True,
        )

    dp_child_env = os.environ.get("TB_DP_CHILD", "") == "1"
    raws = _assign_stable_ids(list(_iter_raw(args.input)))
    if args.max_prompts is not None and args.max_prompts > 0:
        raws = raws[: args.max_prompts]
    n_total = len(raws)


    dp_world_size = int(args.dp_world_size)
    if args.dp:

        try:
            import torch
            dp_world_size = max(1, torch.cuda.device_count())
        except Exception:  # noqa: BLE001

            nv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if nv:
                dp_world_size = max(1, len(nv.split(",")))
            else:
                dp_world_size = 1
        if int(args.dp_world_size) > 0:

            dp_world_size = int(args.dp_world_size)

    if dp_world_size > 0 and int(args.tensor_parallel_size) > 1:
        print("[fnative][WARN] data and tensor parallelism are exclusive; "
              "forcing tp=1 per data-parallel process", flush=True)

    dp_device_tokens: list[str] = []
    if dp_world_size > 1:
        try:
            import torch

            visible_count = int(torch.cuda.device_count())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Stage 0 data parallelism cannot determine visible GPU count"
            ) from exc
        if visible_count < dp_world_size:
            raise RuntimeError(
                "Stage 0 data-parallel processes exceed torch-visible GPUs: "
                f"dp_world_size={dp_world_size} visible_count={visible_count}"
            )
        visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible_spec:
            dp_device_tokens = [
                token.strip() for token in visible_spec.split(",")
                if token.strip()
            ]
        else:
            dp_device_tokens = [str(index) for index in range(visible_count)]
        if len(dp_device_tokens) < dp_world_size:
            raise RuntimeError(
                "Stage 0 data-parallel processes exceed visible GPUs: "
                f"dp_world_size={dp_world_size} visible={dp_device_tokens}"
            )
        dp_device_tokens = dp_device_tokens[:dp_world_size]




    dp_child_rank = int(os.environ.get("TB_DP_RANK", "0")) if dp_child_env else 0
    if dp_world_size <= 1:
        if dp_world_size == 1 and args.dp and not dp_child_env:
            print("[fnative][WARN] --dp found one GPU; using one process", flush=True)
        if dp_child_env:

            args.tensor_parallel_size = 1
        _run_sampling_rank(
            records=raws,
            args=args,
            rank=dp_child_rank,
            out_raw=args.output,
            out_frontier=args.frontier_out,
            is_dp_child=dp_child_env,
        )
        return


    print(f"[fnative][DP] world_size={dp_world_size} tp_per_process=1 "
          f"input_prompts={n_total}", flush=True)

    shards = _shard_indices(n_total, dp_world_size)
    for r, (s, e) in enumerate(shards):
        print(f"[fnative][DP] rank{r}: indices=[{s},{e}) prompts={e - s}", flush=True)


    out_raw_path = Path(args.output)
    out_frontier_path = Path(args.frontier_out)
    rank_raw_paths = [str(out_raw_path) + f".{r}" for r in range(dp_world_size)]
    rank_frontier_paths = [str(out_frontier_path) + f".{r}" for r in range(dp_world_size)]








    import subprocess
    import sys

    procs: list[subprocess.Popen] = []
    shard_tmp_paths: list[str] = []
    for r in range(dp_world_size):
        s, e = shards[r]
        shard_records = raws[s:e]

        shard_tmp = str(out_raw_path) + f".shard{r}.json"
        Path(shard_tmp).parent.mkdir(parents=True, exist_ok=True)
        with open(shard_tmp, "w", encoding="utf-8") as f:
            json.dump(shard_records, f, ensure_ascii=False)
        shard_tmp_paths.append(shard_tmp)

        env = os.environ.copy()
        device_token = dp_device_tokens[r]
        env["CUDA_VISIBLE_DEVICES"] = device_token
        env["TB_DP_CHILD"] = "1"
        env["TB_DP_RANK"] = str(r)

        cmd = [
            sys.executable, "-m", "think_bridge.cli.sample_fnative_vllm",
            "--input", shard_tmp,
            "--output", rank_raw_paths[r],
            "--model", args.model,
            "--n", str(args.n),
            "--seed", str(args.seed),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--max_new_tokens", str(args.max_new_tokens),
            "--max_keep", str(args.max_keep),
            "--backend", args.backend,
            "--tensor_parallel_size", "1",
            "--gpu_memory_utilization", str(args.gpu_memory_utilization),
            "--frontier_out", rank_frontier_paths[r],
            "--batch_prompts", str(args.batch_prompts),
        ]
        if args.trust_remote_code:
            cmd.append("--trust_remote_code")
        if args.no_progress:
            cmd.append("--no-progress")

        print(f"[fnative][DP] spawn rank{r}: CUDA_VISIBLE_DEVICES={device_token} "
              f"shard=[{s},{e}) prompts={e - s} -> {rank_raw_paths[r]}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))


    exit_codes = [p.wait() for p in procs]

    for shard_tmp in shard_tmp_paths:
        try:
            os.remove(shard_tmp)
        except OSError:
            pass
    failed = [(r, c) for r, c in enumerate(exit_codes) if c != 0]
    if failed:
        for r, c in failed:
            print(f"[fnative][DP][ERROR] rank{r} exit_code={c}", flush=True)

        for r in range(dp_world_size):
            for p in (rank_raw_paths[r], rank_frontier_paths[r]):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise SystemExit(1)


    print("[fnative][DP] all ranks complete; merging shards", flush=True)
    merged_records: list[dict[str, Any]] = []
    merged_frontier_ids: list[Any] = []
    n_sampled_total = 0
    n_correct_total = 0
    n_incomplete_total = 0
    bucket_counts = {"all_correct": 0, "mixed": 0, "none_correct": 0}
    for r in range(dp_world_size):
        merged_records.extend(json.loads(Path(rank_raw_paths[r]).read_text(encoding="utf-8")))
        fj = json.loads(Path(rank_frontier_paths[r]).read_text(encoding="utf-8"))
        merged_frontier_ids.extend(fj.get("ids", []))
        n_sampled_total += int(fj.get("n_sampled_total", 0))
        n_correct_total += int(fj.get("n_correct_total", 0))
        n_incomplete_total += int(fj.get("n_incomplete_total", 0))
        for name, value in fj.get("bucket_counts", {}).items():
            if name in bucket_counts:
                bucket_counts[name] += int(value)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, ensure_ascii=False, indent=2)

    frontier_path = Path(args.frontier_out)
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frontier_path, "w", encoding="utf-8") as f:
        json.dump({"bucket": "none_correct", "ids": merged_frontier_ids,
                   "n": len(merged_frontier_ids),
                   "generation_backend": str(args.backend),
                   "generation_model": str(args.model),
                   "generation_seed": int(args.seed),
                   "generation_temperature": float(args.temperature),
                   "generation_top_p": float(args.top_p),
                   "generation_max_new_tokens": int(args.max_new_tokens),
                   "n_rollout": int(args.n),
                   "n_sampled_total": n_sampled_total,
                   "n_correct_total": n_correct_total,
                   "n_incomplete_total": n_incomplete_total,
                   "bucket_counts": bucket_counts}, f, ensure_ascii=False, indent=2)


    for r in range(dp_world_size):
        try:
            os.remove(rank_raw_paths[r])
            os.remove(rank_frontier_paths[r])
        except OSError:
            pass

    rate = n_correct_total / max(n_sampled_total, 1)
    print(f"[fnative][DP] merge complete records={len(merged_records)} output={out_path}", flush=True)
    print(f"[fnative][DP] global buckets all_correct={bucket_counts['all_correct']} "
          f"mixed={bucket_counts['mixed']} none_correct={bucket_counts['none_correct']} "
          f"prompts={sum(bucket_counts.values())}", flush=True)
    print(f"[fnative][DP] global_sample_accuracy={rate:.3f} "
          f"({n_correct_total}/{n_sampled_total})", flush=True)
    print(
        f"[fnative][DP] global_incomplete_generations={n_incomplete_total}/{n_sampled_total}",
        flush=True,
    )
    print(f"[fnative][DP] frontier(none_correct ids)={len(merged_frontier_ids)} -> {frontier_path}",
          flush=True)
    if not merged_records:
        print("[fnative][DP][WARN] no correct rollouts; check model, prompt, and sampling", flush=True)


if __name__ == "__main__":
    main()
