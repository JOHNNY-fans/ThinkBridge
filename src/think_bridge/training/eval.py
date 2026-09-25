"""Sealed free evaluators and checkpoint selectors for isolated ThinkBridge."""

from __future__ import annotations

from think_bridge.training.eval_examples import (
    answer_examples as _answer_examples,
    print_evaluation_examples,
)

from think_bridge.model.reasoner_inference import reasoner_inference_metadata

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as torch_functional

from think_bridge.model.artifact_schema import (
    ROUTE1_SELECTION,
    artifact_header,
)
from think_bridge.eval.answer_match import judge_answer
from think_bridge.model.trajectory import (
    incremental_executor_prefill_batched,
    incremental_executor_step_batched,
)
from think_bridge.model.checkpoint_policy import (
    SealedCheckpoint,
    BridgeCheckpointIdentity,
    checkpoint_artifact_sha256,
    validate_route1_selection_candidates,
    validate_checkpoint_directory,
    validate_checkpoint_seal,
)
from think_bridge.model.training_config import TrainingConfig
from think_bridge.training.progress import (
    iter_progress,
)
from think_bridge.model.contract import (
    NEED_Z_COHORT_DEFINITION,
    OBJECTIVE_VERSION,
    ROUTE1_VALIDATION_REPORT_SCHEMA_VERSION,
    canonical_json_sha256,
    file_sha256,
    normalize_route1_null_mode,
    normalize_route1_standalone_null_mode,
    is_bridge_isolated_path,
    require_manifest_fields,
    require_sha256,
    route1_direct_baseline_identity,
    route1_selector_generation_conditions,
    route1_selector_metrics,
    route1_selector_request_counts,
    route1_population,
    select_route1_evaluation_rows,
    validate_hard_donors,
    validate_matched_hard_donor_manifest,
    validation_randomness_identity,
    write_atomic_json,
)
from think_bridge.training.train import (
    _read_json,
    _read_jsonl,
    validate_prepared_records,
)
from think_bridge.training.checkpoint_manager import (
    checkpoint_run_relative_locator,
    resolve_run_artifact_locator,
)


_ROUTE1_CONDITIONS = (
    "true_z",
    "direct",
    "wrong_z_1",
    "wrong_z_2",
    "wrong_z_3",
    "wrong_z_4",
    "wrong_z_5",
    "wrong_z_6",
    "wrong_z_7",
    "wrong_z_8",
)


@dataclass(frozen=True)
class CausalEvalRequest:
    """One explicit compact-KV decode request in logical prompt coordinates."""

    prompt_ids: tuple[int, ...]
    seed: int
    z: torch.Tensor | None = None
    process_ids: tuple[int, ...] | None = None
    record_id: str | None = None
    condition: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt_ids or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in self.prompt_ids
        ):
            raise ValueError(
                "causal eval request prompt ids must be non-empty integers"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("causal eval request seed must be an integer")


@contextmanager
def evaluation_mode(
    model: Any, *, hf: bool = True, answer_group_size: int | None = None
) -> Iterable[None]:
    """Evaluate current in-memory weights and restore the exact prior mode."""

    modes = [(module, bool(module.training)) for module in model.modules()]
    model.eval()
    from think_bridge.eval.hf_protocol import hf_evaluation_context

    try:
        with (
            hf_evaluation_context(model, answer_group_size=answer_group_size)
            if hf
            else nullcontext()
        ):
            yield
    finally:
        for module, was_training in modes:
            module.training = was_training


def _write_json_transactional(
    path: Path,
    value: Mapping[str, Any],
    *,
    transaction_id: str,
) -> str:
    """Publish one distributed report or exact-reuse a concurrent winner."""

    if not is_bridge_isolated_path(path):
        raise ValueError("ThinkBridge report path must remain run isolated")
    from think_bridge.training.distributed_artifacts import publish_file_once

    expected = json.loads(json.dumps(dict(value), sort_keys=True, ensure_ascii=False))

    def write(temporary: Path) -> None:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")

    def validate(candidate: Path) -> None:
        observed = json.loads(candidate.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("concurrent evaluation report differs from exact content")

    return publish_file_once(
        path,
        transaction_id=transaction_id,
        write_temporary=write,
        validate=validate,
    )


def _write_json_idempotent(path: Path, value: Mapping[str, Any]) -> None:
    """Recover an exact completed seal; reject any semantic drift."""

    if not is_bridge_isolated_path(path):
        raise ValueError("ThinkBridge report/seal path must remain run isolated")
    write_atomic_json(path, value, replace_mismatch=False)


def _write_selection_result(path: Path, value: Mapping[str, Any]) -> None:
    """Refresh a run's best after more evaluations, preserving prior selection."""
    if not path.exists():
        _write_json_idempotent(path, value)
        return
    previous = _read_json(path)
    if previous == dict(value):
        return
    old_reports = previous.get("candidate_reports")
    new_reports = value.get("candidate_reports")
    stable_fields = (
        "artifact_type",
        "schema_version",
        "objective_version",
        "method",
        "route",
        "seed",
        "metric_policy",
        "validation_randomness",
        "evaluation_domain",
        "direct_baseline_identity_sha256",
    )
    if (
        not isinstance(old_reports, list)
        or not old_reports
        or (not isinstance(new_reports, list))
        or (len(new_reports) <= len(old_reports))
        or (new_reports[: len(old_reports)] != old_reports)
        or any((previous.get(key) != value.get(key) for key in stable_fields))
    ):
        raise ValueError(
            "selection refresh requires unchanged earlier reports and an extended candidate history"
        )
    history = (
        path.parent
        / "selection_history"
        / f"{path.stem}-{canonical_json_sha256(previous)}.json"
    )
    _write_json_idempotent(history, previous)
    write_atomic_json(path, value, replace_mismatch=True)


def _distributed_evaluation_shard_identity(
    *,
    route: str,
    split: str,
    checkpoint_sha256: str,
    generation_seed: int,
    route1_null_mode: str | None,
    route1_eval_backend: str | None,
    record_ids: Sequence[str],
    route1_answer_max_tokens: int | None = None,
    evaluation_subset_sha256: str | None = None,
    run_diagnostics: bool = True,
    answer_group_size: int = 8,
    reasoner_batch_size: int = 64,
) -> str:
    """Seal only route-owned inputs to the distributed evaluation shards."""
    if route not in {"route1"}:
        raise ValueError("distributed evaluation shard route is invalid")
    route1_like = True
    payload: dict[str, Any] = {
        **artifact_header("think-bridge.evaluation.distributed-" + route),
        "route": route,
        "diagnostics_executed": bool(run_diagnostics),
        "split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "route1_null_mode": route1_null_mode,
        "route1_eval_backend": route1_eval_backend,
        "record_ids": [str(record_id) for record_id in record_ids],
        "evaluation_subset_sha256": require_sha256(
            str(evaluation_subset_sha256 or ""), "evaluation_subset_sha256"
        ),
    }
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError("evaluation shard requires generation randomness")
    payload["generation_seed"] = generation_seed
    if (
        isinstance(route1_answer_max_tokens, bool)
        or not isinstance(route1_answer_max_tokens, int)
        or route1_answer_max_tokens <= 0
    ):
        raise ValueError("Route1 evaluation shard requires an answer horizon")
    payload["answer_max_tokens"] = int(route1_answer_max_tokens)
    if route1_eval_backend == "torch":
        from think_bridge.eval.hf_protocol import hf_evaluation_protocol

        payload["answer_evaluation"] = hf_evaluation_protocol(
            answer_group_size, reasoner_batch_size
        )
    return canonical_json_sha256(payload)


def _load_bound_evaluation_data(
    config: TrainingConfig, arguments: Any, *, manifest: Mapping[str, Any]
) -> tuple[
    list[dict[str, Any]], dict[str, list[dict[str, Any]]] | None, dict[str, Any] | None
]:
    """Load sealed validation data from an already verified manifest binding."""
    route = str(arguments.route)
    expected_manifest_route = route
    if (
        manifest["method"] != config.method
        or manifest["route"] != expected_manifest_route
    ):
        raise ValueError("evaluation config method/route differs from manifest")
    split = str(arguments.split)
    if split != "validation":
        raise ValueError("Bridge Stage1 evaluation is validation-only")
    records_path = Path(arguments.records)
    rows = _read_jsonl(records_path)
    if not rows:
        raise ValueError("validation dataset is empty")
    validate_prepared_records(
        rows,
        route="route1",
        expected_split=split,
        route2_content_capacity=int(manifest["route2_content_capacity"]),
    )
    if getattr(arguments, "donors", None) is None:
        raise ValueError("Route1 evaluation requires the donor manifest")
    donor_path = Path(arguments.donors)
    donor_payload = _read_json(donor_path)
    donor_records = validate_matched_hard_donor_manifest(
        donor_payload,
        rows,
        split=split,
        content_capacity=int(manifest["route2_content_capacity"]),
    )
    donors_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for item in donor_records:
        anchor_id = str(item["anchor_record_id"])
        if not anchor_id or anchor_id in donors_by_anchor:
            raise ValueError("hard-donor anchors must be unique and non-empty")
        donors = item["donors"]
        for donor in donors:
            prompt_ids = donor["prompt_ids"]
            if (
                not isinstance(prompt_ids, list)
                or not prompt_ids
                or any(
                    (not isinstance(token, int) or token < 0 for token in prompt_ids)
                )
            ):
                raise ValueError("hard donor prompt ids are invalid")
        donors_by_anchor[anchor_id] = [dict(row) for row in donors]
    source_ids = {str(row["record_id"]) for row in rows}
    if not source_ids.issubset(donors_by_anchor):
        raise ValueError(
            "hard-donor manifest does not bind every sealed evaluation record"
        )
    donors_by_anchor = {
        record_id: donors_by_anchor[record_id] for record_id in source_ids
    }
    for row in rows:
        validate_hard_donors(row, donors_by_anchor[str(row["record_id"])])
    (selected_rows, evaluation_domain) = select_route1_evaluation_rows(
        rows, max_eval_samples=getattr(arguments, "max_eval_samples", None)
    )
    selected_ids = {str(row["record_id"]) for row in selected_rows}
    selected_donors = {
        record_id: donors_by_anchor[record_id] for record_id in selected_ids
    }
    return (selected_rows, selected_donors, evaluation_domain)


def _compact_physical_inputs(
    model: Any,
    prompt_ids: Sequence[int],
    *,
    z: torch.Tensor | None = None,
    process_ids: Sequence[int] | None = None,
    answer_prefix: Sequence[int] = (),
    append_boundary: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if z is not None and process_ids is not None:
        raise ValueError("executor process condition must be exactly one of z/CoT/null")
    embedding = model.executor.get_input_embeddings()
    device = embedding.weight.device
    dtype = embedding.weight.dtype
    prompt = torch.tensor(list(prompt_ids), dtype=torch.long, device=device)
    if prompt.numel() == 0:
        raise ValueError("free evaluator prompt cannot be empty")
    prompt_emb = embedding(prompt)
    process = prompt_emb.new_empty((0, prompt_emb.size(-1)))
    if z is not None:
        if z.ndim == 3:
            z = z.squeeze(0)
        if (
            z.ndim != 2
            or z.size(0) not in {32, 64, 128}
            or z.size(1) != prompt_emb.size(-1)
        ):
            raise ValueError("Route1 free z must be [K,dF]")
        process = z.to(device=device, dtype=dtype)
    elif process_ids is not None:
        if not process_ids or len(process_ids) > model.alignment_capacity:
            raise ValueError(
                "generated/source CoT is empty or exceeds the 6144 horizon"
            )
        ids = torch.tensor(list(process_ids), dtype=torch.long, device=device)
        process = embedding(ids)
    boundary = (
        embedding(model.boundary_ids.to(device))
        if append_boundary
        else prompt_emb.new_empty((0, prompt_emb.size(-1)))
    )
    prefix_ids = torch.tensor(list(answer_prefix), dtype=torch.long, device=device)
    prefix = (
        embedding(prefix_ids)
        if prefix_ids.numel()
        else prompt_emb.new_zeros((0, prompt_emb.size(-1)))
    )
    inputs = torch.cat((prompt_emb, process, boundary, prefix), dim=0).unsqueeze(0)
    attention = torch.ones((1, inputs.size(1)), dtype=torch.bool, device=device)
    process_positions = torch.arange(
        prompt.numel(),
        prompt.numel() + process.size(0),
        dtype=torch.long,
        device=device,
    )
    suffix_start = prompt.numel() + process.size(0)
    suffix_positions = torch.arange(
        suffix_start,
        suffix_start + boundary.size(0) + prefix_ids.numel(),
        dtype=torch.long,
        device=device,
    )
    positions = torch.cat(
        (
            torch.arange(prompt.numel(), dtype=torch.long, device=device),
            process_positions,
            suffix_positions,
        )
    ).unsqueeze(0)
    qstar = prompt.numel() + process.size(0) + boundary.size(0) - 1
    return inputs, attention, positions, int(qstar)


def _route1_append_boundary(request: CausalEvalRequest) -> bool:
    """Return whether evaluation must add a physical sealed boundary."""

    if request.condition != "direct":
        return True
    if request.process_ids is not None:
        raise ValueError("direct Route1 request cannot carry generated/source CoT")
    # The prepared direct/no-think ids already encode the complete prompt,
    # including its empty think block and boundary.  The optional zero64
    # diagnostic instead carries the base prompt plus an explicit z row.
    return request.z is not None


@torch.no_grad()
def _reason_many(
    model: Any, prompt_rows: Sequence[Sequence[int]]
) -> list[torch.Tensor]:
    from think_bridge.model.reasoner_inference import reason_eval_prompts

    return reason_eval_prompts(model, prompt_rows)


@torch.no_grad()
def _reason_many_batched(
    model: Any,
    prompt_rows: Sequence[Sequence[int]],
    *,
    local_batch_size: int,
) -> list[torch.Tensor]:
    """Compute current-R latents without exceeding the eval row batch."""

    if (
        isinstance(local_batch_size, bool)
        or not isinstance(local_batch_size, int)
        or local_batch_size <= 0
    ):
        raise ValueError("batched Route1 reasoning batch size must be positive")
    outputs: list[torch.Tensor] = []
    for start in range(0, len(prompt_rows), local_batch_size):
        outputs.extend(
            value.to(device="cpu")
            for value in _reason_many(
                model, prompt_rows[start : start + local_batch_size]
            )
        )
    if len(outputs) != len(prompt_rows):
        raise RuntimeError("batched Route1 reasoning lost a prompt")
    return outputs


def _nucleus(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> int:
    if temperature <= 0.0:
        return int(logits.argmax().item())
    probabilities = torch_functional.softmax(
        logits.float() / float(temperature), dim=-1
    )
    sorted_probability, sorted_index = probabilities.sort(descending=True)
    keep = (sorted_probability.cumsum(0) - sorted_probability) < float(top_p)
    filtered = sorted_probability * keep
    filtered = filtered / filtered.sum().clamp_min(torch.finfo(filtered.dtype).tiny)
    selected = int(torch.multinomial(filtered, 1, generator=generator).item())
    return int(sorted_index[selected].item())


def _compact_physical_inputs_many(
    model: Any,
    requests: Sequence[CausalEvalRequest],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not requests:
        raise ValueError("free decode batch cannot be empty")
    rows = [
        _compact_physical_inputs(
            model,
            request.prompt_ids,
            z=request.z,
            process_ids=request.process_ids,
            append_boundary=_route1_append_boundary(request),
        )
        for request in requests
    ]
    maximum = max(inputs.size(1) for inputs, _, _, _ in rows)
    hidden = int(rows[0][0].size(-1))
    device = rows[0][0].device
    dtype = rows[0][0].dtype
    inputs = torch.zeros((len(rows), maximum, hidden), dtype=dtype, device=device)
    attention = torch.zeros((len(rows), maximum), dtype=torch.bool, device=device)
    positions = torch.zeros((len(rows), maximum), dtype=torch.long, device=device)
    for index, (row_inputs, row_attention, row_positions, _) in enumerate(rows):
        length = int(row_inputs.size(1))
        inputs[index, :length] = row_inputs[0]
        attention[index, :length] = row_attention[0]
        positions[index, :length] = row_positions[0]
    return inputs, attention, positions


def _free_answers_greedy(
    model: Any,
    state: Any,
    logits: torch.Tensor,
    *,
    max_tokens: int,
    eos_token_id: int,
    token_callback: Callable[[int, int], bool | None] | None = None,
) -> list[list[int]]:
    """Decode greedily with one batch-level synchronization per live step."""

    batch_size = int(logits.size(0))
    device = logits.device
    generated = torch.full(
        (batch_size, max_tokens),
        int(eos_token_id),
        dtype=torch.long,
        device=device,
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    eos_tokens = torch.full(
        (batch_size,), int(eos_token_id), dtype=torch.long, device=device
    )
    for token_index in range(max_tokens):
        selected = logits[:, 0].argmax(dim=-1)
        tokens = torch.where(active, selected, eos_tokens)
        callback_stopped = (
            torch.zeros_like(active) if token_callback is not None else None
        )
        if token_callback is not None:
            for row_index in range(batch_size):
                if bool(active[row_index]):
                    callback_stopped[row_index] = (
                        token_callback(row_index, int(tokens[row_index].item())) is True
                    )
        generated[:, token_index] = tokens
        lengths = lengths + active.long()
        active = active & tokens.ne(int(eos_token_id))
        if callback_stopped is not None:
            active = active & ~callback_stopped
        if token_index + 1 >= max_tokens or not bool(active.any()):
            break
        token_embed = model.executor.get_input_embeddings()(tokens.unsqueeze(1))
        state, logits = incremental_executor_step_batched(
            model.executor,
            state,
            token_embed=token_embed,
            active_mask=active,
        )
    packed_rows = torch.cat((lengths.unsqueeze(1), generated), dim=1).cpu().tolist()
    return [row[1 : 1 + row[0]] for row in packed_rows]


def _free_answers_sampled(
    model: Any,
    requests: Sequence[CausalEvalRequest],
    state: Any,
    logits: torch.Tensor,
    *,
    max_tokens: int,
    eos_token_id: int,
    temperature: float,
    top_p: float,
    token_callback: Callable[[int, int], bool | None] | None = None,
) -> list[list[int]]:
    """Preserve the sealed per-request RNG and nucleus-sampling order."""

    device = logits.device
    generators: list[torch.Generator] = []
    for request in requests:
        generator = torch.Generator(device=device)
        generator.manual_seed(request.seed)
        generators.append(generator)
    generated: list[list[int]] = [[] for _ in requests]
    active = torch.ones(len(requests), dtype=torch.bool, device=device)
    for token_index in range(max_tokens):
        tokens: list[int] = []
        for row_index in range(len(requests)):
            if bool(active[row_index]):
                token = _nucleus(
                    logits[row_index, 0],
                    temperature=float(temperature),
                    top_p=float(top_p),
                    generator=generators[row_index],
                )
                generated[row_index].append(token)
                if token_callback is not None:
                    if token_callback(row_index, token) is True:
                        active[row_index] = False
                if token == int(eos_token_id):
                    active[row_index] = False
            else:
                token = int(eos_token_id)
            tokens.append(token)
        if not bool(active.any()) or token_index + 1 >= max_tokens:
            break
        token_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
        token_embed = model.executor.get_input_embeddings()(token_ids)
        state, logits = incremental_executor_step_batched(
            model.executor,
            state,
            token_embed=token_embed,
            active_mask=active,
        )
    return generated


@torch.no_grad()
def _free_answers(
    model: Any,
    tokenizer: Any,
    requests: Sequence[CausalEvalRequest],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    token_callback: Callable[[int, int], bool | None] | None = None,
) -> list[dict[str, Any]]:
    """Generate answers; a callback returning True stops only its current row."""
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("free decode max_tokens must be a positive integer")
    inputs, attention, positions = _compact_physical_inputs_many(model, requests)
    eos = int(tokenizer.eos_token_id)
    state, logits = incremental_executor_prefill_batched(
        model.executor,
        inputs_embeds=inputs,
        attention_mask=attention,
        position_ids=positions,
    )
    generated = (
        _free_answers_greedy(
            model,
            state,
            logits,
            max_tokens=max_tokens,
            eos_token_id=eos,
            token_callback=token_callback,
        )
        if float(temperature) <= 0.0
        else _free_answers_sampled(
            model,
            requests,
            state,
            logits,
            max_tokens=max_tokens,
            eos_token_id=eos,
            token_callback=token_callback,
            temperature=float(temperature),
            top_p=float(top_p),
        )
    )
    outputs: list[dict[str, Any]] = []
    for ids in generated:
        terminated = bool(ids and ids[-1] == eos)
        outputs.append(
            {
                "ids": ids,
                "text": tokenizer.decode(ids, skip_special_tokens=True),
                "terminated": terminated,
                "cap_hit": not terminated and len(ids) >= max_tokens,
            }
        )
    return outputs


def _required_eval_row(row: Mapping[str, Any]) -> tuple[str, str, bool, bool]:
    reference = row.get("reference_answer")
    task_type = row.get("task_type")
    need_z = row.get("need_z")
    direct_correct = row.get("direct_correct")
    if not isinstance(reference, str) or not reference:
        raise ValueError("free evaluator record lacks reference_answer")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError("free evaluator record lacks task_type")
    if not all(isinstance(value, bool) for value in (need_z, direct_correct)):
        raise ValueError("free evaluator requires sealed need/direct booleans")
    return (
        reference,
        task_type,
        need_z,
        direct_correct,
    )


def _route1_row_chunks(
    rows: Sequence[Mapping[str, Any]], local_row_batch: int
) -> tuple[Sequence[Mapping[str, Any]], ...]:
    if (
        isinstance(local_row_batch, bool)
        or not isinstance(local_row_batch, int)
        or local_row_batch <= 0
    ):
        raise ValueError("route1_eval_local_row_batch must be a positive integer")
    return tuple(
        rows[offset : offset + local_row_batch]
        for offset in range(0, len(rows), local_row_batch)
    )


def _route1_eval_transaction_id(
    request_namespace: str, *, rank: int, batch_index: int
) -> str:
    """Build a bounded vLLM request namespace unique to one eval row batch."""

    if not isinstance(request_namespace, str) or not request_namespace:
        raise ValueError("Route1 eval request namespace is empty")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or batch_index < 0
    ):
        raise ValueError("Route1 eval request rank/batch identity is invalid")
    namespace = hashlib.sha256(request_namespace.encode("utf-8")).hexdigest()[:16]
    return f"route1-eval-{namespace}-rank-{rank:02d}-batch-{batch_index:06d}"


def _route1_generation_requests(
    rows: Sequence[Mapping[str, Any]],
    donors_by_anchor: Mapping[str, Sequence[Mapping[str, Any]]],
    reason_cache: Mapping[tuple[int, ...], Any],
    *,
    generation_seed: int,
    null_mode: str = "direct",
    run_diagnostics: bool = True,
) -> list[CausalEvalRequest]:
    null_mode = normalize_route1_standalone_null_mode(null_mode)
    if null_mode != "direct":
        raise ValueError("sparse Route1 selector requires direct/no-think baseline")
    requests: list[CausalEvalRequest] = []
    for row in rows:
        prompt = tuple(int(token) for token in row["prompt_ids"])
        native_correct = row.get("locked_full_correct")
        direct_correct = row.get("direct_correct")
        donors = (
            donors_by_anchor.get(str(row["record_id"]), ()) if run_diagnostics else ()
        )
        condition_names = route1_selector_generation_conditions(
            native_correct,
            direct_correct,
            include_controls=run_diagnostics,
            donor_count=len(donors),
        )
        conditions = {"true_z": (prompt, reason_cache[prompt])}
        if run_diagnostics and route1_population(native_correct, direct_correct) == "C":
            conditions.update(
                {
                    f"wrong_z_{index + 1}": (
                        prompt,
                        reason_cache[
                            tuple(int(token) for token in donor["prompt_ids"])
                        ],
                    )
                    for index, donor in enumerate(donors)
                }
            )
        for condition in condition_names:
            condition_prompt, condition_z = conditions[condition]
            requests.append(
                CausalEvalRequest(
                    record_id=str(row["record_id"]),
                    condition=condition,
                    prompt_ids=condition_prompt,
                    z=condition_z,
                    process_ids=None,
                    seed=int(generation_seed),
                )
            )
    return requests


def _assemble_route1_paired_rows(
    rows: Sequence[Mapping[str, Any]],
    requests: Sequence[CausalEvalRequest],
    outputs: Sequence[Mapping[str, Any]],
    *,
    run_diagnostics: bool = True,
) -> list[dict[str, Any]]:
    if len(outputs) != len(requests):
        raise ValueError("Route1 batched free outputs lost a record/condition row")
    by_record: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for request, output in zip(requests, outputs):
        record_id = str(request.record_id or "")
        condition = str(request.condition or "")
        if not record_id or not condition or condition in by_record[record_id]:
            raise ValueError("Route1 sparse request identity is empty or duplicated")
        by_record[record_id][condition] = output
    paired: list[dict[str, Any]] = []
    for row in rows:
        (
            reference,
            task_type,
            need_z,
            direct_correct,
        ) = _required_eval_row(row)
        native_correct = row.get("locked_full_correct")
        if not isinstance(native_correct, bool):
            raise ValueError("Route1 eval row lacks sealed native correctness")
        population = route1_population(native_correct, direct_correct)
        by_condition = by_record.get(str(row["record_id"]), {})
        donor_count = sum(key.startswith("wrong_z_") for key in by_condition)
        condition_names = route1_selector_generation_conditions(
            native_correct,
            direct_correct,
            include_controls=run_diagnostics,
            donor_count=donor_count,
        )
        if set(by_condition) != set(condition_names):
            raise ValueError("Route1 sparse outputs differ from selector conditions")
        correct = {
            "true_z": bool(
                judge_answer(by_condition["true_z"]["text"], reference, task_type)
            )
        }
        if population == "C":
            correct["direct"] = direct_correct
            correct.update(
                {
                    condition: bool(
                        judge_answer(
                            by_condition[condition]["text"], reference, task_type
                        )
                    )
                    for condition in condition_names
                    if condition != "true_z"
                }
            )
        paired.append(
            {
                "record_id": str(row["record_id"]),
                "prompt_group_id": str(row["prompt_group_id"]),
                "need_z": need_z,
                "direct_correct": direct_correct,
                "population": population,
                "correct": correct,
                "generated_token_ids": {
                    condition: by_condition[condition]["ids"]
                    for condition in condition_names
                },
            }
        )
    for value in paired:
        signature = by_record[value["record_id"]]["true_z"].get("reasoner_input")
        if signature is not None:
            value["reasoner_input"] = dict(signature)
    if set(by_record) != {str(row["record_id"]) for row in rows}:
        raise ValueError("Route1 sparse outputs escaped the selected validation rows")
    return paired


def _route1_free_answers_vllm(
    model: Any,
    tokenizer: Any,
    requests: Sequence[CausalEvalRequest],
    *,
    max_tokens: int,
    request_namespace: str,
    eval_batch_index: int,
    config: Any = None,
) -> list[dict[str, Any]]:
    """Run all paired free-generation rows through one private vLLM service."""

    if config is None:
        raise RuntimeError("Route1 free evaluation lacks its split-runtime config")
    from think_bridge.training.vllm_client import BridgeRoute1ServiceClient
    from think_bridge.training.vllm_runtime import (
        ROUTE1_MAX_HTTP_RESPONSE_BYTES,
        Route1ServiceRequest,
        TensorPayload,
    )

    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    physical = int(config.eval_vllm_physical_chunk_size)
    if physical <= 0:
        raise ValueError("Route1 vLLM physical chunk must be positive")
    if (
        getattr(model, "_feedback_fixed_groups", False)
        and requests
        and all(value.condition.startswith("wrong_z_") for value in requests)
    ):
        # The ordered true-z phase is already complete. Each anchor contributes
        # eight controls in donor order; keep that request intact even when a
        # training/evaluation memory profile uses a smaller physical chunk.
        physical = 8
    template_z = next((value.z for value in requests if value.z is not None), None)
    if template_z is None:
        raise RuntimeError(
            "Route1 vLLM paired-condition transport lacks a z geometry template"
        )
    transaction = _route1_eval_transaction_id(
        request_namespace,
        rank=rank,
        batch_index=eval_batch_index,
    )
    # This is service-local routing metadata, not a scientific sample identity.
    # A rank offset by the request width gives physical=2 ranks the
    # complementary DP pairs and lets every physical=4 request cover DP0..3.
    sample_ordinal_base = rank * physical
    service_requests: list[Route1ServiceRequest] = []
    by_identity: dict[tuple[Any, ...], tuple[int, int]] = {}
    for chunk_id, start in enumerate(range(0, len(requests), physical)):
        real_values = list(requests[start : start + physical])
        if not real_values:
            continue
        values = list(real_values)
        while len(values) < physical:
            values.append(real_values[-1])
        boundary_ids = tuple(
            int(value) for value in model.boundary_ids.detach().cpu().tolist()
        )
        prompt_rows = tuple(
            tuple(int(token) for token in value.prompt_ids) for value in values
        )
        z_present = tuple(value.z is not None for value in values)
        append_boundary = tuple(_route1_append_boundary(value) for value in values)
        z_rows = torch.stack(
            [
                (
                    value.z.detach().reshape(template_z.shape[-2], -1)
                    if value.z is not None
                    else torch.zeros_like(template_z)
                    .detach()
                    .reshape(template_z.shape[-2], -1)
                )
                for value in values
            ],
            dim=0,
        )
        request = Route1ServiceRequest(
            temperature=0.0,
            run_id=transaction,
            transaction_id=transaction,
            rank=int(rank),
            step=0,
            micro_step=0,
            chunk_id=chunk_id,
            sample_ids=tuple(
                range(
                    sample_ordinal_base + start,
                    sample_ordinal_base + start + len(values),
                )
            ),
            sample_keys=tuple(
                f"eval-r{rank}-b{eval_batch_index}-s{start + offset}"
                + ("-padding" if offset >= len(real_values) else "")
                for offset in range(len(values))
            ),
            sample_is_padding=tuple(
                offset >= len(real_values) for offset in range(len(values))
            ),
            prompt_ids=prompt_rows,
            prompt_lengths=tuple(len(row) for row in prompt_rows),
            detached_z=TensorPayload.from_torch(z_rows, wire_dtype="float32"),
            boundary_ids=boundary_ids,
            generation_seed=int(values[0].seed),
            max_prefix_tokens=int(max_tokens),
            z_present=z_present,
            append_boundary=append_boundary,
            eos_token_id=int(tokenizer.eos_token_id),
        )
        service_requests.append(request)
        by_identity[request.identity] = (start, start + len(real_values))
    client = BridgeRoute1ServiceClient(
        host=str(config.eval_vllm_host),
        port=int(config.eval_vllm_port),
        timeout_seconds=float(config.eval_vllm_request_timeout_seconds),
        max_response_bytes=ROUTE1_MAX_HTTP_RESPONSE_BYTES,
        # The shared engine serializes collectives; keep only one active HTTP
        # request per evaluator rank so queue wait is not charged against many
        # already-open request timeouts.
        max_in_flight=1,
    )
    if not client.health():
        raise RuntimeError("Route1 vLLM evaluation service is not healthy")
    outputs: list[dict[str, Any] | None] = [None] * len(requests)
    for response in client.stream(service_requests):
        start, stop = by_identity[response.identity]
        if len(response.prefix_token_ids) < stop - start:
            raise RuntimeError("Route1 vLLM free evaluation lost response rows")
        for offset, ids in enumerate(response.prefix_token_ids[: stop - start]):
            token_ids = list(ids)
            terminated = bool(
                token_ids and token_ids[-1] == int(tokenizer.eos_token_id)
            )
            outputs[start + offset] = {
                "ids": token_ids,
                "text": tokenizer.decode(token_ids, skip_special_tokens=True),
                "terminated": terminated,
                "cap_hit": not terminated and len(token_ids) >= int(max_tokens),
            }
    if any(value is None for value in outputs):
        raise RuntimeError("Route1 vLLM free evaluation response set is incomplete")
    return [value for value in outputs if value is not None]


def _route1_paired_outputs(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    donors_by_anchor: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    split: str,
    generation_seed: int,
    answer_max_tokens: int,
    route1_eval_local_row_batch: int,
    no_progress: bool,
    request_namespace: str,
    route1_null_mode: str = "direct",
    eval_backend: str = "torch",
    eval_backend_config: Any = None,
    fixed_global_rows: Sequence[Mapping[str, Any]] | None = None,
    true_outputs: Mapping[str, Mapping[str, Any]] | None = None,
    initial_reason_cache: Mapping[tuple[int, ...], Any] | None = None,
    run_diagnostics: bool = True,
) -> list[dict[str, Any]]:
    if eval_backend not in {"torch", "vllm"}:
        raise ValueError("Route1 eval backend must be torch or vllm")
    route1_null_mode = normalize_route1_standalone_null_mode(route1_null_mode)
    paired_rows: list[dict[str, Any]] = []
    reason_cache: dict[tuple[int, ...], torch.Tensor] = dict(initial_reason_cache or {})
    fixed_anchors = fixed_donors = None
    if getattr(model, "_feedback_fixed_groups", False):
        from think_bridge.model.evaluation_groups import FixedReasonerGroups

        universe = list(rows if fixed_global_rows is None else fixed_global_rows)
        fixed_anchors = FixedReasonerGroups(
            [r["prompt_ids"] for r in universe],
            lambda ps: _reason_many(model, ps),
            group_size=model._feedback_eval_batch_size,
        )
        extra = []
        seen = set(fixed_anchors.first)
        for anchor in universe if run_diagnostics else ():
            for donor in donors_by_anchor.get(str(anchor["record_id"]), ()):
                key = tuple(donor["prompt_ids"])
                if key not in seen:
                    extra.append(key)
                    seen.add(key)
        fixed_donors = FixedReasonerGroups(
            extra,
            lambda ps: _reason_many(model, ps),
            group_size=model._feedback_eval_batch_size,
        )
    from think_bridge.eval.hf_protocol import (
        hf_evaluation_protocol,
        canonical_reason,
        decode_groups,
    )

    if eval_backend == "torch":
        model._answer_evaluation_protocol = hf_evaluation_protocol(
            route1_eval_local_row_batch, model._feedback_eval_batch_size
        )
    chunks = _route1_row_chunks(rows, route1_eval_local_row_batch)
    request_counts = route1_selector_request_counts(
        rows, include_controls=run_diagnostics, donors_by_anchor=donors_by_anchor
    )
    for eval_batch_index, chunk in enumerate(
        iter_progress(
            chunks,
            total=int(request_counts["total"]),
            desc=f"Route1 {split}",
            unit="request",
            disabled=no_progress,
            update_size=lambda batch: int(
                route1_selector_request_counts(
                    batch,
                    include_controls=run_diagnostics,
                    donors_by_anchor=donors_by_anchor,
                )["total"]
            ),
        )
    ):
        missing: list[tuple[int, ...]] = []
        observed = set(reason_cache)
        for row in chunk:
            prompts = [row["prompt_ids"]]
            if (
                run_diagnostics
                and route1_population(
                    row.get("locked_full_correct"), row.get("direct_correct")
                )
                == "C"
            ):
                prompts.extend(
                    donor["prompt_ids"]
                    for donor in donors_by_anchor[str(row["record_id"])]
                )
            for prompt_ids in prompts:
                key = tuple(int(token) for token in prompt_ids)
                if key not in observed:
                    observed.add(key)
                    missing.append(key)
        if missing:
            if fixed_anchors is None:
                latents = (
                    canonical_reason(model, missing)
                    if eval_backend == "torch"
                    else _reason_many(model, missing)
                )
            else:
                # Controls and answer request width cannot redefine true-z groups.
                latents = [
                    (
                        fixed_anchors if p in fixed_anchors.first else fixed_donors
                    ).lookup([p])[0]
                    for p in missing
                ]
            if len(latents) != len(missing):
                raise RuntimeError("batched Route1 reasoner lost a requested prompt")
            reason_cache.update(zip(missing, latents))
        requests = _route1_generation_requests(
            chunk,
            donors_by_anchor,
            reason_cache,
            generation_seed=int(generation_seed),
            null_mode=route1_null_mode,
            run_diagnostics=run_diagnostics,
        )
        # True answers always decode as one complete configured row group. Controls
        # cannot change true prefill shapes, active sequence count or decode order.
        if eval_backend == "torch" and true_outputs is None:
            true_requests = [r for r in requests if r.condition == "true_z"]
            true_values = decode_groups(
                model, tokenizer, true_requests, max_tokens=answer_max_tokens
            )
            chunk_true = {
                str(r.record_id): value for r, value in zip(true_requests, true_values)
            }
            from think_bridge.eval.answer_protocol import input_signature

            for request in true_requests:
                chunk_true[str(request.record_id)]["reasoner_input"] = input_signature(
                    request.prompt_ids,
                    request.z,
                    model.boundary_ids.tolist(),
                    model.eos_token_id,
                )
        else:
            chunk_true = true_outputs
        pending = [r for r in requests if chunk_true is None or r.condition != "true_z"]
        generated = (
            (
                decode_groups(model, tokenizer, pending, max_tokens=answer_max_tokens)
                if eval_backend == "torch"
                else _route1_free_answers_vllm(
                    model,
                    tokenizer,
                    pending,
                    max_tokens=answer_max_tokens,
                    request_namespace=request_namespace,
                    eval_batch_index=eval_batch_index,
                    config=eval_backend_config,
                )
            )
            if pending
            else []
        )
        remaining = iter(generated)
        outputs = [
            chunk_true[str(r.record_id)]
            if chunk_true is not None and r.condition == "true_z"
            else next(remaining)
            for r in requests
        ]
        paired_rows.extend(
            _assemble_route1_paired_rows(
                chunk, requests, outputs, run_diagnostics=run_diagnostics
            )
        )
    return paired_rows


def _evaluate_route1(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    donors_by_anchor: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    manifest: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_identity: BridgeCheckpointIdentity,
    split: str,
    generation_seed: int,
    answer_max_tokens: int,
    no_progress: bool,
    evaluation_domain: Mapping[str, Any],
    metric_policy: Mapping[str, Any],
    route1_eval_local_row_batch: int = 1,
    route1_null_mode: str = "direct",
    paired_rows_override: Sequence[Mapping[str, Any]] | None = None,
    config: Any = None,
    run_diagnostics: bool = True,
) -> dict[str, Any]:
    paired_rows = (
        [dict(row) for row in paired_rows_override]
        if paired_rows_override is not None
        else _route1_paired_outputs(
            model,
            tokenizer,
            rows,
            donors_by_anchor,
            split=split,
            generation_seed=generation_seed,
            answer_max_tokens=answer_max_tokens,
            route1_eval_local_row_batch=route1_eval_local_row_batch,
            no_progress=no_progress,
            request_namespace=f"checkpoint-{checkpoint_sha256}",
            route1_null_mode=route1_null_mode,
            run_diagnostics=run_diagnostics,
        )
    )
    source_by_record = {str(row["record_id"]): row for row in rows}
    for paired in paired_rows:
        source = source_by_record.get(str(paired.get("record_id", "")))
        if source is None:
            raise ValueError("paired Route1 output escaped its sealed target row")
        for field in (
            "need_z",
            "direct_correct",
        ):
            sealed = source.get(field)
            if not isinstance(sealed, bool):
                raise ValueError(f"sealed validation target lacks {field}")
            observed = paired.get(field)
            if observed is not None and observed is not sealed:
                raise ValueError(f"paired Route1 target label differs: {field}")
            paired[field] = sealed
        paired["population"] = route1_population(
            bool(source["locked_full_correct"]), bool(source["direct_correct"])
        )
    c_rows = [row for row in paired_rows if row["population"] == "C"]
    selector_metrics = route1_selector_metrics(
        paired_rows,
        generation_seed=generation_seed,
        run_seed=checkpoint_identity.seed,
        include_controls=run_diagnostics,
    )
    if len(rows) != len(paired_rows):
        raise ValueError(
            "locked full-accuracy denominator differs from paired validation rows"
        )
    if (
        not isinstance(evaluation_domain, Mapping)
        or evaluation_domain.get("actual_sample_count") != len(rows)
        or not isinstance(evaluation_domain.get("evaluation_subset_sha256"), str)
    ):
        raise ValueError("Route1 evaluation subset identity differs from its rows")
    observed_subset_domain = [
        {
            "record_id": str(row["record_id"]),
            "prompt_group_id": str(row["prompt_group_id"]),
            "population": route1_population(
                row["locked_full_correct"], row["direct_correct"]
            ),
        }
        for row in rows
    ]
    observed_population_denominators = {
        population: sum(
            item["population"] == population for item in observed_subset_domain
        )
        for population in "BCDE"
    }
    if (
        evaluation_domain.get("evaluation_subset_sha256")
        != canonical_json_sha256(observed_subset_domain)
        or evaluation_domain.get("population_denominators")
        != observed_population_denominators
    ):
        raise ValueError("Route1 evaluation subset hash/counts differ from its rows")
    quadrant_rows = {
        quadrant: [row for row in paired_rows if row["population"] == quadrant]
        for quadrant in "BCDE"
    }
    quadrant_accuracy = {
        quadrant: selector_metrics[f"{quadrant.lower()}_accuracy"]
        for quadrant in "BCDE"
    }
    paired_domain = [
        {
            "record_id": row["record_id"],
            "prompt_group_id": row["prompt_group_id"],
            "need_z": row["need_z"],
            "direct_correct": row["direct_correct"],
            "population": row["population"],
            "need_z_definition": NEED_Z_COHORT_DEFINITION,
        }
        for row in paired_rows
    ]
    paired_hash = canonical_json_sha256(paired_domain)
    paired_output_hash = canonical_json_sha256(paired_rows)
    gate_metrics = {
        "step": int(checkpoint_identity.step),
        "seed": int(checkpoint_identity.seed),
        "judge_path": "think_bridge.eval.answer_match.judge_answer",
        "donor_count": max(selector_metrics.get("wrong_donor_counts", []), default=0),
        "bootstrap_ci_generated": selector_metrics.get("robust_g1_ci_low") is not None,
        "diagnostics_executed": bool(run_diagnostics),
        "paired_row_sha256": paired_hash,
        **selector_metrics,
    }
    direct_baseline = route1_direct_baseline_identity(
        manifest,
        rows,
        answer_max_tokens=int(answer_max_tokens),
    )
    generation_request_counts = route1_selector_request_counts(
        rows, include_controls=run_diagnostics, donors_by_anchor=donors_by_anchor
    )
    return {
        **artifact_header(ROUTE1_VALIDATION_REPORT_SCHEMA_VERSION),
        "objective_version": OBJECTIVE_VERSION,
        "method": "bridge",
        "route": "route1",
        "seed": int(checkpoint_identity.seed),
        "step": int(checkpoint_identity.step),
        "checkpoint_path": checkpoint_run_relative_locator(checkpoint_path),
        "checkpoint_sha256": require_sha256(checkpoint_sha256, "checkpoint_sha256"),
        "split": split,
        "free_generation": True,
        "diagnostics_executed": bool(run_diagnostics),
        "reasoner_inference": reasoner_inference_metadata(model),
        "answer_evaluation": getattr(model, "_answer_evaluation_protocol", None),
        "generation": {
            "seed": int(generation_seed),
            "answer_max_tokens": int(answer_max_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "validation_randomness": validation_randomness_identity(
            route="route1", base_seed=int(generation_seed)
        ),
        "gate_metrics": gate_metrics,
        "top_line_metrics": {
            "true_z_full_accuracy": selector_metrics["true_z_full_accuracy"],
            **{
                name: selector_metrics[name]
                for name in ("robust_g1", "robust_g1_ci_low")
                if name in selector_metrics
            },
        },
        "metric_policy": dict(metric_policy),
        "evaluation_domain": dict(evaluation_domain),
        "direct_baseline": direct_baseline,
        "generation_request_counts": generation_request_counts,
        "need_z_cohort_definition": NEED_Z_COHORT_DEFINITION,
        "population_denominators": {
            quadrant: len(quadrant_rows[quadrant]) for quadrant in "BCDE"
        },
        "population_true_z_accuracy": quadrant_accuracy,
        "c_metrics": {
            "a_true": selector_metrics["a_true"],
            "a_direct": selector_metrics["a_direct"],
            **{
                name: selector_metrics[name]
                for name in ("a_wrong", "robust_g1", "robust_g1_ci_low")
                if name in selector_metrics
            },
            "raw_g1": selector_metrics["raw_g1"],
            "prompt_count": len(c_rows),
            "control_available_count": selector_metrics.get(
                "control_available_count", 0
            ),
            "control_unavailable_count": selector_metrics.get(
                "control_unavailable_count", 0
            ),
            "wrong_donor_counts": selector_metrics.get("wrong_donor_counts", []),
            "wrong_pair_count": selector_metrics["wrong_pair_count"],
        },
        "train_c_sample_count": int(manifest["train_c_sample_count"]),
        "train_d_sample_count": int(manifest["train_d_sample_count"]),
        "train_d_excluded_non_eos_count": int(
            manifest["train_d_excluded_non_eos_count"]
        ),
        "paired_row_sha256": paired_hash,
        "paired_output_sha256": paired_output_hash,
        "qualitative_answer_examples": _answer_examples(
            tokenizer,
            rows,
            paired_rows,
            count=2,
        ),
        "paired_rows": paired_rows,
    }


def evaluate_loaded_model(
    config: TrainingConfig,
    arguments: Any,
    *,
    model: Any,
    tokenizer: Any,
    checkpoint_identity: BridgeCheckpointIdentity,
    checkpoint_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    donors: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    evaluation_domain: Mapping[str, Any] | None,
    rank: int,
    world_size: int,
    checkpoint_seal: SealedCheckpoint | None = None,
    capture_live_feedback: bool = False,
) -> Path:
    """Evaluate inside one invocation-owned distributed report transaction."""

    import torch.distributed as distributed
    from think_bridge.training.distributed_artifacts import (
        distributed_artifact_transaction,
    )

    report_path = Path(arguments.report)
    capture_context = nullcontext()
    with (
        capture_context,
        distributed_artifact_transaction(
            distributed,
            rank=rank,
            world_size=world_size,
            parent=report_path.parent / "bridge-eval-transactions",
            label="evaluation",
        ) as transaction,
    ):
        return _evaluate_loaded_model_transaction(
            config,
            arguments,
            model=model,
            tokenizer=tokenizer,
            checkpoint_identity=checkpoint_identity,
            checkpoint_payload=checkpoint_payload,
            manifest=manifest,
            rows=rows,
            donors=donors,
            evaluation_domain=evaluation_domain,
            rank=rank,
            world_size=world_size,
            checkpoint_seal=checkpoint_seal,
            transaction=transaction,
        )


def _evaluate_loaded_model_transaction(
    config: TrainingConfig,
    arguments: Any,
    *,
    model: Any,
    tokenizer: Any,
    checkpoint_identity: BridgeCheckpointIdentity,
    checkpoint_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    donors: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    evaluation_domain: Mapping[str, Any] | None,
    rank: int,
    world_size: int,
    checkpoint_seal: SealedCheckpoint | None,
    transaction: Any,
) -> Path:
    """Evaluate current in-memory phase weights, then restore training mode."""
    from think_bridge.training.distributed_artifacts import (
        contiguous_shard_indices,
        merge_json_shards,
        write_json_shard,
    )
    from think_bridge.training.runtime_backend import (
        broadcast_rank0_result,
        collect_rank_failures,
    )
    import torch.distributed as distributed

    evaluation_started = time.perf_counter()
    if (
        isinstance(arguments.route1_eval_local_row_batch, bool)
        or not isinstance(arguments.route1_eval_local_row_batch, int)
        or arguments.route1_eval_local_row_batch <= 0
    ):
        raise ValueError("route1_eval_local_row_batch must be a positive integer")
    expected_route = str(arguments.route)
    if arguments.route != expected_route:
        raise ValueError("evaluation arm and route disagree")
    if arguments.split != "validation":
        raise ValueError("Bridge Stage1 evaluation is validation-only")
    from think_bridge.training.metric_registry import report_metric_policy

    sealed_metric_policy = report_metric_policy(
        route=expected_route, value=getattr(arguments, "metric_policy", None)
    ).as_dict()
    route1_like = True
    if donors is None:
        raise ValueError("Route1 evaluation requires its donor bundle")
    if not isinstance(evaluation_domain, Mapping):
        raise ValueError("evaluation domain is missing after input validation")
    sampler_state = checkpoint_payload.get("sampler_state")
    if not isinstance(sampler_state, Mapping):
        raise ValueError("evaluation checkpoint lacks sealed sampler state")
    route1_null_mode = normalize_route1_standalone_null_mode(
        getattr(arguments, "route1_null_mode", "direct")
    )
    checkpoint_null_mode = normalize_route1_null_mode(
        sampler_state.get("route1_eval_null_mode", "direct")
    )
    if checkpoint_null_mode != "direct":
        raise ValueError("Route1 checkpoint baseline is not direct/no-think")
    checkpoint_path = Path(arguments.checkpoint)
    if checkpoint_seal is None:
        checkpoint_seal = validate_checkpoint_directory(checkpoint_path)
    else:
        checkpoint_seal = validate_checkpoint_seal(checkpoint_seal)
        if checkpoint_seal.path != checkpoint_path.resolve(strict=True):
            raise ValueError("in-process evaluation checkpoint seal/path mismatch")
    checkpoint_sha256 = checkpoint_seal.artifact_sha256
    report_path = Path(arguments.report)
    shard_root = transaction.root / "evaluation-shards"
    route1_answer_max_tokens = getattr(arguments, "answer_max_tokens", None)
    answer_group_size = int(arguments.route1_eval_local_row_batch)
    shard_identity = _distributed_evaluation_shard_identity(
        answer_group_size=answer_group_size,
        reasoner_batch_size=getattr(model, "_feedback_eval_batch_size", 64) or 64,
        route=expected_route,
        run_diagnostics=bool(getattr(arguments, "eval_diagnostics", True)),
        split=str(arguments.split),
        checkpoint_sha256=checkpoint_sha256,
        generation_seed=int(arguments.generation_seed),
        route1_null_mode=route1_null_mode,
        route1_eval_backend=str(getattr(arguments, "route1_eval_backend", "torch")),
        record_ids=[str(row["record_id"]) for row in rows],
        evaluation_subset_sha256=evaluation_domain.get("evaluation_subset_sha256"),
        **{"route1_answer_max_tokens": route1_answer_max_tokens},
    )
    ordered_answers = (
        getattr(model, "_feedback_fixed_groups", False)
        and getattr(arguments, "route1_eval_backend", "torch") == "vllm"
    )
    if getattr(arguments, "route1_eval_backend", "torch") == "torch" or getattr(
        model, "_feedback_fixed_groups", False
    ):
        from think_bridge.model.evaluation_groups import evaluation_group_indices

        shard_plan = [
            [
                i
                for group in evaluation_group_indices(
                    len(rows),
                    rank=owner,
                    world_size=world_size,
                    group_size=64
                    if ordered_answers
                    else math.lcm(
                        answer_group_size,
                        getattr(model, "_feedback_eval_batch_size", 64) or 64,
                    ),
                )
                for i in group
            ]
            for owner in range(world_size)
        ]
    else:
        shard_plan = [
            list(contiguous_shard_indices(len(rows), rank=owner, world_size=world_size))
            for owner in range(world_size)
        ]
    indices = shard_plan[rank]
    local_rows = [rows[index] for index in indices]
    local_failure: str | None = None
    indexed_rows: list[dict[str, Any]] = []
    with evaluation_mode(
        model,
        hf=getattr(arguments, "route1_eval_backend", "torch") == "torch",
        answer_group_size=answer_group_size,
    ):
        try:
            true_outputs = initial_reason_cache = None
            if ordered_answers:
                from think_bridge.eval.answer_protocol import (
                    ordered_true_z_outputs,
                    answer_protocol,
                )
                from think_bridge.training.vllm_client import BridgeRoute1ServiceClient

                client = BridgeRoute1ServiceClient(
                    host=arguments.eval_vllm_host,
                    port=arguments.eval_vllm_port,
                    timeout_seconds=arguments.eval_vllm_request_timeout_seconds,
                    max_response_bytes=256 * 1024 * 1024,
                    max_in_flight=1,
                )
                model._answer_evaluation_protocol = {
                    **answer_protocol(),
                    "data_parallel_size": getattr(
                        arguments, "eval_vllm_data_parallel_size", None
                    ),
                    "max_num_seqs": getattr(arguments, "eval_vllm_max_num_seqs", 64),
                    "gpu_memory_utilization": getattr(
                        arguments, "eval_vllm_gpu_memory_utilization", None
                    ),
                    "enforce_eager": False,
                    "answer_max_tokens": int(route1_answer_max_tokens),
                    "seed": int(arguments.generation_seed),
                }
                try:
                    (actual_indices, true_outputs, initial_reason_cache) = (
                        ordered_true_z_outputs(
                            model,
                            tokenizer,
                            rows,
                            execute=client.submit,
                            seed=arguments.generation_seed,
                            max_tokens=int(route1_answer_max_tokens),
                            namespace=str(transaction.transaction_id),
                            rank=rank,
                            world_size=world_size,
                            no_progress=bool(arguments.no_progress),
                        )
                    )
                    if indices != actual_indices:
                        raise RuntimeError(
                            "True-z answer plan and evaluation shard plan differ"
                        )
                finally:
                    client.abort_pending()
            values = _route1_paired_outputs(
                model,
                tokenizer,
                local_rows,
                donors,
                split=arguments.split,
                generation_seed=int(arguments.generation_seed),
                answer_max_tokens=int(route1_answer_max_tokens),
                route1_eval_local_row_batch=int(arguments.route1_eval_local_row_batch),
                no_progress=bool(arguments.no_progress) or rank != 0,
                request_namespace=str(transaction.transaction_id),
                route1_null_mode=route1_null_mode,
                eval_backend=str(getattr(arguments, "route1_eval_backend", "torch")),
                eval_backend_config=arguments,
                fixed_global_rows=rows,
                true_outputs=true_outputs,
                initial_reason_cache=initial_reason_cache,
                run_diagnostics=bool(getattr(arguments, "eval_diagnostics", True)),
            )
            if len(values) != len(indices):
                raise ValueError(
                    "evaluation worker output count differs from assigned rows"
                )
            indexed_rows = [
                {"index": index, "payload": value}
                for (index, value) in zip(indices, values)
            ]
            write_json_shard(
                shard_root,
                rank=rank,
                world_size=world_size,
                transaction_id=transaction.transaction_id,
                identity_sha256=shard_identity,
                indices=indices,
                rows=indexed_rows,
            )
        except Exception as exc:
            local_failure = f"rank={rank} eval worker: {type(exc).__name__}: {exc}"
    failures = collect_rank_failures(
        distributed, world_size=world_size, local_failure=local_failure
    )
    if failures:
        raise RuntimeError("; ".join(failures))
    if distributed.is_initialized():
        distributed.barrier()
    rank0_result: dict[str, Any] | None = None
    if rank == 0:
        try:
            with evaluation_mode(
                model,
                hf=getattr(arguments, "route1_eval_backend", "torch") == "torch",
                answer_group_size=answer_group_size,
            ):
                merge_started = time.perf_counter()
                merged_rows = merge_json_shards(
                    shard_root,
                    world_size=world_size,
                    expected_count=len(rows),
                    transaction_id=transaction.transaction_id,
                    identity_sha256=shard_identity,
                    expected_indices_by_rank=shard_plan,
                )
                merge_seconds = time.perf_counter() - merge_started
                report = _build_distributed_evaluation_report(
                    model=model,
                    tokenizer=tokenizer,
                    rows=rows,
                    donors=donors,
                    config=config,
                    manifest=manifest,
                    checkpoint_path=Path(arguments.checkpoint),
                    checkpoint_sha256=checkpoint_sha256,
                    checkpoint_identity=checkpoint_identity,
                    split=arguments.split,
                    generation_seed=int(arguments.generation_seed),
                    expected_route=expected_route,
                    merged_rows=merged_rows,
                    evaluation_domain=evaluation_domain or {},
                    metric_policy=sealed_metric_policy,
                    run_diagnostics=bool(getattr(arguments, "eval_diagnostics", True)),
                    no_progress=bool(arguments.no_progress),
                    evaluation_local_row_batch=int(
                        arguments.route1_eval_local_row_batch
                    ),
                    **{"route1_answer_max_tokens": route1_answer_max_tokens},
                )
            report["epoch"] = int(sampler_state["epoch"])
            report["route1_null_mode"] = route1_null_mode
            report.update(
                {
                    "route1_population": str(
                        sampler_state.get("route1_population", "staged")
                    ),
                    "route1_course_weight": float(config.route1_course_weight),
                    "route1_match_weight": float(config.route1_match_weight),
                    "route1_specific_weight": float(config.route1_specific_weight),
                    "route1_max_optimizer_updates": sampler_state.get(
                        "route1_max_optimizer_updates"
                    ),
                    "route1_eval_backend": str(
                        getattr(arguments, "route1_eval_backend", "torch")
                    ),
                }
            )
            _write_json_transactional(
                Path(arguments.report),
                report,
                transaction_id=transaction.transaction_id,
            )
            print_evaluation_examples(report, report_path=Path(arguments.report))
            rank0_result = {
                "error": None,
                "report_sha256": file_sha256(Path(arguments.report)),
            }
        except Exception as exc:
            rank0_result = {
                "error": f"{type(exc).__name__}: {exc}",
                "report_sha256": None,
            }
    result = broadcast_rank0_result(distributed, rank=rank, local_result=rank0_result)
    if not isinstance(result, Mapping) or result.get("error") is not None:
        raise RuntimeError(f"distributed evaluation publication failed: {result}")
    if distributed.is_initialized():
        distributed.barrier()
    return report_path


def _build_distributed_evaluation_report(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    donors: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    config: TrainingConfig,
    manifest: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_identity: BridgeCheckpointIdentity,
    split: str,
    generation_seed: int,
    expected_route: str,
    merged_rows: Sequence[Mapping[str, Any]],
    evaluation_domain: Mapping[str, Any],
    metric_policy: Mapping[str, Any],
    no_progress: bool,
    evaluation_local_row_batch: int,
    route1_answer_max_tokens: int | None = None,
    run_diagnostics: bool = True,
) -> dict[str, Any]:
    """Assemble one rank-0 report while the caller holds evaluation mode."""
    if bool(model.training):
        raise RuntimeError("distributed evaluation report requires model.eval()")
    from think_bridge.training.metric_registry import report_metric_policy

    metric_policy = report_metric_policy(
        route=expected_route, value=metric_policy
    ).as_dict()
    merged_payload = [row["payload"] for row in merged_rows]
    for index, value in enumerate(merged_payload):
        if (
            not isinstance(value, Mapping)
            or value.get("record_id") != rows[index]["record_id"]
            or value.get("prompt_group_id") != rows[index]["prompt_group_id"]
        ):
            raise ValueError("distributed evaluation row/source identity changed")
    if donors is None:
        raise ValueError("Route1 evaluation requires its donor bundle")
    if (
        isinstance(route1_answer_max_tokens, bool)
        or not isinstance(route1_answer_max_tokens, int)
        or route1_answer_max_tokens <= 0
    ):
        raise ValueError("Route1 report requires an answer horizon")
    return _evaluate_route1(
        model,
        tokenizer,
        rows,
        donors,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_identity=checkpoint_identity,
        split=split,
        generation_seed=int(generation_seed),
        answer_max_tokens=int(route1_answer_max_tokens),
        no_progress=True,
        evaluation_domain=evaluation_domain,
        metric_policy=metric_policy,
        paired_rows_override=merged_payload,
        config=config,
        run_diagnostics=run_diagnostics,
    )


def run_selection(arguments: Any) -> int:
    from think_bridge.training.metric_registry import (
        metric_policy_from_mapping,
        require_report_metric_policy,
        resolve_metric_policy,
        select_best_candidate,
    )

    entrance = {"select-route1": "reasoner-sft"}.get(arguments.command)
    if entrance is None:
        raise ValueError(
            f"unsupported ThinkBridge selection command: {arguments.command}"
        )
    policy_value = getattr(arguments, "metric_policy", None)
    policy = (
        resolve_metric_policy(
            entrance=entrance, metric_for_best_model=None, metric_aggregation="mean"
        )
        if policy_value is None
        else metric_policy_from_mapping(policy_value)
    )
    require_manifest_fields(_read_json(Path(arguments.manifest)))
    if arguments.command in {"select-route1"}:
        expected_schema = ROUTE1_VALIDATION_REPORT_SCHEMA_VERSION
        expected_route = "route1"
        report_pairs = [(path, _read_json(path)) for path in arguments.reports]
        reports = [report for (_, report) in report_pairs]
        if any(
            (
                require_report_metric_policy(report, route=expected_route) != policy
                for report in reports
            )
        ):
            raise ValueError(
                "selection report metric policy differs from the sealed selector"
            )
        report_steps = [int(report.get("step", -1)) for report in reports]
        if not report_steps or report_steps != sorted(set(report_steps)):
            raise ValueError("Route1 selection reports must be nonempty and ordered")
        if any(
            (
                report.get("artifact_type") != expected_schema
                or report.get("schema_version") != 1
                or report.get("objective_version") != OBJECTIVE_VERSION
                or (report.get("split") != "validation")
                or (report.get("free_generation") is not True)
                or (report.get("method") != "bridge")
                or (report.get("route") != expected_route)
                or (int(report.get("seed", -1)) != int(arguments.seed))
                for report in reports
            )
        ):
            raise ValueError(
                "Route1-primary selector received a foreign validation report"
            )
        randomness = validation_randomness_identity(
            route="route1", base_seed=int(arguments.generation_seed)
        )
        if any(
            (report.get("validation_randomness") != randomness for report in reports)
        ):
            raise ValueError(
                "Route1 selector requires one sealed common-randomness identity"
            )
        evaluation_domains = [report.get("evaluation_domain") for report in reports]
        if (
            any((not isinstance(value, Mapping) for value in evaluation_domains))
            or len({canonical_json_sha256(value) for value in evaluation_domains}) != 1
            or any(
                (
                    value.get("actual_sample_count")
                    != report["gate_metrics"].get("true_full_total")
                    or require_sha256(
                        str(value.get("evaluation_subset_sha256", "")),
                        "evaluation_subset_sha256",
                    )
                    != value.get("evaluation_subset_sha256")
                    for (value, report) in zip(evaluation_domains, reports)
                )
            )
        ):
            raise ValueError(
                "Route1 selector reports do not share one evaluation subset"
            )
        direct_baselines = [report.get("direct_baseline") for report in reports]
        if (
            any((not isinstance(value, Mapping) for value in direct_baselines))
            or len(
                {
                    require_sha256(
                        str(value.get("identity_sha256", "")),
                        "direct_baseline_identity_sha256",
                    )
                    for value in direct_baselines
                }
            )
            != 1
        ):
            raise ValueError("Route1 selector reports do not share one direct baseline")
        backends = {report.get("route1_eval_backend") for report in reports}
        if len(backends) != 1 or not backends.issubset({"torch", "vllm"}):
            raise ValueError("Route1 selector reports must use one evaluation backend")
        run_dir = Path(arguments.output).resolve(strict=False).parent
        report_ledger = []
        for path, report in report_pairs:
            resolved = Path(path).resolve(strict=True)
            try:
                relative = resolved.relative_to(run_dir)
            except ValueError as exc:
                raise ValueError("Route1 candidate report escaped its run") from exc
            report_ledger.append(
                {
                    "path": relative.as_posix(),
                    "sha256": file_sha256(resolved),
                    "step": int(report["step"]),
                    "epoch": int(report["epoch"]),
                }
            )
        normalized_gate_metrics = validate_route1_selection_candidates(
            [report["gate_metrics"] for report in reports]
        )
        for report, metrics in zip(reports, normalized_gate_metrics):
            report["gate_metrics"] = dict(metrics)
        decision = select_best_candidate(
            reports, evaluator="stage1.route1", policy=policy
        )
        chosen = [
            pair for pair in report_pairs if int(pair[1]["step"]) == decision.step
        ]
        if len(chosen) != 1:
            raise ValueError("Route1 selected step does not identify one report")
        (chosen_path, chosen_report) = chosen[0]
        checkpoint_path = resolve_run_artifact_locator(
            run_dir,
            chosen_report["checkpoint_path"],
            label="selected Route1 checkpoint path",
        )
        if (
            checkpoint_artifact_sha256(checkpoint_path)
            != chosen_report["checkpoint_sha256"]
        ):
            raise ValueError("selected Route1 checkpoint artifact hash changed")
        seal = {
            **artifact_header(ROUTE1_SELECTION),
            "objective_version": OBJECTIVE_VERSION,
            "method": "bridge",
            "route": expected_route,
            "seed": int(arguments.seed),
            "selected_r_sha256": chosen_report["checkpoint_sha256"],
            "selection_metrics": dict(chosen_report["gate_metrics"]),
            "metric_policy": policy.as_dict(),
            "raw_metrics": dict(decision.raw_metrics),
            "normalized_metrics": dict(decision.normalized_metrics),
            "aggregate_score": float(decision.aggregate_score),
            "tie_break": dict(decision.tie_break),
            "validation_report_sha256": file_sha256(chosen_path),
            "validation_randomness": randomness,
            "evaluation_domain": dict(evaluation_domains[0]),
            "direct_baseline_identity_sha256": direct_baselines[0]["identity_sha256"],
            "candidate_reports": report_ledger,
            "candidate_reports_sha256": canonical_json_sha256(report_ledger),
        }
        _write_selection_result(Path(arguments.output), seal)
        return 0
    raise AssertionError("unreachable ThinkBridge selection command")
