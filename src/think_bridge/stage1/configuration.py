"""Typed configuration for the maintained ThinkBridge training course."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping
from think_bridge.model.contract import (
    OBJECTIVE_VERSION,
    CAUSAL_OBJECTIVE_SCHEMA_VERSION,
)
from think_bridge.model.feedback_precision import (
    QUESTION_READER_INPUT_MODES,
    REASONER_INPUT_MODES,
)

from think_bridge.stage1.methods.bridge.recipe import (
    METHOD_VERSION,
)


@dataclass(frozen=True)
class BridgeConfig:
    schema_version: int
    method: str
    objective_version: str
    route1_objective_version: str
    geometry_schema_version: str
    model_family: str
    model_name_or_path: str
    tokenizer_name_or_path: str
    attn_implementation: str
    reasoner_compute_dtype: str
    reasoner_eval_group_size: int
    seeds: tuple[int, ...]
    latent_slots: int
    latent_steps: int
    latents_per_step: int
    emitter_depth: int
    tap_count: int
    real_frozen_f_appends: int
    latent_bound_scale_init: float
    reasoner_loop_steps: int
    reasoner_zero_init: bool
    reasoner_input_mode: str
    reasoner_output_normalization: str
    reasoner_dropout_p: float
    reasoner_dropout_views: int
    boundary_text: str
    cot_content_capacity: int
    answer_capacity: int
    output_root: str
    cache_root: str
    report_root: str
    checkpoint_root: str
    lm_head_vocab_chunk_size: int
    route1_epochs: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    save_total_limit: int
    generation_seed: int
    route1_generation_temperature: float
    world_size: int
    trainer_gpu_ids: tuple[int, ...]
    route1_world_size: int
    route1_trainer_gpu_ids: tuple[int, ...]
    route1_trainer_master_port: int
    route1_generation_backend: str
    route1_vllm_gpu_ids: tuple[int, ...]
    route1_vllm_data_parallel_size: int
    route1_vllm_tensor_parallel_size: int
    route1_vllm_host: str
    route1_vllm_port: int
    route1_vllm_startup_timeout_seconds: float
    route1_vllm_request_timeout_seconds: float
    route1_vllm_shutdown_timeout_seconds: float
    route1_vllm_backpressure_timeout_seconds: float
    route1_vllm_watchdog_interval_seconds: float
    route1_vllm_physical_chunk_size: int
    route1_vllm_max_in_flight: int
    route1_vllm_max_queued_requests: int
    route1_vllm_max_pending_microsteps: int
    route1_vllm_gpu_memory_utilization: float
    route1_vllm_enforce_eager: bool
    route1_vllm_max_num_seqs: int
    route1_local_samples: int
    route1_local_chunk_size: int
    route1_wrong_control_chunk_size: int
    route1_specificity_donors_per_owner: int
    route1_specificity_wrong_gradient: str
    route1_specificity_include_direct: bool
    route1_distillation_populations: str
    route1_gradient_accumulation_steps: int
    route1_gradient_checkpointing: bool
    route1_eval_local_row_batch: int
    route1_course_steps: int | None
    route1_course_epochs: float
    route1_course_weight: float
    route1_match_weight: float
    route1_specific_weight: float
    route1_specificity_tau: float
    route1_specificity_loss: str
    route1_specificity_margin: float
    route1_specificity_temperature: float
    route1_specificity_negative_kl_cap: float | None
    lr_r: float
    weight_decay: float
    max_grad_norm_r: float
    warmup_updates_r: int
    optimizer_backend: str
    zero_stage: int
    deepspeed_config: str | None
    deepspeed_version_spec: str

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        model_name_or_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        runtime_overrides: Mapping[str, Any] | None = None,
        stage_kind: str | None = None,
    ) -> "BridgeConfig":
        values = dict(raw)
        values.setdefault("reasoner_compute_dtype", "bfloat16")
        values.setdefault("reasoner_eval_group_size", 1)
        values.setdefault("reasoner_loop_steps", 2)
        values.setdefault("reasoner_zero_init", False)
        values.setdefault("reasoner_input_mode", "last-query-reader-self-loop")
        values.setdefault("reasoner_output_normalization", "residual")
        values.setdefault("reasoner_dropout_p", 0.0)
        values.setdefault("reasoner_dropout_views", 2)
        values.setdefault("route1_specificity_donors_per_owner", 2)
        values.setdefault("route1_course_steps", 0)
        values.setdefault("route1_course_epochs", 0.0)
        if values["route1_course_epochs"] is None:
            values["route1_course_epochs"] = 0.0
        values.setdefault("route1_specificity_wrong_gradient", "live")
        values.setdefault("route1_specificity_include_direct", False)
        values.setdefault("route1_distillation_populations", "BC")
        values.setdefault("route1_specificity_tau", 0.1)
        values.setdefault("route1_specificity_loss", "same-prompt-capped-soft-infonce")
        values.setdefault("route1_specificity_margin", 0.1)
        values.setdefault("route1_specificity_temperature", 0.1)
        values.setdefault("route1_specificity_negative_kl_cap", 0.2)
        values.setdefault("route1_generation_temperature", 0.0)
        if stage_kind not in {None, "reasoner-sft"}:
            raise ValueError(f"unsupported private Bridge stage kind: {stage_kind}")
        if model_name_or_path is not None:
            values["model_name_or_path"] = str(model_name_or_path)
            if tokenizer_name_or_path is None:
                values["tokenizer_name_or_path"] = str(model_name_or_path)
        if tokenizer_name_or_path is not None:
            values["tokenizer_name_or_path"] = str(tokenizer_name_or_path)
        allowed_runtime = {
            "reasoner_compute_dtype",
            "reasoner_eval_group_size",
            "seed",
            "generation_seed",
            "route1_generation_temperature",
            "save_steps",
            "eval_steps",
            "logging_steps",
            "save_total_limit",
            "route1_epochs",
            "route1_local_samples",
            "route1_local_chunk_size",
            "route1_wrong_control_chunk_size",
            "route1_specificity_donors_per_owner",
            "route1_specificity_wrong_gradient",
            "route1_specificity_include_direct",
            "route1_distillation_populations",
            "route1_gradient_accumulation_steps",
            "route1_gradient_checkpointing",
            "route1_eval_local_row_batch",
            "lr_r",
            "weight_decay",
            "max_grad_norm_r",
            "warmup_updates_r",
            "route1_course_steps",
            "route1_course_epochs",
            "route1_course_weight",
            "route1_match_weight",
            "route1_specific_weight",
            "route1_specificity_tau",
            "route1_specificity_loss",
            "route1_specificity_margin",
            "route1_specificity_temperature",
            "route1_specificity_negative_kl_cap",
            "latent_slots",
            "latent_steps",
            "latents_per_step",
            "real_frozen_f_appends",
            "geometry_schema_version",
            "reasoner_loop_steps",
            "emitter_depth",
            "reasoner_zero_init",
            "reasoner_input_mode",
            "tap_count",
            "reasoner_output_normalization",
            "reasoner_dropout_p",
            "reasoner_dropout_views",
            "world_size",
            "trainer_gpu_ids",
            "route1_world_size",
            "route1_trainer_gpu_ids",
            "route1_trainer_master_port",
            "route1_generation_backend",
            "route1_vllm_gpu_ids",
            "route1_vllm_data_parallel_size",
            "route1_vllm_tensor_parallel_size",
            "route1_vllm_host",
            "route1_vllm_port",
            "route1_vllm_startup_timeout_seconds",
            "route1_vllm_request_timeout_seconds",
            "route1_vllm_shutdown_timeout_seconds",
            "route1_vllm_backpressure_timeout_seconds",
            "route1_vllm_watchdog_interval_seconds",
            "route1_vllm_physical_chunk_size",
            "route1_vllm_max_in_flight",
            "route1_vllm_max_queued_requests",
            "route1_vllm_max_pending_microsteps",
            "route1_vllm_gpu_memory_utilization",
            "route1_vllm_enforce_eager",
            "route1_vllm_max_num_seqs",
        }
        for key, value in dict(runtime_overrides or {}).items():
            if key not in allowed_runtime:
                raise ValueError(f"unknown Bridge runtime override: {key}")
            values["seeds" if key == "seed" else key] = (
                [value] if key == "seed" else value
            )
        runtime = dict(runtime_overrides or {})
        for route in ("route1",):
            world_field = f"{route}_world_size"
            gpu_field = f"{route}_trainer_gpu_ids"
            if world_field not in runtime:
                if "world_size" in runtime:
                    values[world_field] = values["world_size"]
                else:
                    values.setdefault(world_field, values.get("world_size"))
            if gpu_field not in runtime:
                if "trainer_gpu_ids" in runtime:
                    values[gpu_field] = values["trainer_gpu_ids"]
                else:
                    values.setdefault(gpu_field, values.get("trainer_gpu_ids"))
        values.setdefault(
            "route1_wrong_control_chunk_size", values.get("route1_local_chunk_size")
        )
        values["objective_version"] = OBJECTIVE_VERSION
        values["route1_objective_version"] = CAUSAL_OBJECTIVE_SCHEMA_VERSION
        values["geometry_schema_version"] = (
            f"recursive-t{values.get('latent_steps')}-b{values.get('latents_per_step')}-k{values.get('latent_slots')}-answer-only"
        )
        required = set(cls.__dataclass_fields__)
        missing = sorted(required.difference(values))
        unknown = sorted(set(values).difference(required))
        if missing or unknown:
            raise ValueError(
                f"Bridge config schema mismatch; missing={missing}, unknown={unknown}"
            )
        seeds = values["seeds"]
        if not isinstance(seeds, (list, tuple)):
            raise ValueError("Bridge seeds must be a sequence")
        values["seeds"] = tuple(seeds)
        for key in ("trainer_gpu_ids", "route1_trainer_gpu_ids", "route1_vllm_gpu_ids"):
            raw_ids = values[key]
            if not isinstance(raw_ids, (list, tuple)):
                raise ValueError(f"Bridge {key} must be a sequence")
            values[key] = tuple(raw_ids)
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        required = {
            "reasoner_input_mode": "last-query-reader-self-loop",
            "latent_steps": 1,
            "latent_slots": 64,
            "latents_per_step": 64,
            "objective_version": OBJECTIVE_VERSION,
            "route1_objective_version": CAUSAL_OBJECTIVE_SCHEMA_VERSION,
            "emitter_depth": 2,
            "reasoner_loop_steps": 2,
            "tap_count": 1,
            "reasoner_zero_init": False,
            "reasoner_output_normalization": "residual",
            "route1_specificity_loss": "same-prompt-capped-soft-infonce",
            "route1_specificity_wrong_gradient": "live",
            "route1_specificity_include_direct": False,
            "route1_distillation_populations": "BC",
            "route1_course_epochs": 0.0,
            "route1_course_steps": 0,
        }
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise ValueError(f"unsupported {name}: expected {expected!r}")
        from think_bridge.model.feedback_precision import feedback_policy

        feedback_policy(self)
        if (
            isinstance(self.reasoner_loop_steps, bool)
            or not isinstance(self.reasoner_loop_steps, int)
            or self.reasoner_loop_steps < 1
        ):
            raise ValueError("reasoner_loop_steps must be a positive integer")
        if self.reasoner_input_mode not in REASONER_INPUT_MODES:
            raise ValueError("unknown reasoner_input_mode")
        if self.reasoner_input_mode in QUESTION_READER_INPUT_MODES and (
            self.latent_steps != 1 or self.reasoner_zero_init
        ):
            raise ValueError(
                "question reader requires one external step and nonzero initialization"
            )
        if self.reasoner_output_normalization not in {"residual", "rmsnorm"}:
            raise ValueError(
                "reasoner_output_normalization must be residual or rmsnorm"
            )
        if not isinstance(self.reasoner_zero_init, bool):
            raise ValueError("reasoner_zero_init must be a boolean")
        if (
            isinstance(self.reasoner_dropout_p, bool)
            or not math.isfinite(float(self.reasoner_dropout_p))
            or (not 0.0 <= float(self.reasoner_dropout_p) < 1.0)
        ):
            raise ValueError("reasoner_dropout_p must be finite in [0, 1)")
        if (
            isinstance(self.reasoner_dropout_views, bool)
            or not isinstance(self.reasoner_dropout_views, int)
            or self.reasoner_dropout_views < 1
            or (self.reasoner_dropout_views > 8)
        ):
            raise ValueError("reasoner_dropout_views must be an integer in [1, 8]")
        from think_bridge.training.specificity_sampling import validate_donor_limit

        validate_donor_limit(self.route1_specificity_donors_per_owner)
        if self.route1_specificity_wrong_gradient not in {"live", "stop"}:
            raise ValueError("specificity_wrong_gradient must be live or stop")
        if not isinstance(self.route1_specificity_include_direct, bool):
            raise ValueError("specificity_include_direct must be a boolean")
        if self.route1_distillation_populations not in {"C", "BC", "G"}:
            raise ValueError("route1_distillation_populations must be C or BC")
        if (
            not math.isfinite(float(self.route1_specificity_tau))
            or self.route1_specificity_tau <= 0
        ):
            raise ValueError("specificity_tau must be finite and positive")
        if self.route1_specificity_loss != "same-prompt-capped-soft-infonce":
            raise ValueError("unsupported specificity loss")
        if (
            not math.isfinite(float(self.route1_specificity_margin))
            or self.route1_specificity_margin < 0
        ):
            raise ValueError("specificity_margin must be finite and nonnegative")
        if (
            not math.isfinite(float(self.route1_specificity_temperature))
            or self.route1_specificity_temperature <= 0
        ):
            raise ValueError("specificity_temperature must be finite and positive")
        cap = self.route1_specificity_negative_kl_cap
        if cap is not None and (not math.isfinite(float(cap)) or cap <= 0):
            raise ValueError("specificity_negative_kl_cap must be finite and positive")
        if (
            self.route1_specificity_loss
            in {
                "capped-soft-infonce",
                "normalized-capped-soft-infonce",
                "same-prompt-capped-soft-infonce",
            }
            and cap is None
        ):
            raise ValueError("capped soft InfoNCE requires specificity_negative_kl_cap")
        if (
            not math.isfinite(float(self.route1_generation_temperature))
            or self.route1_generation_temperature < 0
        ):
            raise ValueError(
                "route1_generation_temperature must be finite and nonnegative"
            )
        if self.schema_version != 1:
            raise ValueError("configuration schema_version must be 1")
        if self.method != METHOD_VERSION:
            raise ValueError("configuration method must be bridge")
        if self.model_family not in {"qwen3-0.6b", "qwen3-4b"}:
            raise ValueError("Bridge supports only maintained Qwen3 model families")
        if self.attn_implementation != "sdpa":
            raise ValueError("Bridge requires SDPA")
        if not self.seeds or any(
            (isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds)
        ):
            raise ValueError("seeds must be a nonempty integer sequence")
        if (
            self.latent_slots not in {32, 64, 128}
            or self.latent_steps <= 0
            or self.latents_per_step <= 0
            or (self.latent_steps * self.latents_per_step != self.latent_slots)
            or isinstance(self.emitter_depth, bool)
            or (not isinstance(self.emitter_depth, int))
            or (self.emitter_depth < 1)
            or (self.reasoner_loop_steps > 1 and self.emitter_depth < 2)
            or (
                self.tap_count
                != (1 if self.reasoner_input_mode in QUESTION_READER_INPUT_MODES else 6)
            )
            or (
                self.real_frozen_f_appends
                not in {self.latent_steps - 1, self.latent_steps}
            )
        ):
            raise ValueError(
                "Bridge feedback geometry requires K in {32,64,128}, positive R depth, mode-matched F taps, and one real F append per recurrent step"
            )
        expected_geometry = f"recursive-t{self.latent_steps}-b{self.latents_per_step}-k{self.latent_slots}-answer-only"
        if self.geometry_schema_version != expected_geometry:
            raise ValueError(
                "Bridge geometry_schema_version differs from the configured geometry"
            )
        if (
            not math.isfinite(float(self.latent_bound_scale_init))
            or self.latent_bound_scale_init <= 0
        ):
            raise ValueError("Bridge latent bound scale must be finite and positive")
        if (self.cot_content_capacity, self.answer_capacity) != (6144, 2048):
            raise ValueError(
                "Bridge Route1 content capacities differ from the approved design"
            )
        if (
            isinstance(self.route1_epochs, bool)
            or not isinstance(self.route1_epochs, int)
            or self.route1_epochs <= 0
        ):
            raise ValueError("R epochs must be a positive integer")
        from think_bridge.stage1.methods.bridge.recipe import route1_course_updates

        route1_course_updates(
            1, self.route1_course_epochs, course_steps=self.route1_course_steps
        )
        route1_enabled = self.route1_epochs > 0
        route1_weights = (
            self.route1_course_weight,
            self.route1_match_weight,
            self.route1_specific_weight,
        )
        if route1_enabled and any(
            (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not math.isfinite(float(value)))
                or (float(value) < 0.0)
                for value in route1_weights
            )
        ):
            raise ValueError("Route1 loss weights must be finite and nonnegative")
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size <= 0
        ):
            raise ValueError("Bridge world size must be a positive integer")
        gpu_ids = tuple(self.trainer_gpu_ids)
        if (
            len(gpu_ids) != self.world_size
            or any(
                (
                    isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
                    for gpu in gpu_ids
                )
            )
            or len(set(gpu_ids)) != len(gpu_ids)
        ):
            raise ValueError(
                "Bridge trainer GPU ids must be unique nonnegative integers aligned with world size"
            )
        for enabled, route, route_world, route_gpu_ids in (
            (
                route1_enabled,
                "Route1",
                self.route1_world_size,
                tuple(self.route1_trainer_gpu_ids),
            ),
        ):
            if not enabled:
                continue
            if (
                isinstance(route_world, bool)
                or not isinstance(route_world, int)
                or route_world <= 0
                or (len(route_gpu_ids) != route_world)
                or any(
                    (
                        isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
                        for gpu in route_gpu_ids
                    )
                )
                or (len(set(route_gpu_ids)) != len(route_gpu_ids))
            ):
                raise ValueError(
                    f"{route} trainer GPU ids must be unique nonnegative integers aligned with its world size"
                )
        if route1_enabled:
            route1_batch_values = (
                self.route1_local_samples,
                self.route1_gradient_accumulation_steps,
            )
            if any(
                (
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in route1_batch_values
                )
            ):
                raise ValueError("Route1 local samples and GAS must be positive")
        if route1_enabled and self.route1_generation_backend not in {"torch", "vllm"}:
            raise ValueError("Route1 generation backend must be torch or vllm")
        if route1_enabled and self.route1_generation_backend == "vllm":
            service_gpu_ids = tuple(self.route1_vllm_gpu_ids)
            positive_service_ints = (
                self.route1_vllm_data_parallel_size,
                self.route1_vllm_tensor_parallel_size,
                self.route1_vllm_port,
                self.route1_vllm_physical_chunk_size,
                self.route1_vllm_max_in_flight,
                self.route1_vllm_max_queued_requests,
                self.route1_vllm_max_pending_microsteps,
                self.route1_vllm_max_num_seqs,
            )
            positive_service_times = (
                self.route1_vllm_startup_timeout_seconds,
                self.route1_vllm_request_timeout_seconds,
                self.route1_vllm_shutdown_timeout_seconds,
                self.route1_vllm_backpressure_timeout_seconds,
                self.route1_vllm_watchdog_interval_seconds,
            )
            if (
                not service_gpu_ids
                or any(
                    (
                        isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
                        for gpu in service_gpu_ids
                    )
                )
                or len(set(service_gpu_ids)) != len(service_gpu_ids)
                or (
                    len(service_gpu_ids)
                    != int(self.route1_vllm_data_parallel_size)
                    * int(self.route1_vllm_tensor_parallel_size)
                )
            ):
                raise ValueError("Route1 vLLM GPU ids must match DP x TP")
            if set(service_gpu_ids).intersection(self.route1_trainer_gpu_ids):
                raise ValueError("Route1 vLLM and trainer GPU partitions overlap")
            if any(
                (
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in positive_service_ints
                )
            ):
                raise ValueError("Route1 vLLM positive runtime integer is invalid")
            if any(
                (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or (not math.isfinite(float(value)))
                    or (float(value) <= 0.0)
                    for value in positive_service_times
                )
            ):
                raise ValueError("Route1 vLLM timeout is invalid")
            if (
                self.route1_vllm_host != "127.0.0.1"
                or self.route1_vllm_port > 65535
                or self.route1_vllm_port == self.route1_trainer_master_port
                or (
                    self.route1_vllm_physical_chunk_size > self.route1_vllm_max_num_seqs
                )
                or (not 0.0 < float(self.route1_vllm_gpu_memory_utilization) < 1.0)
                or (not isinstance(self.route1_vllm_enforce_eager, bool))
            ):
                raise ValueError("Route1 vLLM runtime topology is invalid")
        positive_ints = {
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "save_total_limit": self.save_total_limit,
        }
        if route1_enabled:
            positive_ints.update(
                {
                    "route1_local_chunk_size": self.route1_local_chunk_size,
                    "route1_wrong_control_chunk_size": self.route1_wrong_control_chunk_size,
                    "route1_eval_local_row_batch": self.route1_eval_local_row_batch,
                    "route1_trainer_master_port": self.route1_trainer_master_port,
                }
            )
        if any(
            (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in positive_ints.values()
            )
        ):
            raise ValueError(
                f"Bridge positive runtime integer is invalid: {positive_ints}"
            )
        if self.optimizer_backend == "replicated_ddp":
            if self.zero_stage != 0 or self.deepspeed_config is not None:
                raise ValueError("replicated DDP cannot carry a DeepSpeed/ZeRO config")
        elif self.optimizer_backend == "deepspeed_zero1":
            if self.zero_stage != 1 or not self.deepspeed_config:
                raise ValueError("DeepSpeed backend requires an exact ZeRO-1 config")
        else:
            raise ValueError("unsupported Bridge backend")

    @property
    def causal_objective_schema_version(self) -> str:
        return self.route1_objective_version

    @property
    def alignment_capacity(self) -> int:
        return self.cot_content_capacity

    @property
    def bound_scale_init(self) -> float:
        return self.latent_bound_scale_init

    @property
    def route1_microstep_global_batch(self) -> int:
        return self.route1_world_size * self.route1_local_samples

    @property
    def route1_service_gpu_ids(self) -> tuple[int, ...]:
        return (
            tuple(self.route1_vllm_gpu_ids)
            if self.route1_generation_backend == "vllm"
            else ()
        )

    @property
    def route1_optimizer_global_batch(self) -> int:
        return (
            self.route1_microstep_global_batch * self.route1_gradient_accumulation_steps
        )

    @property
    def trajectory_max_steps(self) -> int:
        """Route1 native-match uses the full sealed answer horizon."""
        return self.answer_capacity
