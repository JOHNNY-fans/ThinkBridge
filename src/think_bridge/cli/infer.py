"""Generate a latent-conditioned answer."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="think-bridge infer")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--reasoner_checkpoint", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--local_device", default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 2048:
        parser.error("answer budget must be within 1..2048")
    import torch
    from think_bridge.model.inference import _build_runtime
    from think_bridge.data.templates import build_thinking_prompt
    from think_bridge.model.reasoner_inference import reason_eval_padded
    from think_bridge.model.trajectory import generate_true_z_prefix

    args.attn_implementation = "sdpa"
    runtime = _build_runtime(args)
    model, tokenizer, device = runtime.model, runtime.tokenizer, runtime.device
    prompt = build_thinking_prompt(tokenizer, args.question, task_type="math")
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        z = reason_eval_padded(model, ids, mask)
        generated = generate_true_z_prefix(
            model.executor,
            embedding=model.executor.get_input_embeddings(),
            prompt_ids=ids,
            prompt_mask=mask,
            z=z,
            boundary_ids=model.boundary_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_steps=args.max_new_tokens,
            temperature=0.0,
            seed=args.seed,
        )
        answer_ids = generated.token_ids[0][generated.token_mask[0].bool()].tolist()
        result = {
            "question": args.question,
            "answer": tokenizer.decode(answer_ids, skip_special_tokens=True),
            "answer_token_ids": answer_ids,
            "answer_cap_hit": answer_ids[-1] != tokenizer.eos_token_id,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
