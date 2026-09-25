# Included data

Both Stage0 and Stage1 training data are distributed as small format examples only. The complete training pool, generated rollouts, behavior/direct artifacts, and training validation data are not bundled. The full local research copies remain in the separate `think_bridge/data/` workspace. None of those files are moved or modified by this release.

## Training examples

`stage0/examples/Qwen3-0.6B/native.json` and `stage0/examples/Qwen3-4B/native.json` each contain eight question groups and eight paired native rollouts per group (64 records). They total about 1.63 MB. These are anonymized excerpts for inspecting the format, not the complete data required by the default training commands. Training launchers do not silently fall back to these samples.

`stage1/examples/{Qwen3-0.6B,Qwen3-4B}/` contains a five-file example of the Stage1 inputs: `native.json`, `behavior.json`, `direct.json`, `validation.json`, and `validation_behavior.json`. Each model has the same 64 sampled rollouts, eight matching training behavior/direct records, and eight held-out validation questions with matching native/direct behavior. The records are excerpts of the real local artifacts. Sample manifest metadata identifies the subset and retains a SHA-256 of its parent artifact; full-source counts and fingerprints are not misrepresented as subset metadata.

Stage1 examples show the training **input** format. Compiled token targets, optimizer state, and checkpoints are generated locally and are not included. Tiny examples do not constitute a full training run, and the default global batch may exceed their size.

For a full run, supply your train/validation QA files under `data/raw/` and run `examples/prepare_stage0.sh`, or point `DATA_ROOT` to your already prepared model-specific artifacts. The preparation procedure and required files are documented in the root README. Locally added raw data and complete Stage0 artifacts remain Git-ignored.

All bundled JSON data (both training-example stages plus the tests) totals about 6.58 MB.

## Benchmark tests

The following project test files are included in full, copied byte-for-byte from the research workspace without sampling or rewriting. Together they occupy about 3.04 MB.

| File under `benchmark_dataset/` | Records | Use |
|---|---:|---|
| `gsm_test.json` | 1,319 | GSM test; default independent evaluation input |
| `gsmhard_test.json` | 1,319 | GSM-Hard test |
| `math500_test.json` | 500 | MATH-500 test |
| `multiarith_test.json` | 180 | MultiArith test |
| `svamp_test.json` | 300 | SVAMP test |
| `mathchat_follow_up_test.json` | 3,924 turns / 1,308 conversations | Multi-turn MathChat data; requires a multi-turn evaluator |

The public `eval` command supports the first five as single-turn math QA. It uses the question as model input and the reference answer for judging; retained `cot`/`steps` columns are not fed into the answer path. Use the multi-turn block in `examples/evaluate.sh` for MathChat. It retains complete previous questions and generated replies and computes fresh z each turn; evaluating rows independently does not reproduce this protocol.

To select another single-turn benchmark:

```bash
TEST_DATASET=data/benchmark_dataset/math500_test.json bash examples/evaluate.sh
```

The bundled MultiArith file retains its original 180 rows, including two repeated question/answer pairs. Evaluation keeps those rows in the denominator; wrong-z donors are distinct prompts. No benchmark rows are removed. Benchmark tests are evaluation inputs and are not defaults for training or training validation. Dataset rights remain with the upstream dataset owners; the repository's software license does not relicense these datasets.
