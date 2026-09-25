#!/usr/bin/env bash
# Linux x86_64, Python 3.10+, CUDA 12.9 toolkit and a compatible NVIDIA driver.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install 'setuptools>=64' wheel ninja
"$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu129 'torch==2.11.0'
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip install 'https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=8bc2203995d061e6b988916b71b9dee8a5970f5fdc5f37d4445a877a2fab2cc1'
DS_BUILD_OPS=0 "$PYTHON_BIN" -m pip install --no-build-isolation 'deepspeed==0.17.6'
"$PYTHON_BIN" -m pip install --no-deps .
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -m think_bridge.cli.main --help
