#!/usr/bin/env bash
# One-shot CUDA validation on a rented / borrowed GPU box (e.g. RunPod).
#
#   git clone https://github.com/zanarashidi/bpdecode.git && cd bpdecode
#   ./scripts/gpu_check.sh
#
# Assumes: NVIDIA GPU, CUDA toolkit (nvcc) on PATH, Python 3.10+, a working
# `torch` (the RunPod PyTorch template ships one -- it is used as-is).
set -euo pipefail

echo "== environment =="
nvidia-smi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Install the CUDA toolkit (e.g. 'apt-get install -y cuda-toolkit')"
  echo "or use a RunPod template with the CUDA devel toolkit, then re-run."
  exit 1
fi
nvcc --version

echo "== C++ core (CUDA) + gtests =="
cmake -S csrc -B build-cpp -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cpp -j
ctest --test-dir build-cpp --output-on-failure

echo "== torch extension with the CUDA path =="
python -m pip install --upgrade pip
if ! python -c "import torch" 2>/dev/null; then
  # no preinstalled torch -- grab a CUDA 12.x wheel (works against newer drivers)
  pip install torch --index-url https://download.pytorch.org/whl/cu124
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
pip install scikit-build-core cmake ninja pytest numpy
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native" \
  pip install --no-build-isolation -e .

echo "== CUDA vs CPU differential =="
pytest tests/test_cuda.py tests/test_pda_cuda.py -v

echo "== compute-sanitizer (memcheck) =="
if command -v compute-sanitizer >/dev/null 2>&1; then
  compute-sanitizer --tool memcheck --error-exitcode 1 \
    python -m pytest tests/test_cuda.py tests/test_pda_cuda.py -q
else
  echo "compute-sanitizer not on PATH; skipping (differential above still validates correctness)"
fi

echo "== benchmarks on GPU =="
pip install -q 'transformers>=4.40' outlines xgrammar llguidance jsonschema lark || true
python bench/regex_mask.py --device cuda --batch 64 || echo "(regex bench skipped)"
python bench/json_schema_mask.py || echo "(json-schema bench skipped)"

echo
echo "OK -- CUDA kernels match the CPU reference. Bench output above."
