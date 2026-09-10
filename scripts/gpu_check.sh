#!/usr/bin/env bash
# One-shot CUDA validation on a rented / borrowed GPU box.
#
#   ssh gpu-box
#   git clone <repo> && cd bpdecode
#   python -m venv .venv && . .venv/bin/activate
#   ./scripts/gpu_check.sh
#
# Assumes: NVIDIA GPU, CUDA toolkit (nvcc) on PATH, Python 3.10+.
set -euo pipefail

nvidia-smi
nvcc --version

echo "== C++ core (CUDA) + gtests =="
cmake -S csrc -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp -j
ctest --test-dir build-cpp --output-on-failure

echo "== torch extension with the CUDA path =="
python -m pip install --upgrade pip
# pick the wheel matching your driver; cu124 works on most recent boxes
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install scikit-build-core cmake ninja
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native" \
  pip install --no-build-isolation -e '.[dev]'

echo "== CUDA vs CPU differential =="
pytest tests/test_cuda.py -v

echo "== compute-sanitizer =="
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest tests/test_cuda.py -q

echo "OK -- CUDA kernels match the CPU reference."
