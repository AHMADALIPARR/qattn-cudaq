# MIGRATION_REPORT.md

Python was removed from the production execution path. The native runtime is qattn_runtime.

| Component | Before | After |
|---|---|---|
| Entrypoint | python main.py | qattn_runtime |
| Feature circuit | CUDA-Q Python or NumPy | feature_encode in src/statevector.cpp; CUDA kernel when toolkit present |
| Attention | NumPy | fidelity_attention in src/attention.cpp |
| Entropy | requests | src/entropy.cpp; OS crypto always; ANU/NIST via libcurl when built |
| Build | none | CMake; CPU path also builds with g++ alone |

Python files remain in the repository root as historical provenance. They are not inputs to the native CMake graph.

Attention mathematics unchanged. No algorithmic substitution.

## Verified in this environment

- CPU reference compiles with g++ -std=c++20.
- test_cpu_reference reports 0 failures (14 assertions).
- qattn_runtime produces finite, row-normalized attention.
- OS crypto entropy returns labeled non-quantum bytes.
- Network providers return explicit BLOCKED when libcurl is absent.
- Native binary does not link Python.

## Not verified (BLOCKED)

| Capability | Missing | Tests |
|---|---|---|
| CUDA kernels | nvcc | B, H |
| ArrayFire | ArrayFire + CUDA | C |
| CUDA-Q | CUDA-Q package | D |
| Live ANU/NIST | libcurl-dev + network | G live |
| CMake configure | cmake not installed | A CMake path |
