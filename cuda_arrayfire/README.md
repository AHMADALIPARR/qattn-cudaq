# Q-ATTN-01 CUDA + ArrayFire Engine

Real implementation. CUDA kernels do the gate math. ArrayFire owns the GPU arrays.

## What this is

- `qattn_kernels.cu` — CUDA kernels:
  - `apply_ry_kernel` — Ry(θ) on a target qubit of a 4-qubit statevector
  - `apply_cnot_kernel` — CNOT between control and target
  - `fidelity_kernel` — |⟨ψ|φ⟩|² via shared-memory reduction
- `qattn_engine.cpp` — ArrayFire host:
  - Allocates statevectors with `af::array`
  - Passes raw device pointers to the CUDA kernels
  - Runs the feature circuit (Ry × 4 + CNOT chain)
  - Computes fidelity attention and classical value aggregation

No Python. No imported quantum libraries. The gate application is CUDA.

## Build

```bash
mkdir build && cd build
cmake .. -DArrayFire_DIR=/path/to/arrayfire/share/ArrayFire/cmake
cmake --build .
./qattn_engine
```

Requires:
- CUDA Toolkit (nvcc)
- ArrayFire with CUDA backend (`afcuda`)
- GPU with compute capability ≥ 8.0 (adjust `-arch` in CMakeLists.txt if needed)

## License

SnapKitty Opaque AGPL-3.0  
Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective  
SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
