# Q-ATTN-01

Native quantum-attention runtime. C++20, CUDA, ArrayFire, CUDA-Q.

![Q-ATTN-01 cover](docs/cover.jpg)

Fidelity attention over a 4-qubit feature circuit. Vacuum-fluctuation entropy from ANU is an input, not a circuit. No Python on the production path.

## What it is

| Piece | Role |
|---|---|
| Feature circuit | `Ry(θ_i)` on qubit `i`, then `CNOT(0,1)`, `CNOT(1,2)`, `CNOT(2,3)` |
| Attention | `s_ij = |⟨q_i|q_j⟩|²`, then `a_ij = exp(β s_ij) / Σ_k exp(β s_ik)` |
| Values | `output_i = Σ_j a_ij · value_j` |
| Entropy | ANU Quantum Numbers, NIST Beacon, OS crypto. Labeled. Not a quantum circuit. |

`θ_i = embedding[i] · π`. Qubit 0 is the LSB. Statevector dimension is 16.

## Where the code is

```
cuda_arrayfire/          production tree
  src/                   CPU reference, attention, entropy, entrypoint
  cuda/kernels.cu        GPU kernels (sm_86, RTX 3080)
  arrayfire/             ArrayFire host, compiled only if found
  cudaq/                 CUDA-Q circuits, compiled only if found
  include/qattn/         public headers
  tests/                 CPU reference tests
  docs/                  audit and reports
LICENSE                  SnapKitty Opaque AGPL-3.0
```

Python prototypes are not in this tree. They remain in git history under `archive/python/`.

## Build

CPU reference, no GPU required:

```bash
cd cuda_arrayfire
g++ -std=c++20 -Iinclude -o qattn_runtime \
    src/main.cpp src/statevector.cpp src/attention.cpp src/entropy.cpp
./qattn_runtime
```

CMake, enables CUDA / ArrayFire / CUDA-Q when detected:

```bash
cd cuda_arrayfire
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Network entropy requires libcurl at build time (`-DQATTN_HAS_LIBCURL` or a found `CURL` package). The modern ANU endpoint needs `ANU_API_KEY` in the environment. The key is never embedded.

## Status

| Path | State |
|---|---|
| CPU reference + tests | Verified. CMake build and `ctest` pass. |
| Live NIST Beacon | Verified. |
| Live ANU legacy | Works when the endpoint is up. Failures are reported. |
| CUDA kernels | Source present. Not executed here (no `nvcc`). |
| ArrayFire | Source present. Not executed here. |
| CUDA-Q | Source present. Not executed here. |
| Physical QPU | Not implemented. Not claimed. |

## License

SnapKitty Opaque AGPL-3.0  
Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective  
SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
