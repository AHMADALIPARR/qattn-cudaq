# Q-ATTN-01 Native Runtime

C++20 / CUDA / ArrayFire / CUDA-Q. No Python on the production path.

## Layout

```
include/qattn/     public headers
src/               CPU reference, attention, entropy, main
cuda/kernels.cu    GPU kernels (built if CUDA toolkit found)
arrayfire/         ArrayFire host (built if ArrayFire + CUDA found)
cudaq/             CUDA-Q circuits (built if CUDA-Q found)
tests/             CPU reference tests
docs/              migration, performance, quantum backend reports
NATIVE_AUDIT.md    pre-migration audit
```

## Attention definition

theta_i = embedding[i] * pi
|q> = Ry(theta) x 4 + CNOT chain
s_ij = |<q_i|q_j>|^2
a_ij = exp(beta * s_ij) / sum_k exp(beta * s_ik)

## Build (CPU reference, no GPU required)

```bash
g++ -std=c++20 -Iinclude -o qattn_runtime \
    src/main.cpp src/statevector.cpp src/attention.cpp src/entropy.cpp
./qattn_runtime

g++ -std=c++20 -Iinclude -o test_cpu_reference \
    tests/test_cpu_reference.cpp src/statevector.cpp src/attention.cpp src/entropy.cpp
./test_cpu_reference
```

## Build (CMake, enables optional backends)

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

CUDA targets sm_86 (RTX 3080). ArrayFire and CUDA-Q are compiled only when detected. Network entropy requires libcurl and ANU_API_KEY for the modern endpoint.

## Status in the authoring environment

CPU reference: verified, 0 test failures.
CUDA, ArrayFire, CUDA-Q, live HTTP: BLOCKED (toolchains not present). See docs/.

## License

SnapKitty Opaque AGPL-3.0
Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective
SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
