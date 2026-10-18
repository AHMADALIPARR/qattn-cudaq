# Q-ATTN-01

Native quantum-attention runtime. C++20, CUDA, ArrayFire, CUDA-Q.

Production path: [`cuda_arrayfire/`](cuda_arrayfire/)

Python prototypes are archived under [`archive/python/`](archive/python/) and are not part of the build.

## Build

```bash
cd cuda_arrayfire
g++ -std=c++20 -Iinclude -o qattn_runtime \
    src/main.cpp src/statevector.cpp src/attention.cpp src/entropy.cpp
./qattn_runtime
```

Or with CMake (enables CUDA / ArrayFire / CUDA-Q when found):

```bash
cd cuda_arrayfire
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Status

CPU reference: verified. CUDA, ArrayFire, CUDA-Q, and live HTTP are compiled only when their toolkits are present. See `cuda_arrayfire/docs/`.

## License

SnapKitty Opaque AGPL-3.0  
Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective  
SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
