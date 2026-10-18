# Q-ATTN-01

![Q-ATTN-01](docs/cover.jpg)

A classical Transformer decides which tokens matter by a dot product. Q-ATTN-01 decides by the fidelity of two quantum states.

Each token embedding is encoded into a 4-qubit register. Four `Ry` rotations load the features. A CNOT chain entangles them. The attention score between token `i` and token `j` is the overlap of the two resulting states:

```
s_ij = |⟨q_i | q_j⟩|²
```

Those scores are softened and normalized, then used to mix classical value vectors. The circuit is quantum. The aggregation is classical. The whole thing is a hybrid.

```
θ_i = embedding[i] · π
|q⟩ = Ry(θ) × 4, then CNOT(0,1), CNOT(1,2), CNOT(2,3)
a_ij = exp(β · s_ij) / Σ_k exp(β · s_ik)
y_i  = Σ_j a_ij · value_j
```

Qubit 0 is the least significant bit. The statevector is 16 amplitudes. That is the entire feature map. It is small on purpose: small enough to simulate exactly, small enough to check by hand, small enough to run on an RTX 3080 without pretending to be a fault-tolerant machine.

## The circuit

Every token goes through the same circuit. Only the angles change.

```
|0⟩ ── Ry(θ₀) ─●────────────────────  q0
|0⟩ ── Ry(θ₁) ─⊕──●───────────────  q1
|0⟩ ── Ry(θ₂) ───⊕──●─────────────  q2
|0⟩ ── Ry(θ₃) ─────⊕──────────────  q3
```

The register starts in `|0000⟩`. Each `Ry(θ_i)` is the standard Y-rotation,

```
Ry(θ) = [[ cos(θ/2), -sin(θ/2) ],
         [ sin(θ/2),  cos(θ/2) ]]
```

with `θ_i = embedding[i] · π`. A zero embedding is the identity. `Ry(π)` on a single qubit flips `|0⟩` to `|1⟩`. The four rotations load the token. They do not yet talk to each other.

The CNOT chain is what couples them. Control on qubit 0, target on 1. Then 1 onto 2. Then 2 onto 3. After that, the amplitude of any basis state depends on more than one feature. Two tokens with similar embeddings land in nearby states. Two tokens that differ land farther apart. Fidelity reads that distance.

Zero angles leave the state at `|0000⟩`. The CNOTs do nothing, because every control is off. That is the sanity check. A circuit that cannot sit still when given nothing is wrong.

The same circuit is written three times, on purpose:

- the CPU reference in `src/statevector.cpp`, so the math can be checked without a GPU
- the CUDA kernels in `cuda/kernels.cu`, same pair mapping, same gate order
- the CUDA-Q `__qpu__` function in `cudaq/feature_circuit.cpp`, same gates, simulator only

A Bell circuit (`H` then `CNOT`) lives next to the feature circuit in the CUDA-Q module. It is there to test the backend, not to compute attention.

## The CUDA kernel

The GPU does not call a quantum library. It runs three kernels over a 16-element statevector of `cuDoubleComplex`. The target is `sm_86`, an RTX 3080.

`apply_ry_kernel` launches 8 threads. A 4-qubit state has 8 pairs of amplitudes that differ only in the target bit. Each thread owns one pair. It maps its index onto the basis states where the target bit is clear, reads both amplitudes, and writes the `Ry` mix:

```
new_low  =  c * a + s * b
new_high = -s * a + c * b
```

with `c = cos(θ/2)` and `s = sin(θ/2)`. Same matrix as the CPU reference. Same pair order.

`apply_cnot_kernel` launches 16 threads, one per basis state. A thread whose control bit is off returns. A thread whose control bit is on swaps its amplitude with the state that has the target bit flipped, but only if its index is the lower of the two. That stops the pair from swapping twice.

`fidelity_kernel` is one block of 16. Each thread computes one term of `⟨ψ|φ⟩` into shared memory. A stride reduction folds the real and imaginary parts. Thread 0 writes `|⟨ψ|φ⟩|²`.

The host wrappers are `extern "C"`: `cuda_apply_ry`, `cuda_apply_cnot`, `cuda_fidelity`. ArrayFire locks an `af::array`, hands over the device pointer, and these kernels write in place. No copy of the statevector leaves the device until fidelity needs a single double back on the host.

The kernels are the circuit. They are not a wrapper around something else.

## Why the entropy is separate

ANU's quantum random number generator measures vacuum fluctuations. NIST's beacon publishes a signed public pulse. Both are real sources of randomness. Neither is a quantum computer.

Q-ATTN-01 treats them that way. Entropy can perturb an angle or seed an experiment. It does not execute a gate. The runtime labels every sample: `anu_modern`, `anu_legacy`, `nist_beacon`, or `os_crypto`. Calling ANU does not run the feature circuit. The feature circuit does not claim to be a QPU.

That separation is the point. Quantum attention is a circuit. Quantum entropy is a number. Mixing the two up is how projects start lying about what they did.

## What runs

The production tree is native. C++20 owns the entrypoint, the statevector, the attention, and the entropy client. CUDA kernels implement the same gates on the GPU. ArrayFire owns the device arrays when it is present. CUDA-Q defines the same circuit in `__qpu__` when it is present. A CPU reference always exists so the math can be checked without a GPU.

Python is not on this path. Earlier prototypes remain in git history under `archive/python/`. They are provenance, not the runtime.

```
cuda_arrayfire/
  src/            CPU reference, attention, entropy, main
  cuda/           Ry, CNOT, fidelity kernels (sm_86)
  arrayfire/      ArrayFire host, built only if found
  cudaq/          CUDA-Q feature circuit and Bell circuit
  include/qattn/  headers
  tests/          CPU reference tests
```

## Build

```bash
cd cuda_arrayfire
g++ -std=c++20 -Iinclude -o qattn_runtime \
    src/main.cpp src/statevector.cpp src/attention.cpp src/entropy.cpp
./qattn_runtime
```

Or with CMake, which turns on CUDA, ArrayFire, and CUDA-Q only when it finds them:

```bash
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

Live ANU and NIST need libcurl at build time. The modern ANU endpoint needs `ANU_API_KEY` in the environment. The key is never stored in the tree.

## License

SnapKitty Opaque AGPL-3.0  
Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective  
SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
