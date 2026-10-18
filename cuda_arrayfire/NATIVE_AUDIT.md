# NATIVE_AUDIT.md

Audit of the repository prior to native reconstruction. No findings invented.

## Files inspected

Root Python (historical, not on production path):
- main.py, cudaq_attention.py, qiskit_attention.py, quantum_kernels.py, entropy_sources.py, anu_qrng.py, kernel_demo.py

Prior cuda_arrayfire/:
- qattn_kernels.cu: real CUDA, but host had no CPU reference, no RAII, no tests
- qattn_engine.cpp: required ArrayFire at link time; no entropy; exit(1) on error
- CMakeLists.txt: hard-required ArrayFire and CUDA; no optional path

## Data path (prior)

embeddings -> thetas = emb * pi -> feature_encode (Ry + CNOT) -> statevector -> s_ij = |<psi_i|psi_j>|^2 -> a_ij = exp(beta * s_ij) / sum -> outputs = a @ values

Entropy path is separate HTTP; it does not execute a circuit.

## Execution locations (prior)

Feature encoding: CUDA-Q simulator or NumPy. Fidelity and aggregation: NumPy on host. Entropy: requests. Physical QPU: never.

## Mathematics preserved

4 qubits, qubit 0 = LSB. theta_i = embedding[i] * pi. Circuit: Ry x 4 then CNOT(0,1), CNOT(1,2), CNOT(2,3). s_ij = |<q_i|q_j>|^2. a_ij = exp(beta * s_ij) / sum. Classical baseline: exp(dot) row-normalized.

## Defects

1. Production path was Python.
2. No deterministic CPU reference independent of NumPy.
3. No native HTTP client.
4. No CMake path that builds without Python and without GPU.
5. CUDA kernels existed but were not wired to a testable reference.

## Migration actions

CPU reference in src/statevector.cpp matches CUDA kernels. Native attention in src/attention.cpp. Entropy abstraction with OS crypto always available; network gated on libcurl. CUDA kernels in cuda/kernels.cu (sm_86). ArrayFire and CUDA-Q compiled only if found. Python retained as provenance, not linked.

## Verification in authoring environment

A: PASS for CPU via g++. CMake not installed.
B, C, D, H: BLOCKED (no nvcc, ArrayFire, CUDA-Q, GPU).
E, F (CPU), J: PASS.
G: PASS for OS crypto and BLOCKED reporting; live HTTP untested.
I: PASS (native binary links no Python).
