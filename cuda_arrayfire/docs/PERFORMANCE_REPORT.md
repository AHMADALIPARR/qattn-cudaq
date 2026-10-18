# PERFORMANCE_REPORT.md

## Environment

Sandbox without CUDA toolkit, ArrayFire, CUDA-Q, or cmake. Compiler: g++ 12.2.0. GPU: not present.

## Measured

CPU reference is O(n^2) fidelity evaluations, each O(16). For the 4-token toy input, execution completes in well under one second. No microsecond instrumentation in this pass.

## Not measured (BLOCKED)

Kernel launch latency: no CUDA device. ArrayFire sync cost: not installed. Peak VRAM: no GPU. CUDA-Q shot throughput: not installed.

## Memory bound (design, not measured)

4-qubit statevector is 256 bytes. For n <= 1024, peak is under 1 MB, within 10 GB. Design bound, not a measured allocation.

Out-of-memory behavior not tested. No GPU present. Not claimed.
