# QUANTUM_BACKEND_REPORT.md

| Kind | What it is | Status |
|---|---|---|
| CPU statevector | Deterministic reference of the feature circuit | Verified |
| CUDA kernels | GPU implementation of the same gates | Source present; not compiled (no nvcc) |
| ArrayFire | Owns device arrays; launches CUDA kernels | Source present; not compiled |
| CUDA-Q | Native __qpu__ feature circuit and Bell circuit | Source present; not compiled |
| Physical QPU | Hardware execution | Not implemented. Not claimed. |
| ANU QRNG | Vacuum-fluctuation entropy source | Client present; network BLOCKED without libcurl |
| NIST Beacon | Public audit randomness | Client present; network BLOCKED without libcurl |

Feature circuit (all backends): Ry(theta_i) on qubit i, then CNOT(0,1), CNOT(1,2), CNOT(2,3). theta_i = embedding[i] * pi. Simulator circuit. Not physical quantum computation.

CUDA-Q Bell circuit defined in cudaq/feature_circuit.cpp. Expected correlations not verified. Status: BLOCKED.

ANU and NIST return random bytes. They do not run the feature circuit. OS crypto is labeled non-quantum.

CMake sets QATTN_HAS_CUDA, QATTN_HAS_ARRAYFIRE, QATTN_HAS_CUDAQ at configure time. Absent backends are not silently replaced.
