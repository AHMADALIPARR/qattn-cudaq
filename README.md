# Q-ATTN-01 with CUDA-Q and Qiskit Kernels

Hybrid quantum-classical attention prototype using:

- **CUDA-Q** kernels for GPU-accelerated statevector simulation (via cuQuantum)
- **Qiskit** for circuit construction and comparison
- **ANU QRNG** for physical quantum entropy
- Fidelity-based attention scores as described in the research notes

## Architecture

```
Classical embeddings
        ↓
CUDA-Q kernel: Ry(θ_i) + CNOT chain  →  |q_i⟩
        ↓
Fidelity s_ij = |⟨q_i | k_j⟩|²   (via get_state)
        ↓
Softmax attention weights
        ↓
Classical value aggregation
```

QRNG supplies entropy only (not quantum computation).

## Requirements

```bash
pip install cudaq qiskit requests numpy
```

For GPU acceleration:
```bash
pip install cudaq  # with CUDA 12/13 wheel as documented
cudaq.set_target("nvidia")
```

## ANU API key

Get a free key at https://quantumnumbers.anu.edu.au/

```bash
export ANU_API_KEY=your_key
```

Or set it in code:

```python
from entropy_sources import set_anu_api_key, anu_key_status, fetch_anu_modern

set_anu_api_key("your_key")
print(anu_key_status())          # {"present": True, "source": "set_anu_api_key", ...}
data = fetch_anu_modern(10)      # uses the key
```

Key resolution order (both `entropy_sources` and `anu_qrng`):
1. Explicit `api_key=` argument
2. `set_anu_api_key()` call
3. `ANU_API_KEY` environment variable

If no key is available, the modern endpoint raises; `anu_qrng.fetch()` falls back to the legacy endpoint.

## Files

- `anu_qrng.py` — ANU Quantum Numbers client
- `entropy_sources.py` — Multi-source entropy client (ANU, NIST Beacon, qrngapi)
- `cudaq_attention.py` — CUDA-Q kernels for feature encoding + fidelity attention
- `qiskit_attention.py` — Equivalent Qiskit circuits for comparison
- `quantum_kernels.py` — Quantum kernel methods + kernel ridge regression
- `kernel_demo.py` — Kernel methods demo
- `main.py` — Toy experiment comparing both backends + classical baseline

## Run

```bash
export ANU_API_KEY=your_key
python main.py
python kernel_demo.py
python entropy_sources.py   # test all entropy sources
```
