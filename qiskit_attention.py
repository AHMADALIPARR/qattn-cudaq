"""Qiskit equivalent of the CUDA-Q feature encoding and fidelity attention.

Useful for cross-checking the CUDA-Q kernel results.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from typing import List, Tuple


def feature_circuit(thetas: List[float]) -> QuantumCircuit:
    """Build the 4-qubit feature encoding circuit."""
    qc = QuantumCircuit(4)
    for i in range(4):
        qc.ry(thetas[i], i)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    return qc


def get_statevector(thetas: List[float]) -> np.ndarray:
    qc = feature_circuit(thetas)
    return Statevector(qc).data


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def quantum_attention(
    embeddings: List[List[float]],
    value_vectors: List[List[float]],
    beta: float = 2.0,
    entropy: List[float] | None = None,
) -> Tuple[List[List[float]], List[List[float]], List[np.ndarray]]:
    """Same interface as the CUDA-Q version."""
    n = len(embeddings)
    states = []

    for emb in embeddings:
        thetas = []
        for q in range(4):
            angle = emb[q] * np.pi if q < len(emb) else 0.0
            if entropy and q < len(entropy):
                angle += entropy[q] * 0.1
            thetas.append(angle)
        states.append(get_statevector(thetas))

    attn = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            attn[i, j] = np.exp(beta * fidelity(states[i], states[j]))
        attn[i] /= attn[i].sum()

    values = np.array(value_vectors)
    outputs = attn @ values

    return outputs.tolist(), attn.tolist(), states
