# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective
# SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
# Q-ATTN-01 — hybrid quantum attention with multi-source entropy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""CUDA-Q kernels for hybrid quantum attention.

Feature encoding: Ry(θ) on each qubit + CNOT chain.
Attention scores from state fidelity via get_state.
"""

import cudaq
import numpy as np
from typing import List, Tuple

# Default target: use nvidia if GPU available, else qpp-cpu
try:
    cudaq.set_target("nvidia")
except Exception:
    cudaq.set_target("qpp-cpu")


@cudaq.kernel
def feature_encode(thetas: List[float]):
    """4-qubit feature encoding kernel.

    Ry(θ_i) on qubit i, followed by CNOT chain for entanglement.
    """
    q = cudaq.qvector(4)
    for i in range(4):
        ry(thetas[i], q[i])
    # Entanglement
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])
    x.ctrl(q[2], q[3])


def get_statevector(thetas: List[float]) -> np.ndarray:
    """Return the complex statevector produced by the feature kernel."""
    state = cudaq.get_state(feature_encode, thetas)
    return np.array(state)


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """|⟨ψ|φ⟩|²"""
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def quantum_attention(
    embeddings: List[List[float]],
    value_vectors: List[List[float]],
    beta: float = 2.0,
    entropy: List[float] | None = None,
) -> Tuple[List[List[float]], List[List[float]], List[np.ndarray]]:
    """Full hybrid forward pass.

    Returns (outputs, attention_weights, statevectors).
    """
    n = len(embeddings)
    states = []

    for i, emb in enumerate(embeddings):
        thetas = []
        for q in range(4):
            angle = emb[q] * np.pi if q < len(emb) else 0.0
            if entropy and q < len(entropy):
                angle += entropy[q] * 0.1
            thetas.append(angle)
        states.append(get_statevector(thetas))

    # Attention from fidelities
    attn = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            attn[i, j] = np.exp(beta * fidelity(states[i], states[j]))
        attn[i] /= attn[i].sum()

    # Classical value aggregation
    values = np.array(value_vectors)
    outputs = attn @ values

    return outputs.tolist(), attn.tolist(), states
