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

"""Q-ATTN-01 experiment with CUDA-Q and Qiskit kernels + ANU QRNG."""

import numpy as np
from anu_qrng import floats as anu_floats
import cudaq_attention
import qiskit_attention

# Toy data
embeddings = [
    [0.1, 0.3, 0.5, 0.2],
    [0.4, 0.1, 0.2, 0.6],
    [0.7, 0.8, 0.1, 0.3],
    [0.2, 0.5, 0.9, 0.4],
]
values = [
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
    [0.2, 0.8],
]

print("=== Q-ATTN-01 with CUDA-Q / Qiskit ===\n")

# Fetch ANU entropy
print("Fetching ANU QRNG entropy...")
try:
    entropy = anu_floats(8)
    print(f"Entropy sample: {entropy}")
    source = "anu"
except Exception as e:
    print(f"ANU QRNG unavailable ({e}). Using fixed demo entropy.")
    entropy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    source = "demo"

# CUDA-Q path
print("\n--- CUDA-Q kernel ---")
outs_cq, attn_cq, states_cq = cudaq_attention.quantum_attention(
    embeddings, values, beta=2.0, entropy=entropy
)
print("Attention (CUDA-Q):")
for i, row in enumerate(attn_cq):
    print(f"  token {i}: {np.round(row, 4)}")
print("Outputs:")
for i, o in enumerate(outs_cq):
    print(f"  token {i}: {np.round(o, 4)}")

# Qiskit path
print("\n--- Qiskit circuit ---")
outs_qk, attn_qk, states_qk = qiskit_attention.quantum_attention(
    embeddings, values, beta=2.0, entropy=entropy
)
print("Attention (Qiskit):")
for i, row in enumerate(attn_qk):
    print(f"  token {i}: {np.round(row, 4)}")

# Cross-check fidelity between backends
print("\n--- Backend agreement (token 0 state fidelity) ---")
f = float(np.abs(np.vdot(states_cq[0], states_qk[0])) ** 2)
print(f"CUDA-Q vs Qiskit state fidelity: {f:.6f}")

# Classical baseline
print("\n--- Classical baseline ---")
embs = np.array(embeddings)
dots = embs @ embs.T
class_attn = np.exp(dots)
class_attn /= class_attn.sum(axis=1, keepdims=True)
print("Classical attention:")
for i, row in enumerate(class_attn):
    print(f"  token {i}: {np.round(row, 4)}")

print("\n=== Provenance ===")
print(f"Entropy source: {source}")
print("Kernel: 4-qubit Ry + CNOT chain (CUDA-Q and Qiskit)")
print("Attention: fidelity → softmax → classical values")
print("Done.")
