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

"""Quantum kernel methods for the Q-ATTN feature encoding.

Provides:
- Fidelity kernel matrix from the CUDA-Q feature kernel
- Optional projected / soft kernel variants
- Kernel ridge regression wrapper
- Pure-NumPy fallback when CUDA-Q is unavailable
"""

import numpy as np
from typing import List, Optional, Callable, Tuple

try:
    import cudaq
    CUDAQ_AVAILABLE = True
    try:
        cudaq.set_target("nvidia")
    except Exception:
        cudaq.set_target("qpp-cpu")
except ImportError:
    CUDAQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature map (CUDA-Q kernel + pure-NumPy fallback)
# ---------------------------------------------------------------------------

if CUDAQ_AVAILABLE:
    @cudaq.kernel
    def feature_encode(thetas: List[float]):
        q = cudaq.qvector(4)
        for i in range(4):
            ry(thetas[i], q[i])
        x.ctrl(q[0], q[1])
        x.ctrl(q[1], q[2])
        x.ctrl(q[2], q[3])

    def _state(thetas: List[float]) -> np.ndarray:
        return np.array(cudaq.get_state(feature_encode, thetas))
else:
    def _ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _apply_ry(state: np.ndarray, qubit: int, theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        new = np.zeros_like(state)
        mask = 1 << qubit
        for i in range(16):
            amp = state[i]
            if (i & mask) == 0:
                new[i] += c * amp
                new[i | mask] += s * amp
            else:
                new[i & ~mask] += -s * amp
                new[i] += c * amp
        return new

    def _apply_cnot(state: np.ndarray, ctrl: int, tgt: int) -> np.ndarray:
        new = state.copy()
        cmask, tmask = 1 << ctrl, 1 << tgt
        for i in range(16):
            if (i & cmask) != 0:
                flipped = i ^ tmask
                if i < flipped:
                    new[i], new[flipped] = new[flipped], new[i]
        return new

    def _state(thetas: List[float]) -> np.ndarray:
        state = np.zeros(16, dtype=complex)
        state[0] = 1.0
        for q, th in enumerate(thetas):
            state = _apply_ry(state, q, th)
        state = _apply_cnot(state, 0, 1)
        state = _apply_cnot(state, 1, 2)
        state = _apply_cnot(state, 2, 3)
        return state


def _thetas_from_embedding(x: List[float], entropy: Optional[List[float]] = None) -> List[float]:
    thetas = []
    for q in range(4):
        angle = (x[q] * np.pi) if q < len(x) else 0.0
        if entropy and q < len(entropy):
            angle += entropy[q] * 0.1
        thetas.append(angle)
    return thetas


# ---------------------------------------------------------------------------
# Kernel methods
# ---------------------------------------------------------------------------

def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """Quantum fidelity kernel value |⟨ψ|φ⟩|²."""
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def kernel_matrix(
    X: List[List[float]],
    entropy: Optional[List[float]] = None,
    beta: float = 1.0,
) -> np.ndarray:
    """Compute the full quantum kernel matrix K_ij = exp(β · |⟨φ(x_i)|φ(x_j)⟩|²).

    beta=0 recovers the raw fidelity kernel.
    """
    n = len(X)
    states = [_state(_thetas_from_embedding(x, entropy)) for x in X]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            f = fidelity(states[i], states[j])
            val = np.exp(beta * f) if beta != 0 else f
            K[i, j] = K[j, i] = val
    return K


def projected_kernel(
    X: List[List[float]],
    projectors: Optional[List[np.ndarray]] = None,
    entropy: Optional[List[float]] = None,
) -> np.ndarray:
    """Simple projected kernel: average fidelity after local Pauli projections.

    Default projectors are the four computational-basis projectors on qubit 0.
    """
    if projectors is None:
        # |0><0| and |1><1| on qubit 0 (LSB)
        P0 = np.diag([1, 0] * 8).astype(complex)
        P1 = np.diag([0, 1] * 8).astype(complex)
        projectors = [P0, P1]

    n = len(X)
    states = [_state(_thetas_from_embedding(x, entropy)) for x in X]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            vals = []
            for P in projectors:
                psi = P @ states[i]
                phi = P @ states[j]
                # renormalize if non-zero
                npsi, nphi = np.linalg.norm(psi), np.linalg.norm(phi)
                if npsi > 1e-12 and nphi > 1e-12:
                    vals.append(fidelity(psi / npsi, phi / nphi))
                else:
                    vals.append(0.0)
            K[i, j] = K[j, i] = np.mean(vals)
    return K


class QuantumKernelRidge:
    """Kernel ridge regression using the quantum fidelity kernel."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.X_train_: Optional[List[List[float]]] = None
        self.K_train_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.entropy_: Optional[List[float]] = None

    def fit(self, X: List[List[float]], y: np.ndarray, entropy: Optional[List[float]] = None):
        self.X_train_ = X
        self.entropy_ = entropy
        self.K_train_ = kernel_matrix(X, entropy, self.beta)
        n = len(X)
        # (K + αI) α = y
        self.alpha_ = np.linalg.solve(self.K_train_ + self.alpha * np.eye(n), y)
        return self

    def predict(self, X: List[List[float]]) -> np.ndarray:
        if self.X_train_ is None or self.alpha_ is None:
            raise RuntimeError("Must call fit first")
        # Cross-kernel between test and train
        n_test, n_train = len(X), len(self.X_train_)
        K_cross = np.zeros((n_test, n_train))
        train_states = [_state(_thetas_from_embedding(x, self.entropy_)) for x in self.X_train_]
        for i, x in enumerate(X):
            s = _state(_thetas_from_embedding(x, self.entropy_))
            for j, ts in enumerate(train_states):
                f = fidelity(s, ts)
                K_cross[i, j] = np.exp(self.beta * f) if self.beta != 0 else f
        return K_cross @ self.alpha_


def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Simple kernel-target alignment score (higher is better)."""
    y = y.reshape(-1, 1)
    Y = y @ y.T
    num = np.sum(K * Y)
    den = np.linalg.norm(K) * np.linalg.norm(Y)
    return float(num / den) if den > 0 else 0.0
