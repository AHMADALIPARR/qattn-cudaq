"""Demo of quantum kernel methods on the toy embeddings."""

import numpy as np
from quantum_kernels import (
    kernel_matrix,
    projected_kernel,
    QuantumKernelRidge,
    kernel_alignment,
    CUDAQ_AVAILABLE,
)
from anu_qrng import floats as anu_floats

print("=== Quantum Kernel Methods ===")
print(f"CUDA-Q available: {CUDAQ_AVAILABLE}\n")

# Toy data (same as previous experiments)
X = [
    [0.1, 0.3, 0.5, 0.2],
    [0.4, 0.1, 0.2, 0.6],
    [0.7, 0.8, 0.1, 0.3],
    [0.2, 0.5, 0.9, 0.4],
]
# Synthetic regression targets
y = np.array([0.2, 0.8, 0.5, 0.9])

# Entropy
try:
    entropy = anu_floats(4)
    print(f"ANU entropy: {entropy}")
except Exception as e:
    print(f"ANU unavailable ({e}), using fixed entropy")
    entropy = [0.1, 0.2, 0.3, 0.4]

print("\n--- Fidelity kernel matrix (β=0, raw) ---")
K_raw = kernel_matrix(X, entropy, beta=0.0)
print(np.round(K_raw, 4))

print("\n--- Soft fidelity kernel (β=2) ---")
K_soft = kernel_matrix(X, entropy, beta=2.0)
print(np.round(K_soft, 4))

print("\n--- Projected kernel (local projectors) ---")
K_proj = projected_kernel(X, entropy=entropy)
print(np.round(K_proj, 4))

print("\n--- Kernel-target alignment ---")
print(f"Raw fidelity alignment: {kernel_alignment(K_raw, y):.4f}")
print(f"Soft fidelity alignment: {kernel_alignment(K_soft, y):.4f}")
print(f"Projected alignment: {kernel_alignment(K_proj, y):.4f}")

print("\n--- Quantum Kernel Ridge Regression ---")
krr = QuantumKernelRidge(alpha=0.1, beta=2.0)
krr.fit(X, y, entropy=entropy)
preds = krr.predict(X)
print("Targets: ", np.round(y, 4))
print("Preds:   ", np.round(preds, 4))
print("MSE:     ", np.mean((preds - y) ** 2))

print("\nDone.")
