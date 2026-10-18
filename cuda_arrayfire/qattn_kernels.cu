// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective
// SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
// Q-ATTN-01 — CUDA kernels for 4-qubit feature encoding and fidelity
//
// Real CUDA implementation. No Python. No library calls for the gate math.
// Statevector layout: qubit 0 is LSB, dimension 16 for 4 qubits.

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// Apply Ry(theta) to target qubit in a 4-qubit statevector.
// Kernel is launched with 8 threads (one per pair of amplitudes).
__global__ void apply_ry_kernel(
    cuDoubleComplex* __restrict__ state,
    int target,
    double theta)
{
    int pair = threadIdx.x + blockIdx.x * blockDim.x;
    if (pair >= 8) return;

    int mask = 1 << target;
    // Map pair index to basis states where target bit is 0
    int low = 0;
    int bit = 0;
    for (int i = 0; i < 4; ++i) {
        if (i == target) continue;
        if (pair & (1 << bit)) low |= (1 << i);
        ++bit;
    }
    int high = low | mask;

    double c = cos(theta * 0.5);
    double s = sin(theta * 0.5);

    cuDoubleComplex a = state[low];
    cuDoubleComplex b = state[high];

    // Ry = [[c, -s], [s, c]]
    cuDoubleComplex new_a = make_cuDoubleComplex(c * a.x + s * b.x, c * a.y + s * b.y);
    cuDoubleComplex new_b = make_cuDoubleComplex(-s * a.x + c * b.x, -s * a.y + c * b.y);

    state[low] = new_a;
    state[high] = new_b;
}

// Apply CNOT (control -> target) to 4-qubit statevector.
__global__ void apply_cnot_kernel(
    cuDoubleComplex* __restrict__ state,
    int control,
    int target)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 16) return;

    int cmask = 1 << control;
    int tmask = 1 << target;
    if ((i & cmask) == 0) return;

    int flipped = i ^ tmask;
    if (i < flipped) {
        cuDoubleComplex tmp = state[i];
        state[i] = state[flipped];
        state[flipped] = tmp;
    }
}

// Compute |<psi|phi>|^2 on device. Single block reduction.
__global__ void fidelity_kernel(
    const cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ phi,
    double* __restrict__ result)
{
    __shared__ double re_s[16];
    __shared__ double im_s[16];

    int i = threadIdx.x;
    cuDoubleComplex a = psi[i];
    cuDoubleComplex b = phi[i];
    // conj(a) * b
    re_s[i] = a.x * b.x + a.y * b.y;
    im_s[i] = a.x * b.y - a.y * b.x;
    __syncthreads();

    // Reduce
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (i < stride) {
            re_s[i] += re_s[i + stride];
            im_s[i] += im_s[i + stride];
        }
        __syncthreads();
    }
    if (i == 0) {
        double re = re_s[0];
        double im = im_s[0];
        *result = re * re + im * im;
    }
}

// Host wrappers
extern "C" {

void cuda_apply_ry(cuDoubleComplex* d_state, int target, double theta) {
    apply_ry_kernel<<<1, 8>>>(d_state, target, theta);
}

void cuda_apply_cnot(cuDoubleComplex* d_state, int control, int target) {
    apply_cnot_kernel<<<1, 16>>>(d_state, control, target);
}

void cuda_fidelity(const cuDoubleComplex* d_psi, const cuDoubleComplex* d_phi, double* d_result) {
    fidelity_kernel<<<1, 16>>>(d_psi, d_phi, d_result);
}

} // extern "C"
