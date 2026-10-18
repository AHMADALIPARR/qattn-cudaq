// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective
// CUDA kernels. Compiled only when CUDA toolkit is present.
// Matches src/statevector.cpp exactly. Target sm_86 (RTX 3080).

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

__global__ void apply_ry_kernel(cuDoubleComplex* __restrict__ state, int target, double theta) {
    int pair = threadIdx.x + blockIdx.x * blockDim.x;
    if (pair >= 8) return;
    int mask = 1 << target;
    int low = 0, bit = 0;
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
    state[low]  = make_cuDoubleComplex(c * a.x + s * b.x, c * a.y + s * b.y);
    state[high] = make_cuDoubleComplex(-s * a.x + c * b.x, -s * a.y + c * b.y);
}

__global__ void apply_cnot_kernel(cuDoubleComplex* __restrict__ state, int control, int target) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 16) return;
    int cmask = 1 << control, tmask = 1 << target;
    if ((i & cmask) == 0) return;
    int flipped = i ^ tmask;
    if (i < flipped) {
        cuDoubleComplex tmp = state[i];
        state[i] = state[flipped];
        state[flipped] = tmp;
    }
}

__global__ void fidelity_kernel(const cuDoubleComplex* __restrict__ psi,
                                const cuDoubleComplex* __restrict__ phi,
                                double* __restrict__ result) {
    __shared__ double re_s[16], im_s[16];
    int i = threadIdx.x;
    cuDoubleComplex a = psi[i], b = phi[i];
    re_s[i] = a.x * b.x + a.y * b.y;
    im_s[i] = a.x * b.y - a.y * b.x;
    __syncthreads();
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (i < stride) {
            re_s[i] += re_s[i + stride];
            im_s[i] += im_s[i + stride];
        }
        __syncthreads();
    }
    if (i == 0) *result = re_s[0] * re_s[0] + im_s[0] * im_s[0];
}

extern "C" {
void cuda_apply_ry(cuDoubleComplex* d, int t, double th) { apply_ry_kernel<<<1, 8>>>(d, t, th); }
void cuda_apply_cnot(cuDoubleComplex* d, int c, int t) { apply_cnot_kernel<<<1, 16>>>(d, c, t); }
void cuda_fidelity(const cuDoubleComplex* a, const cuDoubleComplex* b, double* r) {
    fidelity_kernel<<<1, 16>>>(a, b, r);
}
}
