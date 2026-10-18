// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective
// ArrayFire backend. Compiled only when ArrayFire is found.
// Uses native C++ API. Verifies CUDA backend at startup.

#include <arrayfire.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void cuda_apply_ry(cuDoubleComplex* d, int t, double th);
void cuda_apply_cnot(cuDoubleComplex* d, int c, int t);
void cuda_fidelity(const cuDoubleComplex* a, const cuDoubleComplex* b, double* r);
}

namespace qattn {

struct AfSession {
    AfSession() {
        af::setBackend(AF_BACKEND_CUDA);
        if (af::getActiveBackend() != AF_BACKEND_CUDA)
            throw std::runtime_error("ArrayFire CUDA backend not active");
    }
};

af::array af_feature_encode(const double thetas[4]) {
    std::vector<af::cdouble> host(16, {0.0, 0.0});
    host[0] = {1.0, 0.0};
    af::array state(16, host.data());
    state.eval();
    state.lock();
    auto* d = reinterpret_cast<cuDoubleComplex*>(state.device<af::cdouble>());
    for (int q = 0; q < 4; ++q) cuda_apply_ry(d, q, thetas[q]);
    cuda_apply_cnot(d, 0, 1);
    cuda_apply_cnot(d, 1, 2);
    cuda_apply_cnot(d, 2, 3);
    cudaDeviceSynchronize();
    state.unlock();
    return state;
}

double af_fidelity(const af::array& psi, const af::array& phi) {
    double h = 0.0;
    double* d = nullptr;
    cudaMalloc(&d, sizeof(double));
    auto* dp = reinterpret_cast<const cuDoubleComplex*>(psi.device<af::cdouble>());
    auto* dq = reinterpret_cast<const cuDoubleComplex*>(phi.device<af::cdouble>());
    cuda_fidelity(dp, dq, d);
    cudaDeviceSynchronize();
    cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return h;
}

std::string af_info() { return af::infoString(); }

}  // namespace qattn
