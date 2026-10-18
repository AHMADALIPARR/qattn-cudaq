// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective
// SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
// Q-ATTN-01 — ArrayFire host for CUDA quantum attention engine
//
// ArrayFire owns the GPU arrays. CUDA kernels operate on the raw device pointers.
// This is the real engine. No Python.

#include <arrayfire.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" {
void cuda_apply_ry(cuDoubleComplex* d_state, int target, double theta);
void cuda_apply_cnot(cuDoubleComplex* d_state, int control, int target);
void cuda_fidelity(const cuDoubleComplex* d_psi, const cuDoubleComplex* d_phi, double* d_result);
}

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Build a 4-qubit statevector on the GPU via ArrayFire, then run the feature circuit.
af::array feature_encode(const std::vector<double>& thetas) {
    // |0000>
    std::vector<af::cdouble> host(16, {0.0, 0.0});
    host[0] = {1.0, 0.0};

    af::array state = af::array(16, host.data());
    state.eval();

    // Get raw CUDA device pointer from ArrayFire
    cuDoubleComplex* d_state = nullptr;
    state.lock();
    d_state = reinterpret_cast<cuDoubleComplex*>(state.device<af::cdouble>());

    // Ry on each qubit
    for (int q = 0; q < 4 && q < (int)thetas.size(); ++q) {
        cuda_apply_ry(d_state, q, thetas[q]);
    }
    // CNOT chain
    cuda_apply_cnot(d_state, 0, 1);
    cuda_apply_cnot(d_state, 1, 2);
    cuda_apply_cnot(d_state, 2, 3);

    check_cuda(cudaDeviceSynchronize(), "feature_encode sync");
    state.unlock();
    return state;
}

double fidelity(const af::array& psi, const af::array& phi) {
    double h_result = 0.0;
    double* d_result = nullptr;
    check_cuda(cudaMalloc(&d_result, sizeof(double)), "malloc result");

    const cuDoubleComplex* d_psi = reinterpret_cast<const cuDoubleComplex*>(psi.device<af::cdouble>());
    const cuDoubleComplex* d_phi = reinterpret_cast<const cuDoubleComplex*>(phi.device<af::cdouble>());

    cuda_fidelity(d_psi, d_phi, d_result);
    check_cuda(cudaDeviceSynchronize(), "fidelity sync");
    check_cuda(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost), "copy result");
    cudaFree(d_result);
    return h_result;
}

// Quantum attention: embeddings -> thetas -> states -> fidelity softmax -> outputs
void quantum_attention(
    const std::vector<std::vector<double>>& embeddings,
    const std::vector<std::vector<double>>& values,
    double beta,
    std::vector<std::vector<double>>& outputs,
    std::vector<std::vector<double>>& attn)
{
    int n = (int)embeddings.size();
    std::vector<af::array> states;
    states.reserve(n);

    for (const auto& emb : embeddings) {
        std::vector<double> thetas(4, 0.0);
        for (int q = 0; q < 4 && q < (int)emb.size(); ++q) {
            thetas[q] = emb[q] * M_PI;
        }
        states.push_back(feature_encode(thetas));
    }

    attn.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            double f = fidelity(states[i], states[j]);
            attn[i][j] = exp(beta * f);
            sum += attn[i][j];
        }
        for (int j = 0; j < n; ++j) attn[i][j] /= sum;
    }

    int vdim = values.empty() ? 0 : (int)values[0].size();
    outputs.assign(n, std::vector<double>(vdim, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < vdim; ++k) {
                outputs[i][k] += attn[i][j] * values[j][k];
            }
        }
    }
}

int main() {
    af::setBackend(AF_BACKEND_CUDA);
    printf("ArrayFire backend: %s\n", af::infoString().c_str());

    std::vector<std::vector<double>> embeddings = {
        {0.1, 0.3, 0.5, 0.2},
        {0.4, 0.1, 0.2, 0.6},
        {0.7, 0.8, 0.1, 0.3},
        {0.2, 0.5, 0.9, 0.4}
    };
    std::vector<std::vector<double>> values = {
        {1.0, 0.0},
        {0.0, 1.0},
        {0.5, 0.5},
        {0.2, 0.8}
    };

    std::vector<std::vector<double>> outputs, attn;
    quantum_attention(embeddings, values, 2.0, outputs, attn);

    printf("\nAttention (CUDA + ArrayFire):\n");
    for (int i = 0; i < (int)attn.size(); ++i) {
        printf("  token %d:", i);
        for (double a : attn[i]) printf(" %.4f", a);
        printf("\n");
    }
    printf("\nOutputs:\n");
    for (int i = 0; i < (int)outputs.size(); ++i) {
        printf("  token %d:", i);
        for (double o : outputs[i]) printf(" %.4f", o);
        printf("\n");
    }
    return 0;
}
