// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective
// SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
// Native entrypoint. No Python.

#include "qattn/attention.hpp"
#include "qattn/entropy.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void print_matrix(const char* title, const std::vector<std::vector<double>>& m) {
    std::printf("%s\n", title);
    for (size_t i = 0; i < m.size(); ++i) {
        std::printf("  [%zu]", i);
        for (double v : m[i]) std::printf(" %.4f", v);
        std::printf("\n");
    }
}

int main() {
    std::printf("Q-ATTN-01 native runtime\n");
    std::printf("Backend: cpu_reference (CUDA/ArrayFire/CUDA-Q selected at build time)\n");

    std::vector<std::vector<double>> embeddings = {
        {0.1, 0.3, 0.5, 0.2}, {0.4, 0.1, 0.2, 0.6},
        {0.7, 0.8, 0.1, 0.3}, {0.2, 0.5, 0.9, 0.4},
    };
    std::vector<std::vector<double>> values = {
        {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}, {0.2, 0.8},
    };

    auto q = qattn::fidelity_attention(embeddings, values, 2.0);
    if (!q.ok) { std::fprintf(stderr, "attention failed: %s\n", q.error.c_str()); return 1; }
    print_matrix("Fidelity attention (CPU reference):", q.weights);
    print_matrix("Outputs:", q.outputs);

    auto c = qattn::classical_attention(embeddings, values);
    print_matrix("Classical baseline:", c.weights);

    auto os = qattn::fetch_os_crypto(8);
    std::printf("\nEntropy (os_crypto): ok=%d bytes=%zu\n", os.ok, os.bytes.size());

    qattn::EntropyConfig cfg;
    if (const char* k = std::getenv("ANU_API_KEY")) cfg.anu_api_key = k;
    cfg.length = 8;
    for (auto p : {qattn::EntropyProvider::AnuLegacy, qattn::EntropyProvider::AnuModern, qattn::EntropyProvider::NistBeacon}) {
        auto s = qattn::fetch_entropy(p, cfg);
        std::printf("Entropy (%s): ok=%d error=%s\n", s.provider.c_str(), s.ok, s.error.c_str());
    }
    std::printf("\nNote: ANU/NIST supply entropy only. They do not execute quantum circuits.\n");
    return 0;
}
