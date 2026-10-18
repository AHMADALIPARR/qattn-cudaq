// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective

#include "qattn/attention.hpp"
#include "qattn/statevector.hpp"
#include <cmath>

namespace qattn {

AttentionResult fidelity_attention(
    const std::vector<std::vector<double>>& embeddings,
    const std::vector<std::vector<double>>& values,
    double beta)
{
    AttentionResult r;
    r.backend = "cpu_reference";
    const int n = static_cast<int>(embeddings.size());
    if (n == 0) { r.error = "empty embeddings"; return r; }
    if (static_cast<int>(values.size()) != n) { r.error = "values length must match embeddings"; return r; }
    const int vdim = values[0].empty() ? 0 : static_cast<int>(values[0].size());

    std::vector<State> states(n);
    for (int i = 0; i < n; ++i) {
        double th[4];
        thetas_from_embedding(embeddings[i].data(), static_cast<int>(embeddings[i].size()), th);
        states[i] = feature_encode(th);
    }

    r.weights.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            r.weights[i][j] = std::exp(beta * fidelity(states[i], states[j]));
            sum += r.weights[i][j];
        }
        if (sum <= 0.0 || !std::isfinite(sum)) { r.error = "non-finite attention row"; return r; }
        for (int j = 0; j < n; ++j) r.weights[i][j] /= sum;
    }

    r.outputs.assign(n, std::vector<double>(vdim, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < vdim; ++k)
                r.outputs[i][k] += r.weights[i][j] * values[j][k];
    r.ok = true;
    return r;
}

AttentionResult classical_attention(
    const std::vector<std::vector<double>>& embeddings,
    const std::vector<std::vector<double>>& values)
{
    AttentionResult r;
    r.backend = "classical_dot";
    const int n = static_cast<int>(embeddings.size());
    if (n == 0) { r.error = "empty embeddings"; return r; }
    const int vdim = values.empty() ? 0 : static_cast<int>(values[0].size());
    r.weights.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            double dot = 0.0;
            const int m = std::min(embeddings[i].size(), embeddings[j].size());
            for (int k = 0; k < m; ++k) dot += embeddings[i][k] * embeddings[j][k];
            r.weights[i][j] = std::exp(dot);
            sum += r.weights[i][j];
        }
        for (int j = 0; j < n; ++j) r.weights[i][j] /= sum;
    }
    r.outputs.assign(n, std::vector<double>(vdim, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < vdim; ++k)
                r.outputs[i][k] += r.weights[i][j] * values[j][k];
    r.ok = true;
    return r;
}

}  // namespace qattn
