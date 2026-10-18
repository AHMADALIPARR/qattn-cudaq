#pragma once
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective

#include "qattn/types.hpp"
namespace qattn {
// theta_i = embedding[i] * pi
// |q_i> = feature_encode(theta)
// s_ij = |<q_i|q_j>|^2
// a_ij = exp(beta * s_ij) / sum_k exp(beta * s_ik)
// output_i = sum_j a_ij * value_j
AttentionResult fidelity_attention(
    const std::vector<std::vector<double>>& embeddings,
    const std::vector<std::vector<double>>& values,
    double beta);
AttentionResult classical_attention(
    const std::vector<std::vector<double>>& embeddings,
    const std::vector<std::vector<double>>& values);
}  // namespace qattn
