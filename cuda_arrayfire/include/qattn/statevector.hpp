#pragma once
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective

#include "qattn/types.hpp"
namespace qattn {
State zero_state();
void apply_ry(State& s, int target, double theta);
void apply_cnot(State& s, int control, int target);
State feature_encode(const double thetas[4]);
double fidelity(const State& a, const State& b);
void thetas_from_embedding(const double* emb, int n, double out[4]);
}  // namespace qattn
