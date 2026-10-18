// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective

#include "qattn/statevector.hpp"
#include <cmath>

namespace qattn {

State zero_state() {
    State s{};
    s[0] = {1.0, 0.0};
    return s;
}

void apply_ry(State& s, int target, double theta) {
    const int mask = 1 << target;
    const double c = std::cos(theta * 0.5);
    const double s_ = std::sin(theta * 0.5);
    for (int pair = 0; pair < 8; ++pair) {
        int low = 0, bit = 0;
        for (int i = 0; i < 4; ++i) {
            if (i == target) continue;
            if (pair & (1 << bit)) low |= (1 << i);
            ++bit;
        }
        const int high = low | mask;
        const c64 a = s[low], b = s[high];
        s[low]  = c * a + s_ * b;
        s[high] = -s_ * a + c * b;
    }
}

void apply_cnot(State& s, int control, int target) {
    const int cmask = 1 << control;
    const int tmask = 1 << target;
    for (int i = 0; i < 16; ++i) {
        if ((i & cmask) == 0) continue;
        const int flipped = i ^ tmask;
        if (i < flipped) std::swap(s[i], s[flipped]);
    }
}

State feature_encode(const double thetas[4]) {
    State s = zero_state();
    for (int q = 0; q < 4; ++q) apply_ry(s, q, thetas[q]);
    apply_cnot(s, 0, 1);
    apply_cnot(s, 1, 2);
    apply_cnot(s, 2, 3);
    return s;
}

double fidelity(const State& a, const State& b) {
    c64 inner{0.0, 0.0};
    for (int i = 0; i < 16; ++i) inner += std::conj(a[i]) * b[i];
    return std::norm(inner);
}

void thetas_from_embedding(const double* emb, int n, double out[4]) {
    for (int q = 0; q < 4; ++q) out[q] = (q < n) ? emb[q] * M_PI : 0.0;
}

}  // namespace qattn
