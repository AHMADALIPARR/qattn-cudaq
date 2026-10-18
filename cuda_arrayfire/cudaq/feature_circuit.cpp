// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective
// CUDA-Q native C++ circuit. Compiled only when CUDA-Q is found.
// Simulator circuit definition, not physical QPU execution.

#include <cudaq.h>
#include <vector>

namespace qattn {

__qpu__ void feature_circuit(const std::vector<double>& thetas) {
    cudaq::qvector q(4);
    for (int i = 0; i < 4; ++i) ry(thetas[i], q[i]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[1], q[2]);
    x<cudaq::ctrl>(q[2], q[3]);
}

__qpu__ void bell() {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
}

std::vector<std::complex<double>> cudaq_feature_state(const std::vector<double>& thetas) {
    auto st = cudaq::get_state(feature_circuit, thetas);
    std::vector<std::complex<double>> out(st.size());
    for (std::size_t i = 0; i < st.size(); ++i) out[i] = st[i];
    return out;
}

}  // namespace qattn
