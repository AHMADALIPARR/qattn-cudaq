#pragma once
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective

#include <array>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace qattn {
using c64 = std::complex<double>;
using State = std::array<c64, 16>;

struct AttentionResult {
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> outputs;
    std::string backend;
    bool ok = false;
    std::string error;
};

struct EntropySample {
    std::string provider;
    std::vector<uint8_t> bytes;
    std::string provenance;
    bool ok = false;
    std::string error;
    bool is_quantum_entropy = false;
};
}  // namespace qattn
