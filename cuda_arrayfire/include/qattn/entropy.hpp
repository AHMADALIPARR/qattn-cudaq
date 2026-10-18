#pragma once
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective

#include "qattn/types.hpp"
#include <string>
namespace qattn {
// Quantum randomness is not quantum computation.
enum class EntropyProvider { AnuModern, AnuLegacy, NistBeacon, OsCrypto };
struct EntropyConfig {
    int timeout_ms = 15000;
    int retries = 2;
    std::string anu_api_key;
    int length = 16;
};
EntropySample fetch_entropy(EntropyProvider p, const EntropyConfig& cfg);
EntropySample fetch_os_crypto(int n_bytes);
std::string provider_name(EntropyProvider p);
}  // namespace qattn
