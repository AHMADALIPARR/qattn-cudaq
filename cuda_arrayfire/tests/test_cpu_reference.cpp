// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective

#include "qattn/statevector.hpp"
#include "qattn/attention.hpp"
#include "qattn/entropy.hpp"
#include <cmath>
#include <cstdio>

static int g_fails = 0;
static void expect(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++g_fails; }
    else std::printf("PASS: %s\n", msg);
}
static bool near(double a, double b, double tol = 1e-9) { return std::abs(a - b) < tol; }

int main() {
    auto z = qattn::zero_state();
    expect(near(qattn::fidelity(z, z), 1.0), "fidelity(|0>,|0>) == 1");

    auto s = z;
    qattn::apply_ry(s, 0, M_PI);
    expect(near(std::norm(s[1]), 1.0), "Ry(pi) on q0 maps |0> to |1>");
    expect(near(std::norm(s[0]), 0.0), "Ry(pi) on q0 leaves |0> amplitude zero");

    qattn::State x10{};
    x10[1] = {1.0, 0.0};
    qattn::apply_cnot(x10, 0, 1);
    expect(near(std::norm(x10[3]), 1.0), "CNOT(0,1) on |10> yields |11>");

    double th0[4] = {0, 0, 0, 0};
    auto f0 = qattn::feature_encode(th0);
    expect(near(std::norm(f0[0]), 1.0), "feature_encode(0) == |0000>");

    std::vector<std::vector<double>> embs = {{0.1, 0.3, 0.5, 0.2}, {0.4, 0.1, 0.2, 0.6}, {0.7, 0.8, 0.1, 0.3}};
    std::vector<std::vector<double>> vals = {{1, 0}, {0, 1}, {0.5, 0.5}};
    auto a = qattn::fidelity_attention(embs, vals, 2.0);
    expect(a.ok, "fidelity_attention ok");
    bool rows_ok = true;
    for (const auto& row : a.weights) {
        double sum = 0;
        for (double w : row) { if (!std::isfinite(w) || w < 0) rows_ok = false; sum += w; }
        if (!near(sum, 1.0, 1e-9)) rows_ok = false;
    }
    expect(rows_ok, "attention rows sum to 1 and are non-negative");

    double th[4];
    qattn::thetas_from_embedding(embs[0].data(), 4, th);
    auto st = qattn::feature_encode(th);
    expect(near(qattn::fidelity(st, st), 1.0), "encoded state fidelity with self is 1");

    auto c = qattn::classical_attention(embs, vals);
    expect(c.ok, "classical_attention ok");
    double csum = 0; for (double w : c.weights[0]) csum += w;
    expect(near(csum, 1.0), "classical row 0 sums to 1");

    auto os = qattn::fetch_os_crypto(16);
    expect(os.ok && os.bytes.size() == 16, "os_crypto returns 16 bytes");
    expect(!os.is_quantum_entropy, "os_crypto is not labeled quantum");

    qattn::EntropyConfig cfg; cfg.length = 4;
    auto net = qattn::fetch_entropy(qattn::EntropyProvider::AnuLegacy, cfg);
#if defined(QATTN_HAS_LIBCURL)
    std::printf("INFO: libcurl built in; live network result ok=%d\n", net.ok);
#else
    expect(!net.ok, "anu_legacy reports failure when libcurl absent");
    expect(net.error.find("BLOCKED") != std::string::npos, "error is marked BLOCKED");
#endif

    std::printf("\n%d failure(s)\n", g_fails);
    return g_fails ? 1 : 0;
}
