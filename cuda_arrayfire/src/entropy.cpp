// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective

#include "qattn/entropy.hpp"
#include <cstdlib>
#include <fstream>
#include <sstream>

#if defined(QATTN_HAS_LIBCURL)
#include <curl/curl.h>
#endif

namespace qattn {

std::string provider_name(EntropyProvider p) {
    switch (p) {
        case EntropyProvider::AnuModern: return "anu_modern";
        case EntropyProvider::AnuLegacy: return "anu_legacy";
        case EntropyProvider::NistBeacon: return "nist_beacon";
        case EntropyProvider::OsCrypto: return "os_crypto";
    }
    return "unknown";
}

EntropySample fetch_os_crypto(int n_bytes) {
    EntropySample s;
    s.provider = "os_crypto";
    s.is_quantum_entropy = false;
    s.provenance = "OS cryptographic RNG (/dev/urandom). Not quantum entropy.";
    if (n_bytes <= 0) { s.error = "n_bytes must be positive"; return s; }
    s.bytes.resize(n_bytes);
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (!urandom) { s.error = "/dev/urandom unavailable"; return s; }
    urandom.read(reinterpret_cast<char*>(s.bytes.data()), n_bytes);
    if (urandom.gcount() != n_bytes) { s.error = "short read"; return s; }
    s.ok = true;
    return s;
}

#if defined(QATTN_HAS_LIBCURL)
static size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    static_cast<std::string*>(userdata)->append(ptr, size * nmemb);
    return size * nmemb;
}
static bool http_get(const std::string& url, const std::string& hk, const std::string& hv,
                     int timeout_ms, std::string& body, std::string& err) {
    CURL* c = curl_easy_init();
    if (!c) { err = "curl init failed"; return false; }
    struct curl_slist* hdrs = nullptr;
    if (!hk.empty()) {
        std::string h = hk + ": " + hv;
        hdrs = curl_slist_append(hdrs, h.c_str());
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
    }
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(c, CURLOPT_TIMEOUT_MS, timeout_ms);
    curl_easy_setopt(c, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(c, CURLOPT_SSL_VERIFYHOST, 2L);
    CURLcode rc = curl_easy_perform(c);
    long code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
    if (hdrs) curl_slist_free_all(hdrs);
    curl_easy_cleanup(c);
    if (rc != CURLE_OK) { err = curl_easy_strerror(rc); return false; }
    if (code < 200 || code >= 300) { err = "HTTP " + std::to_string(code); return false; }
    return true;
}
static bool extract_uint8(const std::string& body, std::vector<uint8_t>& out, std::string& err) {
    auto pos = body.find("\"data\"");
    if (pos == std::string::npos) { err = "no data"; return false; }
    auto lb = body.find('[', pos), rb = body.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) { err = "no array"; return false; }
    std::stringstream ss(body.substr(lb + 1, rb - lb - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        int v = std::stoi(tok);
        if (v < 0 || v > 255) { err = "range"; return false; }
        out.push_back(static_cast<uint8_t>(v));
    }
    if (out.empty()) { err = "empty"; return false; }
    return true;
}
static bool extract_nist(const std::string& body, std::vector<uint8_t>& out, std::string& err) {
    auto pos = body.find("localRandomValue");
    if (pos == std::string::npos) { err = "no localRandomValue"; return false; }
    auto q1 = body.find('"', pos + 16), q2 = body.find('"', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) { err = "malformed"; return false; }
    std::string hex = body.substr(q1 + 1, q2 - q1 - 1);
    if (hex.size() < 64 || hex.size() % 2) { err = "short hex"; return false; }
    for (size_t i = 0; i < hex.size(); i += 2)
        out.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    return true;
}
#endif

EntropySample fetch_entropy(EntropyProvider p, const EntropyConfig& cfg) {
    if (p == EntropyProvider::OsCrypto) return fetch_os_crypto(cfg.length);
#if !defined(QATTN_HAS_LIBCURL)
    EntropySample s;
    s.provider = provider_name(p);
    s.error = "BLOCKED: libcurl not available at build time. Network entropy providers require -DQATTN_HAS_LIBCURL.";
    s.is_quantum_entropy = (p == EntropyProvider::AnuModern || p == EntropyProvider::AnuLegacy);
    return s;
#else
    EntropySample s;
    s.provider = provider_name(p);
    s.is_quantum_entropy = (p == EntropyProvider::AnuModern || p == EntropyProvider::AnuLegacy);
    std::string url, hk, hv;
    if (p == EntropyProvider::AnuModern) {
        if (cfg.anu_api_key.empty()) { s.error = "ANU_API_KEY required"; return s; }
        url = "https://api.quantumnumbers.anu.edu.au/?length=" + std::to_string(cfg.length) + "&type=uint8";
        hk = "x-api-key"; hv = cfg.anu_api_key;
        s.provenance = "ANU Quantum Numbers. Entropy only; not a circuit.";
    } else if (p == EntropyProvider::AnuLegacy) {
        url = "https://qrng.anu.edu.au/API/jsonI.php?length=" + std::to_string(cfg.length) + "&type=uint8";
        s.provenance = "ANU QRNG legacy. Entropy only; not a circuit.";
    } else {
        url = "https://beacon.nist.gov/beacon/2.0/pulse/last";
        s.provenance = "NIST Beacon. Not quantum computation.";
        s.is_quantum_entropy = false;
    }
    std::string body, err;
    bool ok = false;
    for (int a = 0; a <= cfg.retries && !ok; ++a) { body.clear(); ok = http_get(url, hk, hv, cfg.timeout_ms, body, err); }
    if (!ok) { s.error = err; return s; }
    if (p == EntropyProvider::NistBeacon) { if (!extract_nist(body, s.bytes, err)) { s.error = err; return s; } }
    else { if (!extract_uint8(body, s.bytes, err)) { s.error = err; return s; } }
    s.ok = true;
    return s;
#endif
}

}  // namespace qattn
