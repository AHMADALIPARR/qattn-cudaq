# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bel Esprit D'Accord Irrevocable Trust / SnapKitty Collective
# SnapKitty Research Lab — Ahmad Ali Parr, Jessica Westerhoff
# Q-ATTN-01 — hybrid quantum attention with multi-source entropy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Unified quantum entropy client for multiple public sources.

Endpoints supported:
- ANU Quantum Numbers (modern): https://api.quantumnumbers.anu.edu.au
- ANU QRNG (legacy):            https://qrng.anu.edu.au/API/jsonI.php
- NIST Randomness Beacon 2.0:  https://beacon.nist.gov/beacon/2.0/pulse/last
- qrngapi.com:                 https://qrngapi.com/api/random (requires API key)
"""

import os
import requests
from typing import Dict, Any, Optional


SOURCES = {
    "anu_modern": {
        "url": "https://api.quantumnumbers.anu.edu.au/",
        "needs_key": True,
        "key_env": "ANU_API_KEY",
        "key_header": "x-api-key",
    },
    "anu_legacy": {
        "url": "https://qrng.anu.edu.au/API/jsonI.php",
        "needs_key": False,
    },
    "nist_beacon": {
        "url": "https://beacon.nist.gov/beacon/2.0/pulse/last",
        "needs_key": False,
    },
    "qrngapi": {
        "url": "https://qrngapi.com/api/random",
        "needs_key": True,
        "key_env": "QRNGAPI_KEY",
        "key_header": "x-api-key",
    },
}


# ---------------------------------------------------------------------------
# ANU API key helpers
# ---------------------------------------------------------------------------

_anu_key: Optional[str] = None


def set_anu_api_key(key: str) -> None:
    """Set the ANU Quantum Numbers API key for this process.

    Also stores it in the environment so other modules see it.
    """
    global _anu_key
    _anu_key = key
    os.environ["ANU_API_KEY"] = key


def get_anu_api_key() -> Optional[str]:
    """Return the current ANU API key (process override, then env)."""
    return _anu_key or os.getenv("ANU_API_KEY")


def anu_key_status() -> Dict[str, Any]:
    """Report whether an ANU key is available and from where."""
    key = get_anu_api_key()
    return {
        "present": bool(key),
        "source": "set_anu_api_key" if _anu_key else ("env" if os.getenv("ANU_API_KEY") else None),
        "prefix": (key[:4] + "...") if key and len(key) > 4 else None,
    }


def fetch_anu_modern(length: int = 10, typ: str = "uint8", api_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetch from the modern ANU Quantum Numbers endpoint.

    Key resolution order: explicit api_key argument → set_anu_api_key() → ANU_API_KEY env.
    """
    api_key = api_key or get_anu_api_key()
    if not api_key:
        raise RuntimeError(
            "ANU_API_KEY required for modern ANU endpoint. "
            "Call set_anu_api_key('...') or export ANU_API_KEY."
        )
    url = f"https://api.quantumnumbers.anu.edu.au/?length={length}&type={typ}"
    r = requests.get(url, headers={"x-api-key": api_key}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"ANU modern failed: {data}")
    return {"source": "anu_modern", "data": data["data"], "raw": data}


def fetch_anu_legacy(length: int = 10, typ: str = "uint8") -> Dict[str, Any]:
    url = f"https://qrng.anu.edu.au/API/jsonI.php?length={length}&type={typ}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"ANU legacy failed: {data}")
    return {"source": "anu_legacy", "data": data["data"], "raw": data}


def fetch_nist_beacon() -> Dict[str, Any]:
    """Fetch the latest NIST Randomness Beacon 2.0 pulse."""
    r = requests.get("https://beacon.nist.gov/beacon/2.0/pulse/last", timeout=15)
    r.raise_for_status()
    data = r.json()
    pulse = data.get("pulse", {})
    local = pulse.get("localRandomValue", "")
    return {
        "source": "nist_beacon",
        "data": local,
        "pulse_index": pulse.get("pulseIndex"),
        "timestamp": pulse.get("timeStamp"),
        "raw": data,
    }


def fetch_qrngapi(bytes_count: int = 32, fmt: str = "hex", api_key: Optional[str] = None) -> Dict[str, Any]:
    api_key = api_key or os.getenv("QRNGAPI_KEY")
    if not api_key:
        raise RuntimeError("QRNGAPI_KEY required for qrngapi.com")
    url = f"https://qrngapi.com/api/random?bytes={bytes_count}&format={fmt}"
    headers = {"x-api-key": api_key}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text}
    return {"source": "qrngapi", "data": body, "raw": body}


def fetch_all(anu_length: int = 10) -> Dict[str, Any]:
    """Fetch from every endpoint. Sources that fail or need a key are recorded as errors."""
    results = {}
    for name, fn in [
        ("anu_modern", lambda: fetch_anu_modern(anu_length)),
        ("anu_legacy", lambda: fetch_anu_legacy(anu_length)),
        ("nist_beacon", fetch_nist_beacon),
        ("qrngapi", fetch_qrngapi),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def test_all(anu_length: int = 10, timeout: float = 15.0) -> Dict[str, Any]:
    """Test every registered endpoint and return a structured report."""
    import time
    tests = [
        ("anu_modern", lambda: fetch_anu_modern(anu_length), True),
        ("anu_legacy", lambda: fetch_anu_legacy(anu_length), False),
        ("nist_beacon", fetch_nist_beacon, False),
        ("qrngapi", fetch_qrngapi, True),
    ]
    report = {}
    for name, fn, needs_key in tests:
        t0 = time.perf_counter()
        try:
            result = fn()
            latency = (time.perf_counter() - t0) * 1000
            data = result.get("data")
            if isinstance(data, list):
                sample = str(data[:5])
            elif isinstance(data, str):
                sample = data[:32] + ("..." if len(data) > 32 else "")
            else:
                sample = str(data)[:64]
            report[name] = {
                "ok": True,
                "latency_ms": round(latency, 1),
                "sample": sample,
                "needs_key": needs_key,
                "error": None,
            }
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            report[name] = {
                "ok": False,
                "latency_ms": round(latency, 1),
                "sample": None,
                "needs_key": needs_key,
                "error": str(e),
            }
    return report


def test_all_entropy_sources(anu_length: int = 10) -> Dict[str, Any]:
    """Test every entropy source and validate the returned data.

    Returns a report with:
      - per-source status (ok, latency, sample, validation notes)
      - summary counts (passed, failed, skipped_key)
    """
    import time

    def validate(name: str, result: Dict[str, Any]) -> Optional[str]:
        data = result.get("data")
        if name == "nist_beacon":
            if not isinstance(data, str) or len(data) < 64:
                return "NIST localRandomValue missing or too short"
        elif name in ("anu_modern", "anu_legacy"):
            if not isinstance(data, list) or len(data) == 0:
                return "ANU data list empty"
            if not all(isinstance(x, int) for x in data):
                return "ANU data not all integers"
        elif name == "qrngapi":
            if data is None:
                return "qrngapi returned no data"
        return None

    tests = [
        ("anu_modern", lambda: fetch_anu_modern(anu_length), True),
        ("anu_legacy", lambda: fetch_anu_legacy(anu_length), False),
        ("nist_beacon", fetch_nist_beacon, False),
        ("qrngapi", fetch_qrngapi, True),
    ]

    report: Dict[str, Any] = {"sources": {}, "summary": {"passed": 0, "failed": 0, "skipped_key": 0}}

    for name, fn, needs_key in tests:
        t0 = time.perf_counter()
        entry: Dict[str, Any] = {
            "needs_key": needs_key,
            "ok": False,
            "latency_ms": 0.0,
            "sample": None,
            "error": None,
            "validation": None,
        }
        try:
            result = fn()
            entry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            data = result.get("data")
            if isinstance(data, list):
                entry["sample"] = str(data[:5])
            elif isinstance(data, str):
                entry["sample"] = data[:32] + ("..." if len(data) > 32 else "")
            else:
                entry["sample"] = str(data)[:64]
            note = validate(name, result)
            if note:
                entry["error"] = note
                entry["validation"] = "fail"
                report["summary"]["failed"] += 1
            else:
                entry["ok"] = True
                entry["validation"] = "pass"
                report["summary"]["passed"] += 1
        except Exception as e:
            entry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            entry["error"] = str(e)
            if needs_key and "API_KEY" in str(e):
                entry["validation"] = "skipped_key"
                report["summary"]["skipped_key"] += 1
            else:
                entry["validation"] = "fail"
                report["summary"]["failed"] += 1
        report["sources"][name] = entry

    return report


def print_entropy_test(report: Optional[Dict[str, Any]] = None) -> None:
    """Pretty-print the result of test_all_entropy_sources()."""
    if report is None:
        report = test_all_entropy_sources()
    print(f"{'Source':<14} {'Status':<12} {'Latency':>10}  Sample / Error")
    print("-" * 80)
    for name, r in report["sources"].items():
        status = r["validation"] or ("pass" if r["ok"] else "fail")
        lat = f"{r['latency_ms']:.1f} ms"
        detail = r["sample"] if r["ok"] else r["error"]
        print(f"{name:<14} {status:<12} {lat:>10}  {detail}")
    s = report["summary"]
    print("-" * 80)
    print(f"Passed: {s['passed']}  Failed: {s['failed']}  Skipped (no key): {s['skipped_key']}")


if __name__ == "__main__":
    print("=== Entropy source test ===")
    print_entropy_test()
