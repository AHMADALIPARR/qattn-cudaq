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

"""ANU Quantum Numbers (QRNG) client."""

import os
import requests
from typing import List, Dict, Any, Optional

MODERN_URL = "https://api.quantumnumbers.anu.edu.au/"
LEGACY_URL = "https://qrng.anu.edu.au/API/jsonI.php"

# Shared key store (same process as entropy_sources if imported together)
_anu_key: Optional[str] = None


def set_anu_api_key(key: str) -> None:
    """Set the ANU Quantum Numbers API key for this process."""
    global _anu_key
    _anu_key = key
    os.environ["ANU_API_KEY"] = key


def get_anu_api_key() -> Optional[str]:
    """Return the current ANU API key (process override, then env)."""
    return _anu_key or os.getenv("ANU_API_KEY")


def fetch(
    length: int = 16,
    typ: str = "uint8",
    size: int = 1,
    api_key: Optional[str] = None,
    legacy: bool = False,
) -> Dict[str, Any]:
    """Fetch random numbers from ANU QRNG.

    Key resolution: explicit api_key → set_anu_api_key() → ANU_API_KEY env.
    If no key is available, falls back to the legacy endpoint.
    Raises on failure. Never silently substitutes PRNG.
    """
    if not 1 <= length <= 1024:
        raise ValueError("length must be between 1 and 1024")

    api_key = api_key or get_anu_api_key()

    if legacy or not api_key:
        url = f"{LEGACY_URL}?length={length}&type={typ}"
        if typ in ("hex16", "hex8"):
            url += f"&size={size}"
        headers = {}
        source = "anu-legacy"
    else:
        url = f"{MODERN_URL}?length={length}&type={typ}"
        if typ in ("hex16", "hex8"):
            url += f"&size={size}"
        headers = {"x-api-key": api_key}
        source = "anu-modern"

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("success"):
        raise RuntimeError(f"ANU QRNG failure: {data}")

    return {
        "success": True,
        "type": data.get("type"),
        "length": int(data.get("length", length)),
        "data": data["data"],
        "source": source,
        "raw": data,
    }


def floats(n: int = 8, api_key: Optional[str] = None) -> List[float]:
    """Return n floats in [0, 1) derived from uint16 quantum entropy."""
    r = fetch(length=n, typ="uint16", api_key=api_key)
    return [v / 65535.0 for v in r["data"]]
