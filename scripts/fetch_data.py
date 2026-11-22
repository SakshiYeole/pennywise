"""Download the open dataset (pinned commit, ~1.6 MB) into data/raw/. Safe to re-run."""
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

COMMIT = "3e8d052cb7b362fb1a04b6064d2331982553ea56"
URL = ("https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2/"
       f"resolve/{COMMIT}/transactions-synthetic.csv")
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "transactions-synthetic.csv"


def main() -> int:
    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        print(f"already present: {DEST} ({DEST.stat().st_size:,} bytes)")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "pennywise-fetch"})
    # python.org builds of Python on macOS ship without CA certificates; certifi (installed
    # with the openai package) provides them, so TLS verification stays on.
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=60, context=context) as resp:
        data = resp.read()
    head = data[:60].decode("utf-8", "replace").splitlines()[0] if data else ""
    if head.strip() != "description,category" or len(data) < 1_000_000:
        print(f"unexpected file (header {head!r}, {len(data):,} bytes); not saved", file=sys.stderr)
        return 1
    DEST.write_bytes(data)
    print(f"saved {DEST} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
