"""Read the ZIP64 central directory of the 93 GB raw_data.zip via HTTP range."""
from __future__ import annotations

import argparse
import struct
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/cross-frequency_dataset/Raw%20Data/raw_data.zip"
ROOT = Path(__file__).resolve().parents[1]


def fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-mb", type=int, default=8)
    args = parser.parse_args()
    size = content_length(URL)
    print(f"zip size: {size} bytes")
    tail = fetch_range(URL, max(0, size - args.tail_mb * 1024 * 1024), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    zip64_eocd = tail.rfind(b"PK\x06\x06")
    zip64_loc = tail.rfind(b"PK\x06\x07")
    print(f"EOCD offset in tail: {eocd}, ZIP64 EOCD: {zip64_eocd}, locator: {zip64_loc}")
    out = ROOT / "notes" / "raw_zip_tail_scan.txt"
    out.write_text(
        f"size={size}\neocd={eocd}\nzip64_eocd={zip64_eocd}\nzip64_loc={zip64_loc}\ntail_len={len(tail)}\n",
        encoding="utf-8",
    )
    if zip64_eocd >= 0:
        rec = tail[zip64_eocd:]
        # ZIP64 EOCD: after 12-byte header+size, disk numbers, then n_entries and cd size/offset
        if len(rec) >= 56:
            n_entries, cd_size, cd_offset = struct.unpack("<QQQ", rec[32:56])
            print(f"ZIP64 entries={n_entries} cd_size={cd_size} cd_offset={cd_offset}")
            out.write_text(out.read_text(encoding="utf-8") + f"entries={n_entries}\ncd_size={cd_size}\ncd_offset={cd_offset}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
