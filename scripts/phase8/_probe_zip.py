#!/usr/bin/env python3
"""One-shot zip probe on the data node. No extract."""
from __future__ import annotations

import hashlib
import os
import zipfile

PATH = os.environ.get("CI4R_RAW_ZIP", os.environ.get("RADARMIND_CI4R_ZIP", ""))
if not PATH:
    raise SystemExit("set CI4R_RAW_ZIP or RADARMIND_CI4R_ZIP")


def main() -> None:
    st = os.stat(PATH)
    print(f"path={PATH}")
    print(f"size_bytes={st.st_size}")
    print(f"size_gb={st.st_size / (1024**3):.6f}")
    print(f"mtime={st.st_mtime}")
    with zipfile.ZipFile(PATH, "r") as z:
        names = z.namelist()
        print(f"n_members={len(names)}")
        print(f"zip_ok=1")
        # do not hash the 86 GB file; hash is optional and too slow
        sample = "Cross-frequency/10ghz/05_Walking Towards the radar/06050010_1573836822.dat"
        info = z.getinfo(sample)
        print(f"sample_member={sample}")
        print(f"sample_compress={info.compress_size}")
        print(f"sample_uncompress={info.file_size}")
        with z.open(sample) as f:
            head = f.read(64)
        print(f"sample_head_hex={head[:32].hex()}")
        print(f"sample_stream_ok=1")
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
