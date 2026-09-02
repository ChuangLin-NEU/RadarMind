#!/usr/bin/env python3
import zipfile

import os

ARCHIVE = os.environ.get("CI4R_RAW_ZIP", os.environ.get("RADARMIND_CI4R_ZIP", ""))
if not ARCHIVE:
    raise SystemExit("set CI4R_RAW_ZIP or RADARMIND_CI4R_ZIP")
MEMBER = "Cross-frequency/24ghz/05_walking_towards/01050010_1573836822.dat"

with zipfile.ZipFile(ARCHIVE) as z:
    raw = z.read(MEMBER)
print("nbytes", len(raw))
print("head_hex", raw[:128].hex())
print("head_ascii")
print(repr(raw[:400]))
# count newlines / printable
print("n_newline", raw.count(b"\n"), "n_cr", raw.count(b"\r"))
print("tail_hex", raw[-64:].hex())
# try float32 / float64 interpretations of the body after first newline block
text_end = 0
for i, b in enumerate(raw[:500]):
    if b == 0:
        text_end = i
        break
print("first_null", raw.find(b"\x00"))
# split possible ascii header
nl = raw.find(b"\n")
print("first_nl", nl, "line0", raw[: max(nl, 80)])
