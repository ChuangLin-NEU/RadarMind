#!/usr/bin/env python3
"""Selective zip member I/O. Archive stays read-only. No full extract."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


class ArchiveReader:
    def __init__(self, archive_path: str):
        self.archive_path = str(archive_path)
        if not os.path.isfile(self.archive_path):
            raise FileNotFoundError(self.archive_path)
        self._zip: Optional[zipfile.ZipFile] = None
        self._names: Optional[set] = None

    def __enter__(self) -> "ArchiveReader":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive_path, "r")
            self._names = set(self._zip.namelist())

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def _z(self) -> zipfile.ZipFile:
        if self._zip is None:
            self.open()
        assert self._zip is not None
        return self._zip

    def member_exists(self, member: str) -> bool:
        self.open()
        assert self._names is not None
        return member in self._names

    def member_info(self, member: str) -> dict:
        info = self._z().getinfo(member)
        return {
            "member": member,
            "file_size": int(info.file_size),
            "compress_size": int(info.compress_size),
            "crc": int(info.CRC),
        }

    def read_member_bytes(self, member: str) -> bytes:
        with self._z().open(member, "r") as fh:
            return fh.read()

    def checksum_member(self, member: str, algo: str = "sha256") -> str:
        h = hashlib.new(algo)
        with self._z().open(member, "r") as fh:
            while True:
                chunk = fh.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def list_matching(self, needle: str) -> list:
        self.open()
        assert self._names is not None
        key = needle.lower()
        return sorted(n for n in self._names if key in n.lower())

    @contextmanager
    def temporary_extract(self, member: str, dest_dir: str) -> Iterator[str]:
        """Extract one member under dest_dir, then delete it."""
        os.makedirs(dest_dir, exist_ok=True)
        base = os.path.basename(member.replace("\\", "/"))
        fd, tmp = tempfile.mkstemp(prefix=base + ".", dir=dest_dir)
        os.close(fd)
        try:
            with self._z().open(member, "r") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            yield tmp
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def locate_archive(candidates: list) -> dict:
    found = []
    for p in candidates:
        if p and os.path.isfile(p):
            st = os.stat(p)
            rec = {
                "path": os.path.abspath(p),
                "exists": True,
                "size_bytes": int(st.st_size),
                "readable": False,
                "error": "",
            }
            try:
                with zipfile.ZipFile(p, "r") as z:
                    z.infolist()[0]
                rec["readable"] = True
            except Exception as exc:
                rec["error"] = str(exc)
            found.append(rec)
    return {
        "status": "FOUND" if found else "NOT_FOUND",
        "matches": found,
    }
