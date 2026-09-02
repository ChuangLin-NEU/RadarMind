"""Download CI4R-MULTI3 spectrogram PNGs from the public GCS bucket."""
from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "cross-frequency_dataset"
MEDIA = "https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{name}?alt=media"


def download_one(object_name: str, dest: Path, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    url = MEDIA.format(bucket=BUCKET, name=urllib.parse.quote(object_name, safe=""))
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
    if last_err:
        raise last_err


def iter_png_names(list_file: Path, freqs: set[str]) -> list[str]:
    names = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        _, name = line.split("\t", 1)
        if not name.startswith("Spectrograms/11_class_activity_data/"):
            continue
        if not name.lower().endswith(".png"):
            continue
        freq = name.split("/")[2]
        if freq in freqs:
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freqs", default="77GHz,24GHz,Xethru")
    parser.add_argument("--list-file", default=str(ROOT / "notes" / "gcs_object_list.tsv"))
    args = parser.parse_args()
    freqs = {item.strip() for item in args.freqs.split(",") if item.strip()}
    names = iter_png_names(Path(args.list_file), freqs)
    out_root = ROOT / "data" / "CI4R_MULTI3" / "spectrograms"
    print(f"to download: {len(names)}")

    def job(name: str) -> str:
        rel = Path(*name.split("/")[1:])
        download_one(name, out_root / rel)
        return name

    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(job, name) for name in names]
        for fut in as_completed(futures):
            name = fut.result()
            done += 1
            if done % 50 == 0 or done == len(names):
                print(f"{done}/{len(names)} {name}")


if __name__ == "__main__":
    main()
