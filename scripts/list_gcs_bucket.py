"""List objects in the public CI4R-MULTI3 GCS bucket."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "cross-frequency_dataset"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"


def list_objects() -> list[dict]:
    items: list[dict] = []
    page_token = None
    while True:
        query = {"maxResults": "1000"}
        if page_token:
            query["pageToken"] = page_token
        url = API + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        items.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return items


def main() -> None:
    items = list_objects()
    out_dir = Path(__file__).resolve().parents[1] / "notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        name = item.get("name", "")
        size = int(item.get("size", 0))
        rows.append(f"{size}\t{name}")
    text = "\n".join(rows)
    (out_dir / "gcs_object_list.tsv").write_text(text, encoding="utf-8")
    print(f"objects: {len(items)}")
    suffixes: dict[str, int] = {}
    prefixes: dict[str, int] = {}
    for item in items:
        name = item.get("name", "")
        if name.endswith("/"):
            continue
        suffix = Path(name).suffix.lower() or "<none>"
        suffixes[suffix] = suffixes.get(suffix, 0) + 1
        prefix = name.split("/")[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print("prefixes:", json.dumps(prefixes, ensure_ascii=False, indent=2))
    print("suffixes:", json.dumps(suffixes, ensure_ascii=False, indent=2))
    bin_like = [item["name"] for item in items if item.get("name", "").lower().endswith(".bin")]
    print(f".bin files: {len(bin_like)}")
    for name in bin_like[:30]:
        print(" ", name)


if __name__ == "__main__":
    main()
