"""CI4R-MULTI3 spectrogram naming and official AIRHAR split helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = [
    "Towards",
    "Away",
    "Pick",
    "Bend",
    "Sit",
    "Kneel",
    "Crawl",
    "Limp",
    "Toes",
    "SStep",
    "Scissor",
]

# Folder name -> official activity code in the dataset paper
CLASS_TO_CODE = {
    "Towards": "WLKT",
    "Away": "WALKA",
    "Pick": "PICK",
    "Bend": "BEND",
    "Sit": "SIT",
    "Kneel": "KNEEL",
    "Crawl": "CRWL",
    "Limp": "LIMP",
    "Toes": "WTOES",
    "SStep": "SHTEPS",
    "Scissor": "SCSSR",
}

FREQ_TO_DIR = {
    "77GHz": "77GHz",
    "24GHz": "24GHz",
    "10GHz": "Xethru",
    "Xethru": "Xethru",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    freq: str
    class_name: str
    class_index: int
    subject: str
    filename: str


def parse_subject(filename: str) -> str:
    stem = Path(filename).stem
    token = stem.split("_")[0]
    if len(token) >= 8:
        return token[-4:]
    return "unknown"


def list_samples(spec_root: Path, freq: str) -> list[Sample]:
    folder = spec_root / FREQ_TO_DIR[freq]
    samples: list[Sample] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = folder / class_name
        if not class_dir.is_dir():
            continue
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() == ".png")
        for path in files:
            samples.append(
                Sample(
                    path=path,
                    freq=freq,
                    class_name=class_name,
                    class_index=class_index,
                    subject=parse_subject(path.name),
                    filename=path.name,
                )
            )
    return samples


def official_last12_split(samples: list[Sample]) -> list[tuple[Sample, str]]:
    """Match AIRHAR: last 12 files of each class go to test. Files are sorted for repeatability."""
    out: list[tuple[Sample, str]] = []
    by_class: dict[int, list[Sample]] = {}
    for sample in samples:
        by_class.setdefault(sample.class_index, []).append(sample)
    for class_index in sorted(by_class):
        items = by_class[class_index]
        train_size = max(len(items) - 12, 0)
        for i, sample in enumerate(items):
            split = "train" if i < train_size else "test"
            out.append((sample, split))
    return out
