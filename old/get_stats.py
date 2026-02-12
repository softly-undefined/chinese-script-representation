# i have data in data/disambiguation-data/base_data.json
# I want to augment this json with the quantity of each of the characters present in data/clean_texts.txt
# and save the augmented json to data/disambiguation-data/augmented_data.json

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False


def count_characters(text_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    bar = tqdm(unit="line", desc="count") if _TQDM else None

    with io.open(text_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if bar is not None:
                bar.update(1)
            for ch in line.rstrip("\n"):
                if not ch.isspace():
                    counts[ch] += 1

    if bar is not None:
        bar.close()
    return counts


def augment_records(records: list[dict], counts: Counter[str]) -> list[dict]:
    augmented: list[dict] = []
    for item in records:
        simplified = item.get("simplified", "")
        traditional = item.get("traditional", []) or []

        simplified_count = counts.get(simplified, 0)
        traditional_counts = {char: counts.get(char, 0) for char in traditional}
        total_traditional_count = sum(traditional_counts.values())

        new_item = dict(item)
        new_item["simplified_count"] = simplified_count
        new_item["traditional_counts"] = traditional_counts
        new_item["total_traditional_count"] = total_traditional_count
        new_item["total_count_all_forms"] = simplified_count + total_traditional_count
        augmented.append(new_item)
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment disambiguation JSON with character counts from clean text."
    )
    parser.add_argument(
        "--base-json",
        default="data/disambiguation-data/base_data.json",
        help="Path to base_data.json",
    )
    parser.add_argument(
        "--texts",
        default="data/clean_texts.txt",
        help="Path to cleaned text lines",
    )
    parser.add_argument(
        "--out",
        default="data/disambiguation-data/augmented_data.json",
        help="Output path for augmented JSON",
    )
    args = parser.parse_args()

    base_path = Path(args.base_json)
    text_path = Path(args.texts)
    out_path = Path(args.out)

    if not base_path.exists():
        raise FileNotFoundError(f"Base JSON not found: {base_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    records = json.loads(base_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"Expected list JSON in {base_path}, got {type(records).__name__}")

    counts = count_characters(text_path)
    augmented = augment_records(records, counts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as handle:
        json.dump(augmented, handle, ensure_ascii=False, indent=2)

    print(f"Input base JSON: {base_path}")
    print(f"Input texts: {text_path}")
    print(f"Output JSON: {out_path}")
    print(f"Entries: {len(augmented)}")
    print(f"Unique counted characters: {len(counts)}")


if __name__ == "__main__":
    main()
