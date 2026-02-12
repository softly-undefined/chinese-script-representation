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


def load_base_data(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Base data not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def collect_target_chars(records: list[dict]) -> set[str]:
    chars: set[str] = set()
    for item in records:
        simplified = item.get("simplified")
        if isinstance(simplified, str) and simplified:
            chars.add(simplified)
        for trad in item.get("traditional", []) or []:
            if isinstance(trad, str) and trad:
                chars.add(trad)
    return chars


def count_chars_in_file(path: Path, target_chars: set[str], desc: str) -> Counter[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    counts: Counter[str] = Counter()
    bar = tqdm(unit="line", desc=desc) if _TQDM else None
    with io.open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if bar is not None:
                bar.update(1)
            for ch in line:
                if ch in target_chars:
                    counts[ch] += 1
    if bar is not None:
        bar.close()
    return counts


def build_report(records: list[dict], hans_counts: Counter[str], hant_counts: Counter[str]) -> dict:
    augmented_entries: list[dict] = []

    for item in records:
        simplified = item.get("simplified", "")
        traditional = item.get("traditional", []) or []

        simplified_hans = hans_counts.get(simplified, 0)
        simplified_hant = hant_counts.get(simplified, 0)

        traditional_hans = {ch: hans_counts.get(ch, 0) for ch in traditional}
        traditional_hant = {ch: hant_counts.get(ch, 0) for ch in traditional}

        entry = {
            "simplified": simplified,
            "traditional": traditional,
            "hans": {
                "simplified_count": simplified_hans,
                "traditional_counts": traditional_hans,
                "total_all_forms": simplified_hans + sum(traditional_hans.values()),
            },
            "hant": {
                "simplified_count": simplified_hant,
                "traditional_counts": traditional_hant,
                "total_all_forms": simplified_hant + sum(traditional_hant.values()),
            },
        }
        augmented_entries.append(entry)

    summary = {
        "entries": len(augmented_entries),
        "target_characters": len(set(hans_counts.keys()) | set(hant_counts.keys())),
        "hans_total_hits": sum(hans_counts.values()),
        "hant_total_hits": sum(hant_counts.values()),
    }
    return {"summary": summary, "entries": augmented_entries}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count base_data characters in clean_zh_hans.txt and clean_zh_hant.txt "
            "and save a report."
        )
    )
    parser.add_argument(
        "--base-json",
        default="data/disambiguation-data/base_data.json",
        help="Path to base_data.json",
    )
    parser.add_argument(
        "--hans",
        default="data/clean_zh_hans.txt",
        help="Path to split simplified-like corpus",
    )
    parser.add_argument(
        "--hant",
        default="data/clean_zh_hant.txt",
        help="Path to split traditional-like corpus",
    )
    parser.add_argument(
        "--out",
        default="data/disambiguation-data/observe_counts.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    base_path = Path(args.base_json)
    hans_path = Path(args.hans)
    hant_path = Path(args.hant)
    out_path = Path(args.out)

    records = load_base_data(base_path)
    target_chars = collect_target_chars(records)

    hans_counts = count_chars_in_file(hans_path, target_chars, desc="count_hans")
    hant_counts = count_chars_in_file(hant_path, target_chars, desc="count_hant")

    report = build_report(records, hans_counts, hant_counts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Base JSON: {base_path}")
    print(f"Hans file: {hans_path}")
    print(f"Hant file: {hant_path}")
    print(f"Target chars: {len(target_chars)}")
    print(f"Output: {out_path}")
    print(f"Hans total hits: {report['summary']['hans_total_hits']}")
    print(f"Hant total hits: {report['summary']['hant_total_hits']}")


if __name__ == "__main__":
    main()
