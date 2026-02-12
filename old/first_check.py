# this file checks out base_data.json, printing to (base_data_summary.txt)
# it should also reorganize base_data.json so that the first attribute is labeled simplified and the list is labeled traditoinal

# it should check:
# total number of entries
# total number of unique entries
# number of entries by amount of traditional counterparts
# 
# number of entries and percentage of entries where one of the traditional counterparts is identical to the simplified one
# 

from collections import Counter
import json
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "base_data.json"
SUMMARY_PATH = BASE_DIR / "base_data_summary.txt"
REORGANIZED_PATH = BASE_DIR / "base_data_reorganized.json"


def load_data(path: Path) -> list:
    """Load the raw list-of-dicts structure from base_data.json."""

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize(records: list) -> dict:
    """Merge duplicate simplified entries and de-duplicate variant lists."""

    merged: dict = {}

    for entry in records:
        simplified = entry.get("simplified")
        variants = entry.get("traditional", []) or []

        if not simplified:
            continue

        if simplified not in merged:
            merged[simplified] = []

        existing = merged[simplified]
        seen = set(existing)

        for char in variants:
            if char not in seen:
                existing.append(char)
                seen.add(char)

    return merged


def summarize(data: dict) -> dict:
    total_entries = len(data)
    unique_chars = set()
    traditional_counts = Counter()
    identical_to_simplified = 0

    for simplified, variants in data.items():
        unique_chars.add(simplified)

        unique_variants = []
        seen = set()
        for char in variants:
            if char not in seen:
                unique_variants.append(char)
                seen.add(char)
                unique_chars.add(char)

        traditional_only = [char for char in unique_variants if char != simplified]
        traditional_counts[len(traditional_only)] += 1

        if simplified in seen:
            identical_to_simplified += 1

    percentage_identical = (identical_to_simplified / total_entries) * 100 if total_entries else 0

    return {
        "total_entries": total_entries,
        "unique_characters": len(unique_chars),
        "traditional_distribution": traditional_counts,
        "identical_count": identical_to_simplified,
        "identical_percentage": percentage_identical,
    }


def write_summary(stats: dict, path: Path) -> None:
    lines = [
        f"Total entries: {stats['total_entries']}",
        f"Total unique characters across all forms: {stats['unique_characters']}",
        "",
        "Entries by number of traditional counterparts:",
    ]

    for count in sorted(stats["traditional_distribution"]):
        entries = stats["traditional_distribution"][count]
        label = "counterpart" if count == 1 else "counterparts"
        lines.append(f"  {count} {label}: {entries}")

    lines.extend(
        [
            "",
            "Entries where a traditional counterpart matches the simplified form:",
            f"  Count: {stats['identical_count']}",
            f"  Percentage: {stats['identical_percentage']:.2f}%",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def reorganize(data: dict) -> list:
    """Return a sorted list of entries with de-duplicated variants."""

    reorganized = []
    for simplified, variants in sorted(data.items()):
        seen = set()
        cleaned = []
        for char in variants:
            if char not in seen:
                cleaned.append(char)
                seen.add(char)

        reorganized.append({"simplified": simplified, "traditional": cleaned})

    return reorganized


def write_reorganized(data: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    records = load_data(DATA_PATH)
    normalized = normalize(records)

    stats = summarize(normalized)
    write_summary(stats, SUMMARY_PATH)

    reorganized = reorganize(normalized)
    write_reorganized(reorganized, REORGANIZED_PATH)


if __name__ == "__main__":
    main()
