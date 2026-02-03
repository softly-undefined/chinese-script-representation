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


def load_data(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize(data: dict) -> dict:
    total_entries = len(data)
    unique_chars = set()
    traditional_counts = Counter()
    identical_to_simplified = 0

    for simplified, variants in data.items():
        unique_chars.update(variants)
        unique_variants = list(dict.fromkeys(variants))  # preserve order, drop dupes

        traditional_only = {char for char in unique_variants if char != simplified}
        traditional_counts[len(traditional_only)] += 1

        if simplified in variants:
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
    reorganized = []
    for simplified, variants in sorted(data.items()):
        seen = set()
        traditional_list = []
        for char in variants:
            if char == simplified:
                continue
            if char not in seen:
                seen.add(char)
                traditional_list.append(char)

        if len(traditional_list) == 1:
            traditional_list.append(simplified)

        reorganized.append(
            {
                "simplified": simplified,
                "traditional": traditional_list,
            }
        )
    return reorganized


def write_reorganized(data: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    data = load_data(DATA_PATH)
    stats = summarize(data)
    write_summary(stats, SUMMARY_PATH)

    reorganized = reorganize(data)
    write_reorganized(reorganized, REORGANIZED_PATH)


if __name__ == "__main__":
    main()
