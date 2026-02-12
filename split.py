# split's data/clean_texts.txt into two .txt files, one that is in simplified chinese and one that is in traditional
# saves to data/clean_zh_hans.txt and data/clean_zh_hant.txt

# split: 15699367line [00:15, 999058.88line/s] 
# Input: data/clean_texts.txt
# Output zh-Hans: data/clean_zh_hans.txt
# Output zh-Hant: data/clean_zh_hant.txt
# Traditional-exclusive JSON: data/disambiguation-data/opencc_traditional_exclusive_chars_tokenized_clean.json
# Lines total: 15699367
# Lines zh-Hans: 6548199
# Lines zh-Hant: 9151168
# Traditional-exclusive character count: 1699
# Amount in data/clean_zh_hans.txt: 6548199
# Amount in data/clean_zh_hant.txt: 9151168

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False

def load_traditional_exclusive_chars(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Traditional-exclusive JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chars = payload.get("characters", []) if isinstance(payload, dict) else []
    if not isinstance(chars, list):
        raise TypeError(f"`characters` must be a list in {path}")
    cleaned = {ch for ch in chars if isinstance(ch, str) and len(ch) == 1}
    if not cleaned:
        raise ValueError(f"No usable traditional-exclusive characters found in {path}")
    return cleaned


def classify_line(text: str, traditional_exclusive_chars: set[str]) -> str:
    # Traditional if line contains at least one traditional-exclusive char.
    if any(ch in traditional_exclusive_chars for ch in text):
        return "hant"
    # Simplified otherwise.
    return "hans"


def split_file(
    in_path: Path,
    out_hans: Path,
    out_hant: Path,
    traditional_exclusive_json: Path,
) -> dict[str, int]:
    traditional_exclusive_chars = load_traditional_exclusive_chars(
        traditional_exclusive_json
    )

    out_hans.parent.mkdir(parents=True, exist_ok=True)
    out_hant.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    hans = 0
    hant = 0

    bar = tqdm(unit="line", desc="split") if _TQDM else None

    with io.open(in_path, "r", encoding="utf-8") as in_f, io.open(
        out_hans, "w", encoding="utf-8"
    ) as hans_f, io.open(out_hant, "w", encoding="utf-8") as hant_f:
        for line in in_f:
            total += 1
            if bar is not None:
                bar.update(1)

            text = line.rstrip("\n")
            label = classify_line(
                text=text,
                traditional_exclusive_chars=traditional_exclusive_chars,
            )

            if label == "hans":
                hans_f.write(text)
                hans_f.write("\n")
                hans += 1
            else:
                hant_f.write(text)
                hant_f.write("\n")
                hant += 1

    if bar is not None:
        bar.close()

    return {
        "total": total,
        "hans": hans,
        "hant": hant,
        "traditional_exclusive_chars": len(traditional_exclusive_chars),
    }


def count_lines(path: Path) -> int:
    with io.open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split cleaned Chinese lines into zh-Hans / zh-Hant files."
    )
    parser.add_argument("--in", dest="in_path", default="data/clean_texts.txt")
    parser.add_argument("--out-hans", default="data/clean_zh_hans.txt")
    parser.add_argument("--out-hant", default="data/clean_zh_hant.txt")
    parser.add_argument(
        "--traditional-exclusive-json",
        default="data/disambiguation-data/opencc_traditional_exclusive_chars_tokenized_clean.json",
        help="JSON with `characters` list used to mark traditional lines.",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    stats = split_file(
        in_path=in_path,
        out_hans=Path(args.out_hans),
        out_hant=Path(args.out_hant),
        traditional_exclusive_json=Path(args.traditional_exclusive_json),
    )

    print(f"Input: {in_path}")
    print(f"Output zh-Hans: {args.out_hans}")
    print(f"Output zh-Hant: {args.out_hant}")
    print(f"Traditional-exclusive JSON: {args.traditional_exclusive_json}")
    print(f"Lines total: {stats['total']}")
    print(f"Lines zh-Hans: {stats['hans']}")
    print(f"Lines zh-Hant: {stats['hant']}")
    print(
        "Traditional-exclusive character count: "
        f"{stats['traditional_exclusive_chars']}"
    )
    print(f"Amount in {args.out_hans}: {count_lines(Path(args.out_hans))}")
    print(f"Amount in {args.out_hant}: {count_lines(Path(args.out_hant))}")


if __name__ == "__main__":
    main()
