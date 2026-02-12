# cleans data/zhwiki-base/texts.txt into data/clean_texts.txt
# removes any lines with:
# 1. any english/english puncuation (numbers are okay)
# 2. any unclosed chinese parenthesis
# 3. any sentences <7 characters
# save the rest to data/clean_texts.txt

# clean: 21545265line [00:40, 525552.18line/s]
# Input: data/zhwiki-base/texts.txt
# Output: data/clean_texts.txt
# Total lines: 21545265
# Kept lines: 15699367
# Dropped empty: 0
# Dropped english/ascii punctuation: 5576109
# Dropped unclosed chinese parenthesis: 269789

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False


ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
ASCII_PUNCT_RE = re.compile(r"""[!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~]""")

OPEN_TO_CLOSE = {
    "（": "）",
    "【": "】",
    "「": "」",
    "『": "』",
    "《": "》",
}
CLOSE_TO_OPEN = {close: open_ for open_, close in OPEN_TO_CLOSE.items()}


def default_in_path() -> str:
    if Path("texts.txt").exists():
        return "texts.txt"
    return "data/zhwiki-base/texts.txt"


def has_english_or_ascii_punctuation(text: str) -> bool:
    return bool(ASCII_LETTER_RE.search(text) or ASCII_PUNCT_RE.search(text))


def has_unclosed_chinese_parenthesis(text: str) -> bool:
    stack: list[str] = []
    for ch in text:
        if ch in OPEN_TO_CLOSE:
            stack.append(ch)
            continue
        if ch in CLOSE_TO_OPEN:
            if not stack or stack[-1] != CLOSE_TO_OPEN[ch]:
                return True
            stack.pop()
    return len(stack) > 0


def clean_file(in_path: str, out_path: str, keep_empty: bool, min_chars: int) -> dict[str, int]:
    src = Path(in_path)
    dst = Path(out_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    dropped_empty = 0
    dropped_too_short = 0
    dropped_english = 0
    dropped_parenthesis = 0

    bar = tqdm(unit="line", desc="clean") if _TQDM else None

    with io.open(src, "r", encoding="utf-8") as in_f, io.open(
        dst, "w", encoding="utf-8"
    ) as out_f:
        for raw_line in in_f:
            total += 1
            if bar is not None:
                bar.update(1)

            line = raw_line.rstrip("\n")
            if not keep_empty and not line.strip():
                dropped_empty += 1
                continue

            if len([ch for ch in line if not ch.isspace()]) < min_chars:
                dropped_too_short += 1
                continue

            if has_english_or_ascii_punctuation(line):
                dropped_english += 1
                continue

            if has_unclosed_chinese_parenthesis(line):
                dropped_parenthesis += 1
                continue

            out_f.write(line)
            out_f.write("\n")
            kept += 1

    if bar is not None:
        bar.close()

    return {
        "total": total,
        "kept": kept,
        "dropped_empty": dropped_empty,
        "dropped_too_short": dropped_too_short,
        "dropped_english_or_ascii_punctuation": dropped_english,
        "dropped_unclosed_chinese_parenthesis": dropped_parenthesis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean scraped lines and save filtered output."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default=None,
        help="Input path (default: texts.txt if present, otherwise data/zhwiki-base/texts.txt)",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="data/clean_texts.txt",
        help="Output path",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep empty lines (default drops empty lines)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=7,
        help="Minimum number of non-whitespace characters required to keep a line",
    )
    args = parser.parse_args()

    in_path = args.in_path if args.in_path is not None else default_in_path()
    stats = clean_file(
        in_path=in_path,
        out_path=args.out_path,
        keep_empty=args.keep_empty,
        min_chars=args.min_chars,
    )

    print(f"Input: {in_path}")
    print(f"Output: {args.out_path}")
    print(f"Total lines: {stats['total']}")
    print(f"Kept lines: {stats['kept']}")
    print(f"Dropped empty: {stats['dropped_empty']}")
    print(f"Dropped too short (<{args.min_chars} chars): {stats['dropped_too_short']}")
    print(
        "Dropped english/ascii punctuation: "
        f"{stats['dropped_english_or_ascii_punctuation']}"
    )
    print(
        "Dropped unclosed chinese parenthesis: "
        f"{stats['dropped_unclosed_chinese_parenthesis']}"
    )


if __name__ == "__main__":
    main()
