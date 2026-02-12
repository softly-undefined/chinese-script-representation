# saves the first 2000 lines of zhwiki-latest-pages-articles.xml to data/check_xml.xml

from __future__ import annotations

import argparse
import io

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False


def copy_first_n_lines(in_path: str, out_path: str, limit: int) -> int:
    copied = 0
    bar = tqdm(total=limit, unit="line", desc="copy") if _TQDM else None

    with io.open(in_path, "r", encoding="utf-8") as in_f, io.open(
        out_path, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            if copied >= limit:
                break
            out_f.write(line)
            copied += 1
            if bar is not None:
                bar.update(1)

    if bar is not None:
        bar.close()

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the first N lines of the zhwiki XML dump to a check file."
    )
    parser.add_argument("--xml", default="zhwiki-latest-pages-articles.xml")
    parser.add_argument("--out", default="data/check_xml.xml")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

    copied = copy_first_n_lines(args.xml, args.out, args.limit)
    print(f"Wrote {copied} lines to {args.out}")


if __name__ == "__main__":
    main()
