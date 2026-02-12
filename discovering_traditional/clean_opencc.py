# cleans data in opencc_one_to_one_pairs.json that aren't recognized
# # ex. {
#       "traditional": "䴬",
#       "simplified": "𪎈"
#     },
# 
# 𪎈 isn't recognized by VScode or something so don't use it
#
# do the same with opencc_traditional_exclusive_chars.json
# # examples
#   "𧎈",
#     "𧒯",
#     "𧔥",
#     "𧕟",

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


def is_allowed_char(ch: str, keep_non_bmp: bool) -> bool:
    if not isinstance(ch, str) or len(ch) != 1:
        return False
    code = ord(ch)
    if 0xD800 <= code <= 0xDFFF:  # surrogate range should never appear in valid scalar text
        return False
    if not keep_non_bmp and code > 0xFFFF:
        return False
    return ch.isprintable()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def clean_pairs(
    payload: dict,
    keep_non_bmp: bool,
) -> tuple[list[dict[str, str]], dict[str, int], list[dict[str, str]]]:
    raw_pairs = payload.get("pairs", [])
    if not isinstance(raw_pairs, list):
        raise TypeError("`pairs` must be a list in opencc_one_to_one_pairs.json")

    cleaned: list[dict[str, str]] = []
    removed: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    stats = {
        "input_pairs": len(raw_pairs),
        "removed_bad_shape": 0,
        "removed_unrecognized_char": 0,
        "removed_duplicate": 0,
    }

    iterator = raw_pairs
    if _TQDM:
        iterator = tqdm(raw_pairs, desc="clean_pairs", unit="pair")

    for item in iterator:
        if not isinstance(item, dict):
            stats["removed_bad_shape"] += 1
            continue
        trad = item.get("traditional")
        simp = item.get("simplified")
        if not (isinstance(trad, str) and isinstance(simp, str) and len(trad) == 1 and len(simp) == 1):
            stats["removed_bad_shape"] += 1
            removed.append({"traditional": str(trad), "simplified": str(simp), "reason": "bad_shape"})
            continue
        if not (is_allowed_char(trad, keep_non_bmp) and is_allowed_char(simp, keep_non_bmp)):
            stats["removed_unrecognized_char"] += 1
            removed.append({"traditional": trad, "simplified": simp, "reason": "unrecognized_char"})
            continue
        key = (trad, simp)
        if key in seen:
            stats["removed_duplicate"] += 1
            continue
        seen.add(key)
        cleaned.append({"traditional": trad, "simplified": simp})

    return cleaned, stats, removed


def clean_exclusive_chars(
    payload: dict,
    keep_non_bmp: bool,
) -> tuple[list[str], dict[str, int], list[str]]:
    raw_chars = payload.get("characters", [])
    if not isinstance(raw_chars, list):
        raise TypeError("`characters` must be a list in opencc_traditional_exclusive_chars.json")

    cleaned: list[str] = []
    removed: list[str] = []
    seen: set[str] = set()
    stats = {
        "input_characters": len(raw_chars),
        "removed_bad_shape": 0,
        "removed_unrecognized_char": 0,
        "removed_duplicate": 0,
    }

    iterator = raw_chars
    if _TQDM:
        iterator = tqdm(raw_chars, desc="clean_exclusive", unit="char")

    for ch in iterator:
        if not (isinstance(ch, str) and len(ch) == 1):
            stats["removed_bad_shape"] += 1
            continue
        if not is_allowed_char(ch, keep_non_bmp):
            stats["removed_unrecognized_char"] += 1
            removed.append(ch)
            continue
        if ch in seen:
            stats["removed_duplicate"] += 1
            continue
        seen.add(ch)
        cleaned.append(ch)
    return cleaned, stats, removed


def write_outputs(
    out_pairs_json: Path,
    out_exclusive_json: Path,
    out_exclusive_txt: Path,
    out_report_json: Path,
    source_pairs_payload: dict,
    source_exclusive_payload: dict,
    cleaned_pairs: list[dict[str, str]],
    cleaned_exclusive: list[str],
    pair_stats: dict[str, int],
    exclusive_stats: dict[str, int],
    removed_pairs: list[dict[str, str]],
    removed_exclusive: list[str],
    keep_non_bmp: bool,
) -> None:
    out_pairs_json.parent.mkdir(parents=True, exist_ok=True)

    pair_payload = dict(source_pairs_payload)
    pair_payload["pairs"] = cleaned_pairs
    pair_payload["num_pairs"] = len(cleaned_pairs)
    pair_payload["cleaning"] = {"keep_non_bmp": keep_non_bmp}
    with io.open(out_pairs_json, "w", encoding="utf-8") as handle:
        json.dump(pair_payload, handle, ensure_ascii=False, indent=2)

    exclusive_payload = dict(source_exclusive_payload)
    exclusive_payload["characters"] = cleaned_exclusive
    exclusive_payload["num_characters"] = len(cleaned_exclusive)
    exclusive_payload["cleaning"] = {"keep_non_bmp": keep_non_bmp}
    with io.open(out_exclusive_json, "w", encoding="utf-8") as handle:
        json.dump(exclusive_payload, handle, ensure_ascii=False, indent=2)

    with io.open(out_exclusive_txt, "w", encoding="utf-8") as handle:
        for ch in cleaned_exclusive:
            handle.write(ch)
            handle.write("\n")

    report = {
        "keep_non_bmp": keep_non_bmp,
        "pair_stats": pair_stats,
        "exclusive_stats": exclusive_stats,
        "removed_pairs_sample": removed_pairs[:50],
        "removed_exclusive_sample": removed_exclusive[:200],
    }
    with io.open(out_report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean OpenCC extracted outputs by removing unrecognized/non-renderable "
            "characters (non-BMP by default), deduplicating, and writing cleaned files."
        )
    )
    parser.add_argument(
        "--pairs-json",
        default="discovering_traditional/opencc_one_to_one_pairs.json",
        help="Input 1:1 pairs JSON",
    )
    parser.add_argument(
        "--exclusive-json",
        default="discovering_traditional/opencc_traditional_exclusive_chars.json",
        help="Input traditional-exclusive JSON",
    )
    parser.add_argument(
        "--out-pairs-json",
        default="discovering_traditional/opencc_one_to_one_pairs_cleaned.json",
        help="Output cleaned 1:1 pairs JSON",
    )
    parser.add_argument(
        "--out-exclusive-json",
        default="discovering_traditional/opencc_traditional_exclusive_chars_cleaned.json",
        help="Output cleaned exclusive JSON",
    )
    parser.add_argument(
        "--out-exclusive-txt",
        default="discovering_traditional/opencc_traditional_exclusive_chars_cleaned.txt",
        help="Output cleaned exclusive TXT",
    )
    parser.add_argument(
        "--out-report-json",
        default="discovering_traditional/opencc_clean_report.json",
        help="Output cleaning report JSON",
    )
    parser.add_argument(
        "--keep-non-bmp",
        action="store_true",
        help="Keep non-BMP characters (default removes them for editor compatibility).",
    )
    args = parser.parse_args()

    pairs_payload = load_json(Path(args.pairs_json))
    exclusive_payload = load_json(Path(args.exclusive_json))

    cleaned_pairs, pair_stats, removed_pairs = clean_pairs(
        pairs_payload, keep_non_bmp=args.keep_non_bmp
    )
    cleaned_exclusive, exclusive_stats, removed_exclusive = clean_exclusive_chars(
        exclusive_payload, keep_non_bmp=args.keep_non_bmp
    )

    write_outputs(
        out_pairs_json=Path(args.out_pairs_json),
        out_exclusive_json=Path(args.out_exclusive_json),
        out_exclusive_txt=Path(args.out_exclusive_txt),
        out_report_json=Path(args.out_report_json),
        source_pairs_payload=pairs_payload,
        source_exclusive_payload=exclusive_payload,
        cleaned_pairs=cleaned_pairs,
        cleaned_exclusive=cleaned_exclusive,
        pair_stats=pair_stats,
        exclusive_stats=exclusive_stats,
        removed_pairs=removed_pairs,
        removed_exclusive=removed_exclusive,
        keep_non_bmp=args.keep_non_bmp,
    )

    print(f"Input pairs: {args.pairs_json}")
    print(f"Input exclusive: {args.exclusive_json}")
    print(f"Output cleaned pairs: {args.out_pairs_json}")
    print(f"Output cleaned exclusive JSON: {args.out_exclusive_json}")
    print(f"Output cleaned exclusive TXT: {args.out_exclusive_txt}")
    print(f"Output report: {args.out_report_json}")
    print(f"Pairs kept: {len(cleaned_pairs)} / {pair_stats['input_pairs']}")
    print(
        f"Exclusive chars kept: {len(cleaned_exclusive)} / "
        f"{exclusive_stats['input_characters']}"
    )


if __name__ == "__main__":
    main()
