from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False


def find_opencc_data_roots() -> list[Path]:
    roots: list[Path] = []

    env_root = os.environ.get("OPENCC_DATA_DIR", "").strip()
    if env_root:
        roots.append(Path(env_root))

    try:
        import opencc  # type: ignore

        pkg_root = Path(opencc.__file__).resolve().parent
        roots.extend(
            [
                pkg_root,
                pkg_root / "data",
                pkg_root / "resources",
                pkg_root.parent / "share" / "opencc",
            ]
        )
    except Exception:  # noqa: BLE001
        pass

    roots.extend(
        [
            Path("/opt/homebrew/share/opencc"),
            Path("/usr/local/share/opencc"),
            Path("/usr/share/opencc"),
        ]
    )

    seen: set[Path] = set()
    unique: list[Path] = []
    for r in roots:
        rr = r.resolve() if r.exists() else r
        if rr in seen:
            continue
        seen.add(rr)
        unique.append(rr)
    return unique


def find_config_file(config_name: str, roots: list[Path]) -> Path | None:
    candidates = []
    for root in roots:
        candidates.extend(
            [
                root / config_name,
                root / "config" / config_name,
                root / "opencc" / "config" / config_name,
            ]
        )
    for c in candidates:
        if c.exists():
            return c
    return None


def normalize_dict_filename(value: str) -> str:
    # OpenCC config can include absolute-like slashes; we normalize to basename.
    return Path(value).name


def collect_dict_files(node: Any) -> list[str]:
    files: list[str] = []
    if isinstance(node, dict):
        if "file" in node and isinstance(node["file"], str):
            files.append(normalize_dict_filename(node["file"]))
        for v in node.values():
            files.extend(collect_dict_files(v))
    elif isinstance(node, list):
        for item in node:
            files.extend(collect_dict_files(item))
    return files


def find_dictionary_file(dict_name: str, config_dir: Path, roots: list[Path]) -> Path | None:
    candidates = [
        config_dir / dict_name,
        config_dir.parent / "dictionary" / dict_name,
        config_dir.parent / "dict" / dict_name,
        config_dir.parent / dict_name,
    ]
    for root in roots:
        candidates.extend(
            [
                root / dict_name,
                root / "dictionary" / dict_name,
                root / "dict" / dict_name,
                root / "opencc" / "dictionary" / dict_name,
            ]
        )
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_dictionary_tsv(path: Path) -> list[tuple[str, str]]:
    """
    Parse OpenCC txt dictionary rows into (src_char, dst_char) char-level pairs.
    Keeps only rows where both sides are single CJK characters.
    """
    pairs: list[tuple[str, str]] = []
    with io.open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                continue
            src, dst = line.split("\t", 1)
            src = src.strip()
            if len(src) != 1:
                continue
            for candidate in dst.strip().split():
                if len(candidate) == 1:
                    pairs.append((src, candidate))
    return pairs


def extract_t2s_char_pairs_from_configs(roots: list[Path]) -> tuple[list[tuple[str, str]], list[Path]]:
    """
    Read t2s-oriented OpenCC config JSON and extract char-level (traditional, simplified) pairs.
    """
    config_names = ["t2s.json", "tw2s.json", "twp2s.json", "hk2s.json"]
    all_pairs: list[tuple[str, str]] = []
    used_files: list[Path] = []

    for cfg_name in config_names:
        cfg_path = find_config_file(cfg_name, roots)
        if cfg_path is None:
            continue

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        dict_names = collect_dict_files(cfg)
        if not dict_names:
            continue

        for dname in dict_names:
            dpath = find_dictionary_file(dname, cfg_path.parent, roots)
            if dpath is None:
                continue
            if dpath.suffix.lower() != ".txt":
                # system opencc may use .ocd2 binary dicts; skip direct parsing.
                continue
            pairs = parse_dictionary_tsv(dpath)
            all_pairs.extend(pairs)
            used_files.append(dpath)

    # De-duplicate while keeping deterministic order.
    unique_pairs = sorted(set(all_pairs))
    unique_files = sorted(set(used_files))
    return unique_pairs, unique_files


def create_opencc_converter():
    import opencc  # type: ignore

    for cfg in ("t2s.json", "t2s"):
        try:
            conv = opencc.OpenCC(cfg)
            _ = conv.convert("雲")
            return conv
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("Unable to initialize OpenCC converter with t2s config")


def han_codepoint_ranges() -> list[tuple[int, int]]:
    # Major CJK ideograph blocks; inclusive ranges.
    return [
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0x20000, 0x2A6DF),  # Extension B
        (0x2A700, 0x2B73F),  # Extension C
        (0x2B740, 0x2B81F),  # Extension D
        (0x2B820, 0x2CEAF),  # Extension E
        (0x2CEB0, 0x2EBEF),  # Extension F
        (0x30000, 0x3134F),  # Extension G
    ]


def extract_t2s_char_pairs_by_unicode_scan() -> tuple[list[tuple[str, str]], list[Path]]:
    """
    Fallback when OpenCC dictionary files are unavailable (e.g., binary packaged dicts).
    Uses OpenCC conversion over Unicode Han ranges to derive char-level mappings.
    """
    converter = create_opencc_converter()
    pairs: set[tuple[str, str]] = set()

    total = sum(end - start + 1 for start, end in han_codepoint_ranges())
    iterator = range(total)
    bar = tqdm(total=total, desc="scan_han", unit="char") if _TQDM else None

    for start, end in han_codepoint_ranges():
        for cp in range(start, end + 1):
            ch = chr(cp)
            out = converter.convert(ch)
            if len(out) == 1:
                pairs.add((ch, out))
            if bar is not None:
                bar.update(1)

    if bar is not None:
        bar.close()
    return sorted(pairs), []


def build_outputs(
    pairs: list[tuple[str, str]], include_identical: bool
) -> tuple[list[dict[str, str]], list[str]]:
    """
    1) 1:1 traditional:simplified pairs:
       - traditional char maps to exactly one simplified char
       - simplified char maps back to exactly one traditional char
    2) traditional-exclusive chars:
       - chars that appear on traditional side
       - never appear on simplified side
       - and are not identity mappings (trad != simp)
    """
    trad_to_simp: dict[str, set[str]] = {}
    simp_to_trad: dict[str, set[str]] = {}
    filtered_pairs: list[tuple[str, str]] = []
    for trad, simp in pairs:
        if not include_identical and trad == simp:
            continue
        filtered_pairs.append((trad, simp))

    for trad, simp in filtered_pairs:
        trad_to_simp.setdefault(trad, set()).add(simp)
        simp_to_trad.setdefault(simp, set()).add(trad)

    one_to_one: list[dict[str, str]] = []
    for trad in sorted(trad_to_simp):
        simp_set = trad_to_simp[trad]
        if len(simp_set) != 1:
            continue
        simp = next(iter(simp_set))
        if len(simp_to_trad.get(simp, set())) != 1:
            continue
        one_to_one.append({"traditional": trad, "simplified": simp})

    traditional_chars = set(trad_to_simp.keys())
    simplified_chars = set(simp_to_trad.keys())
    exclusive = sorted(ch for ch in traditional_chars if ch not in simplified_chars)
    exclusive = [ch for ch in exclusive if any(s != ch for s in trad_to_simp.get(ch, set()))]
    return one_to_one, exclusive


def save_outputs(
    out_pairs_json: Path,
    out_exclusive_json: Path,
    out_exclusive_txt: Path,
    one_to_one: list[dict[str, str]],
    exclusive: list[str],
    sources: list[Path],
    method: str,
) -> None:
    out_pairs_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": method,
        "num_pairs": len(one_to_one),
        "pairs": one_to_one,
        "sources": [str(p) for p in sources],
    }
    with io.open(out_pairs_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    exclusive_payload = {
        "method": method,
        "num_characters": len(exclusive),
        "characters": exclusive,
        "sources": [str(p) for p in sources],
    }
    with io.open(out_exclusive_json, "w", encoding="utf-8") as handle:
        json.dump(exclusive_payload, handle, ensure_ascii=False, indent=2)

    with io.open(out_exclusive_txt, "w", encoding="utf-8") as handle:
        for ch in exclusive:
            handle.write(ch)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract OpenCC-derived traditional:simplified 1:1 pairs and "
            "traditional-exclusive characters."
        )
    )
    parser.add_argument(
        "--out-pairs-json",
        default="discovering_traditional/opencc_one_to_one_pairs.json",
        help="Output JSON file for 1:1 traditional:simplified pairs",
    )
    parser.add_argument(
        "--out-exclusive-json",
        default="discovering_traditional/opencc_traditional_exclusive_chars.json",
        help="Output JSON file for traditional-exclusive characters",
    )
    parser.add_argument(
        "--out-exclusive-txt",
        default="discovering_traditional/opencc_traditional_exclusive_chars.txt",
        help="Output TXT file (one char per line) for traditional-exclusive characters",
    )
    parser.add_argument(
        "--include-identical",
        action="store_true",
        help="Include identity pairs where traditional == simplified in 1:1 output",
    )
    args = parser.parse_args()

    roots = find_opencc_data_roots()
    pairs, sources = extract_t2s_char_pairs_from_configs(roots)
    method = "config_dict_parse"
    if not pairs:
        pairs, sources = extract_t2s_char_pairs_by_unicode_scan()
        method = "unicode_scan_via_opencc"
    if not pairs:
        raise RuntimeError("No OpenCC char-level pairs were found.")

    one_to_one, exclusive = build_outputs(pairs, include_identical=args.include_identical)
    save_outputs(
        out_pairs_json=Path(args.out_pairs_json),
        out_exclusive_json=Path(args.out_exclusive_json),
        out_exclusive_txt=Path(args.out_exclusive_txt),
        one_to_one=one_to_one,
        exclusive=exclusive,
        sources=sources,
        method=method,
    )

    print(f"Extraction method: {method}")
    print(f"OpenCC sources used: {len(sources)} dictionary files")
    print(f"Raw char-level pairs extracted: {len(pairs)}")
    print(f"1:1 traditional:simplified pairs: {len(one_to_one)}")
    print(f"Traditional-exclusive characters: {len(exclusive)}")
    print(f"Saved: {args.out_pairs_json}")
    print(f"Saved: {args.out_exclusive_json}")
    print(f"Saved: {args.out_exclusive_txt}")


if __name__ == "__main__":
    main()
