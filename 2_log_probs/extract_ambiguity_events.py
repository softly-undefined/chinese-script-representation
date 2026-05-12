from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False


@dataclass(frozen=True)
class MappingPair:
    simplified: str
    traditional: str
    mapping_source: str
    mapping_kind: str


@dataclass
class ReservoirBucket:
    limit: int
    rng: random.Random
    seen: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event: dict[str, Any]) -> None:
        self.seen += 1
        if len(self.events) < self.limit:
            self.events.append(event)
            return

        replacement_index = self.rng.randrange(self.seen)
        if replacement_index < self.limit:
            self.events[replacement_index] = event


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def output_root() -> Path:
    return script_dir() / "out"


def ensure_under_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed_root = output_root().resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Output path must be inside {allowed_root}: {path}") from exc
    return resolved


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_unique(chars: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ch in chars:
        if ch in seen:
            continue
        seen.add(ch)
        out.append(ch)
    return out


def classify_wiki_mapping_kind(
    simplified: str,
    original_traditional: list[str],
    usable_traditional: list[str],
) -> str:
    has_identity = simplified in original_traditional
    if len(usable_traditional) > 1:
        return "one_simplified_to_many_traditional"
    if has_identity and usable_traditional:
        return "same_plus_distinct_traditional"
    return "wiki_one_to_one_non_ambiguous"


def load_wiki_pairs(path: Path, include_wiki_one_to_one: bool) -> list[MappingPair]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list JSON in {path}, got {type(payload).__name__}")

    pairs: list[MappingPair] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        simplified = item.get("simplified")
        raw_traditional = item.get("traditional", []) or []
        if not (isinstance(simplified, str) and len(simplified) == 1):
            continue
        if not isinstance(raw_traditional, list):
            continue

        traditional = ordered_unique(
            [ch for ch in raw_traditional if isinstance(ch, str) and len(ch) == 1]
        )
        usable_traditional = [ch for ch in traditional if ch != simplified]
        if not usable_traditional:
            continue

        is_ambiguous_entry = len(traditional) > 1
        if not include_wiki_one_to_one and not is_ambiguous_entry:
            continue

        mapping_kind = classify_wiki_mapping_kind(
            simplified=simplified,
            original_traditional=traditional,
            usable_traditional=usable_traditional,
        )
        for trad in usable_traditional:
            key = (simplified, trad, "wiki_one_to_multi", mapping_kind)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                MappingPair(
                    simplified=simplified,
                    traditional=trad,
                    mapping_source="wiki_one_to_multi",
                    mapping_kind=mapping_kind,
                )
            )

    if not pairs:
        raise ValueError(f"No usable wiki mapping pairs found in {path}")
    return pairs


def load_control_pairs(path: Path, existing_pairs: set[tuple[str, str]]) -> list[MappingPair]:
    payload = load_json(path)
    raw_pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
    if not isinstance(raw_pairs, list):
        raise TypeError(f"`pairs` must be a list in {path}")

    controls: list[MappingPair] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        traditional = item.get("traditional")
        simplified = item.get("simplified")
        if not (
            isinstance(traditional, str)
            and isinstance(simplified, str)
            and len(traditional) == 1
            and len(simplified) == 1
        ):
            continue
        pair_key = (simplified, traditional)
        if pair_key in existing_pairs or pair_key in seen:
            continue
        seen.add(pair_key)
        controls.append(
            MappingPair(
                simplified=simplified,
                traditional=traditional,
                mapping_source="opencc_one_to_one",
                mapping_kind="opencc_one_to_one_control",
            )
        )
    return controls


def load_traditional_exclusive_chars(path: Path) -> set[str]:
    payload = load_json(path)
    chars = payload.get("characters", []) if isinstance(payload, dict) else []
    if not isinstance(chars, list):
        raise TypeError(f"`characters` must be a list in {path}")
    return {ch for ch in chars if isinstance(ch, str) and len(ch) == 1}


def load_simplified_chars_from_opencc(path: Path) -> set[str]:
    payload = load_json(path)
    raw_pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
    if not isinstance(raw_pairs, list):
        return set()
    chars: set[str] = set()
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        simplified = item.get("simplified")
        if isinstance(simplified, str) and len(simplified) == 1:
            chars.add(simplified)
    return chars


def count_script_composition(
    text: str,
    target_pair: MappingPair,
    traditional_exclusive_chars: set[str],
    simplified_indicator_chars: set[str],
) -> dict[str, int | bool]:
    non_space_chars = [ch for ch in text if not ch.isspace()]
    traditional_exclusive_count = sum(1 for ch in non_space_chars if ch in traditional_exclusive_chars)
    simplified_indicator_count = sum(1 for ch in non_space_chars if ch in simplified_indicator_chars)
    target_simplified_count = sum(1 for ch in text if ch == target_pair.simplified)
    target_traditional_count = sum(1 for ch in text if ch == target_pair.traditional)
    return {
        "line_chars_no_space": len(non_space_chars),
        "traditional_exclusive_count": traditional_exclusive_count,
        "simplified_indicator_count": simplified_indicator_count,
        "target_simplified_count": target_simplified_count,
        "target_traditional_count": target_traditional_count,
        "mixed_script_by_indicators": traditional_exclusive_count > 0
        and simplified_indicator_count > 0,
    }


def make_event(
    *,
    context_script: str,
    source_file: Path,
    line_index: int,
    line: str,
    char_index: int,
    observed_char: str,
    pair: MappingPair,
    prefix_chars: int,
    suffix_chars: int,
    traditional_exclusive_chars: set[str],
    simplified_indicator_chars: set[str],
) -> dict[str, Any]:
    full_prefix = line[:char_index]
    prefix = full_prefix[-prefix_chars:] if prefix_chars > 0 else full_prefix
    suffix_end = char_index + 1 + suffix_chars
    suffix = line[char_index + 1 : suffix_end] if suffix_chars > 0 else ""
    script_counts = count_script_composition(
        text=line,
        target_pair=pair,
        traditional_exclusive_chars=traditional_exclusive_chars,
        simplified_indicator_chars=simplified_indicator_chars,
    )
    return {
        "context_script": context_script,
        "source_file": str(source_file),
        "line_index": line_index,
        "char_index": char_index,
        "prefix": prefix,
        "observed_char": observed_char,
        "suffix": suffix,
        "simplified": pair.simplified,
        "traditional": pair.traditional,
        "mapping_source": pair.mapping_source,
        "mapping_kind": pair.mapping_kind,
        "full_prefix_chars": len(full_prefix),
        "prefix_truncated": len(prefix) < len(full_prefix),
        "suffix_truncated": suffix_end < len(line),
        "candidate_relation": (
            "observed_traditional"
            if context_script == "hant"
            else "observed_simplified_candidate_traditional_unknown"
        ),
        "script_counts": script_counts,
    }


def build_target_lookup(
    pairs: list[MappingPair],
    context_script: str,
) -> dict[str, list[MappingPair]]:
    lookup: dict[str, list[MappingPair]] = {}
    for pair in pairs:
        observed = pair.traditional if context_script == "hant" else pair.simplified
        lookup.setdefault(observed, []).append(pair)
    return lookup


def scan_corpus(
    *,
    corpus_path: Path,
    context_script: str,
    pairs: list[MappingPair],
    buckets: dict[tuple[str, str, str], ReservoirBucket],
    per_pair_limit: int,
    rng: random.Random,
    max_lines: int | None,
    min_prefix_chars: int,
    prefix_chars: int,
    suffix_chars: int,
    traditional_exclusive_chars: set[str],
    simplified_indicator_chars: set[str],
) -> dict[str, int]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    target_lookup = build_target_lookup(pairs, context_script=context_script)
    total_lines = 0
    matched_positions = 0
    candidate_events_seen = 0
    skipped_short_prefix = 0

    iterator_desc = f"scan_{context_script}"
    bar = tqdm(unit="line", desc=iterator_desc) if _TQDM else None
    with io.open(corpus_path, "r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            if max_lines is not None and total_lines >= max_lines:
                break
            total_lines += 1
            if bar is not None:
                bar.update(1)

            line = raw_line.rstrip("\n")
            for char_index, observed_char in enumerate(line):
                mapping_pairs = target_lookup.get(observed_char)
                if not mapping_pairs:
                    continue
                matched_positions += 1
                if char_index < min_prefix_chars:
                    skipped_short_prefix += len(mapping_pairs)
                    continue

                for pair in mapping_pairs:
                    candidate_events_seen += 1
                    key = (context_script, pair.simplified, pair.traditional)
                    bucket = buckets.get(key)
                    if bucket is None:
                        bucket = ReservoirBucket(limit=per_pair_limit, rng=rng)
                        buckets[key] = bucket
                    bucket.add(
                        make_event(
                            context_script=context_script,
                            source_file=corpus_path,
                            line_index=line_index,
                            line=line,
                            char_index=char_index,
                            observed_char=observed_char,
                            pair=pair,
                            prefix_chars=prefix_chars,
                            suffix_chars=suffix_chars,
                            traditional_exclusive_chars=traditional_exclusive_chars,
                            simplified_indicator_chars=simplified_indicator_chars,
                        )
                    )

    if bar is not None:
        bar.close()

    return {
        "total_lines_scanned": total_lines,
        "matched_positions": matched_positions,
        "candidate_events_seen": candidate_events_seen,
        "skipped_short_prefix": skipped_short_prefix,
    }


def sorted_sampled_events(
    buckets: dict[tuple[str, str, str], ReservoirBucket]
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket_events = sorted(
            buckets[key].events,
            key=lambda event: (
                event["source_file"],
                int(event["line_index"]),
                int(event["char_index"]),
                event["simplified"],
                event["traditional"],
            ),
        )
        events.extend(bucket_events)
    for index, event in enumerate(events, start=1):
        event["event_id"] = f"evt_{index:08d}"
    return events


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    pairs: list[MappingPair],
    buckets: dict[tuple[str, str, str], ReservoirBucket],
    scan_stats: dict[str, dict[str, int]],
    event_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_source_kind = Counter((pair.mapping_source, pair.mapping_kind) for pair in pairs)
    sampled_by_context = Counter(key[0] for key in buckets)
    seen_by_context = Counter()
    for key, bucket in buckets.items():
        seen_by_context[key[0]] += bucket.seen

    payload = {
        "events_written": event_count,
        "reservoir_keys_with_samples": len(buckets),
        "per_pair_limit": args.per_pair_limit,
        "seed": args.seed,
        "contexts": args.contexts,
        "prefix_chars": args.prefix_chars,
        "suffix_chars": args.suffix_chars,
        "min_prefix_chars": args.min_prefix_chars,
        "mapping_pairs_total": len(pairs),
        "mapping_pairs_by_source_kind": {
            f"{source}:{kind}": count for (source, kind), count in sorted(by_source_kind.items())
        },
        "candidate_events_seen_by_context": dict(sorted(seen_by_context.items())),
        "sampled_keys_by_context": dict(sorted(sampled_by_context.items())),
        "scan_stats": scan_stats,
        "inputs": {
            "wiki_mappings": args.wiki_mappings,
            "control_pairs": args.control_pairs,
            "traditional_exclusive": args.traditional_exclusive,
            "opencc_pairs_for_script_counts": args.opencc_pairs_for_script_counts,
            "hant_corpus": args.hant_corpus,
            "hans_corpus": args.hans_corpus,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    out = output_root()
    parser = argparse.ArgumentParser(
        description=(
            "Extract script-choice ambiguity events for next-token log-prob experiments."
        )
    )
    parser.add_argument(
        "--wiki-mappings",
        default=str(root / "data" / "disambiguation-data" / "wiki_one_to_multi_tokenized_clean.json"),
        help="Cleaned wiki one-to-many mapping JSON.",
    )
    parser.add_argument(
        "--control-pairs",
        default=str(root / "data" / "disambiguation-data" / "opencc_one_to_one_pairs_tokenized_clean.json"),
        help="OpenCC one-to-one pairs JSON for optional controls.",
    )
    parser.add_argument(
        "--traditional-exclusive",
        default=str(
            root
            / "data"
            / "disambiguation-data"
            / "opencc_traditional_exclusive_chars_tokenized_clean.json"
        ),
        help="Traditional-exclusive character JSON used for script composition counts.",
    )
    parser.add_argument(
        "--opencc-pairs-for-script-counts",
        default=str(root / "data" / "disambiguation-data" / "opencc_one_to_one_pairs_tokenized_clean.json"),
        help="OpenCC pairs JSON used to count simplified indicator characters.",
    )
    parser.add_argument(
        "--hant-corpus",
        default=str(root / "data" / "zhwiki-clean" / "clean_zh_hant.txt"),
        help="Cleaned traditional-like corpus.",
    )
    parser.add_argument(
        "--hans-corpus",
        default=str(root / "data" / "zhwiki-clean" / "clean_zh_hans.txt"),
        help="Cleaned simplified-like corpus.",
    )
    parser.add_argument(
        "--contexts",
        choices=["both", "hant", "hans"],
        default="both",
        help="Which corpus contexts to extract.",
    )
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="Include OpenCC one-to-one mappings as controls.",
    )
    parser.add_argument(
        "--include-wiki-one-to-one",
        action="store_true",
        help=(
            "Also include non-ambiguous wiki entries with a single distinct traditional form. "
            "Default keeps entries whose wiki mapping has multiple traditional options."
        ),
    )
    parser.add_argument(
        "--per-pair-limit",
        type=int,
        default=100,
        help="Reservoir sample size per (context_script, simplified, traditional).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional maximum lines to scan per selected corpus, useful for smoke tests.",
    )
    parser.add_argument(
        "--min-prefix-chars",
        type=int,
        default=1,
        help="Skip events with fewer than this many characters before the observed character.",
    )
    parser.add_argument(
        "--prefix-chars",
        type=int,
        default=256,
        help="Maximum left-context characters stored in each event.",
    )
    parser.add_argument(
        "--suffix-chars",
        type=int,
        default=64,
        help="Maximum right-context characters stored for inspection.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Reservoir sampling seed.")
    parser.add_argument(
        "--out-jsonl",
        default=str(out / "ambiguity_events.jsonl"),
        help="Output JSONL path. Must be inside 2_log_probs/out/.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(out / "ambiguity_events_summary.json"),
        help="Output summary JSON path. Must be inside 2_log_probs/out/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_pair_limit <= 0:
        raise ValueError("--per-pair-limit must be positive")
    if args.max_lines is not None and args.max_lines <= 0:
        raise ValueError("--max-lines must be positive when provided")
    if args.prefix_chars < 0 or args.suffix_chars < 0 or args.min_prefix_chars < 0:
        raise ValueError("Prefix/suffix length arguments must be non-negative")

    out_jsonl = ensure_under_output_root(Path(args.out_jsonl))
    summary_out = ensure_under_output_root(Path(args.summary_out))

    wiki_pairs = load_wiki_pairs(
        Path(args.wiki_mappings),
        include_wiki_one_to_one=bool(args.include_wiki_one_to_one),
    )
    all_pairs = list(wiki_pairs)
    if args.include_controls:
        existing = {(pair.simplified, pair.traditional) for pair in all_pairs}
        all_pairs.extend(load_control_pairs(Path(args.control_pairs), existing_pairs=existing))

    traditional_exclusive_chars = load_traditional_exclusive_chars(Path(args.traditional_exclusive))
    simplified_indicator_chars = load_simplified_chars_from_opencc(
        Path(args.opencc_pairs_for_script_counts)
    )
    simplified_indicator_chars.update(pair.simplified for pair in all_pairs)

    rng = random.Random(args.seed)
    buckets: dict[tuple[str, str, str], ReservoirBucket] = {}
    scan_stats: dict[str, dict[str, int]] = {}

    contexts = ["hant", "hans"] if args.contexts == "both" else [args.contexts]
    for context_script in contexts:
        corpus_path = Path(args.hant_corpus if context_script == "hant" else args.hans_corpus)
        scan_stats[context_script] = scan_corpus(
            corpus_path=corpus_path,
            context_script=context_script,
            pairs=all_pairs,
            buckets=buckets,
            per_pair_limit=args.per_pair_limit,
            rng=rng,
            max_lines=args.max_lines,
            min_prefix_chars=args.min_prefix_chars,
            prefix_chars=args.prefix_chars,
            suffix_chars=args.suffix_chars,
            traditional_exclusive_chars=traditional_exclusive_chars,
            simplified_indicator_chars=simplified_indicator_chars,
        )

    events = sorted_sampled_events(buckets)
    write_jsonl(out_jsonl, events)
    write_summary(
        summary_out,
        args=args,
        pairs=all_pairs,
        buckets=buckets,
        scan_stats=scan_stats,
        event_count=len(events),
    )

    print(f"Mapping pairs: {len(all_pairs)}")
    print(f"Reservoir keys with samples: {len(buckets)}")
    print(f"Events written: {len(events)}")
    print(f"Output JSONL: {out_jsonl}")
    print(f"Summary JSON: {summary_out}")


if __name__ == "__main__":
    main()
