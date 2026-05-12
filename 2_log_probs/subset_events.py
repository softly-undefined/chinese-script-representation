from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with io.open(path, "r", encoding="utf-8") as handle:
        iterator = handle
        if _TQDM:
            iterator = tqdm(handle, desc="read_events", unit="event")
        for line_number, line in enumerate(iterator, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"Expected JSON object on line {line_number} in {path}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def event_sort_key(row: dict[str, Any]) -> tuple[str, int, int, str, str, str]:
    return (
        str(row.get("source_file", "")),
        int(row.get("line_index", -1)),
        int(row.get("char_index", -1)),
        str(row.get("context_script", "")),
        str(row.get("simplified", "")),
        str(row.get("traditional", "")),
    )


def parse_args() -> argparse.Namespace:
    out = output_root()
    parser = argparse.ArgumentParser(
        description="Create a reproducible random subset of extracted ambiguity events."
    )
    parser.add_argument(
        "--events",
        default=str(out / "ambiguity_events_with_controls.jsonl"),
        help="Input events JSONL.",
    )
    parser.add_argument(
        "--out-jsonl",
        default=str(out / "ambiguity_events_with_controls_50k.jsonl"),
        help="Output subset JSONL. Must be inside 2_log_probs/out/.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50_000,
        help="Number of events to sample.",
    )
    parser.add_argument("--seed", type=int, default=53, help="Random sample seed.")
    parser.add_argument(
        "--preserve-input-order",
        action="store_true",
        help="Write sampled rows in their input order instead of source-position order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive")

    out_jsonl = ensure_under_output_root(Path(args.out_jsonl))
    rows = load_jsonl(Path(args.events))
    if not rows:
        raise ValueError(f"No events found in {args.events}")
    if args.n >= len(rows):
        subset = list(rows)
    else:
        rng = random.Random(args.seed)
        selected_indices = set(rng.sample(range(len(rows)), args.n))
        subset = [row for index, row in enumerate(rows) if index in selected_indices]

    if not args.preserve_input_order:
        subset.sort(key=event_sort_key)

    write_jsonl(out_jsonl, subset)
    print(f"Input events: {len(rows)}")
    print(f"Subset events: {len(subset)}")
    print(f"Seed: {args.seed}")
    print(f"Output JSONL: {out_jsonl}")


if __name__ == "__main__":
    main()
