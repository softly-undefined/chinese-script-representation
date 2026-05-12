from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


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
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def as_float(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else float("nan")
    return float(statistics.pstdev(values))


def pct(count: int, total: int) -> float:
    return float(count / total) if total else float("nan")


def group_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def summarize_group(rows: list[dict[str, Any]], key_fields: tuple[str, ...], key: tuple[str, ...]) -> dict[str, Any]:
    deltas = [as_float(row, "delta_logp") for row in rows]
    deltas = [value for value in deltas if value is not None]
    simplified_logprobs = [as_float(row, "simplified_logprob") for row in rows]
    simplified_logprobs = [value for value in simplified_logprobs if value is not None]
    traditional_logprobs = [as_float(row, "traditional_logprob") for row in rows]
    traditional_logprobs = [value for value in traditional_logprobs if value is not None]
    simplified_num_tokens = [
        float(row["simplified_num_tokens"])
        for row in rows
        if isinstance(row.get("simplified_num_tokens"), int)
    ]
    traditional_num_tokens = [
        float(row["traditional_num_tokens"])
        for row in rows
        if isinstance(row.get("traditional_num_tokens"), int)
    ]

    out: dict[str, Any] = {field: value for field, value in zip(key_fields, key)}
    out.update(
        {
            "n": len(rows),
            "mean_delta_logp": mean(deltas),
            "std_delta_logp": stdev(deltas),
            "median_delta_logp": median(deltas),
            "mean_abs_delta_logp": mean([abs(value) for value in deltas]),
            "min_delta_logp": min(deltas) if deltas else float("nan"),
            "max_delta_logp": max(deltas) if deltas else float("nan"),
            "pct_traditional_favored": pct(sum(1 for value in deltas if value > 0), len(deltas)),
            "pct_simplified_favored": pct(sum(1 for value in deltas if value < 0), len(deltas)),
            "pct_tie": pct(sum(1 for value in deltas if value == 0), len(deltas)),
            "mean_simplified_logprob": mean(simplified_logprobs),
            "mean_traditional_logprob": mean(traditional_logprobs),
            "mean_simplified_num_tokens": mean(simplified_num_tokens),
            "mean_traditional_num_tokens": mean(traditional_num_tokens),
            "single_token_pair_count": sum(1 for row in rows if bool(row.get("single_token_pair"))),
            "pct_single_token_pair": pct(sum(1 for row in rows if bool(row.get("single_token_pair"))), len(rows)),
        }
    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_by(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row, key_fields)].append(row)

    summaries = [
        summarize_group(group_rows, key_fields=key_fields, key=key)
        for key, group_rows in grouped.items()
    ]
    summaries.sort(key=lambda row: tuple(str(row[field]) for field in key_fields))
    return summaries


def strongest_examples(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        delta = as_float(row, "delta_logp")
        if delta is None:
            continue
        scored.append((abs(delta), row))
    scored.sort(key=lambda item: item[0], reverse=True)

    examples: list[dict[str, Any]] = []
    for _, row in scored[:top_k]:
        prefix = str(row.get("prefix", ""))
        suffix = str(row.get("suffix", ""))
        examples.append(
            {
                "event_id": row.get("event_id", ""),
                "context_script": row.get("context_script", ""),
                "mapping_source": row.get("mapping_source", ""),
                "mapping_kind": row.get("mapping_kind", ""),
                "simplified": row.get("simplified", ""),
                "traditional": row.get("traditional", ""),
                "observed_char": row.get("observed_char", ""),
                "single_token_pair": row.get("single_token_pair", ""),
                "delta_logp": row.get("delta_logp", ""),
                "simplified_logprob": row.get("simplified_logprob", ""),
                "traditional_logprob": row.get("traditional_logprob", ""),
                "prefix_tail": prefix[-80:],
                "suffix_head": suffix[:40],
                "line_index": row.get("line_index", ""),
                "char_index": row.get("char_index", ""),
                "source_file": row.get("source_file", ""),
            }
        )
    return examples


def import_matplotlib():
    try:
        import os

        mpl_config_dir = output_root() / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        xdg_cache_dir = output_root() / ".cache"
        xdg_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = [
            "PingFang SC",
            "Heiti TC",
            "Songti SC",
            "Arial Unicode MS",
            "Noto Sans CJK SC",
            "Noto Sans CJK TC",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        return plt, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def group_label(row: dict[str, Any]) -> str:
    token_label = "1tok" if str(row.get("single_token_pair")) == "True" else "multi"
    kind = str(row.get("mapping_kind", ""))
    kind = (
        kind.replace("one_simplified_to_many_traditional", "1S-manyT")
        .replace("same_plus_distinct_traditional", "same+distinct")
        .replace("opencc_one_to_one_control", "OpenCC control")
    )
    return f"{row.get('context_script')} | {kind} | {token_label}"


def plot_delta_distribution(rows: list[dict[str, Any]], out_dir: Path, plt) -> Path:
    out_path = out_dir / "delta_distribution.png"
    by_context: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        delta = as_float(row, "delta_logp")
        if delta is not None:
            by_context[str(row.get("context_script", ""))].append(delta)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for context, deltas in sorted(by_context.items()):
        ax.hist(deltas, bins=32, alpha=0.55, label=context)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Delta Log Prob Distribution")
    ax.set_xlabel("delta_logp = logP(traditional) - logP(simplified)")
    ax.set_ylabel("Events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_context_corrected_delta_distribution(
    rows: list[dict[str, Any]], out_dir: Path, plt
) -> Path:
    out_path = out_dir / "context_corrected_delta_distribution.png"
    by_context: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        delta = as_float(row, "delta_logp")
        if delta is None:
            continue
        context = str(row.get("context_script", ""))
        if context == "hans":
            corrected_delta = -delta
        elif context == "hant":
            corrected_delta = delta
        else:
            continue
        by_context[context].append(corrected_delta)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for context, deltas in sorted(by_context.items()):
        ax.hist(deltas, bins=32, alpha=0.55, label=context)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("How Strongly the Model Chooses the Expected Script")
    ax.set_xlabel("Preference for the context's script; >0 = expected script chosen")
    ax.set_ylabel("Events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_aggregate_delta(aggregate_rows: list[dict[str, Any]], out_dir: Path, plt) -> Path:
    out_path = out_dir / "aggregate_mean_delta.png"
    rows = sorted(
        aggregate_rows,
        key=lambda row: (
            str(row.get("context_script", "")),
            str(row.get("mapping_source", "")),
            str(row.get("mapping_kind", "")),
            str(row.get("single_token_pair", "")),
        ),
    )
    labels = [group_label(row) for row in rows]
    values = [float(row["mean_delta_logp"]) for row in rows]
    errors = [float(row["std_delta_logp"]) for row in rows]
    colors = ["tab:green" if value >= 0 else "tab:red" for value in values]

    fig_width = max(7.5, 0.75 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    ax.bar(range(len(rows)), values, yerr=errors, color=colors, alpha=0.78, capsize=3)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Mean Delta by Context and Mapping Type")
    ax.set_ylabel("Mean delta_logp")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_top_pair_delta(
    pair_rows: list[dict[str, Any]],
    out_dir: Path,
    plt,
    top_k: int,
) -> Path:
    out_path = out_dir / "top_pair_mean_delta.png"
    rows = sorted(
        pair_rows,
        key=lambda row: abs(float(row["mean_delta_logp"])),
        reverse=True,
    )[:top_k]
    rows.reverse()
    labels = [
        f"{row.get('simplified')}→{row.get('traditional')} | {row.get('context_script')}"
        for row in rows
    ]
    values = [float(row["mean_delta_logp"]) for row in rows]
    colors = ["tab:green" if value >= 0 else "tab:red" for value in values]

    fig_height = max(4.5, 0.28 * len(rows))
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    ax.barh(range(len(rows)), values, color=colors, alpha=0.78)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title(f"Top {len(rows)} Pairs by Absolute Mean Delta")
    ax.set_xlabel("Mean delta_logp")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_single_token_share(aggregate_rows: list[dict[str, Any]], out_dir: Path, plt) -> Path:
    out_path = out_dir / "single_token_share.png"
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"single": 0, "multi": 0})
    for row in aggregate_rows:
        key = (
            str(row.get("context_script", "")),
            str(row.get("mapping_source", "")),
            str(row.get("mapping_kind", "")),
        )
        if str(row.get("single_token_pair")) == "True":
            grouped[key]["single"] += int(row["n"])
        else:
            grouped[key]["multi"] += int(row["n"])

    keys = sorted(grouped)
    labels = [
        group_label(
            {
                "context_script": key[0],
                "mapping_kind": key[2],
                "single_token_pair": "True",
            }
        ).replace(" | 1tok", "")
        for key in keys
    ]
    single_values = [grouped[key]["single"] for key in keys]
    multi_values = [grouped[key]["multi"] for key in keys]

    fig_width = max(7.5, 0.75 * len(keys))
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    ax.bar(range(len(keys)), single_values, label="single-token pair", color="tab:blue", alpha=0.78)
    ax.bar(
        range(len(keys)),
        multi_values,
        bottom=single_values,
        label="multi-token pair",
        color="tab:orange",
        alpha=0.78,
    )
    ax.set_title("Tokenization Shape by Group")
    ax.set_ylabel("Events")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def make_plots(
    *,
    valid_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    out_dir: Path,
    top_k_pairs: int,
) -> tuple[list[Path], str | None]:
    plt, error = import_matplotlib()
    if plt is None:
        return [], error

    plot_paths = [
        plot_delta_distribution(valid_rows, out_dir, plt),
        plot_context_corrected_delta_distribution(valid_rows, out_dir, plt),
        plot_aggregate_delta(aggregate_rows, out_dir, plt),
        plot_top_pair_delta(pair_rows, out_dir, plt, top_k=top_k_pairs),
        plot_single_token_share(aggregate_rows, out_dir, plt),
    ]
    return plot_paths, None


def format_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def write_quick_report(
    *,
    out_dir: Path,
    run_summary: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    plot_paths: list[Path],
    plot_error: str | None,
) -> Path:
    out_path = out_dir / "quick_report.md"
    top_aggregate = sorted(
        aggregate_rows,
        key=lambda row: abs(float(row["mean_delta_logp"])),
        reverse=True,
    )[:8]
    top_pairs = sorted(
        pair_rows,
        key=lambda row: abs(float(row["mean_delta_logp"])),
        reverse=True,
    )[:12]

    lines = [
        "# Log-Prob Summary",
        "",
        f"- Input rows: {run_summary['input_rows']}",
        f"- Valid scored rows: {run_summary['valid_scored_rows']}",
        f"- Aggregate groups: {run_summary['aggregate_groups']}",
        f"- Pair groups: {run_summary['pair_groups']}",
        "",
        "Positive `delta_logp` means traditional was favored; negative means simplified was favored.",
        "",
        "## Strongest Aggregate Groups",
        "",
        "| group | n | mean_delta_logp | pct_traditional_favored |",
        "|---|---:|---:|---:|",
    ]
    for row in top_aggregate:
        lines.append(
            "| "
            + group_label(row)
            + f" | {row['n']} | {format_float(row['mean_delta_logp'])} | "
            + f"{format_float(row['pct_traditional_favored'])} |"
        )

    lines.extend(
        [
            "",
            "## Strongest Pairs",
            "",
            "| pair | context | n | mean_delta_logp | pct_traditional_favored |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in top_pairs:
        pair = f"{row.get('simplified')}→{row.get('traditional')}"
        lines.append(
            f"| {pair} | {row.get('context_script')} | {row['n']} | "
            f"{format_float(row['mean_delta_logp'])} | "
            f"{format_float(row['pct_traditional_favored'])} |"
        )

    lines.extend(["", "## Plots", ""])
    if plot_error is not None:
        lines.append(f"Plots were skipped because matplotlib could not be imported: `{plot_error}`")
    else:
        for path in plot_paths:
            lines.append(f"- `{path.name}`")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    out = output_root()
    parser = argparse.ArgumentParser(
        description="Summarize simplified-vs-traditional next-token log-prob scores."
    )
    parser.add_argument(
        "--scored-jsonl",
        default=str(out / "scored_logprobs.jsonl"),
        help="Input scored JSONL from score_next_token_logprobs.py.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(out / "summary"),
        help="Output directory for summary CSVs. Must be inside 2_log_probs/out/.",
    )
    parser.add_argument(
        "--top-k-examples",
        type=int,
        default=50,
        help="Number of strongest absolute-delta examples to write.",
    )
    parser.add_argument(
        "--plot-top-k-pairs",
        type=int,
        default=30,
        help="Number of strongest pair groups to include in the top-pair plot.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG visualization outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k_examples <= 0:
        raise ValueError("--top-k-examples must be positive")
    if args.plot_top_k_pairs <= 0:
        raise ValueError("--plot-top-k-pairs must be positive")

    out_dir = ensure_under_output_root(Path(args.out_dir))
    rows = load_jsonl(Path(args.scored_jsonl))
    valid_rows = [row for row in rows if as_float(row, "delta_logp") is not None]
    skipped_rows = len(rows) - len(valid_rows)
    if not valid_rows:
        raise ValueError(f"No scored rows with numeric delta_logp found in {args.scored_jsonl}")

    aggregate_fields = ("context_script", "mapping_source", "mapping_kind", "single_token_pair")
    pair_fields = (
        "context_script",
        "mapping_source",
        "mapping_kind",
        "single_token_pair",
        "simplified",
        "traditional",
    )
    aggregate_rows = summarize_by(valid_rows, aggregate_fields)
    pair_rows = summarize_by(valid_rows, pair_fields)
    example_rows = strongest_examples(valid_rows, top_k=args.top_k_examples)

    metric_fields = [
        "n",
        "mean_delta_logp",
        "std_delta_logp",
        "median_delta_logp",
        "mean_abs_delta_logp",
        "min_delta_logp",
        "max_delta_logp",
        "pct_traditional_favored",
        "pct_simplified_favored",
        "pct_tie",
        "mean_simplified_logprob",
        "mean_traditional_logprob",
        "mean_simplified_num_tokens",
        "mean_traditional_num_tokens",
        "single_token_pair_count",
        "pct_single_token_pair",
    ]
    write_csv(
        out_dir / "aggregate_summary.csv",
        aggregate_rows,
        fieldnames=list(aggregate_fields) + metric_fields,
    )
    write_csv(
        out_dir / "pair_summary.csv",
        pair_rows,
        fieldnames=list(pair_fields) + metric_fields,
    )
    write_csv(
        out_dir / "strongest_examples.csv",
        example_rows,
        fieldnames=[
            "event_id",
            "context_script",
            "mapping_source",
            "mapping_kind",
            "simplified",
            "traditional",
            "observed_char",
            "single_token_pair",
            "delta_logp",
            "simplified_logprob",
            "traditional_logprob",
            "prefix_tail",
            "suffix_head",
            "line_index",
            "char_index",
            "source_file",
        ],
    )

    run_summary = {
        "input_rows": len(rows),
        "valid_scored_rows": len(valid_rows),
        "skipped_rows": skipped_rows,
        "aggregate_groups": len(aggregate_rows),
        "pair_groups": len(pair_rows),
        "strongest_examples": len(example_rows),
    }
    plot_paths: list[Path] = []
    plot_error = None
    if not args.no_plots:
        plot_paths, plot_error = make_plots(
            valid_rows=valid_rows,
            aggregate_rows=aggregate_rows,
            pair_rows=pair_rows,
            out_dir=out_dir,
            top_k_pairs=args.plot_top_k_pairs,
        )
    run_summary["plots"] = [str(path) for path in plot_paths]
    run_summary["plot_error"] = plot_error
    quick_report = write_quick_report(
        out_dir=out_dir,
        run_summary=run_summary,
        aggregate_rows=aggregate_rows,
        pair_rows=pair_rows,
        plot_paths=plot_paths,
        plot_error=plot_error,
    )
    run_summary["quick_report"] = str(quick_report)
    (out_dir / "summary_report.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Input rows: {len(rows)}")
    print(f"Valid scored rows: {len(valid_rows)}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Output directory: {out_dir}")
    print(f"Quick report: {quick_report}")
    if plot_error is not None:
        print(f"Plots skipped: {plot_error}")
    elif plot_paths:
        print(f"Plots written: {len(plot_paths)}")


if __name__ == "__main__":
    main()
