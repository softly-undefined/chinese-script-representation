# this file analyzes the results of one_to_one_mappings.py
# it should visualize L2 distance, cosine distance, and cosine similarity across layers for all pairs
# it should run two ways depending on args:
# - on a specific pair, creating specific plots for that pair
# - on the whole dataset, creating aggregate plots and statistics

from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

# Headless backend for CLI/script usage.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False

METRICS = ("l2_distance", "cosine_similarity")
LABELS = {
    "l2_distance": "L2 Distance",
    "cosine_similarity": "Cosine Similarity",
}
# ACL 2-column width is ~6.9in, so half-page target is ~3.45in.
HALF_PAGE_WIDTH_IN = 3.35
HALF_PAGE_HEIGHT_IN = 4.0
AXIS_LABEL_FONTSIZE = 10


def default_csv_path() -> Path:
    return Path(__file__).resolve().parent / "one_to_one_layerwise_distances.csv"


def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with io.open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV appears empty: {csv_path}")

        iterator = reader
        if _TQDM:
            iterator = tqdm(reader, desc="read_csv", unit="row")

        for row in iterator:
            row["pair_index"] = int(row["pair_index"])
            row["layer_index"] = int(row["layer_index"])
            for metric in METRICS:
                row[metric] = float(row[metric])
            rows.append(row)

    if not rows:
        raise ValueError(f"No data rows in CSV: {csv_path}")
    return rows


def group_by_pair(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["traditional"], row["simplified"])
        grouped[key].append(row)
    for pair in grouped:
        grouped[pair].sort(key=lambda r: r["layer_index"])
    return dict(grouped)


def layer_template(rows: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    layers = sorted({int(r["layer_index"]) for r in rows})
    layer_names_by_index = {int(r["layer_index"]): str(r["layer_name"]) for r in rows}
    names = [layer_names_by_index[i] for i in layers]
    return layers, names


def plot_single_pair(
    pair_rows: list[dict[str, Any]],
    traditional: str,
    simplified: str,
    out_dir: Path,
) -> None:
    layers = [int(r["layer_index"]) for r in pair_rows]

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN),
        sharex=True,
    )
    if n_metrics == 1:
        axes = [axes]
    for i, metric in enumerate(METRICS):
        vals = [float(r[metric]) for r in pair_rows]
        axes[i].plot(layers, vals, marker="o", linewidth=2)
        axes[i].set_title(LABELS[metric], fontsize=AXIS_LABEL_FONTSIZE)
        axes[i].grid(alpha=0.3)

    axes[-1].set_xticks(layers)
    axes[-1].set_xticklabels([str(l) for l in layers], rotation=0, ha="center")
    axes[-1].set_xlabel("Layer", fontsize=AXIS_LABEL_FONTSIZE)

    fig.tight_layout(rect=[0, 0.02, 1, 1.0], pad=0.6)

    out_path = out_dir / f"pair_{traditional}_{simplified}_metrics.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def aggregate_by_layer(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, float]]]:
    by_metric_layer: dict[str, dict[int, list[float]]] = {
        metric: defaultdict(list) for metric in METRICS
    }
    for row in rows:
        layer_idx = int(row["layer_index"])
        for metric in METRICS:
            by_metric_layer[metric][layer_idx].append(float(row[metric]))

    stats: dict[str, dict[int, dict[str, float]]] = {metric: {} for metric in METRICS}
    for metric in METRICS:
        for layer_idx, vals in by_metric_layer[metric].items():
            arr = np.asarray(vals, dtype=float)
            stats[metric][layer_idx] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
            }
    return stats


def plot_aggregate(rows: list[dict[str, Any]], out_dir: Path, num_pairs: int) -> None:
    layers, _ = layer_template(rows)
    stats = aggregate_by_layer(rows)

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN),
        sharex=True,
    )
    if n_metrics == 1:
        axes = [axes]
    for i, metric in enumerate(METRICS):
        means = [stats[metric][l]["mean"] for l in layers]
        stds = [stats[metric][l]["std"] for l in layers]
        axes[i].plot(layers, means, marker="o", linewidth=2, label="mean")
        axes[i].fill_between(layers, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2, label="±1 std")
        axes[i].set_title(LABELS[metric], fontsize=AXIS_LABEL_FONTSIZE)
        axes[i].grid(alpha=0.3)
        axes[i].legend(loc="best")

    axes[-1].set_xticks(layers)
    axes[-1].set_xticklabels([str(l) for l in layers], rotation=0, ha="center")
    axes[-1].set_xlabel("Layer", fontsize=AXIS_LABEL_FONTSIZE)
    fig.tight_layout(rect=[0, 0.02, 1, 1.0], pad=0.6)
    fig.savefig(out_dir / "aggregate_layerwise_metrics.png", dpi=180)
    plt.close(fig)

    with io.open(out_dir / "aggregate_layerwise_stats.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer_index", "layer_name", "metric", "mean", "std", "median"])
        for layer_idx in layers:
            for metric in METRICS:
                writer.writerow(
                    [
                        layer_idx,
                        str(layer_idx),
                        metric,
                        stats[metric][layer_idx]["mean"],
                        stats[metric][layer_idx]["std"],
                        stats[metric][layer_idx]["median"],
                    ]
                )


def write_top_pairs(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    out_dir: Path,
    top_k: int,
) -> None:
    rows: list[dict[str, Any]] = []

    iterator = grouped.items()
    if _TQDM:
        iterator = tqdm(list(grouped.items()), desc="score_pairs", unit="pair")

    for (traditional, simplified), pair_rows in iterator:
        avg_l2 = float(np.mean([float(r["l2_distance"]) for r in pair_rows]))
        avg_cos_sim = float(np.mean([float(r["cosine_similarity"]) for r in pair_rows]))
        last = pair_rows[-1]
        rows.append(
            {
                "traditional": traditional,
                "simplified": simplified,
                "avg_l2_distance": avg_l2,
                "avg_cosine_similarity": avg_cos_sim,
                "last_layer_name": str(last["layer_name"]),
                "last_layer_l2_distance": float(last["l2_distance"]),
                "last_layer_cosine_similarity": float(last["cosine_similarity"]),
            }
        )

    rows_by_l2 = sorted(rows, key=lambda r: r["avg_l2_distance"], reverse=True)
    rows_by_sim = sorted(rows, key=lambda r: r["avg_cosine_similarity"])

    out_csv = out_dir / "top_pairs_summary.csv"
    with io.open(out_csv, "w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    def _write_subset(name: str, subset: list[dict[str, Any]]) -> None:
        with io.open(out_dir / name, "w", encoding="utf-8", newline="") as handle:
            fields = list(subset[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in subset[:top_k]:
                writer.writerow(r)

    _write_subset("top_by_avg_l2_distance.csv", rows_by_l2)
    _write_subset("bottom_by_avg_cosine_similarity.csv", rows_by_sim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph one-to-one mBERT layer-wise distance metrics."
    )
    parser.add_argument(
        "--csv",
        default=str(default_csv_path()),
        help="Path to one_to_one_layerwise_distances.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "graphs"),
        help="Directory for generated graphs/statistics",
    )
    parser.add_argument(
        "--traditional",
        default=None,
        help="Traditional character for single-pair mode (requires --simplified).",
    )
    parser.add_argument(
        "--simplified",
        default=None,
        help="Simplified character for single-pair mode (requires --traditional).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k pair count for aggregate ranking outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    grouped = group_by_pair(rows)

    single_mode = args.traditional is not None or args.simplified is not None
    if single_mode:
        if not (args.traditional and args.simplified):
            raise ValueError("Single-pair mode requires both --traditional and --simplified.")
        key = (args.traditional, args.simplified)
        if key not in grouped:
            sample = ", ".join([f"{t}->{s}" for t, s in list(grouped.keys())[:10]])
            raise KeyError(f"Pair not found: {args.traditional}->{args.simplified}. Sample pairs: {sample}")
        plot_single_pair(grouped[key], args.traditional, args.simplified, out_dir)
        print(f"Single-pair graph written to: {out_dir}")
        return

    plot_aggregate(rows, out_dir, num_pairs=len(grouped))
    write_top_pairs(grouped, out_dir, top_k=max(1, args.top_k))
    print(f"Aggregate outputs written to: {out_dir}")
    print(f"Pairs analyzed: {len(grouped)}")
    print(f"Rows analyzed: {len(rows)}")


if __name__ == "__main__":
    main()
