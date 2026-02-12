from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore

    _TQDM = True
except Exception:  # noqa: BLE001
    tqdm = None
    _TQDM = False

METRICS = ("cag", "cosine_s_t1", "cosine_s_t2", "cosine_t1_t2")
LABELS = {
    "cag": "CAG",
    "cosine_s_t1": "cos(s,t1)",
    "cosine_s_t2": "cos(s,t2)",
    "cosine_t1_t2": "cos(t1,t2)",
}
HALF_PAGE_WIDTH_IN = 3.35
HALF_PAGE_HEIGHT_IN = 4.8
AXIS_LABEL_FONTSIZE = 10


def default_csv_path() -> Path:
    return Path(__file__).resolve().parent / "one_to_two_layerwise_cag.csv"


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
            row["mapping_index"] = int(row["mapping_index"])
            row["layer_index"] = int(row["layer_index"])
            for metric in METRICS:
                row[metric] = float(row[metric])
            rows.append(row)

    if not rows:
        raise ValueError(f"No data rows in CSV: {csv_path}")
    return rows


def group_by_mapping(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["simplified"], row["traditional_1"], row["traditional_2"])
        grouped[key].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda r: r["layer_index"])
    return dict(grouped)


def layer_template(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(r["layer_index"]) for r in rows})


def plot_single_mapping(
    mapping_rows: list[dict[str, Any]],
    simplified: str,
    traditional_1: str,
    traditional_2: str,
    out_dir: Path,
) -> None:
    layers = [int(r["layer_index"]) for r in mapping_rows]
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
        vals = [float(r[metric]) for r in mapping_rows]
        axes[i].plot(layers, vals, marker="o", linewidth=2)
        axes[i].set_title(LABELS[metric], fontsize=AXIS_LABEL_FONTSIZE)
        axes[i].grid(alpha=0.3)

    axes[-1].set_xticks(layers)
    axes[-1].set_xticklabels([str(l) for l in layers], rotation=0, ha="center")
    axes[-1].set_xlabel("Layer", fontsize=AXIS_LABEL_FONTSIZE)

    fig.tight_layout(rect=[0, 0.02, 1, 1.0], pad=0.6)
    out_path = out_dir / f"mapping_{simplified}_{traditional_1}_{traditional_2}.png"
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


def plot_aggregate(rows: list[dict[str, Any]], out_dir: Path, num_mappings: int) -> None:
    layers = layer_template(rows)
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
        axes[i].fill_between(
            layers,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
            label="±1 std",
        )
        axes[i].set_title(LABELS[metric], fontsize=AXIS_LABEL_FONTSIZE)
        axes[i].grid(alpha=0.3)
        axes[i].legend(loc="best")

    axes[-1].set_xticks(layers)
    axes[-1].set_xticklabels([str(l) for l in layers], rotation=0, ha="center")
    axes[-1].set_xlabel("Layer", fontsize=AXIS_LABEL_FONTSIZE)
    fig.tight_layout(rect=[0, 0.02, 1, 1.0], pad=0.6)
    fig.savefig(out_dir / "aggregate_cag_metrics.png", dpi=180)
    plt.close(fig)

    with io.open(out_dir / "aggregate_cag_stats.csv", "w", encoding="utf-8", newline="") as handle:
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


def write_top_mappings(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    out_dir: Path,
    top_k: int,
) -> None:
    rows: list[dict[str, Any]] = []

    iterator = grouped.items()
    if _TQDM:
        iterator = tqdm(list(grouped.items()), desc="score_mappings", unit="mapping")

    for (s, t1, t2), mapping_rows in iterator:
        avg_cag = float(np.mean([float(r["cag"]) for r in mapping_rows]))
        last = mapping_rows[-1]
        rows.append(
            {
                "simplified": s,
                "traditional_1": t1,
                "traditional_2": t2,
                "avg_cag": avg_cag,
                "avg_cosine_s_t1": float(np.mean([float(r["cosine_s_t1"]) for r in mapping_rows])),
                "avg_cosine_s_t2": float(np.mean([float(r["cosine_s_t2"]) for r in mapping_rows])),
                "avg_cosine_t1_t2": float(np.mean([float(r["cosine_t1_t2"]) for r in mapping_rows])),
                "last_layer_name": str(last["layer_name"]),
                "last_layer_cag": float(last["cag"]),
                "last_layer_cosine_s_t1": float(last["cosine_s_t1"]),
                "last_layer_cosine_s_t2": float(last["cosine_s_t2"]),
                "last_layer_cosine_t1_t2": float(last["cosine_t1_t2"]),
            }
        )

    rows_by_cag = sorted(rows, key=lambda r: r["avg_cag"], reverse=True)
    out_csv = out_dir / "mapping_summary.csv"
    with io.open(out_csv, "w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with io.open(out_dir / "top_by_avg_cag.csv", "w", encoding="utf-8", newline="") as handle:
        fields = list(rows_by_cag[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows_by_cag[: top_k]:
            writer.writerow(r)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph one-to-two mBERT CAG metrics.")
    parser.add_argument(
        "--csv",
        default=str(default_csv_path()),
        help="Path to one_to_two_layerwise_cag.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "graphs"),
        help="Directory for generated graphs/statistics",
    )
    parser.add_argument(
        "--simplified",
        default=None,
        help="Simplified char for single-mapping mode",
    )
    parser.add_argument(
        "--traditional-1",
        default=None,
        help="Traditional char 1 for single-mapping mode (requires --traditional-2)",
    )
    parser.add_argument(
        "--traditional-2",
        default=None,
        help="Traditional char 2 for single-mapping mode (requires --traditional-1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k mappings for ranking outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    grouped = group_by_mapping(rows)

    single_mode = any(x is not None for x in [args.simplified, args.traditional_1, args.traditional_2])
    if single_mode:
        if not (args.simplified and args.traditional_1 and args.traditional_2):
            raise ValueError(
                "Single-mapping mode requires --simplified, --traditional-1, and --traditional-2."
            )
        # Canonical order for matching since generation sorted t1/t2.
        t1, t2 = sorted([args.traditional_1, args.traditional_2])
        key = (args.simplified, t1, t2)
        if key not in grouped:
            sample = ", ".join([f"{s}:{a},{b}" for s, a, b in list(grouped.keys())[:10]])
            raise KeyError(f"Mapping not found: {args.simplified}:{t1},{t2}. Samples: {sample}")
        plot_single_mapping(grouped[key], args.simplified, t1, t2, out_dir)
        print(f"Single-mapping graph written to: {out_dir}")
        return

    plot_aggregate(rows, out_dir, num_mappings=len(grouped))
    write_top_mappings(grouped, out_dir, top_k=max(1, args.top_k))
    print(f"Aggregate outputs written to: {out_dir}")
    print(f"Mappings analyzed: {len(grouped)}")
    print(f"Rows analyzed: {len(rows)}")


if __name__ == "__main__":
    main()
