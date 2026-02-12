from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

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

HALF_PAGE_WIDTH_IN = 3.35
HALF_PAGE_HEIGHT_IN = 2.35
AXIS_LABEL_FONTSIZE = 10


def default_csv_path() -> Path:
    return Path(__file__).resolve().parent / "one_to_two_layerwise_cag.csv"


def default_out_path() -> Path:
    return Path(__file__).resolve().parent / "graphs" / "aggregate_hi_lo_cosine.png"


def default_assignments_txt_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_assignments.txt")


def read_rows(csv_path: Path) -> list[dict[str, float | int | str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[dict[str, float | int | str]] = []
    with io.open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV appears empty: {csv_path}")

        iterator = reader
        if _TQDM:
            iterator = tqdm(reader, desc="read_csv", unit="row")

        for row in iterator:
            rows.append(
                {
                    "mapping_index": int(row["mapping_index"]),
                    "simplified": row["simplified"],
                    "traditional_1": row["traditional_1"],
                    "traditional_2": row["traditional_2"],
                    "layer_index": int(row["layer_index"]),
                    "cosine_s_t1": float(row["cosine_s_t1"]),
                    "cosine_s_t2": float(row["cosine_s_t2"]),
                }
            )

    if not rows:
        raise ValueError(f"No data rows in CSV: {csv_path}")
    return rows


def aggregate_hi_lo_cosine(
    rows: list[dict[str, float | int | str]],
) -> tuple[list[int], dict[str, list[float]], list[dict[str, str | int | float]]]:
    # Determine lower/higher once at layer 0 for each mapping, then keep that
    # assignment fixed when aggregating all other layers.
    by_mapping: dict[int, dict[str, str | dict[int, tuple[float, float]]]] = {}
    iterator = rows
    if _TQDM:
        iterator = tqdm(rows, desc="group_rows", unit="row")
    for row in iterator:
        mapping_idx = int(row["mapping_index"])
        layer = int(row["layer_index"])
        c1 = float(row["cosine_s_t1"])
        c2 = float(row["cosine_s_t2"])
        if mapping_idx not in by_mapping:
            by_mapping[mapping_idx] = {
                "simplified": str(row["simplified"]),
                "traditional_1": str(row["traditional_1"]),
                "traditional_2": str(row["traditional_2"]),
                "layers": {},
            }
        layers_dict = by_mapping[mapping_idx]["layers"]
        assert isinstance(layers_dict, dict)
        layers_dict[layer] = (c1, c2)

    by_layer_hi: dict[int, list[float]] = defaultdict(list)
    by_layer_lo: dict[int, list[float]] = defaultdict(list)
    assignments: list[dict[str, str | int | float]] = []

    mapping_iter = by_mapping.items()
    if _TQDM:
        mapping_iter = tqdm(list(by_mapping.items()), desc="aggregate_mappings", unit="mapping")
    for mapping_idx, payload in mapping_iter:
        layer_vals = payload["layers"]
        assert isinstance(layer_vals, dict)
        if 0 not in layer_vals:
            continue
        c1_l0, c2_l0 = layer_vals[0]
        t1 = str(payload["traditional_1"])
        t2 = str(payload["traditional_2"])
        if c1_l0 >= c2_l0:
            closer = t1
            farther = t2
        else:
            closer = t2
            farther = t1
        assignments.append(
            {
                "mapping_index": mapping_idx,
                "simplified": str(payload["simplified"]),
                "closer": closer,
                "farther": farther,
                "cos_s_t1_layer0": c1_l0,
                "cos_s_t2_layer0": c2_l0,
            }
        )

        c1_is_lower = c1_l0 <= c2_l0
        for layer, (c1, c2) in layer_vals.items():
            if c1_is_lower:
                by_layer_lo[layer].append(c1)
                by_layer_hi[layer].append(c2)
            else:
                by_layer_lo[layer].append(c2)
                by_layer_hi[layer].append(c1)

    layers = sorted(by_layer_hi.keys())
    hi_mean = [float(np.mean(by_layer_hi[l])) for l in layers]
    hi_std = [float(np.std(by_layer_hi[l])) for l in layers]
    lo_mean = [float(np.mean(by_layer_lo[l])) for l in layers]
    lo_std = [float(np.std(by_layer_lo[l])) for l in layers]

    stats = {
        "hi_mean": hi_mean,
        "hi_std": hi_std,
        "lo_mean": lo_mean,
        "lo_std": lo_std,
    }
    assignments.sort(key=lambda a: int(a["mapping_index"]))
    return layers, stats, assignments


def write_stats_csv(layers: list[int], stats: dict[str, list[float]], out_path: Path) -> Path:
    stats_path = out_path.with_suffix(".csv")
    with io.open(stats_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer_index", "metric", "mean", "std"])
        for i, layer in enumerate(layers):
            writer.writerow([layer, "higher_cosine_fixed_from_layer0", stats["hi_mean"][i], stats["hi_std"][i]])
            writer.writerow([layer, "lower_cosine_fixed_from_layer0", stats["lo_mean"][i], stats["lo_std"][i]])
    return stats_path


def write_assignments_txt(assignments: list[dict[str, str | int | float]], out_path: Path) -> Path:
    txt_path = default_assignments_txt_path(out_path)
    with io.open(txt_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("simplified\tcloser\tfarther\n")
        for item in assignments:
            handle.write(f"{item['simplified']}\t{item['closer']}\t{item['farther']}\n")
    return txt_path


def plot_hi_lo(layers: list[int], stats: dict[str, list[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN))

    hi_mean = np.asarray(stats["hi_mean"], dtype=float)
    hi_std = np.asarray(stats["hi_std"], dtype=float)
    lo_mean = np.asarray(stats["lo_mean"], dtype=float)
    lo_std = np.asarray(stats["lo_std"], dtype=float)
    x = np.asarray(layers, dtype=int)

    ax.plot(x, hi_mean, marker="o", linewidth=2, label="Higher cos(s,t) at layer 0")
    ax.fill_between(x, hi_mean - hi_std, hi_mean + hi_std, alpha=0.2)

    ax.plot(x, lo_mean, marker="o", linewidth=2, label="Lower cos(s,t) at layer 0")
    ax.fill_between(x, lo_mean - lo_std, lo_mean + lo_std, alpha=0.2)

    ax.set_xticks(layers)
    ax.set_xticklabels([str(v) for v in layers], rotation=0, ha="center")
    ax.set_xlabel("Layer", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Cosine Similarity", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=[0, 0.02, 1, 1.0], pad=0.6)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one figure with two layer-wise lines:\n"
            "(1) mean higher (chosen once at layer 0) with std band\n"
            "(2) mean lower (chosen once at layer 0) with std band"
        )
    )
    parser.add_argument(
        "--csv",
        default=str(default_csv_path()),
        help="Path to one_to_two_layerwise_cag.csv",
    )
    parser.add_argument(
        "--out",
        default=str(default_out_path()),
        help="Output PNG path for the graph",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)

    rows = read_rows(csv_path)
    layers, stats, assignments = aggregate_hi_lo_cosine(rows)
    plot_hi_lo(layers, stats, out_path)
    stats_path = write_stats_csv(layers, stats, out_path)
    assignments_path = write_assignments_txt(assignments, out_path)

    mappings = len({int(r["mapping_index"]) for r in rows})
    print(f"Input CSV: {csv_path}")
    print(f"Rows analyzed: {len(rows)}")
    print(f"Mappings analyzed: {mappings}")
    print(f"Plot written: {out_path}")
    print(f"Stats CSV written: {stats_path}")
    print(f"Assignments TXT written: {assignments_path}")


if __name__ == "__main__":
    main()
