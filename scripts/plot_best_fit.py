import argparse
import glob
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def iter_input_files(patterns):
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        else:
            files.append(pattern)

    unique = []
    seen = set()
    for file in files:
        path = Path(file)
        if path not in seen:
            seen.add(path)
            unique.append(path)

    return unique


def get_numeric_field(record, field):
    value = record.get(field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reduce_values(values, mode):
    if mode == "last":
        return values[-1]
    if mode == "max":
        return max(values)
    if mode == "min":
        return min(values)
    if mode == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unknown dedupe mode: {mode}")


def load_xy_points(path, x_field, y_field, events=None, dedupe="last"):
    points = {}
    raw_points = []
    meta = {}
    allowed_events = set(events) if events else None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            if record.get("event") == "meta":
                meta = record
                continue

            if allowed_events is not None and record.get("event") not in allowed_events:
                continue

            x_value = get_numeric_field(record, x_field)
            y_value = get_numeric_field(record, y_field)
            if x_value is None or y_value is None:
                continue

            if dedupe == "none":
                raw_points.append((x_value, y_value))
            else:
                points.setdefault(x_value, []).append(y_value)

    if dedupe == "none":
        if not raw_points:
            raise ValueError(f"No {x_field}/{y_field} points found in {path}")
        raw_points.sort(key=lambda point: point[0])
        x_values = [point[0] for point in raw_points]
        y_values = [point[1] for point in raw_points]
        return x_values, y_values, meta

    if not points:
        raise ValueError(f"No {x_field}/{y_field} points found in {path}")

    x_values = sorted(points)
    y_values = [reduce_values(points[x_value], dedupe) for x_value in x_values]
    return x_values, y_values, meta


def run_label(path, meta):
    seed = meta.get("seed")
    if seed is not None:
        return f"seed={seed}"
    return path.stem


def forward_fill_series(series_by_x, x_values):
    filled = []
    last = np.nan
    for x_value in x_values:
        if x_value in series_by_x:
            last = series_by_x[x_value]
        filled.append(last)
    return np.array(filled, dtype=float)


def run_should_be_labeled(run_count, aggregate, legend):
    if legend == "none" or legend == "summary":
        return False
    if legend == "all":
        return True
    return not aggregate and run_count <= 10


def summary_should_be_labeled(legend):
    return legend != "none"


def apply_layout(fig):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        fig.tight_layout()

    if any("Tight layout not applied" in str(warning.message) for warning in caught):
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.90)


def plot_runs(runs, output, title, aggregate, show, x_label, y_label, legend):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    label_runs = run_should_be_labeled(len(runs), aggregate, legend)
    plot_individual_runs = not aggregate or len(runs) == 1

    if plot_individual_runs:
        for run in runs:
            label = run["label"] if label_runs else "_nolegend_"
            ax.plot(run["x"], run["y"], linewidth=1.4, alpha=0.9, label=label)

    if aggregate and len(runs) > 1:
        all_x_values = sorted({x_value for run in runs for x_value in run["x"]})
        rows = []
        for run in runs:
            series = dict(zip(run["x"], run["y"]))
            rows.append(forward_fill_series(series, all_x_values))

        matrix = np.vstack(rows)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)

        mean_label = "mean" if summary_should_be_labeled(legend) else "_nolegend_"
        std_label = "±1 std" if summary_should_be_labeled(legend) else "_nolegend_"
        ax.plot(all_x_values, mean, color="black", linewidth=2.4, label=mean_label)
        ax.fill_between(all_x_values, mean - std, mean + std, color="black", alpha=0.12, label=std_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    visible = [(handle, label) for handle, label in zip(handles, labels) if label != "_nolegend_"]
    if visible:
        ax.legend(*zip(*visible))

    apply_layout(fig)

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot selected numeric fields from CoSO JSONL result files.")
    parser.add_argument("inputs", nargs="+", help="Input .jsonl files or glob patterns, e.g. results_*.jsonl")
    parser.add_argument("--x", default="generation", help="JSON field for X axis, e.g. generation, full_evals, evals, elapsed_s")
    parser.add_argument("--y", default="best_fit", help="JSON field for Y axis, e.g. best_fit, current_fit, fit")
    parser.add_argument("--event", action="append", help="Only use records with this event name; can be used multiple times")
    parser.add_argument("--dedupe", choices=["last", "max", "min", "mean", "none"], default="last", help="How to handle multiple Y values for the same X")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument("--title", help="Plot title")
    parser.add_argument("--aggregate", action="store_true", help="For multiple runs, plot only the mean line and +/- 1 std band")
    parser.add_argument("--legend", choices=["auto", "all", "summary", "none"], default="auto", help="Legend mode; auto hides per-run labels when aggregating many runs")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window")
    args = parser.parse_args()

    title = args.title or f"{args.y} by {args.x}"
    output = args.output or f"plots/{args.y}_by_{args.x}.png"

    files = iter_input_files(args.inputs)
    if not files:
        raise SystemExit("No input files found")

    runs = []
    for path in files:
        x_values, y_values, meta = load_xy_points(path, args.x, args.y, events=args.event, dedupe=args.dedupe)
        runs.append({
            "path": path,
            "label": run_label(path, meta),
            "x": x_values,
            "y": y_values,
            "meta": meta,
        })

    plot_runs(runs, output, title, args.aggregate, args.show, args.x, args.y, args.legend)


if __name__ == "__main__":
    main()

# Użycie:
# python scripts/plot_best_fit.py "results_*.jsonl" --output "plots/best_fit_by_generation.png"
# python scripts/plot_best_fit.py "results_*.jsonl" --x full_evals --y best_fit --aggregate
# python scripts/plot_best_fit.py "results_*.jsonl" --x elapsed_s --y best_fit --event progress
# python scripts/plot_best_fit.py "results_*.jsonl" --aggregate --legend summary

# Dla wielu runów z linią średnią i pasmem ±1 std:
# python scripts/plot_best_fit.py "results_*.jsonl" --output "plots/best_fit_by_generation.png" --aggregate

# python scripts\plot_best_fit.py "results_*.jsonl" --x full_evals --y best_fit --output "plots\5000_e10_del1_cog45.png"
# Saved

