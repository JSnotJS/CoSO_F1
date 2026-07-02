import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_best_fit import apply_layout, forward_fill_series, load_xy_points


def discover_config_dirs(results_dir, config_paths):
    if config_paths:
        return [Path(path) for path in config_paths]

    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    return sorted(path for path in results_dir.iterdir() if path.is_dir())


def list_run_files(config_dir, pattern):
    return sorted(path for path in config_dir.glob(pattern) if path.is_file())


def aggregate_config_runs(run_files, x_field, y_field, events, dedupe):
    runs = []
    for path in run_files:
        try:
            x_values, y_values, _ = load_xy_points(path, x_field, y_field, events=events, dedupe=dedupe)
        except ValueError as exc:
            warnings.warn(str(exc), stacklevel=2)
            continue

        runs.append({
            "x": x_values,
            "y": y_values,
            "path": path,
        })

    if not runs:
        return None

    all_x_values = sorted({x_value for run in runs for x_value in run["x"]})
    rows = []
    for run in runs:
        series = dict(zip(run["x"], run["y"]))
        rows.append(forward_fill_series(series, all_x_values))

    matrix = np.vstack(rows)
    mean = np.nanmean(matrix, axis=0)
    valid = ~np.isnan(mean)

    if not np.any(valid):
        return None

    return {
        "x": np.array(all_x_values, dtype=float)[valid],
        "mean": mean[valid],
        "run_count": len(runs),
        "final_mean": float(mean[valid][-1]),
    }


def load_config_series(config_dirs, pattern, x_field, y_field, events, dedupe):
    series = []
    for config_dir in config_dirs:
        run_files = list_run_files(config_dir, pattern)
        if not run_files:
            warnings.warn(f"No run files found in {config_dir} using pattern {pattern!r}", stacklevel=2)
            continue

        aggregate = aggregate_config_runs(run_files, x_field, y_field, events, dedupe)
        if aggregate is None:
            warnings.warn(f"No plottable data found in {config_dir}", stacklevel=2)
            continue

        aggregate["label"] = config_dir.name
        aggregate["config_dir"] = config_dir
        series.append(aggregate)

    return series


def plot_config_series(series, output, title, x_label, y_label, show):
    best = max(series, key=lambda item: item["final_mean"])
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(len(series), 2)))

    for index, item in enumerate(series):
        is_best = item is best
        ax.plot(
            item["x"],
            item["mean"],
            color=colors[index],
            linewidth=3.0 if is_best else 1.8,
            alpha=1.0 if is_best else 0.9,
            label=item["label"],
        )
        ax.scatter(item["x"][-1], item["mean"][-1], color=colors[index], s=24 if is_best else 14)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or f"Mean {y_label} by {x_label}; best: {best['label']}")
    ax.grid(True, alpha=0.35)
    ax.legend(
        title="configuration folder",
        fontsize=8,
        title_fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    apply_layout(fig)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Saved plot to {output}")
    print(f"Best final mean: {best['label']} = {best['final_mean']:.6g}")

    print("Final mean ranking:")
    for rank, item in enumerate(sorted(series, key=lambda value: value["final_mean"], reverse=True), start=1):
        print(f"{rank:02d}. {item['label']}: {item['final_mean']:.6g} ({item['run_count']} runs)")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot one comparison chart with one mean best_fit line per experiment configuration folder."
    )
    parser.add_argument("configs", nargs="*", help="Configuration directories. If omitted, uses child directories of --results-dir.")
    parser.add_argument("--results-dir", default="results", help="Directory containing 00_..., 01_... configuration folders")
    parser.add_argument("--pattern", default="*.jsonl", help="Run file glob used inside each configuration directory")
    parser.add_argument("--x", default="full_evals", help="JSON field for X axis")
    parser.add_argument("--y", default="best_fit", help="JSON field for Y axis")
    parser.add_argument("--event", action="append", help="Only use records with this event name; default: progress")
    parser.add_argument("--all-events", action="store_true", help="Use all events instead of the default progress events")
    parser.add_argument("--dedupe", choices=["last", "max", "min", "mean", "none"], default="last", help="How to handle multiple Y values for the same X inside one run")
    parser.add_argument("--output", "-o", default="plots/config_comparison_best_fit.png", help="Output image path")
    parser.add_argument("--title", help="Plot title")
    parser.add_argument("--expected-configs", type=int, default=16, help="Warn if a different number of configuration series is plotted")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window")
    args = parser.parse_args()

    events = None if args.all_events else (args.event or ["progress"])
    config_dirs = discover_config_dirs(args.results_dir, args.configs)
    if not config_dirs:
        raise SystemExit(f"No configuration directories found in {args.results_dir}")

    series = load_config_series(config_dirs, args.pattern, args.x, args.y, events, args.dedupe)
    if not series:
        raise SystemExit("No plottable configuration series found")

    if args.expected_configs > 0 and len(series) != args.expected_configs:
        warnings.warn(f"Plotting {len(series)} configuration series, expected {args.expected_configs}", stacklevel=1)

    plot_config_series(series, args.output, args.title, args.x, args.y, args.show)


if __name__ == "__main__":
    main()


# Example:
# python scripts/plot_config_comparison.py --results-dir results --output plots/config_comparison_best_fit.png
