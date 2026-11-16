import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

# Save the figures without opening another window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "ga_global", "ga_local", "ga_mixed", "random_search",
    "hill_climb", "random_target", "all_patches",
)
SEARCH_METHODS = METHODS[:5]
LABELS = {
    "ga_global": "Global GA",
    "ga_local": "Local GA",
    "ga_mixed": "Mixed GA",
    "random_search": "Random search",
    "hill_climb": "Hill climbing",
    "random_target": "Random target patches",
    "all_patches": "All 500 target patches",
}
COLORS = {
    "ga_global": "#0072B2",
    "ga_local": "#009E73",
    "ga_mixed": "#CC79A7",
    "random_search": "#E69F00",
    "hill_climb": "#D55E00",
    "random_target": "#64748B",
    "all_patches": "#263238",
}


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def load_runs(results_dir):
    runs = {}
    for k in (10, 20, 40):
        folder = results_dir / f"k{k}_full"
        config = json.loads((folder / "config.json").read_text())
        if not config.get("complete") or config["k"] != k:
            raise ValueError(f"The run in {folder} is incomplete or has the wrong k.")

        rows = read_csv(folder / "summary.csv")
        keys = [(row["method"], int(row["seed"])) for row in rows]
        expected = {(method, seed) for method in METHODS for seed in config["seeds"]}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f"Missing or repeated summary rows in {folder}.")
        if any(int(row["n_slides"]) != 40 for row in rows):
            raise ValueError(f"Expected 40 held-out predictions per method and seed in {folder}.")
        runs[k] = {"folder": folder, "config": config, "summary": rows}
    return runs


def summarize_runs(runs):
    combined = []
    for k, run in runs.items():
        for method in METHODS:
            rows = [row for row in run["summary"] if row["method"] == method]
            accuracy = np.array([float(row["accuracy"]) for row in rows])
            macro_f1 = np.array([float(row["macro_f1"]) for row in rows])
            combined.append({
                "k": k,
                "method": method,
                "n_seeds": len(rows),
                "n_slides_per_seed": 40,
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_sd": float(accuracy.std(ddof=1)) if len(rows) > 1 else 0.0,
                "macro_f1_mean": float(macro_f1.mean()),
                "macro_f1_sd": float(macro_f1.std(ddof=1)) if len(rows) > 1 else 0.0,
            })
    return combined


def save_figure(figure, output_dir, name):
    for extension in ("png", "pdf"):
        path = output_dir / f"{name}.{extension}"
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close(figure)


def plot_search_and_transfer(run, output_dir):
    history = read_csv(run["folder"] / "search_history.csv")
    traces = defaultdict(list)
    for row in history:
        key = (row["method"], row["seed"], row["fold"], row["slide"])
        traces[key].append((int(row["evaluations"]), float(row["best_fitness"])))

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    for method in SEARCH_METHODS:
        gains = []
        evaluations = None
        for key, points in traces.items():
            if key[0] != method:
                continue
            points.sort()
            counts = np.array([point[0] for point in points])
            if evaluations is None:
                evaluations = counts
            elif not np.array_equal(evaluations, counts):
                raise ValueError("The search histories use different evaluation checkpoints.")
            values = np.array([point[1] for point in points])
            # Measure improvement from each search's own starting fitness.
            gains.append(values - values[0])
        if not gains:
            raise ValueError(f"No search history found for {method}.")
        axes[0].plot(
            evaluations, np.mean(gains, axis=0),
            label=LABELS[method], color=COLORS[method], linewidth=2,
        )

    axes[0].set_title("Source fitness during search")
    axes[0].set_xlabel("Candidate evaluations")
    axes[0].set_ylabel("Mean gain in best source fitness")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=9, frameon=False)

    for position, method in enumerate(METHODS):
        rows = [row for row in run["summary"] if row["method"] == method]
        rows.sort(key=lambda row: int(row["seed"]))
        accuracy = np.array([float(row["accuracy"]) * 100 for row in rows])
        mean = accuracy.mean()
        axes[1].barh(position, mean, color=COLORS[method], alpha=0.7, height=0.62)
        offsets = np.linspace(-0.16, 0.16, len(rows))
        axes[1].scatter(accuracy, position + offsets, color="black", s=20, zorder=3)
        # Keep the average separate from the dots for individual seeds.
        axes[1].text(
            1.02, position, f"{mean:.1f}",
            transform=axes[1].get_yaxis_transform(),
            ha="left", va="center", fontsize=9, clip_on=False,
        )

    axes[1].set_yticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].text(
        1.02, 1.025, "Mean", transform=axes[1].transAxes,
        ha="left", va="bottom", fontsize=9, clip_on=False,
    )
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].set_title("Accuracy on held-out slides")
    axes[1].set_axisbelow(True)
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle("20-patch search and transfer", fontsize=14)
    figure.text(
        0.02, 0.015,
        "Curves average 240 source searches per method. Bars show seed means; dots show individual seeds.\n"
        "Each seed evaluates the same 40 held-out slides across five folds. Seeds are not independent cohorts.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.95), w_pad=2.5)
    save_figure(figure, output_dir, "search_and_transfer")


def plot_patch_counts(combined, output_dir):
    lookup = {(row["k"], row["method"]): row for row in combined}
    patch_counts = (10, 20, 40)
    accuracy = np.array([
        [lookup[k, method]["accuracy_mean"] * 100 for k in patch_counts]
        for method in METHODS
    ])
    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    image = axis.imshow(accuracy, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(3), patch_counts)
    axis.set_yticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
    axis.set_xlabel("Patch count setting (k)")
    axis.set_title("Mean held-out accuracy by patch count")
    for row in range(len(METHODS)):
        for column in range(3):
            axis.text(
                column, row, f"{accuracy[row, column]:.1f}%",
                ha="center", va="center", color="white", fontsize=11,
            )
    figure.colorbar(image, ax=axis, label="Accuracy (%)", fraction=0.045, pad=0.04)
    figure.text(
        0.02, 0.015,
        "Means use three seeds on the same 40 slides. Ridge is retrained using k-patch training bags for each setting.\n"
        "The all-500 reference pools every target patch using that setting's ridge model.",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    save_figure(figure, output_dir, "patch_count_comparison")


def main():
    default_results = Path(__file__).resolve().parents[1] / "results"
    parser = argparse.ArgumentParser(description="Plot completed patch-selection experiments.")
    parser.add_argument("--results", type=Path, default=default_results)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_dir = args.output if args.output is not None else args.results / "figures"

    # Read the saved experiment results.
    runs = load_runs(args.results)
    combined = summarize_runs(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "combined_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"Saved: {summary_path}")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plot_search_and_transfer(runs[20], output_dir)
    plot_patch_counts(combined, output_dir)
    print("Figures and combined summary saved.")


if __name__ == "__main__":
    main()
