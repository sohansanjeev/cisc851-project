import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from patch_data import load_slides, pool_features
from ridge_baseline import make_splits, train_ridge
from ga_search import run_ga
from search_baselines import run_search_baseline
from prototype_transfer import predict_slide


SEARCH_METHODS = (
    "ga_global", "ga_local", "ga_mixed",
    "random_search", "hill_climb",
)


def write_csv(path, rows):
    """Save records with one result per row."""
    if not rows:
        return

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(predictions):
    """Calculate held-out metrics separately for each method and seed."""
    summaries = []
    groups = sorted({
        (row["method"], row["seed"]) for row in predictions
    })

    for method, seed in groups:
        rows = [
            row for row in predictions
            if row["method"] == method and row["seed"] == seed
        ]
        actual = [row["actual"] for row in rows]
        predicted = [row["predicted"] for row in rows]
        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        summaries.append({
            "method": method,
            "seed": seed,
            "n_slides": len(rows),
            "accuracy": accuracy_score(actual, predicted),
            "macro_f1": f1_score(
                actual, predicted, average="macro", zero_division=0
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return summaries


def prediction_row(fold, seed, method, slide, result):
    """Record the prediction and the patch indices used to produce it."""
    return {
        "fold": fold + 1,
        "seed": seed,
        "method": method,
        "slide": slide["id"],
        "actual": slide["label"],
        "predicted": result["prediction"],
        "score": result["score"],
        "patches": len(result["indices"]),
        "selected_indices": " ".join(map(str, result["indices"])),
    }


def save_tables(output, predictions, sources, histories):
    """Update the saved tables after each completed comparison."""
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "source_results.csv", sources)
    write_csv(output / "search_history.csv", histories)
    write_csv(output / "summary.csv", summarize(predictions))


def run_experiments(args):
    if not 1 <= args.fold_limit <= 5:
        raise ValueError("fold-limit must be between 1 and 5")
    if args.population < 3 or not 1 <= args.elites < args.population:
        raise ValueError("Invalid population or elite count")
    if args.generations < 0 or args.threads < 1:
        raise ValueError("Invalid generation or thread count")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Search seeds must be different")

    data_dir = Path(__file__).resolve().parents[1] / "data" / "embeddings"
    slides = load_slides(data_dir)

    if not 1 <= args.k < min(len(s["features"]) for s in slides):
        raise ValueError("Invalid selected patch count")

    # Keep the split seed fixed while changing the search seeds.
    splits = list(make_splits(slides, seed=851))[:args.fold_limit]
    budget = (
        args.population
        + args.generations * (args.population - args.elites)
    )

    # A fresh folder prevents an earlier run from being overwritten.
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {
        **vars(args),
        "output": str(args.output),
        "split_seed": 851,
        "split_unit": "slide",
        "candidate_budget": budget,
        "bags_per_slide": 24,
        "ridge_alpha": 10,
        "tournament_size": 3,
        "crossover_rate": 0.8,
        "mutation_rate": 1.0,
        "local_neighbors": 10,
        "complete": False,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(settings, indent=2))

    split_records = [{
        "fold": fold + 1,
        "classifier_slides": [slides[i]["id"] for i in train_ids],
        "source_slides": [slides[i]["id"] for i in source_ids],
        "test_slides": [slides[i]["id"] for i in test_ids],
    } for fold, train_ids, source_ids, test_ids in splits]

    (args.output / "splits.json").write_text(
        json.dumps(split_records, indent=2)
    )

    predictions = []
    sources = []
    histories = []
    started = time.perf_counter()

    for fold, train_ids, source_ids, test_ids in splits:
        assert set(train_ids).isdisjoint(source_ids)
        assert set(train_ids).isdisjoint(test_ids)
        assert set(source_ids).isdisjoint(test_ids)

        # Every method and search seed in this fold uses this same model.
        model = train_ridge(
            slides, train_ids, k=args.k,
            bags_per_slide=24, alpha=10, seed=851 + fold
        )

        for seed in args.seeds:
            initial_scores = {}

            for method in SEARCH_METHODS:
                print(
                    f"\nFold {fold + 1}/5 | seed {seed} | {method}",
                    flush=True,
                )
                bank_vectors = []

                for position, index in enumerate(source_ids):
                    slide = slides[index]
                    source_seed = seed + 101 * (int(index) + 1)

                    if method.startswith("ga_"):
                        result = run_ga(
                            slide, model,
                            k=args.k,
                            population_size=args.population,
                            generations=args.generations,
                            elites=args.elites,
                            tournament_size=3,
                            crossover_rate=0.8,
                            mutation_mode=method.split("_", 1)[1],
                            seed=source_seed,
                        )
                    else:
                        result = run_search_baseline(
                            slide, model, method,
                            k=args.k,
                            budget=budget,
                            population_size=args.population,
                            batch_size=args.population - args.elites,
                            seed=source_seed,
                        )

                    assert result["evaluations"] == budget
                    assert len(np.unique(result["indices"])) == args.k

                    # Confirm that all searches started from the same best score.
                    initial = result["history"][0]["best_fitness"]
                    if int(index) in initial_scores:
                        assert np.isclose(initial, initial_scores[int(index)])
                    else:
                        initial_scores[int(index)] = initial

                    bank_vectors.append(
                        slide["features"][result["indices"]]
                    )
                    sources.append({
                        "fold": fold + 1,
                        "seed": seed,
                        "method": method,
                        "slide": slide["id"],
                        "initial_fitness": initial,
                        "final_fitness": result["fitness"],
                        "evaluations": result["evaluations"],
                        "selected_indices": " ".join(
                            map(str, result["indices"])
                        ),
                    })

                    for row in result["history"]:
                        histories.append({
                            "fold": fold + 1,
                            "seed": seed,
                            "method": method,
                            "slide": slide["id"],
                            "evaluations": row["evaluations"],
                            "best_fitness": row["best_fitness"],
                        })

                    if (position + 1) % 4 == 0:
                        print(
                            f"  Sources completed: "
                            f"{position + 1}/{len(source_ids)}",
                            flush=True,
                        )

                bank = np.concatenate(bank_vectors, axis=0)

                for index in test_ids:
                    slide = slides[index]
                    result = predict_slide(
                        slide["features"], bank, model, k=args.k
                    )
                    predictions.append(
                        prediction_row(fold, seed, method, slide, result)
                    )

                save_tables(
                    args.output, predictions, sources, histories
                )

            # These references select target patches without source search.
            rng = np.random.default_rng(seed + 1000 + fold)

            for index in test_ids:
                slide = slides[index]
                choices = {
                    "random_target": rng.choice(
                        len(slide["features"]), args.k, replace=False
                    ),
                    "all_patches": np.arange(len(slide["features"])),
                }

                for method, selected in choices.items():
                    pooled = pool_features(slide["features"][selected])
                    score = float(
                        model.decision_function(pooled.reshape(1, -1))[0]
                    )
                    result = {
                        "indices": selected,
                        "score": score,
                        "prediction": int(score > 0),
                    }
                    predictions.append(
                        prediction_row(fold, seed, method, slide, result)
                    )

            save_tables(args.output, predictions, sources, histories)

    summaries = summarize(predictions)
    expected_slides = sum(len(split[3]) for split in splits)

    assert len(summaries) == 7 * len(args.seeds)
    for row in summaries:
        assert row["n_slides"] == expected_slides
        matching = [
            p for p in predictions
            if p["method"] == row["method"] and p["seed"] == row["seed"]
        ]
        assert len({p["slide"] for p in matching}) == expected_slides

    settings["complete"] = True
    settings["elapsed_seconds"] = time.perf_counter() - started
    config_path.write_text(json.dumps(settings, indent=2))

    print("\nRun complete")
    for row in summaries:
        print(
            f"{row['method']} | seed {row['seed']} | "
            f"slides {row['n_slides']} | "
            f"accuracy {row['accuracy']:.3f} | "
            f"macro-F1 {row['macro_f1']:.3f}"
        )

    print(f"\nFiles saved in: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[851, 852, 853])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--fold-limit", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    with threadpool_limits(limits=args.threads):
        run_experiments(args)