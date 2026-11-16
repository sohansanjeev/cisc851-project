from pathlib import Path
import numpy as np

from patch_data import load_slides, pool_features
from ridge_baseline import make_splits, train_ridge
from ga_operators import create_population, mutate
from ga_search import score_subsets


def run_search_baseline(
    slide, model, method, k=20, budget=932,
    population_size=32, batch_size=30, seed=851
):
    """Search with the same candidate budget as the GA."""
    if method not in ("random_search", "hill_climb"):
        raise ValueError("Unknown search method")
    if not 1 <= population_size <= budget or batch_size < 1:
        raise ValueError("Invalid search budget or batch size")

    rng = np.random.default_rng(seed)
    n_patches = len(slide["features"])
    cache = {}

    # Use the same starting population as the GA for this seed.
    initial = create_population(n_patches, k, population_size, rng)
    fitness = score_subsets(initial, slide, model, cache)

    best_index = int(np.argmax(fitness))
    best = initial[best_index].copy()
    best_fitness = float(fitness[best_index])
    evaluations = population_size

    history = [{
        "evaluations": evaluations,
        "best_fitness": best_fitness,
    }]

    while evaluations < budget:
        count = min(batch_size, budget - evaluations)

        if method == "random_search":
            # Draw fresh subsets without using parents or crossover.
            candidates = create_population(n_patches, k, count, rng)
            values = score_subsets(candidates, slide, model, cache)
            winner = int(np.argmax(values))

            if values[winner] > best_fitness:
                best = candidates[winner].copy()
                best_fitness = float(values[winner])

        else:
            # Accept an improving swap before making the next proposal.
            for _ in range(count):
                candidate = mutate(
                    best, n_patches, rng, mode="global"
                )
                value = float(
                    score_subsets([candidate], slide, model, cache)[0]
                )

                if value > best_fitness:
                    best = candidate.copy()
                    best_fitness = value

        # Count attempted candidates even when their scores are cached.
        evaluations += count
        history.append({
            "evaluations": evaluations,
            "best_fitness": best_fitness,
        })

    return {
        "indices": best,
        "fitness": best_fitness,
        "history": history,
        "evaluations": evaluations,
    }


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data" / "embeddings"
    slides = load_slides(data_dir)

    fold, train_ids, source_ids, test_ids = next(make_splits(slides))
    model = train_ridge(slides, train_ids, seed=851 + fold)
    source = slides[source_ids[0]]
    starting_scores = []

    print(f"Source slide: {source['id']}")

    for method in ("random_search", "hill_climb"):
        result = run_search_baseline(source, model, method)
        selected = result["indices"]
        scores = [row["best_fitness"] for row in result["history"]]

        assert len(selected) == len(np.unique(selected)) == 20
        assert np.all((selected >= 0) & (selected < len(source["features"])))
        assert result["evaluations"] == 932
        assert np.all(np.diff(scores) >= -1e-12)

        pooled = pool_features(source["features"][selected])
        direct_score = model.decision_function(pooled.reshape(1, -1))[0]
        expected = (2 * source["label"] - 1) * direct_score
        assert np.isclose(result["fitness"], expected)

        starting_scores.append(scores[0])
        print(
            f"{method}: initial fitness = {scores[0]:.4f}, "
            f"final fitness = {result['fitness']:.4f}, "
            f"evaluations = {result['evaluations']}"
        )

    assert np.isclose(starting_scores[0], starting_scores[1])
    print("Search baseline checks passed")