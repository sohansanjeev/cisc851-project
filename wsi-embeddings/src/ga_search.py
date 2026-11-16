from pathlib import Path
import numpy as np

from patch_data import load_slides, pool_features
from ridge_baseline import make_splits, train_ridge
from ga_operators import (
    create_population,
    tournament_select,
    set_crossover,
    find_neighbors,
    mutate,
)


def score_subsets(population, slide, model, cache):
    """Score selections using the source slide's known class."""
    keys = [tuple(individual) for individual in population]

    # Calculate scores only for selections not already stored.
    missing = list(dict.fromkeys(
        key for key in keys if key not in cache
    ))

    if missing:
        pooled = np.array([
            pool_features(slide["features"][list(key)])
            for key in missing
        ])

        scores = model.decision_function(pooled)

        # Reverse normal-slide scores so higher fitness is always better.
        direction = 2 * slide["label"] - 1
        values = direction * scores

        for key, value in zip(missing, values):
            cache[key] = float(value)

    return np.array([cache[key] for key in keys])


def run_ga(
    slide,
    model,
    k=20,
    population_size=32,
    generations=30,
    elites=2,
    tournament_size=3,
    crossover_rate=0.8,
    mutation_mode="mixed",
    seed=851,
):
    """Evolve a fixed-size patch selection on one source slide."""
    if not 1 <= elites < population_size:
        raise ValueError("Elites must be between 1 and population_size - 1")
    if not 1 <= tournament_size <= population_size:
        raise ValueError("Invalid tournament size")
    if generations < 0 or not 0 <= crossover_rate <= 1:
        raise ValueError("Invalid generation count or crossover rate")
    if mutation_mode not in ("global", "local", "mixed"):
        raise ValueError("Unknown mutation mode")

    rng = np.random.default_rng(seed)
    n_patches = len(slide["features"])
    population = create_population(
        n_patches, k, population_size, rng
    )

    neighbors = None
    if mutation_mode in ("local", "mixed"):
        neighbors = find_neighbors(slide["coords"])

    # This cache belongs to this source slide and fitted classifier.
    cache = {}
    fitness = score_subsets(population, slide, model, cache)
    evaluations = len(population)
    history = []

    for generation in range(generations + 1):
        best_index = int(np.argmax(fitness))
        history.append({
            "generation": generation,
            "best_fitness": float(fitness[best_index]),
            "mean_fitness": float(fitness.mean()),
            "evaluations": evaluations,
        })

        if generation == generations:
            break

        # Keep the best old selections without changing or rescoring them.
        elite_ids = np.argsort(-fitness, kind="stable")[:elites]
        elite_population = population[elite_ids].copy()
        elite_fitness = fitness[elite_ids].copy()

        offspring = []
        for _ in range(population_size - elites):
            parent_a = tournament_select(
                population, fitness, rng, tournament_size
            )
            parent_b = tournament_select(
                population, fitness, rng, tournament_size
            )

            if rng.random() < crossover_rate:
                child = set_crossover(parent_a, parent_b, rng)
            else:
                child = parent_a.copy()

            # Replace one patch while keeping k distinct selections.
            child = mutate(
                child, n_patches, rng,
                mode=mutation_mode, neighbors=neighbors
            )
            offspring.append(child)

        offspring = np.asarray(offspring)
        offspring_fitness = score_subsets(
            offspring, slide, model, cache
        )

        # Count every candidate attempt, including cached selections.
        evaluations += len(offspring)

        population = np.concatenate([elite_population, offspring])
        fitness = np.concatenate([elite_fitness, offspring_fitness])

    return {
        "indices": population[best_index].copy(),
        "fitness": float(fitness[best_index]),
        "history": history,
        "evaluations": evaluations,
    }


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data" / "embeddings"
    slides = load_slides(data_dir)

    fold, train_ids, source_ids, test_ids = next(make_splits(slides))
    assert set(train_ids).isdisjoint(source_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(source_ids).isdisjoint(test_ids)

    model = train_ridge(slides, train_ids, seed=851 + fold)
    source = slides[source_ids[0]]

    result = run_ga(source, model)
    selected = result["indices"]
    best_scores = [
        row["best_fitness"] for row in result["history"]
    ]

    # Check subset validity, elitism and the evaluation budget.
    assert len(selected) == len(np.unique(selected)) == 20
    assert np.all((selected >= 0) & (selected < len(source["features"])))
    assert np.all(np.diff(best_scores) >= -1e-12)
    assert result["evaluations"] == 932

    # Compare the recorded fitness with a direct classifier call.
    pooled = pool_features(source["features"][selected])
    direct_score = model.decision_function(pooled.reshape(1, -1))[0]
    expected_fitness = (2 * source["label"] - 1) * direct_score
    assert np.isclose(result["fitness"], expected_fitness)

    print(f"Source slide: {source['id']}")
    print(f"Selected patches: {len(selected)}")
    print(f"Candidate evaluations: {result['evaluations']}")

    for row in result["history"]:
        if row["generation"] % 5 == 0:
            print(
                f"Generation {row['generation']}: "
                f"best fitness = {row['best_fitness']:.4f}"
            )

    print("Fitness and generation-loop checks passed")