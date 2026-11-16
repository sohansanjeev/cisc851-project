from pathlib import Path
import numpy as np

from patch_data import load_slides


def create_population(n_patches, k, population_size, rng):
    """Create random selections with k different patches each."""
    if not 1 <= k < n_patches:
        raise ValueError("k must be between 1 and n_patches - 1")

    return np.array([
        np.sort(rng.choice(n_patches, size=k, replace=False))
        for _ in range(population_size)
    ])


def tournament_select(population, fitness, rng, tournament_size=3):
    """Choose the highest-scoring individual from a random tournament."""
    fitness = np.asarray(fitness)
    candidates = rng.choice(
        len(population), size=tournament_size, replace=False
    )
    winner = candidates[np.argmax(fitness[candidates])]

    return population[winner].copy()


def set_crossover(parent_a, parent_b, rng):
    """Keep shared patches and fill the remaining places from the parents."""
    shared = np.intersect1d(parent_a, parent_b)
    remaining = np.setdiff1d(np.union1d(parent_a, parent_b), shared)
    needed = len(parent_a) - len(shared)

    if needed == 0:
        return shared.copy()

    selected = rng.choice(remaining, size=needed, replace=False)
    return np.sort(np.concatenate([shared, selected]))


def find_neighbors(coords):
    """Order patch locations by distance from each patch."""
    locations = np.asarray(coords, dtype=np.float64)
    differences = locations[:, None, :] - locations[None, :, :]
    distances_squared = (differences ** 2).sum(axis=2)

    return np.argsort(distances_squared, axis=1, kind="stable")


def mutate(
    individual, n_patches, rng, mode="global",
    neighbors=None, local_neighbors=10
):
    """Replace one selected patch with an unselected patch."""
    if mode not in ("global", "local", "mixed"):
        raise ValueError("Unknown mutation mode")

    if len(individual) >= n_patches:
        raise ValueError("Mutation needs at least one unselected patch")

    child = individual.copy()
    position = rng.integers(len(child))
    removed_patch = child[position]

    # Exclude every selected patch, including the one being removed.
    available = np.ones(n_patches, dtype=bool)
    available[child] = False

    use_local = mode == "local"
    if mode == "mixed":
        use_local = rng.random() < 0.5

    if use_local:
        if neighbors is None or local_neighbors < 1:
            raise ValueError("Local mutation needs valid neighbor settings")

        ordered = neighbors[removed_patch]
        choices = ordered[available[ordered]][:local_neighbors]
    else:
        choices = np.flatnonzero(available)

    child[position] = rng.choice(choices)
    return np.sort(child)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data" / "embeddings"
    slide = load_slides(data_dir)[0]
    rng = np.random.default_rng(851)

    n_patches = len(slide["features"])
    k = 20
    population = create_population(n_patches, k, 32, rng)
    neighbors = find_neighbors(slide["coords"])

    for individual in population:
        assert len(np.unique(individual)) == k
        assert np.all((individual >= 0) & (individual < n_patches))

    # Known scores check that tournament selection chooses the best candidate.
    test_scores = np.arange(len(population), dtype=float)
    winner = tournament_select(
        population, test_scores, rng, tournament_size=len(population)
    )
    assert np.array_equal(winner, population[-1])

    parent_a = tournament_select(population, test_scores, rng)
    parent_b = tournament_select(population, test_scores, rng)
    original_parent = parent_a.copy()

    for _ in range(25):
        child = set_crossover(parent_a, parent_b, rng)
        assert len(np.unique(child)) == k
        assert np.isin(child, np.union1d(parent_a, parent_b)).all()
        assert np.isin(np.intersect1d(parent_a, parent_b), child).all()

        for mode in ("global", "local", "mixed"):
            changed = mutate(
                parent_a, n_patches, rng, mode, neighbors
            )
            assert len(np.unique(changed)) == k
            assert np.all((changed >= 0) & (changed < n_patches))
            assert len(np.setdiff1d(parent_a, changed)) == 1
            assert len(np.setdiff1d(changed, parent_a)) == 1

    assert np.array_equal(parent_a, original_parent)

    # Known coordinates check that local mutation uses spatial distance.
    test_coords = np.array([
        [0, 0], [100, 100], [0, 1],
        [300, 300], [0, 2], [200, 200]
    ])
    test_neighbors = find_neighbors(test_coords)

    for _ in range(25):
        changed = mutate(
            np.array([0, 1]), 6, rng, mode="local",
            neighbors=test_neighbors, local_neighbors=1
        )
        assert set(changed) in ({1, 2}, {0, 4})

    print(f"Population shape: {population.shape}")
    print("Population, selection, crossover and mutation checks passed")