from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from patch_data import load_slides, pool_features
from ridge_baseline import make_splits, train_ridge
from ga_search import run_ga


def normalize_vectors(features):
    """Give each vector a length of one for cosine matching."""
    values = np.asarray(features, dtype=np.float32)
    lengths = np.linalg.norm(values, axis=1, keepdims=True)

    return values / np.maximum(lengths, 1e-12)


def select_similar_patches(target_features, bank, k=20):
    """Select target patches most similar to the source bank."""
    if not 1 <= k <= len(target_features):
        raise ValueError("Invalid target patch count")
    if len(bank) == 0:
        raise ValueError("The prototype bank is empty")
    if target_features.shape[1] != bank.shape[1]:
        raise ValueError("Target and source feature dimensions differ")

    target_vectors = normalize_vectors(target_features)
    source_vectors = normalize_vectors(bank)

    # Each row compares one target patch with all source examples.
    similarities = target_vectors @ source_vectors.T
    best_matches = similarities.max(axis=1)

    # Select different target patches using their strongest matches.
    selected = np.argsort(-best_matches, kind="stable")[:k]
    return np.sort(selected)


def predict_slide(target_features, bank, model, k=20):
    """Select patches and predict the slide without using its label."""
    selected = select_similar_patches(target_features, bank, k)
    pooled = pool_features(target_features[selected])
    score = float(model.decision_function(pooled.reshape(1, -1))[0])

    return {
        "indices": selected,
        "score": score,
        "prediction": int(score > 0),
    }


def build_prototype_bank(
    slides, source_ids, model, k=20, seed=851,
    mutation_mode="mixed", verbose=True, **ga_settings
):
    """Evolve source selections and collect their actual feature vectors."""
    vectors = []
    records = []

    for position, index in enumerate(source_ids):
        slide = slides[index]
        source_seed = seed + 101 * (int(index) + 1)

        result = run_ga(
            slide,
            model,
            k=k,
            seed=source_seed,
            mutation_mode=mutation_mode,
            **ga_settings,
        )

        # Retrieve the vectors identified by the evolved patch indices.
        vectors.append(slide["features"][result["indices"]])
        records.append({
            "slide": slide["id"],
            "label": slide["label"],
            "indices": result["indices"].tolist(),
            "initial_fitness": result["history"][0]["best_fitness"],
            "final_fitness": result["fitness"],
            "evaluations": result["evaluations"],
            "history": result["history"],
        })

        if verbose:
            print(
                f"Source {position + 1}/{len(source_ids)}: "
                f"{slide['id']}, fitness = {result['fitness']:.4f}",
                flush=True,
            )

    return np.concatenate(vectors, axis=0), records


if __name__ == "__main__":
    # Known vectors check that cosine matching selects the expected patches.
    test_features = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.9, 0.1],
    ])
    test_bank = np.array([[1.0, 0.0]])
    selected = select_similar_patches(test_features, test_bank, k=2)
    assert np.array_equal(selected, [0, 3])

    data_dir = Path(__file__).resolve().parents[1] / "data" / "embeddings"
    slides = load_slides(data_dir)
    k = 20

    fold, train_ids, source_ids, test_ids = next(make_splits(slides))
    assert set(train_ids).isdisjoint(source_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(source_ids).isdisjoint(test_ids)

    model = train_ridge(slides, train_ids, k=k, seed=851 + fold)
    bank, source_records = build_prototype_bank(
        slides, source_ids, model, k=k
    )

    expected_shape = (len(source_ids) * k, slides[0]["features"].shape[1])
    assert bank.shape == expected_shape
    print(f"\nPrototype bank shape: {bank.shape}")

    actual_labels = []
    predictions = []
    names = {0: "normal", 1: "tumour"}

    for index in test_ids:
        slide = slides[index]

        # Prediction receives the target features, not the target label.
        result = predict_slide(slide["features"], bank, model, k=k)
        assert len(np.unique(result["indices"])) == k

        # Use the true label afterward to evaluate the prediction.
        actual_labels.append(slide["label"])
        predictions.append(result["prediction"])

        print(
            f"{slide['id']}: actual = {names[slide['label']]}, "
            f"predicted = {names[result['prediction']]}, "
            f"ridge score = {result['score']:.3f}"
        )

    print(f"\nHeld-out slides: {len(predictions)}")
    print(f"Accuracy: {accuracy_score(actual_labels, predictions):.3f}")
    print(
        f"Macro-F1: "
        f"{f1_score(actual_labels, predictions, average='macro', zero_division=0):.3f}"
    )
    print("Confusion matrix: rows actual, columns predicted")
    print("Class order: normal, tumour")
    print(confusion_matrix(actual_labels, predictions, labels=[0, 1]))
    print("Cosine matching and transfer checks passed")