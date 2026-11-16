from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from patch_data import load_slides, pool_features


def make_splits(slides, folds=5, seed=851):
    """Separate classifier training, source evolution and testing."""
    labels = np.array([slide["label"] for slide in slides])
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )

    for fold, (development, test_ids) in enumerate(
        splitter.split(np.zeros(len(slides)), labels)
    ):
        # Reserve half the development slides for the EA.
        train_ids, source_ids = train_test_split(
            development,
            test_size=0.5,
            stratify=labels[development],
            random_state=seed + fold + 1,
        )

        yield (
            fold,
            np.sort(train_ids),
            np.sort(source_ids),
            np.sort(test_ids),
        )


def train_ridge(
    slides, train_ids, k=20, bags_per_slide=24, alpha=10, seed=851
):
    """Fit ridge using random patch subsets from training slides."""
    rng = np.random.default_rng(seed)
    inputs = []
    labels = []

    for index in train_ids:
        slide = slides[index]

        for _ in range(bags_per_slide):
            # Choose k different patches and pool their feature vectors.
            selected = rng.choice(
                len(slide["features"]), size=k, replace=False
            )
            inputs.append(pool_features(slide["features"][selected]))
            labels.append(slide["label"])

    # Fit both scaling and classification using the training bags only.
    model = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=alpha),
    )
    model.fit(np.asarray(inputs), np.asarray(labels))

    return model


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data" / "embeddings"
    slides = load_slides(data_dir)

    k = 20
    seed = 851
    actual_labels = []
    predictions = []

    for fold, train_ids, source_ids, test_ids in make_splits(slides):
        model = train_ridge(
            slides, train_ids, k=k, seed=seed + fold
        )
        rng = np.random.default_rng(seed + 1000 + fold)

        print(
            f"Fold {fold + 1}: {len(train_ids)} training, "
            f"{len(source_ids)} reserved for EA, "
            f"{len(test_ids)} testing"
        )

        for index in test_ids:
            slide = slides[index]

            # Random target sampling provides the baseline prediction.
            selected = rng.choice(
                len(slide["features"]), size=k, replace=False
            )
            pooled = pool_features(slide["features"][selected])
            prediction = int(model.predict(pooled.reshape(1, -1))[0])

            actual_labels.append(slide["label"])
            predictions.append(prediction)

    actual_labels = np.asarray(actual_labels)
    predictions = np.asarray(predictions)
    correct = int((actual_labels == predictions).sum())

    print(f"\nHeld-out predictions: {len(predictions)}")
    print(f"Correct predictions: {correct}/{len(predictions)}")
    print(f"Accuracy: {accuracy_score(actual_labels, predictions):.3f}")
    print(
        f"Macro-F1: "
        f"{f1_score(actual_labels, predictions, average='macro'):.3f}"
    )
    print("Confusion matrix: rows actual, columns predicted")
    print("Class order: normal, tumour")
    print(confusion_matrix(actual_labels, predictions, labels=[0, 1]))