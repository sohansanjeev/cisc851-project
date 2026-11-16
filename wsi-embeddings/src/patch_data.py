from pathlib import Path
import numpy as np


def load_slides(data_dir):
    """Load the patch features, coordinates and label for each slide."""
    slides = []

    for file_path in sorted(Path(data_dir).glob("*.npz")):
        with np.load(file_path, allow_pickle=False) as data:
            slide_id = str(data["slide_id"].item())
            label_text = str(data["label"].item()).lower()
            features = data["features"]
            coords = data["coords"]

        if label_text not in ("normal", "tumor", "tumour"):
            raise ValueError(f"Unknown label in {file_path.name}")

        # Check that the feature values and patch locations are valid.
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError(f"Invalid features in {file_path.name}")

        if coords.shape != (len(features), 2):
            raise ValueError(f"Invalid coordinates in {file_path.name}")

        if not np.isfinite(coords).all():
            raise ValueError(f"Missing coordinates in {file_path.name}")

        # Use 0 for normal and 1 for tumour.
        label = int(label_text in ("tumor", "tumour"))

        slides.append({
            "id": slide_id,
            "label": label,
            "features": features,
            "coords": coords,
        })

    if not slides:
        raise ValueError(f"No embedding files found in {data_dir}")

    return slides


def pool_features(features):
    """Combine selected patch vectors using mean and standard deviation."""
    values = np.asarray(features, dtype=np.float64)

    # Calculate each feature's average and variation across patches.
    means = values.mean(axis=0)
    stds = values.std(axis=0)

    return np.concatenate([means, stds])


if __name__ == "__main__":
    # Find the embeddings folder relative to this Python file.
    data_dir = Path(__file__).resolve().parents[1] / "data" / "embeddings"
    slides = load_slides(data_dir)

    normal_count = sum(slide["label"] == 0 for slide in slides)
    tumour_count = sum(slide["label"] == 1 for slide in slides)

    print(f"Slides loaded: {len(slides)}")
    print(f"Normal: {normal_count}, tumour: {tumour_count}")
    print(f"First slide feature shape: {slides[0]['features'].shape}")

    # Check pooling with the first 20 patches from one slide.
    pooled = pool_features(slides[0]["features"][:20])
    print(f"20-patch pooled feature shape: {pooled.shape}")