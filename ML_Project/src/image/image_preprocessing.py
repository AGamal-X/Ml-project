from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split

from config import PROJECT_CONFIG


def discover_image_classes() -> List[str]:
    image_config = PROJECT_CONFIG["datasets"]["image"]
    data_dir = Path(image_config["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError("Image dataset folder was not found at datasets/images/.")

    selected_classes = image_config.get("selected_classes", [])
    available_classes = sorted([path.name for path in data_dir.iterdir() if path.is_dir()])
    if selected_classes:
        classes = [label for label in selected_classes if label in available_classes]
    else:
        classes = available_classes[: image_config["max_classes"]]

    if len(classes) == 0:
        raise ValueError("No image classes were found. Expected folders like datasets/images/class_name/image.jpg.")
    if len(classes) > image_config["max_classes"]:
        classes = classes[: image_config["max_classes"]]
    return classes


def collect_image_paths() -> Tuple[List[Path], List[str], List[str]]:
    image_config = PROJECT_CONFIG["datasets"]["image"]
    data_dir = Path(image_config["data_dir"])
    extensions = {extension.lower() for extension in image_config["supported_extensions"]}
    class_labels = discover_image_classes()

    paths: List[Path] = []
    labels: List[str] = []
    for class_label in class_labels:
        class_dir = data_dir / class_label
        class_files = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions
        )
        for path in class_files:
            paths.append(path)
            labels.append(class_label)

    if not paths:
        raise ValueError("No supported image files were found in datasets/images/.")
    return paths, labels, class_labels


def split_image_dataset() -> Dict[str, object]:
    paths, labels, class_labels = collect_image_paths()
    splits = PROJECT_CONFIG["splits"]
    random_state = PROJECT_CONFIG["random_state"]
    test_size = splits["test_size"]
    validation_fraction_of_train_val = splits["validation_size"] / (
        splits["train_size"] + splits["validation_size"]
    )

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=validation_fraction_of_train_val,
        random_state=random_state,
        stratify=train_val_labels,
    )

    class_counts = {label: labels.count(label) for label in class_labels}
    metadata = {
        "dataset_name": PROJECT_CONFIG["datasets"]["image"]["dataset_name"],
        "source": PROJECT_CONFIG["datasets"]["image"]["source"],
        "data_dir": str(PROJECT_CONFIG["datasets"]["image"]["data_dir"]),
        "image_size": list(PROJECT_CONFIG["datasets"]["image"]["image_size"]),
        "number_of_classes": len(class_labels),
        "class_labels": class_labels,
        "class_counts": class_counts,
        "split_counts": {
            "train": len(train_paths),
            "validation": len(val_paths),
            "test": len(test_paths),
            "total": len(paths),
        },
    }

    return {
        "train_paths": train_paths,
        "validation_paths": val_paths,
        "test_paths": test_paths,
        "train_labels": train_labels,
        "validation_labels": val_labels,
        "test_labels": test_labels,
        "class_labels": class_labels,
        "metadata": metadata,
    }


def load_and_resize_image(path: Path) -> np.ndarray:
    image_size = PROJECT_CONFIG["datasets"]["image"]["image_size"]
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(image_size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def generate_light_augmentations(path: Path) -> List[np.ndarray]:
    image_size = PROJECT_CONFIG["datasets"]["image"]["image_size"]
    augmented_images: List[np.ndarray] = []
    with Image.open(path) as image:
        base = ImageOps.exif_transpose(image).convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
        variants = [
            ImageOps.mirror(base),
            ImageEnhance.Brightness(base).enhance(1.08),
            base.rotate(8, resample=Image.Resampling.BILINEAR),
        ]
        for variant in variants:
            augmented_images.append(np.asarray(variant, dtype=np.float32) / 255.0)
    return augmented_images
