from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config import PROJECT_CONFIG
from src.image.image_preprocessing import load_and_resize_image


RGB_HISTOGRAM_BINS = 16
GRAYSCALE_HISTOGRAM_BINS = 32
LOCAL_GRID_SIZE = 4
HOG_CELL_SIZE = 16
HOG_ORIENTATION_BINS = 9


def get_image_feature_names() -> List[str]:
    names: List[str] = []
    for channel in ["red", "green", "blue"]:
        names.extend([f"{channel}_hist_bin_{index}" for index in range(RGB_HISTOGRAM_BINS)])
    names.extend([f"grayscale_hist_bin_{index}" for index in range(GRAYSCALE_HISTOGRAM_BINS)])
    for channel in ["red", "green", "blue"]:
        names.extend(
            [
                f"{channel}_mean",
                f"{channel}_std",
                f"{channel}_min",
                f"{channel}_max",
            ]
        )
    names.extend(["grayscale_mean", "grayscale_std", "gradient_mean", "gradient_std", "edge_density"])

    for row in range(LOCAL_GRID_SIZE):
        for column in range(LOCAL_GRID_SIZE):
            for channel in ["red", "green", "blue"]:
                names.extend(
                    [
                        f"grid_{row}_{column}_{channel}_mean",
                        f"grid_{row}_{column}_{channel}_std",
                    ]
                )

    image_width, image_height = PROJECT_CONFIG["datasets"]["image"]["image_size"]
    hog_cells_x = image_width // HOG_CELL_SIZE
    hog_cells_y = image_height // HOG_CELL_SIZE
    for row in range(hog_cells_y):
        for column in range(hog_cells_x):
            names.extend(
                [f"hog_cell_{row}_{column}_bin_{bin_index}" for bin_index in range(HOG_ORIENTATION_BINS)]
            )
    return names


def extract_local_color_features(image: np.ndarray) -> List[float]:
    features: List[float] = []
    height, width, _ = image.shape
    cell_height = height // LOCAL_GRID_SIZE
    cell_width = width // LOCAL_GRID_SIZE

    for row in range(LOCAL_GRID_SIZE):
        for column in range(LOCAL_GRID_SIZE):
            patch = image[
                row * cell_height : (row + 1) * cell_height,
                column * cell_width : (column + 1) * cell_width,
                :,
            ]
            for channel_index in range(3):
                channel = patch[:, :, channel_index]
                features.append(float(np.mean(channel)))
                features.append(float(np.std(channel)))
    return features


def extract_simple_hog_features(grayscale: np.ndarray) -> List[float]:
    gradient_y, gradient_x = np.gradient(grayscale)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orientation = (np.degrees(np.arctan2(gradient_y, gradient_x)) + 180.0) % 180.0

    height, width = grayscale.shape
    cells_y = height // HOG_CELL_SIZE
    cells_x = width // HOG_CELL_SIZE
    features: List[float] = []

    for row in range(cells_y):
        for column in range(cells_x):
            magnitude_patch = magnitude[
                row * HOG_CELL_SIZE : (row + 1) * HOG_CELL_SIZE,
                column * HOG_CELL_SIZE : (column + 1) * HOG_CELL_SIZE,
            ]
            orientation_patch = orientation[
                row * HOG_CELL_SIZE : (row + 1) * HOG_CELL_SIZE,
                column * HOG_CELL_SIZE : (column + 1) * HOG_CELL_SIZE,
            ]
            histogram, _ = np.histogram(
                orientation_patch,
                bins=HOG_ORIENTATION_BINS,
                range=(0.0, 180.0),
                weights=magnitude_patch,
            )
            norm = np.linalg.norm(histogram) + 1e-8
            features.extend((histogram / norm).tolist())
    return features


def extract_single_image_features_from_array(image: np.ndarray) -> np.ndarray:
    features: List[float] = []

    for channel_index in range(3):
        histogram, _ = np.histogram(
            image[:, :, channel_index],
            bins=RGB_HISTOGRAM_BINS,
            range=(0.0, 1.0),
            density=True,
        )
        features.extend(histogram.tolist())

    grayscale = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )
    grayscale_histogram, _ = np.histogram(
        grayscale,
        bins=GRAYSCALE_HISTOGRAM_BINS,
        range=(0.0, 1.0),
        density=True,
    )
    features.extend(grayscale_histogram.tolist())

    for channel_index in range(3):
        channel = image[:, :, channel_index]
        features.extend(
            [
                float(np.mean(channel)),
                float(np.std(channel)),
                float(np.min(channel)),
                float(np.max(channel)),
            ]
        )

    horizontal_gradient = np.diff(grayscale, axis=1)
    vertical_gradient = np.diff(grayscale, axis=0)
    horizontal_core = horizontal_gradient[:-1, :]
    vertical_core = vertical_gradient[:, :-1]
    gradient_magnitude = np.sqrt(horizontal_core**2 + vertical_core**2)
    edge_density = float(np.mean(gradient_magnitude > 0.20))

    features.extend(
        [
            float(np.mean(grayscale)),
            float(np.std(grayscale)),
            float(np.mean(gradient_magnitude)),
            float(np.std(gradient_magnitude)),
            edge_density,
        ]
    )
    features.extend(extract_local_color_features(image))
    features.extend(extract_simple_hog_features(grayscale))

    return np.asarray(features, dtype=np.float32)


def extract_single_image_features(path: Path) -> np.ndarray:
    image = load_and_resize_image(path)
    return extract_single_image_features_from_array(image)


def extract_features_from_paths(paths: Iterable[Path]) -> Tuple[np.ndarray, List[str]]:
    feature_rows = [extract_single_image_features(path) for path in paths]
    return np.vstack(feature_rows), get_image_feature_names()


def describe_image_feature_extraction(metadata: Dict[str, object], feature_names: List[str]) -> str:
    width, height = metadata["image_size"]
    lines = [
        "Image feature extraction summary",
        "--------------------------------",
        f"Images were resized to {width}x{height} pixels and converted to RGB.",
        "The final feature vector combines grayscale HOG-style features for shape and texture with simple color histogram and global image statistics.",
        f"Number of extracted features: {len(feature_names)}",
        f"Feature names: {', '.join(feature_names)}",
    ]
    return "\n".join(lines)
