from __future__ import annotations

import inspect
from itertools import product
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import PROJECT_CONFIG
from src.image.evaluate_classification import classification_metrics
from src.image.feature_extraction import extract_single_image_features_from_array
from src.image.image_preprocessing import generate_light_augmentations


def build_logistic_regression_pipeline(config: Dict[str, object]) -> Pipeline:
    classifier_parameters = {
        "C": config["C"],
        "solver": config["solver"],
        "max_iter": config["max_iter"],
        "class_weight": config["class_weight"],
        "random_state": PROJECT_CONFIG["random_state"],
    }
    if "penalty" in inspect.signature(LogisticRegression).parameters and config.get("penalty") not in (None, "l2"):
        classifier_parameters["penalty"] = config["penalty"]
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        classifier_parameters["multi_class"] = config["multi_class"]

    steps = [("scaler", StandardScaler())]
    pca_components = config.get("pca_n_components")
    if pca_components:
        steps.append(("pca", PCA(n_components=int(pca_components), random_state=PROJECT_CONFIG["random_state"])))
    steps.append(("classifier", LogisticRegression(**classifier_parameters)))
    return Pipeline(steps=steps)


def build_augmented_training_features(
    train_paths,
    train_labels: List[str],
    X_train: np.ndarray,
) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    config = PROJECT_CONFIG["image_models"]["logistic_regression"]
    if not config["use_light_augmentation"]:
        return X_train, list(train_labels), {"used": False, "added_samples": 0}

    augmentation_limit = int(config["augmentation_limit"])
    augmented_rows: List[np.ndarray] = []
    augmented_labels: List[str] = []

    for path, label in list(zip(train_paths, train_labels))[:augmentation_limit]:
        for augmented_image in generate_light_augmentations(path)[:2]:
            augmented_rows.append(extract_single_image_features_from_array(augmented_image))
            augmented_labels.append(label)

    if not augmented_rows:
        return X_train, list(train_labels), {"used": False, "added_samples": 0}

    X_augmented = np.vstack([X_train, np.vstack(augmented_rows)])
    y_augmented = list(train_labels) + augmented_labels
    return X_augmented, y_augmented, {"used": True, "added_samples": len(augmented_rows)}


def select_best_logistic_configuration(
    X_train: np.ndarray,
    y_train: List[str],
    X_val: np.ndarray,
    y_val: List[str],
    class_labels: List[str],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    base_config = PROJECT_CONFIG["image_models"]["logistic_regression"]
    search_records = []
    best_result = None

    for C_value, max_iter, class_weight, pca_components in product(
        base_config["candidate_C_values"],
        base_config["candidate_max_iter"],
        base_config["candidate_class_weight"],
        base_config["candidate_pca_components"],
    ):
        candidate_config = {
            "C": C_value,
            "penalty": base_config["penalty"],
            "solver": base_config["solver"],
            "max_iter": max_iter,
            "multi_class": base_config["multi_class"],
            "class_weight": class_weight,
            "pca_n_components": pca_components,
        }
        pipeline = build_logistic_regression_pipeline(candidate_config)
        pipeline.fit(X_train, y_train)
        validation_predictions = pipeline.predict(X_val).tolist()
        validation_metrics = classification_metrics(y_val, validation_predictions, class_labels)
        record = {
            "C": C_value,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "pca_n_components": pca_components,
            "validation_accuracy": validation_metrics["accuracy"],
        }
        search_records.append(record)
        if best_result is None or validation_metrics["accuracy"] > best_result["validation_metrics"]["accuracy"]:
            best_result = {
                "config": candidate_config,
                "validation_metrics": validation_metrics,
                "validation_predictions": validation_predictions,
            }

    if best_result is None:
        raise ValueError("Logistic Regression configuration search did not produce any result.")

    best_result["search_records"] = sorted(
        search_records,
        key=lambda item: item["validation_accuracy"],
        reverse=True,
    )
    return best_result["config"], best_result


def train_logistic_regression_classifier(
    X_train,
    y_train: List[str],
    X_val,
    y_val: List[str],
    X_test,
    y_test: List[str],
    class_labels: List[str],
    train_paths,
) -> Dict[str, object]:
    augmented_X_train, augmented_y_train, augmentation_summary = build_augmented_training_features(
        train_paths,
        y_train,
        X_train,
    )
    best_config, selection_result = select_best_logistic_configuration(
        augmented_X_train,
        augmented_y_train,
        X_val,
        y_val,
        class_labels,
    )

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = list(y_train) + list(y_val)
    if augmentation_summary["used"]:
        extra_X, extra_y, _ = build_augmented_training_features(train_paths, y_train, X_train)
        if len(extra_y) > len(y_train):
            X_train_val = np.vstack([X_train_val, extra_X[len(y_train) :]])
            y_train_val = y_train_val + extra_y[len(y_train) :]

    model = build_logistic_regression_pipeline(best_config)
    model.fit(X_train_val, y_train_val)

    test_predictions = model.predict(X_test).tolist()
    test_metrics = classification_metrics(y_test, test_predictions, class_labels)

    model_path = PROJECT_CONFIG["outputs"]["models_dir"] / "logistic_regression_image_classifier.joblib"
    scaler_path = PROJECT_CONFIG["outputs"]["models_dir"] / "logistic_regression_image_scaler.joblib"
    pca_path = PROJECT_CONFIG["outputs"]["models_dir"] / "logistic_regression_image_pca.joblib"
    joblib.dump(model, model_path)
    joblib.dump(model.named_steps["scaler"], scaler_path)
    if "pca" in model.named_steps:
        joblib.dump(model.named_steps["pca"], pca_path)

    hyperparameters = dict(PROJECT_CONFIG["image_models"]["logistic_regression"])
    hyperparameters.update(
        {
            "selected_C": best_config["C"],
            "selected_max_iter": best_config["max_iter"],
            "selected_class_weight": best_config["class_weight"],
            "selected_pca_n_components": best_config["pca_n_components"],
            "augmentation_used": augmentation_summary["used"],
            "augmentation_added_samples": augmentation_summary["added_samples"],
        }
    )

    return {
        "model_name": "Logistic Regression",
        "model": model,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "pca_path": str(pca_path) if "pca" in model.named_steps else None,
        "validation_predictions": selection_result["validation_predictions"],
        "test_predictions": test_predictions,
        "validation_metrics": selection_result["validation_metrics"],
        "test_metrics": test_metrics,
        "search_results": selection_result["search_records"],
        "hyperparameters": hyperparameters,
    }
