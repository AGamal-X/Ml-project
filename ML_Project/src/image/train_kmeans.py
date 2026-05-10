from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Dict, List

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import PROJECT_CONFIG
from src.image.evaluate_classification import classification_metrics


def build_cluster_label_mapping(cluster_ids: np.ndarray, labels: List[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for cluster_id in sorted(set(cluster_ids.tolist())):
        labels_in_cluster = [label for label, cluster in zip(labels, cluster_ids) if cluster == cluster_id]
        if labels_in_cluster:
            mapping[int(cluster_id)] = Counter(labels_in_cluster).most_common(1)[0][0]
    return mapping


def predict_labels_from_clusters(cluster_ids: np.ndarray, cluster_mapping: Dict[int, str], fallback_label: str) -> List[str]:
    return [cluster_mapping.get(int(cluster_id), fallback_label) for cluster_id in cluster_ids]


def build_kmeans_pipeline(class_count: int, pca_components, n_init: int) -> Pipeline:
    config = PROJECT_CONFIG["image_models"]["kmeans"]
    steps = [("scaler", StandardScaler())]
    if pca_components:
        steps.append(("pca", PCA(n_components=int(pca_components), random_state=PROJECT_CONFIG["random_state"])))
    steps.append(
        (
            "kmeans",
            KMeans(
                n_clusters=class_count,
                init=config["init"],
                n_init=n_init,
                max_iter=config["max_iter"],
                algorithm=config["algorithm"],
                random_state=PROJECT_CONFIG["random_state"],
            ),
        )
    )
    return Pipeline(steps=steps)


def select_best_kmeans_configuration(
    X_train,
    y_train: List[str],
    X_val,
    y_val: List[str],
    class_labels: List[str],
) -> Dict[str, object]:
    config = PROJECT_CONFIG["image_models"]["kmeans"]
    search_records = []
    best_result = None
    fallback_label = Counter(y_train).most_common(1)[0][0]

    for pca_components, n_init in product(
        config["candidate_pca_components"],
        config["candidate_n_init"],
    ):
        model = build_kmeans_pipeline(len(class_labels), pca_components, n_init)
        model.fit(X_train)
        train_clusters = model.predict(X_train)
        cluster_mapping = build_cluster_label_mapping(train_clusters, y_train)
        validation_clusters = model.predict(X_val)
        validation_predictions = predict_labels_from_clusters(validation_clusters, cluster_mapping, fallback_label)
        validation_metrics = classification_metrics(y_val, validation_predictions, class_labels)
        record = {
            "pca_n_components": pca_components,
            "n_init": n_init,
            "validation_accuracy": validation_metrics["accuracy"],
        }
        search_records.append(record)
        if best_result is None or validation_metrics["accuracy"] > best_result["validation_metrics"]["accuracy"]:
            best_result = {
                "model": model,
                "cluster_mapping": cluster_mapping,
                "validation_predictions": validation_predictions,
                "validation_metrics": validation_metrics,
                "config": {
                    "pca_n_components": pca_components,
                    "n_init": n_init,
                },
            }

    if best_result is None:
        raise ValueError("KMeans configuration search did not produce any result.")

    best_result["search_records"] = sorted(
        search_records,
        key=lambda item: item["validation_accuracy"],
        reverse=True,
    )
    best_result["fallback_label"] = fallback_label
    return best_result


def train_kmeans_classifier(
    X_train,
    y_train: List[str],
    X_val,
    y_val: List[str],
    X_test,
    y_test: List[str],
    class_labels: List[str],
) -> Dict[str, object]:
    selection_result = select_best_kmeans_configuration(
        X_train,
        y_train,
        X_val,
        y_val,
        class_labels,
    )

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = list(y_train) + list(y_val)
    final_model = build_kmeans_pipeline(
        len(class_labels),
        selection_result["config"]["pca_n_components"],
        selection_result["config"]["n_init"],
    )
    final_model.fit(X_train_val)

    train_val_clusters = final_model.predict(X_train_val)
    cluster_mapping = build_cluster_label_mapping(train_val_clusters, y_train_val)
    fallback_label = Counter(y_train_val).most_common(1)[0][0]

    test_clusters = final_model.predict(X_test)
    test_predictions = predict_labels_from_clusters(test_clusters, cluster_mapping, fallback_label)
    test_metrics = classification_metrics(y_test, test_predictions, class_labels)

    model_path = PROJECT_CONFIG["outputs"]["models_dir"] / "kmeans_image_classifier.joblib"
    scaler_path = PROJECT_CONFIG["outputs"]["models_dir"] / "kmeans_image_scaler.joblib"
    pca_path = PROJECT_CONFIG["outputs"]["models_dir"] / "kmeans_image_pca.joblib"
    mapping_path = PROJECT_CONFIG["outputs"]["models_dir"] / "kmeans_cluster_mapping.joblib"
    joblib.dump(
        {
            "model": final_model,
            "cluster_to_label_mapping": cluster_mapping,
            "fallback_label": fallback_label,
        },
        model_path,
    )
    joblib.dump(final_model.named_steps["scaler"], scaler_path)
    if "pca" in final_model.named_steps:
        joblib.dump(final_model.named_steps["pca"], pca_path)
    joblib.dump(cluster_mapping, mapping_path)

    hyperparameters = dict(PROJECT_CONFIG["image_models"]["kmeans"])
    hyperparameters.update(
        {
            "selected_pca_n_components": selection_result["config"]["pca_n_components"],
            "selected_n_init": selection_result["config"]["n_init"],
            "n_clusters": len(class_labels),
        }
    )

    return {
        "model_name": "KMeans",
        "model": final_model,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "pca_path": str(pca_path) if "pca" in final_model.named_steps else None,
        "cluster_mapping_path": str(mapping_path),
        "cluster_to_label_mapping": cluster_mapping,
        "fallback_label": fallback_label,
        "validation_predictions": selection_result["validation_predictions"],
        "test_predictions": test_predictions,
        "validation_metrics": selection_result["validation_metrics"],
        "test_metrics": test_metrics,
        "search_results": selection_result["search_records"],
        "hyperparameters": hyperparameters,
    }
