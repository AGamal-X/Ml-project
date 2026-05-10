from pathlib import Path
from typing import Dict

from config import PROJECT_CONFIG
from src.common.helpers import save_json, write_text_report
from src.image.evaluate_classification import (
    plot_classification_model_comparison,
    plot_confusion_matrix,
)
from src.image.feature_extraction import (
    describe_image_feature_extraction,
    extract_features_from_paths,
)
from src.image.image_preprocessing import split_image_dataset
from src.image.train_kmeans import train_kmeans_classifier
from src.image.train_logistic_regression import train_logistic_regression_classifier


def run_image_pipeline() -> Dict[str, object]:
    split_data = split_image_dataset()
    metadata = split_data["metadata"]
    class_labels = split_data["class_labels"]

    X_train, feature_names = extract_features_from_paths(split_data["train_paths"])
    X_val, _ = extract_features_from_paths(split_data["validation_paths"])
    X_test, _ = extract_features_from_paths(split_data["test_paths"])

    metadata["number_of_extracted_features"] = len(feature_names)
    metadata["feature_names"] = feature_names
    metadata["feature_dimensions"] = {
        "train": list(X_train.shape),
        "validation": list(X_val.shape),
        "test": list(X_test.shape),
    }

    logistic_result = train_logistic_regression_classifier(
        X_train,
        split_data["train_labels"],
        X_val,
        split_data["validation_labels"],
        X_test,
        split_data["test_labels"],
        class_labels,
        split_data["train_paths"],
    )
    kmeans_result = train_kmeans_classifier(
        X_train,
        split_data["train_labels"],
        X_val,
        split_data["validation_labels"],
        X_test,
        split_data["test_labels"],
        class_labels,
    )

    confusion_dir = PROJECT_CONFIG["outputs"]["confusion_matrices_dir"]
    reports_dir = PROJECT_CONFIG["outputs"]["reports_dir"]

    plot_confusion_matrix(
        split_data["test_labels"],
        logistic_result["test_predictions"],
        class_labels,
        "Logistic Regression Confusion Matrix",
        confusion_dir / "logistic_regression_confusion_matrix.png",
    )
    plot_confusion_matrix(
        split_data["test_labels"],
        kmeans_result["test_predictions"],
        class_labels,
        "KMeans Cluster-to-Label Confusion Matrix",
        confusion_dir / "kmeans_confusion_matrix.png",
    )
    plot_classification_model_comparison(
        {
            "Logistic Regression": logistic_result["test_metrics"],
            "KMeans": kmeans_result["test_metrics"],
        },
        PROJECT_CONFIG["outputs"]["figures_dir"] / "image_model_comparison.png",
    )

    feature_summary = describe_image_feature_extraction(metadata, feature_names)
    report_lines = [
        "Image Classification Results",
        "============================",
        "",
        feature_summary,
        "",
        "Classification models are evaluated using accuracy, confusion matrix, and classification report.",
        "Regression metrics such as MAE, MSE, RMSE, and R2 are not used for classification.",
        "",
        "Logistic Regression Test Accuracy:",
        str(logistic_result["test_metrics"]["accuracy"]),
        "Logistic Regression Validation Search Summary:",
        str(logistic_result["search_results"][:5]),
        "",
        "KMeans Test Accuracy:",
        str(kmeans_result["test_metrics"]["accuracy"]),
        "KMeans Validation Search Summary:",
        str(kmeans_result["search_results"]),
        "",
        "KMeans Cluster-to-Label Mapping:",
        str(kmeans_result["cluster_to_label_mapping"]),
    ]
    write_text_report("\n".join(report_lines), reports_dir / "image_classification_report.txt")

    serializable_result = {
        "metadata": metadata,
        "logistic_regression": {
            "validation_metrics": logistic_result["validation_metrics"],
            "test_metrics": logistic_result["test_metrics"],
            "search_results": logistic_result["search_results"],
            "hyperparameters": logistic_result["hyperparameters"],
            "model_path": logistic_result["model_path"],
            "scaler_path": logistic_result["scaler_path"],
            "pca_path": logistic_result["pca_path"],
        },
        "kmeans": {
            "validation_metrics": kmeans_result["validation_metrics"],
            "test_metrics": kmeans_result["test_metrics"],
            "cluster_to_label_mapping": kmeans_result["cluster_to_label_mapping"],
            "fallback_label": kmeans_result["fallback_label"],
            "search_results": kmeans_result["search_results"],
            "hyperparameters": kmeans_result["hyperparameters"],
            "model_path": kmeans_result["model_path"],
            "scaler_path": kmeans_result["scaler_path"],
            "pca_path": kmeans_result["pca_path"],
            "cluster_mapping_path": kmeans_result["cluster_mapping_path"],
        },
        "confusion_matrix_figures": [
            str(Path("outputs/confusion_matrices/logistic_regression_confusion_matrix.png")),
            str(Path("outputs/confusion_matrices/kmeans_confusion_matrix.png")),
        ],
        "figures": [
            str(Path("outputs/figures/image_model_comparison.png")),
        ],
    }
    save_json(serializable_result, reports_dir / "image_classification_results.json")
    return serializable_result
