from pathlib import Path
from typing import Dict

from config import PROJECT_CONFIG
from src.common.helpers import save_json, write_text_report
from src.numerical.data_preprocessing import split_numerical_dataset
from src.numerical.evaluate_regression import (
    plot_predicted_vs_actual,
    plot_regression_model_comparison,
    plot_residuals,
    plot_target_distribution,
)
from src.numerical.feature_engineering import describe_numerical_feature_engineering
from src.numerical.train_knn_regressor import train_knn_regressor
from src.numerical.train_linear_regression import train_linear_regression


def run_numerical_pipeline() -> Dict[str, object]:
    data = split_numerical_dataset()
    metadata = data["metadata"]

    linear_result = train_linear_regression(data)
    knn_result = train_knn_regressor(data)
    model_results = {
        "Linear Regression": linear_result["test_metrics"],
        "KNN Regressor": knn_result["test_metrics"],
    }

    figures_dir = PROJECT_CONFIG["outputs"]["figures_dir"]
    reports_dir = PROJECT_CONFIG["outputs"]["reports_dir"]

    plot_target_distribution(
        data["y_train_val"].to_numpy(),
        "Numerical Target Distribution",
        figures_dir / "numerical_target_distribution.png",
    )

    plot_predicted_vs_actual(
        data["y_test"],
        linear_result["test_predictions"],
        "Linear Regression: Predicted vs Actual",
        figures_dir / "linear_regression_predicted_vs_actual.png",
    )
    plot_residuals(
        data["y_test"],
        linear_result["test_predictions"],
        "Linear Regression: Residual Plot",
        figures_dir / "linear_regression_residuals.png",
    )
    plot_predicted_vs_actual(
        data["y_test"],
        knn_result["test_predictions"],
        "KNN Regressor: Predicted vs Actual",
        figures_dir / "knn_regressor_predicted_vs_actual.png",
    )
    plot_residuals(
        data["y_test"],
        knn_result["test_predictions"],
        "KNN Regressor: Residual Plot",
        figures_dir / "knn_regressor_residuals.png",
    )
    plot_regression_model_comparison(
        model_results,
        figures_dir / "regression_model_comparison.png",
    )

    feature_summary = describe_numerical_feature_engineering(metadata)
    report_lines = [
        "Numerical Regression Results",
        "============================",
        "",
        feature_summary,
        "",
        "Regression models are evaluated using MAE, MSE, RMSE, and R2.",
        "Accuracy and confusion matrix are classification metrics and are not used for regression.",
        "",
        "Dataset inspection summary:",
        str(
            {
                "missing_values": metadata["missing_values"],
                "numeric_features": metadata["numeric_features"],
                "categorical_features": metadata["categorical_features"],
                "target_skewness": metadata["target_skewness"],
                "target_transform_decision": metadata["target_transform_decision"],
                "target_outlier_bounds": metadata["target_outlier_bounds"],
            }
        ),
        "",
        "Linear Regression Test Metrics:",
        str(linear_result["test_metrics"]),
        "Linear Regression Validation Metrics:",
        str(linear_result["validation_metrics"]),
        "",
        "KNN Regressor Test Metrics:",
        str(knn_result["test_metrics"]),
        "KNN Regressor Validation Metrics:",
        str(knn_result["validation_metrics"]),
        "",
        "KNN Controlled Search Results:",
        str(knn_result["validation_results_by_search"]),
    ]
    write_text_report("\n".join(report_lines), reports_dir / "numerical_regression_report.txt")

    serializable_result = {
        "metadata": metadata,
        "linear_regression": {
            "validation_metrics": linear_result["validation_metrics"],
            "validation_trials": linear_result["validation_trials"],
            "test_metrics": linear_result["test_metrics"],
            "hyperparameters": linear_result["hyperparameters"],
            "model_path": linear_result["model_path"],
            "preprocessor_path": linear_result["preprocessor_path"],
        },
        "knn_regressor": {
            "validation_metrics": knn_result["validation_metrics"],
            "test_metrics": knn_result["test_metrics"],
            "validation_results_by_search": knn_result["validation_results_by_search"],
            "hyperparameters": knn_result["hyperparameters"],
            "model_path": knn_result["model_path"],
            "preprocessor_path": knn_result["preprocessor_path"],
        },
        "figures": [
            str(Path("outputs/figures/numerical_target_distribution.png")),
            str(Path("outputs/figures/linear_regression_predicted_vs_actual.png")),
            str(Path("outputs/figures/linear_regression_residuals.png")),
            str(Path("outputs/figures/knn_regressor_predicted_vs_actual.png")),
            str(Path("outputs/figures/knn_regressor_residuals.png")),
            str(Path("outputs/figures/regression_model_comparison.png")),
        ],
    }
    save_json(serializable_result, reports_dir / "numerical_regression_results.json")
    return serializable_result
