from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.75)
    min_value = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.75)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_regression_model_comparison(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(results.keys())
    metrics = ["MAE", "RMSE", "R2"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric in zip(axes, metrics):
        values = [results[model][metric] for model in model_names]
        sns.barplot(x=model_names, y=values, ax=axis)
        axis.set_title(metric)
        axis.set_xlabel("Model")
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Regression Model Comparison on Test Data")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_target_distribution(y_values: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.histplot(y_values, kde=True, bins=30, color="#2563eb")
    plt.xlabel("Target Value")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
