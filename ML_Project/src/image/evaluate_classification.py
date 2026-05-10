from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    class_labels: List[str],
) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=class_labels,
            target_names=class_labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_labels).tolist(),
    }


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    class_labels: List[str],
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_classification_model_comparison(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] * 100.0 for name in model_names]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=accuracies, hue=model_names, palette="Blues_d", legend=False)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Model")
    plt.title("Image Model Comparison on Test Data")
    plt.ylim(0, max(100, max(accuracies) + 10))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
