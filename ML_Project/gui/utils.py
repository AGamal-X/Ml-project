import json
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_CONFIG


MODEL_FILES = {
    "Linear Regression": PROJECT_CONFIG["outputs"]["models_dir"] / "linear_regression_model.joblib",
    "KNN Regressor": PROJECT_CONFIG["outputs"]["models_dir"] / "knn_regressor_model.joblib",
    "Logistic Regression": PROJECT_CONFIG["outputs"]["models_dir"] / "logistic_regression_image_classifier.joblib",
    "KMeans": PROJECT_CONFIG["outputs"]["models_dir"] / "kmeans_image_classifier.joblib",
}


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_complete_results() -> Dict[str, Any]:
    return load_json_file(PROJECT_CONFIG["outputs"]["reports_dir"] / "complete_results.json")


def safe_load_model(name: str) -> Any:
    path = MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


def list_files(folder: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    if not folder.exists():
        return []
    if extensions:
        allowed = {extension.lower() for extension in extensions}
        return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in allowed)
    return sorted(path for path in folder.iterdir() if path.is_file() and path.name != ".gitkeep")


def show_error(title: str, error: Exception | str) -> None:
    messagebox.showerror(title, str(error))


def show_info(title: str, message: str) -> None:
    messagebox.showinfo(title, message)


def open_file(path: Path) -> None:
    try:
        os.startfile(path)
    except Exception as exc:
        show_error("Could not open file", exc)


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget, *args: Any, **kwargs: Any) -> None:
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, background="#111827")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self._resize_inner)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def _resize_inner(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)


def set_status(status_var: tk.StringVar, message: str) -> None:
    status_var.set(message)
