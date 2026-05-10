import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, ImageTk

from config import PROJECT_CONFIG
from gui.styles import COLORS, FONTS
from gui.utils import ScrollableFrame, get_complete_results, safe_load_model, set_status, show_error
from src.image.feature_extraction import extract_single_image_features


class ImagePredictionTab(ttk.Frame):
    def __init__(self, parent: tk.Widget, status_var: tk.StringVar) -> None:
        super().__init__(parent, padding=18)
        self.status_var = status_var
        self.results = get_complete_results()
        self.metadata = self.results.get("image", {}).get("metadata", {})
        self.model_choice = tk.StringVar(value="Logistic Regression")
        self.result_var = tk.StringVar(value="No image selected")
        self.detail_var = tk.StringVar(value="Upload an image, choose a model, then click Predict.")
        self.metrics_var = tk.StringVar(value="Model evaluation will appear here after prediction.")
        self.image_path: Optional[Path] = None
        self.preview_image: Optional[ImageTk.PhotoImage] = None
        self.matrix_image: Optional[ImageTk.PhotoImage] = None
        self._build()

    def _build(self) -> None:
        left = ttk.Frame(self, style="Card.TFrame", padding=18)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right_scroll = ScrollableFrame(self)
        right_scroll.pack(side="right", fill="both", expand=True, padx=(10, 0))
        right = ttk.Frame(right_scroll.inner, style="Card.TFrame", padding=18)
        right.pack(fill="both", expand=True)

        ttk.Label(left, text="Image Upload and Preview", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="The uploaded image is resized and transformed using the same feature extraction pipeline used during training.",
            style="CardMuted.TLabel",
            wraplength=520,
        ).pack(anchor="w", pady=(4, 14))

        preview_frame = ttk.Frame(left, style="AltCard.TFrame", padding=12)
        preview_frame.pack(fill="both", expand=True)
        self.preview_label = tk.Label(
            preview_frame,
            text="No image selected",
            bg=COLORS["panel_alt"],
            fg=COLORS["muted"],
            font=FONTS["body"],
            width=52,
            height=18,
        )
        self.preview_label.pack(fill="both", expand=True)

        controls = ttk.Frame(left, style="Card.TFrame")
        controls.pack(fill="x", pady=(16, 0))
        ttk.Button(controls, text="Upload Image", style="Accent.TButton", command=self.upload_image).pack(side="left", padx=4)
        ttk.Button(controls, text="Clear Image", command=self.clear_image).pack(side="left", padx=4)

        ttk.Label(right, text="Image Prediction", style="Heading.TLabel").pack(anchor="w")
        model_row = ttk.Frame(right, style="Card.TFrame")
        model_row.pack(fill="x", pady=(14, 10))
        ttk.Label(model_row, text="Model", style="Card.TLabel").pack(side="left")
        ttk.Combobox(
            model_row,
            textvariable=self.model_choice,
            values=["Logistic Regression", "KMeans"],
            state="readonly",
            width=24,
        ).pack(side="left", padx=10)
        ttk.Button(model_row, text="Predict", style="Accent.TButton", command=self.predict_image).pack(side="left", padx=4)

        result_panel = ttk.Frame(right, style="AltCard.TFrame", padding=20)
        result_panel.pack(fill="x", pady=(16, 14))
        tk.Label(
            result_panel,
            textvariable=self.result_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["accent"],
            font=FONTS["result"],
            wraplength=430,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            right,
            textvariable=self.detail_var,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONTS["body"],
            wraplength=460,
            justify="left",
        ).pack(anchor="w")

        ttk.Label(right, text="Saved Test Evaluation", style="Heading.TLabel").pack(anchor="w", pady=(20, 8))
        metrics_panel = ttk.Frame(right, style="AltCard.TFrame", padding=14)
        metrics_panel.pack(fill="x")
        tk.Label(
            metrics_panel,
            textvariable=self.metrics_var,
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            font=FONTS["body"],
            wraplength=460,
            justify="left",
        ).pack(anchor="w")

        ttk.Label(right, text="Confusion Matrix", style="Heading.TLabel").pack(anchor="w", pady=(20, 8))
        self.matrix_panel = ttk.Frame(right, style="AltCard.TFrame", padding=12)
        self.matrix_panel.pack(fill="x", expand=False)
        tk.Label(
            self.matrix_panel,
            text="Confusion matrix appears after prediction.",
            bg=COLORS["panel_alt"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        ).pack()

    def upload_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        try:
            self.image_path = Path(file_path)
            with Image.open(self.image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((520, 360), Image.Resampling.LANCZOS)
                self.preview_image = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=self.preview_image, text="")
            self.result_var.set("Image ready")
            self.detail_var.set(f"Selected file: {self.image_path.name}")
            set_status(self.status_var, f"Loaded image: {self.image_path.name}")
        except Exception as exc:
            self.image_path = None
            show_error("Image upload failed", exc)
            set_status(self.status_var, "Image upload failed.")

    def clear_image(self) -> None:
        self.image_path = None
        self.preview_image = None
        self.preview_label.configure(image="", text="No image selected")
        self.result_var.set("No image selected")
        self.detail_var.set("Upload an image, choose a model, then click Predict.")
        self.metrics_var.set("Model evaluation will appear here after prediction.")
        self._clear_matrix("Confusion matrix appears after prediction.")
        set_status(self.status_var, "Image cleared.")

    def predict_image(self) -> None:
        if not self.image_path:
            show_error("Missing image", "Please upload an image before predicting.")
            return
        try:
            features = extract_single_image_features(self.image_path).reshape(1, -1)
            model_name = self.model_choice.get()
            if model_name == "Logistic Regression":
                self._predict_logistic(features)
            else:
                self._predict_kmeans(features)
            set_status(self.status_var, f"{model_name} image prediction completed.")
        except Exception as exc:
            show_error("Image prediction failed", exc)
            set_status(self.status_var, "Image prediction failed.")

    def _predict_logistic(self, features: np.ndarray) -> None:
        model = safe_load_model("Logistic Regression")
        label = str(model.predict(features)[0])
        detail_lines = [
            f"Model: Logistic Regression",
            f"Image size used during training: {self.metadata.get('image_size', 'unknown')}",
            f"Extracted feature count: {features.shape[1]}",
        ]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = [str(label_value) for label_value in model.classes_]
            probability_pairs = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
            top_probability = probability_pairs[0][1]
            detail_lines.append(f"Confidence: {top_probability * 100:.2f}%")
            detail_lines.append("Class probabilities:")
            detail_lines.extend([f"- {class_name}: {probability * 100:.2f}%" for class_name, probability in probability_pairs])
        self.result_var.set(f"Predicted class: {label}")
        self.detail_var.set("\n".join(detail_lines))
        self._show_evaluation("logistic_regression")

    def _predict_kmeans(self, features: np.ndarray) -> None:
        bundle = safe_load_model("KMeans")
        if not isinstance(bundle, dict) or "model" not in bundle or "cluster_to_label_mapping" not in bundle:
            raise ValueError("KMeans model file is missing the cluster-to-label mapping.")
        cluster_id = int(bundle["model"].predict(features)[0])
        mapping = {int(key): value for key, value in bundle["cluster_to_label_mapping"].items()}
        fallback_label = bundle.get("fallback_label", "Unknown")
        label = mapping.get(cluster_id, fallback_label)
        self.result_var.set(f"Mapped class: {label}")
        self.detail_var.set(
            f"Model: KMeans\n"
            f"Predicted cluster id: {cluster_id}\n"
            f"Mapped label: {label}\n"
            f"Extracted feature count: {features.shape[1]}\n"
            f"Cluster mapping: {mapping}"
        )
        self._show_evaluation("kmeans")

    def _show_evaluation(self, key: str) -> None:
        model_results = self.results.get("image", {}).get(key, {})
        metrics = model_results.get("test_metrics", {})
        accuracy = metrics.get("accuracy")
        if isinstance(accuracy, (int, float)):
            self.metrics_var.set(
                f"Test accuracy: {accuracy * 100:.2f}%\n"
                f"Classes: {', '.join(self.metadata.get('class_labels', []))}\n"
                f"Image samples: {self.metadata.get('split_counts', {}).get('total', 'Not available')}\n\n"
                "This is saved test-set accuracy. The uploaded image prediction itself has no known true label unless you provide one."
            )
        else:
            self.metrics_var.set("No saved accuracy was found. Run python main.py first.")

        matrix_name = (
            "logistic_regression_confusion_matrix.png"
            if key == "logistic_regression"
            else "kmeans_confusion_matrix.png"
        )
        self._show_matrix(PROJECT_CONFIG["outputs"]["confusion_matrices_dir"] / matrix_name)

    def _clear_matrix(self, message: str = "") -> None:
        for widget in self.matrix_panel.winfo_children():
            widget.destroy()
        self.matrix_image = None
        if message:
            tk.Label(
                self.matrix_panel,
                text=message,
                bg=COLORS["panel_alt"],
                fg=COLORS["muted"],
                font=FONTS["body"],
            ).pack()

    def _show_matrix(self, path: Path) -> None:
        self._clear_matrix()
        if not path.exists():
            self._clear_matrix("No confusion matrix found. Run python main.py to generate it.")
            return
        try:
            with Image.open(path) as image:
                image.thumbnail((560, 380), Image.Resampling.LANCZOS)
                self.matrix_image = ImageTk.PhotoImage(image)
            tk.Label(self.matrix_panel, image=self.matrix_image, bg=COLORS["panel_alt"]).pack(anchor="center")
            tk.Label(
                self.matrix_panel,
                text=path.name,
                bg=COLORS["panel_alt"],
                fg=COLORS["muted"],
                font=FONTS["body"],
            ).pack(pady=(8, 0))
        except Exception as exc:
            self._clear_matrix(f"Could not load confusion matrix: {exc}")
