import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import List

from PIL import Image, ImageTk

from config import PROJECT_CONFIG
from gui.styles import COLORS, FONTS
from gui.utils import ScrollableFrame, get_complete_results, list_files, open_file, set_status


class ResultsTab(ttk.Frame):
    def __init__(self, parent: tk.Widget, status_var: tk.StringVar) -> None:
        super().__init__(parent, padding=18)
        self.status_var = status_var
        self.thumbnail_refs: List[ImageTk.PhotoImage] = []
        self.results = get_complete_results()
        self._build()

    def _build(self) -> None:
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        container = scroll.inner

        self._section_title(container, "Project Summary")
        summary = self._build_summary_text()
        self._text_card(container, summary)

        self._section_title(container, "Trained Models")
        self._file_list(container, PROJECT_CONFIG["outputs"]["models_dir"])

        self._section_title(container, "Evaluation Reports")
        self._file_list(container, PROJECT_CONFIG["outputs"]["reports_dir"], [".txt", ".json"])

        self._section_title(container, "Regression Figures")
        self._image_gallery(container, PROJECT_CONFIG["outputs"]["figures_dir"])

        self._section_title(container, "Confusion Matrices")
        self._image_gallery(container, PROJECT_CONFIG["outputs"]["confusion_matrices_dir"])

    def _section_title(self, parent: tk.Widget, text: str) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=(16, 12))
        frame.pack(fill="x", pady=(0, 8))
        ttk.Label(frame, text=text, style="Heading.TLabel").pack(anchor="w")

    def _text_card(self, parent: tk.Widget, text: str) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=16)
        frame.pack(fill="x", pady=(0, 18))
        tk.Label(
            frame,
            text=text,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            justify="left",
            wraplength=950,
            font=FONTS["body"],
        ).pack(anchor="w")

    def _build_summary_text(self) -> str:
        numerical = self.results.get("numerical", {})
        image = self.results.get("image", {})
        numerical_meta = numerical.get("metadata", {})
        image_meta = image.get("metadata", {})
        logistic_accuracy = image.get("logistic_regression", {}).get("test_metrics", {}).get("accuracy")
        kmeans_accuracy = image.get("kmeans", {}).get("test_metrics", {}).get("accuracy")
        linear_r2 = numerical.get("linear_regression", {}).get("test_metrics", {}).get("R2")
        knn_r2 = numerical.get("knn_regressor", {}).get("test_metrics", {}).get("R2")

        linear_r2_text = f"{linear_r2:.4f}" if isinstance(linear_r2, (int, float)) else "Not available"
        knn_r2_text = f"{knn_r2:.4f}" if isinstance(knn_r2, (int, float)) else "Not available"
        logistic_text = (
            f"{logistic_accuracy * 100:.2f}%"
            if isinstance(logistic_accuracy, (int, float))
            else "Not available"
        )
        kmeans_text = (
            f"{kmeans_accuracy * 100:.2f}%"
            if isinstance(kmeans_accuracy, (int, float))
            else "Not available"
        )

        return (
            f"Numerical dataset: {numerical_meta.get('dataset_name', 'Not available')}\n"
            f"Target column: {numerical_meta.get('target_column', 'Not available')}\n"
            f"Numerical samples: {numerical_meta.get('split_counts', {}).get('total', 'Not available')}\n"
            f"Linear Regression R2: {linear_r2_text}\n"
            f"KNN Regressor R2: {knn_r2_text}\n\n"
            f"Image dataset: {image_meta.get('dataset_name', 'Not available')}\n"
            f"Classes: {', '.join(image_meta.get('class_labels', []))}\n"
            f"Image samples: {image_meta.get('split_counts', {}).get('total', 'Not available')}\n"
            f"Logistic Regression accuracy: {logistic_text}\n"
            f"KMeans accuracy: {kmeans_text}"
        )

    def _file_list(self, parent: tk.Widget, folder: Path, extensions: List[str] | None = None) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=16)
        frame.pack(fill="x", pady=(0, 18))
        files = list_files(folder, extensions)
        if not files:
            ttk.Label(frame, text=f"No files found in {folder}", style="CardMuted.TLabel").pack(anchor="w")
            return
        for file_path in files:
            row = ttk.Frame(frame, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=f"{file_path.name} ({file_path.stat().st_size:,} bytes)", style="Card.TLabel").pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="Open", command=lambda path=file_path: self._open_and_status(path)).pack(side="right")

    def _image_gallery(self, parent: tk.Widget, folder: Path) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=16)
        frame.pack(fill="x", pady=(0, 18))
        images = list_files(folder, [".png", ".jpg", ".jpeg"])
        if not images:
            ttk.Label(frame, text=f"No image outputs found in {folder}", style="CardMuted.TLabel").pack(anchor="w")
            return

        columns = 2
        for index, image_path in enumerate(images):
            item = ttk.Frame(frame, style="AltCard.TFrame", padding=12)
            item.grid(row=index // columns, column=index % columns, sticky="nsew", padx=8, pady=8)
            try:
                with Image.open(image_path) as image:
                    image.thumbnail((360, 220), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    self.thumbnail_refs.append(photo)
                    tk.Label(item, image=photo, bg=COLORS["panel_alt"]).pack()
            except Exception:
                tk.Label(item, text="Preview unavailable", bg=COLORS["panel_alt"], fg=COLORS["muted"]).pack()
            tk.Label(
                item,
                text=image_path.name,
                bg=COLORS["panel_alt"],
                fg=COLORS["text"],
                font=FONTS["body_bold"],
                wraplength=340,
            ).pack(pady=(8, 4))
            ttk.Button(item, text="Open", command=lambda path=image_path: self._open_and_status(path)).pack()

        for column in range(columns):
            frame.columnconfigure(column, weight=1)

    def _open_and_status(self, path: Path) -> None:
        open_file(path)
        set_status(self.status_var, f"Opened {path.name}")
