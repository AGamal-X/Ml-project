import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Dict, List

import pandas as pd
from PIL import Image, ImageTk

from gui.styles import COLORS, FONTS
from gui.utils import ScrollableFrame, get_complete_results, safe_load_model, set_status, show_error
from config import PROJECT_CONFIG


class NumericalPredictionTab(ttk.Frame):
    def __init__(self, parent: tk.Widget, status_var: tk.StringVar) -> None:
        super().__init__(parent, padding=18)
        self.status_var = status_var
        self.results = get_complete_results()
        self.metadata = self.results.get("numerical", {}).get("metadata", {})
        self.inputs: Dict[str, tk.Variable] = {}
        self.feature_columns: List[str] = self.metadata.get("raw_feature_columns", [])
        self.numeric_features: List[str] = self.metadata.get("numeric_features", [])
        self.categorical_features: List[str] = self.metadata.get("categorical_features", [])
        self.category_values = self._load_category_values()
        self.numeric_defaults = self._load_numeric_defaults()
        self.model_choice = tk.StringVar(value="Linear Regression")
        self.result_var = tk.StringVar(value="No prediction yet")
        self.detail_var = tk.StringVar(value="Enter feature values, then click Predict.")
        self.metrics_var = tk.StringVar(value="Model evaluation will appear here after prediction.")
        self.graph_refs: List[ImageTk.PhotoImage] = []
        self._build()

    def _load_dataset(self) -> pd.DataFrame:
        dataset_file = self.metadata.get("dataset_file")
        if dataset_file:
            return pd.read_csv(dataset_file)
        data_dir = PROJECT_CONFIG["datasets"]["numerical"]["data_dir"]
        file_name = PROJECT_CONFIG["datasets"]["numerical"]["file_name"]
        return pd.read_csv(data_dir / file_name)

    def _load_category_values(self) -> Dict[str, List[str]]:
        values: Dict[str, List[str]] = {}
        try:
            df = self._load_dataset()
            for column in self.categorical_features:
                if column in df.columns:
                    values[column] = sorted(str(value) for value in df[column].dropna().unique())
        except Exception:
            for column in self.categorical_features:
                values[column] = []
        return values

    def _load_numeric_defaults(self) -> Dict[str, float]:
        defaults: Dict[str, float] = {}
        try:
            df = self._load_dataset()
            for column in self.numeric_features:
                if column in df.columns:
                    defaults[column] = float(df[column].median())
        except Exception:
            pass
        return defaults

    def _build(self) -> None:
        left = ttk.Frame(self, style="Card.TFrame", padding=18)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right_scroll = ScrollableFrame(self)
        right_scroll.pack(side="right", fill="both", expand=True, padx=(10, 0))
        right = ttk.Frame(right_scroll.inner, style="Card.TFrame", padding=18)
        right.pack(fill="both", expand=True)

        ttk.Label(left, text="Numerical Regression Input", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Values are transformed using the same saved preprocessor from model training.",
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(4, 16))

        form = ttk.Frame(left, style="Card.TFrame")
        form.pack(fill="x", expand=False)
        for row, column in enumerate(self.feature_columns):
            ttk.Label(form, text=column, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=6, padx=(0, 8))
            if column in self.categorical_features:
                variable = tk.StringVar()
                values = self.category_values.get(column, [])
                combo = ttk.Combobox(form, textvariable=variable, values=values, state="readonly" if values else "normal")
                if values:
                    variable.set(values[0])
                combo.grid(row=row, column=1, sticky="ew", pady=6)
            else:
                default_value = self.numeric_defaults.get(column, 0.0)
                variable = tk.StringVar(value=str(default_value))
                entry = tk.Entry(form, textvariable=variable, bg=COLORS["entry"], fg=COLORS["entry_text"], relief="flat")
                entry.grid(row=row, column=1, sticky="ew", pady=6, ipady=5)
            self.inputs[column] = variable
        form.columnconfigure(1, weight=1)

        controls = ttk.Frame(left, style="Card.TFrame")
        controls.pack(fill="x", pady=(16, 0))
        ttk.Label(controls, text="Model", style="Card.TLabel").pack(side="left")
        ttk.Combobox(
            controls,
            textvariable=self.model_choice,
            values=["Linear Regression", "KNN Regressor"],
            state="readonly",
            width=24,
        ).pack(side="left", padx=10)
        ttk.Button(controls, text="Predict", style="Accent.TButton", command=self.predict_selected).pack(side="left", padx=4)
        ttk.Button(controls, text="Compare Both", command=self.compare_models).pack(side="left", padx=4)
        ttk.Button(controls, text="Reset", command=self.reset_inputs).pack(side="left", padx=4)

        ttk.Label(right, text="Prediction Result", style="Heading.TLabel").pack(anchor="w")
        result_panel = ttk.Frame(right, style="AltCard.TFrame", padding=20)
        result_panel.pack(fill="x", pady=(18, 16))
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
            justify="left",
            wraplength=460,
        ).pack(anchor="w")

        ttk.Label(right, text="Model Graphs", style="Heading.TLabel").pack(anchor="w", pady=(20, 8))
        self.graph_panel = ttk.Frame(right, style="AltCard.TFrame", padding=12)
        self.graph_panel.pack(fill="x", expand=False)
        tk.Label(
            self.graph_panel,
            text="Graphs appear after prediction.",
            bg=COLORS["panel_alt"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        ).pack()

    def _collect_input_dataframe(self) -> pd.DataFrame:
        row = {}
        for column in self.feature_columns:
            value = self.inputs[column].get()
            if column in self.numeric_features:
                try:
                    row[column] = float(value)
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value for {column}: {value}") from exc
            else:
                if not str(value).strip():
                    raise ValueError(f"Missing value for {column}")
                row[column] = str(value)
        return pd.DataFrame([row], columns=self.feature_columns)

    def _predict(self, model_name: str) -> float:
        bundle = safe_load_model(model_name)
        if not isinstance(bundle, dict) or "model" not in bundle or "preprocessor" not in bundle:
            raise ValueError(f"{model_name} file does not contain the expected model/preprocessor bundle.")
        input_df = self._collect_input_dataframe()
        processed = bundle["preprocessor"].transform(input_df)
        prediction = bundle["model"].predict(processed)[0]
        return float(prediction)

    def predict_selected(self) -> None:
        try:
            model_name = self.model_choice.get()
            prediction = self._predict(model_name)
            self.result_var.set(f"{prediction:,.2f}")
            self.detail_var.set(f"Selected model: {model_name}\nTarget: {self.metadata.get('target_column', 'configured target')}")
            self._show_evaluation(model_name)
            set_status(self.status_var, f"{model_name} prediction completed.")
        except Exception as exc:
            show_error("Prediction failed", exc)
            set_status(self.status_var, "Numerical prediction failed.")

    def compare_models(self) -> None:
        try:
            linear = self._predict("Linear Regression")
            knn = self._predict("KNN Regressor")
            self.result_var.set(f"Linear: {linear:,.2f}\nKNN: {knn:,.2f}")
            self.detail_var.set("Comparison completed using both saved regression models.")
            self._show_comparison_evaluation()
            set_status(self.status_var, "Regression model comparison completed.")
        except Exception as exc:
            show_error("Comparison failed", exc)
            set_status(self.status_var, "Regression comparison failed.")

    def reset_inputs(self) -> None:
        for column, variable in self.inputs.items():
            if column in self.numeric_features:
                variable.set(str(self.numeric_defaults.get(column, 0.0)))
            else:
                values = self.category_values.get(column, [])
                variable.set(values[0] if values else "")
        self.result_var.set("No prediction yet")
        self.detail_var.set("Inputs reset.")
        self.metrics_var.set("Model evaluation will appear here after prediction.")
        self._clear_graphs("Graphs appear after prediction.")
        set_status(self.status_var, "Numerical inputs reset.")

    def _metric_text(self, metrics: Dict[str, float]) -> str:
        if not metrics:
            return "No saved metrics were found. Run python main.py first."
        return (
            f"MAE: {metrics.get('MAE', 0):,.2f}\n"
            f"MSE: {metrics.get('MSE', 0):,.2f}\n"
            f"RMSE: {metrics.get('RMSE', 0):,.2f}\n"
            f"R2 score: {metrics.get('R2', 0):.4f}\n\n"
            "These are saved test-set regression metrics. A single manual prediction cannot be evaluated unless its true target value is known."
        )

    def _show_evaluation(self, model_name: str) -> None:
        key = "linear_regression" if model_name == "Linear Regression" else "knn_regressor"
        metrics = self.results.get("numerical", {}).get(key, {}).get("test_metrics", {})
        self.metrics_var.set(self._metric_text(metrics))

        if model_name == "Linear Regression":
            graph_paths = [
                PROJECT_CONFIG["outputs"]["figures_dir"] / "linear_regression_predicted_vs_actual.png",
                PROJECT_CONFIG["outputs"]["figures_dir"] / "linear_regression_residuals.png",
            ]
        else:
            graph_paths = [
                PROJECT_CONFIG["outputs"]["figures_dir"] / "knn_regressor_predicted_vs_actual.png",
                PROJECT_CONFIG["outputs"]["figures_dir"] / "knn_regressor_residuals.png",
            ]
        self._show_graphs(graph_paths)

    def _show_comparison_evaluation(self) -> None:
        linear_metrics = self.results.get("numerical", {}).get("linear_regression", {}).get("test_metrics", {})
        knn_metrics = self.results.get("numerical", {}).get("knn_regressor", {}).get("test_metrics", {})
        self.metrics_var.set(
            "Linear Regression\n"
            f"RMSE: {linear_metrics.get('RMSE', 0):,.2f} | R2: {linear_metrics.get('R2', 0):.4f}\n\n"
            "KNN Regressor\n"
            f"RMSE: {knn_metrics.get('RMSE', 0):,.2f} | R2: {knn_metrics.get('R2', 0):.4f}\n\n"
            "Lower RMSE is better; higher R2 is better."
        )
        self._show_graphs([PROJECT_CONFIG["outputs"]["figures_dir"] / "regression_model_comparison.png"])

    def _clear_graphs(self, message: str = "") -> None:
        for widget in self.graph_panel.winfo_children():
            widget.destroy()
        self.graph_refs.clear()
        if message:
            tk.Label(
                self.graph_panel,
                text=message,
                bg=COLORS["panel_alt"],
                fg=COLORS["muted"],
                font=FONTS["body"],
            ).pack()

    def _show_graphs(self, graph_paths: List[Path]) -> None:
        self._clear_graphs()
        shown = False
        for path in graph_paths:
            if not path.exists():
                continue
            try:
                with Image.open(path) as image:
                    image.thumbnail((560, 320), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                self.graph_refs.append(photo)
                tk.Label(self.graph_panel, image=photo, bg=COLORS["panel_alt"]).pack(anchor="center", pady=(0, 8))
                tk.Label(
                    self.graph_panel,
                    text=path.name,
                    bg=COLORS["panel_alt"],
                    fg=COLORS["muted"],
                    font=FONTS["body"],
                ).pack(pady=(0, 10))
                shown = True
            except Exception:
                continue
        if not shown:
            self._clear_graphs("No graph image found. Run python main.py to generate plots.")
