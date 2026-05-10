import tkinter as tk
from tkinter import ttk

from gui.image_tab import ImagePredictionTab
from gui.numerical_tab import NumericalPredictionTab
from gui.styles import COLORS, FONTS, apply_app_style


class MLProjectApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ML University Project GUI")
        self.geometry("1180x760")
        self.minsize(980, 650)
        apply_app_style(self)

        self.status_var = tk.StringVar(value="Ready")
        self._build_layout()

    def _build_layout(self) -> None:
        header = ttk.Frame(self, padding=(22, 18, 22, 8))
        header.pack(fill="x")

        ttk.Label(header, text="Machine Learning Project Dashboard", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Regression and image classification GUI using the trained models from your project.",
            style="Muted.TLabel",
            font=FONTS["subtitle"],
        ).pack(anchor="w", pady=(4, 0))

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=18, pady=12)

        notebook.add(NumericalPredictionTab(notebook, self.status_var), text="Numerical Regression")
        notebook.add(ImagePredictionTab(notebook, self.status_var), text="Image Recognition")

        status = tk.Label(
            self,
            textvariable=self.status_var,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            anchor="w",
            padx=16,
            pady=7,
            font=FONTS["body"],
        )
        status.pack(fill="x", side="bottom")
