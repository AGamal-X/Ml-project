import tkinter as tk
from tkinter import ttk


COLORS = {
    "bg": "#111827",
    "panel": "#1F2937",
    "panel_alt": "#243244",
    "text": "#F9FAFB",
    "muted": "#CBD5E1",
    "accent": "#38BDF8",
    "accent_dark": "#0284C7",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "entry": "#F8FAFC",
    "entry_text": "#111827",
}


FONTS = {
    "title": ("Segoe UI", 22, "bold"),
    "subtitle": ("Segoe UI", 13),
    "heading": ("Segoe UI", 15, "bold"),
    "body": ("Segoe UI", 10),
    "body_bold": ("Segoe UI", 10, "bold"),
    "result": ("Segoe UI", 18, "bold"),
}


def apply_app_style(root: tk.Tk) -> None:
    root.configure(bg=COLORS["bg"])
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=COLORS["panel"],
        foreground=COLORS["muted"],
        padding=(18, 10),
        font=FONTS["body_bold"],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", COLORS["accent_dark"])],
        foreground=[("selected", COLORS["text"])],
    )

    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Card.TFrame", background=COLORS["panel"], relief="flat")
    style.configure("AltCard.TFrame", background=COLORS["panel_alt"], relief="flat")
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=FONTS["body"])
    style.configure("Card.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=FONTS["body"])
    style.configure("Muted.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=FONTS["body"])
    style.configure("CardMuted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=FONTS["body"])
    style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=FONTS["title"])
    style.configure("Heading.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=FONTS["heading"])
    style.configure("Result.TLabel", background=COLORS["panel_alt"], foreground=COLORS["accent"], font=FONTS["result"])

    style.configure(
        "Accent.TButton",
        background=COLORS["accent_dark"],
        foreground=COLORS["text"],
        font=FONTS["body_bold"],
        padding=(12, 8),
        borderwidth=0,
    )
    style.map("Accent.TButton", background=[("active", COLORS["accent"])])

    style.configure(
        "TButton",
        background=COLORS["panel_alt"],
        foreground=COLORS["text"],
        font=FONTS["body_bold"],
        padding=(10, 7),
        borderwidth=0,
    )
    style.map("TButton", background=[("active", COLORS["accent_dark"])])

    style.configure(
        "TCombobox",
        fieldbackground=COLORS["entry"],
        background=COLORS["entry"],
        foreground=COLORS["entry_text"],
        arrowsize=14,
        padding=4,
    )
