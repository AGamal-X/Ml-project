import json
from pathlib import Path
from typing import Any, Dict

from config import PROJECT_CONFIG, PROJECT_ROOT


def ensure_project_directories() -> None:
    """Create all output directories required by the project."""
    for path in PROJECT_CONFIG["outputs"].values():
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_text_report(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def as_percent(value: float) -> str:
    return f"{value * 100:.2f}%"
