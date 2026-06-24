from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.yml"


def load_config() -> dict[str, Any]:
    """Load JSON-compatible YAML without requiring PyYAML at runtime."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Config is not JSON-compatible and PyYAML is not installed."
            ) from exc
        return yaml.safe_load(text)


def project_path(relative_path: str | Path) -> Path:
    return PROJECT_ROOT / relative_path


def configured_path(config: dict[str, Any], key: str) -> Path:
    return project_path(config["paths"][key])


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ["processed_dir", "tables_dir", "figures_dir", "reports_dir"]:
        configured_path(config, key).mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    display = df.copy()
    display = display.where(pd.notna(display), "")
    columns = [str(c) for c in display.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(rows)


def split_name(year: int, config: dict[str, Any]) -> str:
    split = config["splits"]
    if split["train_start"] <= year <= split["train_end"]:
        return "train"
    if split["validation_start"] <= year <= split["validation_end"]:
        return "validation"
    if split["test_start"] <= year <= split["test_end"]:
        return "test"
    return "outside"


def max_consecutive_true(values: pd.Series) -> int:
    best = 0
    current = 0
    for value in values.fillna(False).astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def rolling_sum_max(values: pd.Series, window: int) -> float:
    if values.empty:
        return float("nan")
    return float(values.fillna(0).rolling(window=window, min_periods=1).sum().max())


def clean_name(value: str) -> str:
    return (
        value.lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_metric(func, y_true, y_pred) -> float:
    try:
        value = func(y_true, y_pred)
    except Exception:
        return float("nan")
    return safe_float(value)


def add_metric(
    rows: list[dict[str, Any]],
    forecast_window: str,
    target: str,
    task: str,
    model: str,
    metric: str,
    value: Any,
    n_test: int,
) -> None:
    rows.append(
        {
            "forecast_window": forecast_window,
            "target": target,
            "task": task,
            "model": model,
            "metric": metric,
            "value": safe_float(value),
            "n_test": int(n_test),
        }
    )


def finite_or_nan(value: Any) -> float:
    value = safe_float(value)
    return value if np.isfinite(value) else float("nan")
