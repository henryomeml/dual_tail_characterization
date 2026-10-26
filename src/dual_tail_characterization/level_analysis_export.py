"""Strict JSON and CSV tables for internal diagnostics; missing values stay null."""
import csv
import json
import math
import numpy as np
from pathlib import Path


def _safe(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def export_table(records, output_stem):
    """JSON null and empty CSV cells denote unavailable values, never zero."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    rows = _safe(list(records))
    output_stem.with_suffix(".json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if fields:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
