"""Strict JSON and CSV tables for internal diagnostics; missing values stay null."""
import json
import math
import numpy as np
from pathlib import Path
from .validation_workbook import RunTable


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
    table = RunTable.from_records(output_stem.with_suffix(".csv").name, output_stem.stem, rows, kind="internal level diagnostics")
    table.write_csv(output_stem.parent)
    return table
