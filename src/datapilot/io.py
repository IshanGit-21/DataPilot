"""Reading user files robustly."""
from __future__ import annotations

import pandas as pd


def _is_number(x) -> bool:
    try:
        float(str(x).strip())
        return True
    except ValueError:
        return False


EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")


def is_excel(source) -> bool:
    name = getattr(source, "name", source)
    return isinstance(name, str) and name.lower().endswith(EXCEL_SUFFIXES)


def excel_sheets(source) -> list[str]:
    """Sheet names of an Excel workbook (the file is rewound afterwards)."""
    names = list(pd.ExcelFile(source).sheet_names)
    if hasattr(source, "seek"):
        source.seek(0)
    return names


def read_any(source, sheet=None, **kwargs) -> pd.DataFrame:
    """Read a CSV or an Excel sheet (the first sheet unless `sheet` is given)."""
    if is_excel(source):
        return pd.read_excel(source, sheet_name=sheet if sheet is not None else 0, **kwargs)
    return pd.read_csv(source, **kwargs)


def read_table(source, sheet=None, **kwargs) -> tuple[pd.DataFrame, bool]:
    """Read a CSV or Excel sheet. Returns (df, headerless).

    Some public datasets (UCI Wine, for example) ship without a header row, so
    pandas turns the first data row into column names. If every first-row cell
    is numeric, no real header is plausible (column names are almost never
    numbers), so re-read with generic names col_0, col_1, ...
    """
    df = read_any(source, sheet, **kwargs)
    if len(df.columns) and all(_is_number(c) for c in df.columns):
        if hasattr(source, "seek"):
            source.seek(0)
        df = read_any(source, sheet, header=None, **kwargs)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        return df, True
    return df, False
