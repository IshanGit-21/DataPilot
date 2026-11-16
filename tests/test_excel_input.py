"""Excel files are read like CSVs, with a choice of sheet."""
import io

import numpy as np
import pandas as pd

from datapilot.io import excel_sheets, is_excel, read_any, read_table


class Upload(io.BytesIO):   # what Streamlit hands over: a file-like object with a name
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def _workbook():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.normal(size=50), "y": rng.integers(0, 2, 50)})
    b = pd.DataFrame({"p": [1, 2, 3], "q": ["a", "b", "c"]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        a.to_excel(w, sheet_name="data", index=False)
        b.to_excel(w, sheet_name="notes", index=False)
    return buf.getvalue(), a, b


def test_excel_is_detected_and_sheets_listed():
    raw, a, b = _workbook()
    up = Upload(raw, "sales.xlsx")
    assert is_excel(up) and not is_excel(Upload(b"a,b\n1,2", "sales.csv"))
    assert excel_sheets(up) == ["data", "notes"]


def test_chosen_sheet_is_read_and_csv_still_works():
    raw, a, b = _workbook()
    df, headerless = read_table(Upload(raw, "sales.xlsx"), "notes")
    assert list(df.columns) == ["p", "q"] and not headerless
    df0, _ = read_table(Upload(raw, "sales.xlsx"))          # first sheet by default
    assert list(df0.columns) == ["x", "y"] and len(df0) == 50
    csv = read_any(Upload(b"u,v\n1,2\n3,4\n", "t.csv"))
    assert csv.shape == (2, 2)


def test_headerless_excel_gets_generic_names():
    buf = io.BytesIO()
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).to_excel(buf, header=False, index=False)
    df, headerless = read_table(Upload(buf.getvalue(), "raw.xlsx"))
    assert headerless and list(df.columns) == ["col_0", "col_1"] and len(df) == 3
