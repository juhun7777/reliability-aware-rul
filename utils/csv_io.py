"""Read standard sensor CSVs without depending on a Windows default encoding."""

import pandas as pd


def read_csv(path):
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="cp949")
    frame.columns = frame.columns.str.replace("\ufeff", "").str.strip()
    return frame
