import pandas as pd


def load_csv(file_obj) -> pd.DataFrame:
    if file_obj is None:
        raise ValueError("CSV 파일이 선택되지 않았습니다.")

    try:
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="cp949")


def candidate_datetime_columns(df: pd.DataFrame) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    cols = []
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.8:
            cols.append(c)
    return cols


def candidate_numeric_columns(df: pd.DataFrame) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
