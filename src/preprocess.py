import pandas as pd


def make_series(df: pd.DataFrame, time_col: str, value_col: str, freq: str) -> pd.Series:
    if df is None or df.empty:
        raise ValueError("입력 데이터가 비어 있습니다.")
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError("선택한 컬럼이 데이터에 없습니다.")

    work = df[[time_col, value_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna().sort_values(time_col)
    work = work.drop_duplicates(subset=[time_col], keep="last")

    if work.empty:
        raise ValueError("유효한 시계열 데이터가 없습니다.")

    y = work.set_index(time_col)[value_col]

    if freq == "auto":
        inferred = pd.infer_freq(y.index)
        if inferred:
            y = y.asfreq(inferred)
    else:
        y = y.asfreq(freq)

    # Keep time continuity for modeling.
    y = y.interpolate(method="linear").ffill().bfill()
    return y


def split_train_test(y: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series]:
    if y is None or len(y) < 2:
        raise ValueError("train/test 분할을 위한 데이터 길이가 부족합니다.")
    test_size = max(1, min(test_size, len(y) - 1))
    return y.iloc[:-test_size], y.iloc[-test_size:]
