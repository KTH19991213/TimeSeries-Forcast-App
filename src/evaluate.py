import numpy as np
import pandas as pd


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, np.nan, denom)
    return float(np.nanmean(2.0 * np.abs(y_true - y_pred) / denom))


def _mase(y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, m: int = 1) -> float:
    if len(y_train) <= m:
        return np.nan
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale < 1e-8:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def calc_metrics(y_train: pd.Series, y_true: pd.Series, y_pred: pd.Series) -> dict:
    paired = pd.concat([y_true.rename("y"), y_pred.rename("p")], axis=1).dropna()
    if paired.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "SMAPE": np.nan, "MASE": np.nan}

    yt = paired["y"].to_numpy(dtype=float)
    yp = paired["p"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": _safe_mape(yt, yp),
        "SMAPE": _smape(yt, yp),
        "MASE": _mase(y_train.to_numpy(dtype=float), yt, yp, m=1),
    }


def compare_models(y_train: pd.Series, y_test: pd.Series, pred_dict: dict) -> pd.DataFrame:
    if not pred_dict:
        return pd.DataFrame(columns=["MAE", "RMSE", "MAPE", "SMAPE", "MASE"])

    rows = []
    for name, pred in pred_dict.items():
        m = calc_metrics(y_train, y_test, pred)
        m["model"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("model")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.sort_values("MAPE")
