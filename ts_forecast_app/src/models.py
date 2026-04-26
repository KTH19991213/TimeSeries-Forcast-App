import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


def _safe_fit_predict(model, y_train: pd.Series, fh: ForecastingHorizon):
    try:
        model.fit(y_train)
        return model.predict(fh)
    except Exception:
        return None


def fit_and_predict(y_train: pd.Series, y_test: pd.Series, horizon: int) -> dict:
    if y_train is None or y_test is None or len(y_train) < 2 or len(y_test) < 1:
        return {}

    h = max(1, min(int(horizon), len(y_test)))
    fh = ForecastingHorizon(range(1, h + 1), is_relative=True)
    pred = {}

    p1 = _safe_fit_predict(NaiveForecaster(strategy="last"), y_train, fh)
    if p1 is not None:
        pred["naive_last"] = p1

    p2 = _safe_fit_predict(
        NaiveForecaster(strategy="mean", window_length=min(12, len(y_train))),
        y_train,
        fh,
    )
    if p2 is not None:
        pred["naive_mean12"] = p2

    try:
        model3 = ExponentialSmoothing(trend="add", seasonal=None)
        p3 = _safe_fit_predict(model3, y_train, fh)
        if p3 is not None:
            pred["exp_smoothing"] = p3
    except Exception:
        pass

    try:
        model4 = ExponentialSmoothing(trend="add", seasonal="add", sp=12)
        p4 = _safe_fit_predict(model4, y_train, fh)
        if p4 is not None:
            pred["exp_smoothing_seasonal12"] = p4
    except Exception:
        pass

    return pred
