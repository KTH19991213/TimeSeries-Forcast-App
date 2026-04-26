import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_train_test_pred(y_train: pd.Series, y_test: pd.Series, pred_dict: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode="lines", name="train"))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode="lines", name="test"))
    for name, pred in pred_dict.items():
        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode="lines", name=name))
    fig.update_layout(height=460, title="Forecast Overview", xaxis_title="time", yaxis_title="value")
    return fig


def plot_metric_bars(metric_df: pd.DataFrame):
    if metric_df is None or metric_df.empty:
        return go.Figure()
    d = metric_df.reset_index().melt(id_vars=["model"], var_name="metric", value_name="value")
    fig = px.bar(d, x="model", y="value", color="metric", barmode="group", height=420)
    fig.update_layout(title="Model Metrics")
    return fig


def plot_residual_hist(y_true: pd.Series, y_pred: pd.Series):
    resid = (y_true - y_pred).dropna()
    fig = px.histogram(resid, nbins=24, title="Residual Distribution")
    fig.update_layout(height=320)
    return fig
