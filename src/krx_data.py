from __future__ import annotations

import pandas as pd
from pykrx import stock


def get_etf_name(ticker: str) -> str:
    try:
        name = stock.get_etf_ticker_name(ticker)
        return name if isinstance(name, str) else ""
    except Exception:
        return ""


def fetch_etf_close_csv_schema(start_date: str, end_date: str, ticker: str) -> pd.DataFrame:
    try:
        ohlcv = stock.get_etf_ohlcv_by_date(start_date, end_date, ticker)
    except Exception as e:
        raise RuntimeError(
            "ETF 시세 조회 실패. 티커 또는 KRX 인증 설정(KRX_ID/KRX_PW)을 확인하세요."
        ) from e

    if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        return pd.DataFrame(columns=["date", "value"])

    out = ohlcv.reset_index()[["날짜", "종가"]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out
