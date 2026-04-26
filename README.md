# TimeSeries-Forcast-App

# Time Series Forecast Web App

Upload a univariate CSV and run automatic forecasting with dashboard metrics.

## Run
1. Install dependencies:
   pip install -r requirements.txt
2. Start app:
   streamlit run app.py

## Features
- CSV upload (univariate)
- KRX ETF daily close via pykrx API
- Forecast horizon control
- Re-forecast when file or parameters change
- Model comparison dashboard
- Metrics: MAE, RMSE, MAPE, SMAPE, MASE

## Sample CSV files
- sample_data/monthly_demand_sample.csv
- sample_data/daily_traffic_sample.csv
- sample_data/airline_official_sktime.csv (from `sktime.datasets.load_airline` used in lectures)

Both files use this schema:
- date: timestamp column
- value: numeric target column

## KRX API mode
1. Select `KRX ETF API` in the app.
2. Choose start/end date and ETF list base date.
3. Select ETF and click `KRX 데이터 불러오기`.
4. Run forecasting with the fetched `date,value` series.

## KRX API troubleshooting
- In some environments, `pykrx` requires KRX credentials via `KRX_ID` and `KRX_PW`.
- You can provide them in app inputs, or set environment variables before running Streamlit.
- If KRX API fails temporarily, switch to `CSV 업로드` mode for demo continuity.
