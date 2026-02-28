from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# statsmodels for ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Prophet is optional (fbprophet / prophet)
try:
    from prophet import Prophet  # type: ignore
    _HAS_PROPHET = True
except ImportError:  # pragma: no cover
    _HAS_PROPHET = False

logger = logging.getLogger(__name__)


def naive_forecast(series: pd.Series, horizon: int) -> pd.Series:
    """Trả về dự báo đơn giản: giá trị cuối cùng được lặp lại."""
    last = series.iloc[-1]
    idx = pd.date_range(series.index[-1], periods=horizon + 1, freq=series.index.freq or "MS")[1:]
    return pd.Series([last] * horizon, index=idx)


def train_arima(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> ARIMA:
    """Huấn luyện mô hình ARIMA và trả về kết quả đã fit."""
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def forecast_arima(model: ARIMA, steps: int) -> pd.Series:
    """Sinh dự báo từ mô hình ARIMA đã được huấn luyện."""
    res = model.get_forecast(steps=steps)
    pred = res.predicted_mean
    # đảm bảo tần số giống chỉ mục huấn luyện
    return pd.Series(pred, index=pd.date_range(model.data.endog.index[-1], periods=steps + 1, freq=model.data.endog.index.freq or "MS")[1:])


def train_prophet(df: pd.DataFrame, date_col: str = "ds", value_col: str = "y") -> Any:
    """Huấn luyện mô hình Prophet. df phải có cột ds và y."""
    if not _HAS_PROPHET:
        raise ImportError("Chưa cài Prophet")
    m = Prophet()
    m.fit(df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"}))
    return m


def forecast_prophet(model: Any, periods: int, freq: str = "MS") -> pd.DataFrame:
    """Tạo DataFrame dự báo bằng mô hình Prophet đã huấn luyện."""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    # chỉ trả về phần dự báo
    return forecast.set_index("ds")["yhat"].iloc[-periods:]


def save_model(model: Any, path: str) -> None:
    try:
        import joblib
        joblib.dump(model, path)
    except ImportError:  # pragma: no cover
        raise


def load_model(path: str) -> Any:
    import joblib

    return joblib.load(path)
