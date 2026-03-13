from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)


def classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
    average: str = "weighted",
) -> dict:
    """Tính một bộ các chỉ số phân loại phổ biến.

    Parameters
    ----------
    y_true : array-like
        Nhãn thực sự.
    y_pred : array-like
        Nhãn dự đoán.
    y_proba : array-like, optional
        Xác suất dự đoán cho lớp tích cực. Cần thiết nếu muốn tính ROC-AUC.
    average : str
        Phương pháp lấy trung bình truyền cho precision/recall/f1 ("binary",
        "macro", "micro", v.v.).

    Returns
    -------
    dict
        Từ điển có các key: accuracy, precision, recall, f1, roc_auc
        (nếu y_proba được cung cấp).
    """

    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            metrics["roc_auc"] = np.nan
    return metrics


def forecast_metrics(y_true, y_pred) -> dict:
    """Tính các chỉ số đánh giá dự báo đơn giản.

    Parameters
    ----------
    y_true : array-like
        Giá trị thực tế.
    y_pred : array-like
        Giá trị dự báo.

    Returns
    -------
    dict
        Từ điển chứa mae, rmse, mape.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # avoid divide-by-zero
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}
