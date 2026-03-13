from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# xgboost optional
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

import joblib

logger = logging.getLogger(__name__)


# ================================
# CLASSIFIERS
# ================================

def _get_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Trả về dict chứa các model classification."""
    
    classifiers: Dict[str, Any] = {
        "logistic": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
    }

    if _HAS_XGB:
        classifiers["xgboost"] = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
        )

    return classifiers


# ================================
# FEATURE PREPARATION
# ================================

def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tách features và target.
    Tự động one-hot encode các biến categorical.
    """

    if drop_cols is None:
        drop_cols = []

    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")

    # one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    y = df[target_col].copy()

    return X, y


# ================================
# TRAIN / TEST SPLIT
# ================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    stratify = y if len(np.unique(y)) > 1 else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


# ================================
# TRAIN MODELS
# ================================

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithms: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:

    """Huấn luyện nhiều model classification."""

    if algorithms is None:
        algorithms = ["logistic", "random_forest"]
        if _HAS_XGB:
            algorithms.append("xgboost")

    classifiers = _get_classifiers(random_state)

    trained: Dict[str, Any] = {}

    for name in algorithms:

        if name not in classifiers:
            logger.warning("Thuật toán %s không khả dụng.", name)
            continue

        model = classifiers[name]

        logger.info("Training model: %s", name)

        model.fit(X_train, y_train)

        trained[name] = model

    return trained


# ================================
# EVALUATION
# ================================

def evaluate_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:

    """Tính metrics cho từng model."""

    from src.evaluation.metrics import classification_metrics

    records = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        y_proba = None

        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                pass

        metrics = classification_metrics(
            y_test,
            y_pred,
            y_proba=y_proba,
        )

        metrics["model"] = name

        records.append(metrics)

    df = pd.DataFrame(records).set_index("model")

    return df


# ================================
# FEATURE IMPORTANCE
# ================================

def feature_importance(
    model: Any,
    feature_names: List[str],
) -> pd.Series:

    """Tính importance cho feature."""

    if hasattr(model, "feature_importances_"):

        vals = model.feature_importances_

    elif hasattr(model, "coef_"):

        vals = model.coef_.ravel()

    else:

        return pd.Series(dtype=float)

    return (
        pd.Series(vals, index=feature_names)
        .sort_values(ascending=False)
    )


# ================================
# BEST MODEL
# ================================

def select_best_model(
    metrics_df: pd.DataFrame,
    criterion: str = "roc_auc",
    higher_is_better: bool = True,
) -> str:

    """Chọn model tốt nhất theo metric."""

    if criterion not in metrics_df.columns:
        raise ValueError(f"Criterion {criterion} not found in metrics")

    if higher_is_better:
        return metrics_df[criterion].idxmax()
    else:
        return metrics_df[criterion].idxmin()


# ================================
# SAVE / LOAD
# ================================

def save_model(model: Any, path: str) -> None:

    joblib.dump(model, path)


def load_model(path: str) -> Any:

    return joblib.load(path)