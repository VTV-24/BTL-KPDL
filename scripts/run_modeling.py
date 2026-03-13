"""
scripts/run_modeling.py
=======================
Đào tạo mô hình phân loại từ file cluster_input.parquet.
Output:
  - outputs/models/best_model.pkl
  - outputs/tables/model_metrics.csv
  - outputs/figures/confusion_matrix.png
  - outputs/figures/feature_importance.png
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ensure src directory is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.models import supervised
from src.evaluation import metrics

warnings.filterwarnings("ignore")


def plot_confusion(y_true, y_pred, output_path: str):
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title("Ma trận nhầm lẫn")
    plt.tight_layout()
    disp.figure_.savefig(output_path, dpi=150)
    plt.close(disp.figure_)


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "params.yaml"))
    mdl_cfg = cfg.get("modeling", {})
    target_col = mdl_cfg.get("target", "target")
    drop_cols = mdl_cfg.get("drop_cols", [])
    algorithms = mdl_cfg.get("algorithms", ["logistic", "random_forest", "xgboost"])
    test_size = mdl_cfg.get("test_size", 0.2)
    random_state = mdl_cfg.get("random_state", 42)
    criterion = mdl_cfg.get("selection_criterion", "roc_auc")

    # load data
    processed_dir = os.path.join(ROOT, cfg["paths"]["processed_dir"])
    input_path = os.path.join(processed_dir, "cluster_input.parquet")
    df = pd.read_parquet(input_path)
    print(f"[INFO] đã tải dữ liệu đầu vào phân cụm: {df.shape}")
    print(df.columns)
    # prepare features
    X, y = supervised.prepare_features(df, target_col=target_col, drop_cols=drop_cols)
    X_train, X_test, y_train, y_test = supervised.split_data(X, y, test_size=test_size, random_state=random_state)
    print(f"[INFO] train/test split: {X_train.shape}, {X_test.shape}")

    # fit models
    models = supervised.train_models(X_train, y_train, algorithms=algorithms, random_state=random_state)
    print(f"[INFO] đã huấn luyện các mô hình: {list(models.keys())}")

    # evaluate
    metrics_df = supervised.evaluate_models(models, X_test, y_test)
    print(metrics_df)

    # output dirs
    output_dir = os.path.join(ROOT, cfg["paths"].get("output_dir", "outputs"))
    models_dir = os.path.join(output_dir, "models")
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # save metrics
    metrics_df.to_csv(os.path.join(tables_dir, "model_metrics.csv"))
    print(f"[LƯU] metrics vào {tables_dir}/model_metrics.csv")

    # choose best
    best_name = supervised.select_best_model(metrics_df, criterion=criterion)
    best_model = models[best_name]
    supervised.save_model(best_model, os.path.join(models_dir, "best_model.pkl"))
    print(f"[LƯU] mô hình tốt nhất ({best_name})")

    # confusion matrix
    y_pred_best = best_model.predict(X_test)
    plot_confusion(y_test, y_pred_best, os.path.join(figures_dir, "confusion_matrix.png"))
    print(f"[LƯU] ma trận nhầm lẫn")

    # feature importance plot
    feat_imp = supervised.feature_importance(best_model, list(X.columns))
    if not feat_imp.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        feat_imp.head(20).plot(kind="barh", ax=ax)
        ax.set_title(f"Feature importance ({best_name})")
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "feature_importance.png"), dpi=150)
        plt.close(fig)
        print(f"[LƯU] tầm quan trọng đặc trưng")

    print("\n[DONE] modeling pipeline complete.")


if __name__ == "__main__":
    main()
