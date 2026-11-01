"""Step 3 - Model building with experiment tracking (MLflow).

Loads the train / test splits (artifact from the data-prep job), tunes an
XGBoost classifier with GridSearchCV, logs every parameter combination and the
final metrics to MLflow, evaluates the best model and saves it into
tourism_project/deployment/ so the workflow can commit it to the repository.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import joblib

os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")   # keep the log short (54+ nested runs)
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SPLITS_DIR = Path(os.getenv("SPLITS_DIR", "."))
MODEL_DIR = Path(os.getenv(
    "MODEL_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "deployment"))
MODEL_FILE = "best_tourism_model_v1.joblib"
THRESHOLD = 0.45            # probability cut-off: favours catching likely buyers
RANDOM_STATE = 42


def configure_mlflow() -> None:
    """Use the MLflow server started by the workflow; fall back to a local DB."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        for _ in range(20):
            try:
                urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            print("MLflow server not reachable - logging to a local database instead.")
            uri = None
    mlflow.set_tracking_uri(uri or "sqlite:///mlflow.db")
    mlflow.set_experiment("tourism-wellness-package")
    print("MLflow tracking URI:", mlflow.get_tracking_uri())


def evaluate(model, X, y) -> dict:
    pred = (model.predict_proba(X)[:, 1] >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


def main() -> None:
    # ---- load the splits produced by the data-prep job ----
    Xtrain = pd.read_csv(SPLITS_DIR / "Xtrain.csv")
    Xtest = pd.read_csv(SPLITS_DIR / "Xtest.csv")
    ytrain = pd.read_csv(SPLITS_DIR / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(SPLITS_DIR / "ytest.csv").squeeze("columns")
    print("Train:", Xtrain.shape, "| Test:", Xtest.shape)

    # ---- preprocessing + model ----
    numeric = Xtrain.select_dtypes(include="number").columns.tolist()
    categorical = Xtrain.select_dtypes(exclude="number").columns.tolist()
    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        (make_pipeline(SimpleImputer(strategy="most_frequent"),
                       OneHotEncoder(handle_unknown="ignore")), categorical),
    )
    # class imbalance: ~4 non-buyers for every buyer
    scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                              random_state=RANDOM_STATE, n_jobs=2)
    pipeline = make_pipeline(preprocessor, model)

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200, 300],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.05, 0.1, 0.2],
        "xgbclassifier__colsample_bytree": [0.6, 0.8],
    }

    configure_mlflow()
    with mlflow.start_run(run_name="xgboost_grid_search"):
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=2)
        grid.fit(Xtrain, ytrain)

        # ---- log every tuned parameter combination as a nested run ----
        results = grid.cv_results_
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"combo_{i:02d}", nested=True):
                mlflow.log_params({k.split("__")[1]: v for k, v in params.items()})
                mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
                mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

        # ---- evaluate the best model ----
        best = grid.best_estimator_
        mlflow.log_params({f"best_{k.split('__')[1]}": v for k, v in grid.best_params_.items()})
        mlflow.log_param("classification_threshold", THRESHOLD)
        train_m, test_m = evaluate(best, Xtrain, ytrain), evaluate(best, Xtest, ytest)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_metric("best_cv_f1", grid.best_score_)

        print("\nBest parameters:", grid.best_params_)
        print("Best CV F1: %.3f" % grid.best_score_)
        print("Train metrics:", {k: round(v, 3) for k, v in train_m.items()})
        print("Test metrics: ", {k: round(v, 3) for k, v in test_m.items()})
        test_pred = (best.predict_proba(Xtest)[:, 1] >= THRESHOLD).astype(int)
        print("\nClassification report (test set):\n",
              classification_report(ytest, test_pred, target_names=["No purchase", "Purchase"]))

        # ---- save the best model where the Streamlit app expects it ----
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / MODEL_FILE
        joblib.dump(best, model_path)
        with open(MODEL_DIR / "model_metrics.json", "w") as f:
            json.dump({"threshold": THRESHOLD, "test": test_m, "train": train_m,
                       "best_params": {k.split("__")[1]: v for k, v in grid.best_params_.items()}},
                      f, indent=2, default=float)
        try:
            mlflow.log_artifact(str(model_path))
        except Exception as exc:      # tracking-server artifact problems must not fail the pipeline
            print("Warning: could not log the model artifact to MLflow:", exc)
        print("Saved model to", model_path)


if __name__ == "__main__":
    main()
