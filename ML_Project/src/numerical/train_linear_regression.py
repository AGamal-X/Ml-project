from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

from config import PROJECT_CONFIG
from src.numerical.data_preprocessing import (
    build_preprocessor,
    cap_target_values,
    evaluate_target_transform_need,
    get_feature_names,
)
from src.numerical.evaluate_regression import regression_metrics


def _fit_linear_model(X_train, y_train, X_eval, use_log_transform: bool, config: Dict[str, object]):
    model = LinearRegression(
        fit_intercept=config["fit_intercept"],
        positive=config["positive"],
    )
    if use_log_transform:
        model = TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)
    return model, predictions


def train_linear_regression(data: Dict[str, object]) -> Dict[str, object]:
    config = PROJECT_CONFIG["numerical_models"]["linear_regression"]
    metadata = data["metadata"]

    preview_preprocessor, numeric_features, categorical_features = build_preprocessor(data["X_train"])
    X_train_processed = preview_preprocessor.fit_transform(data["X_train"])
    X_val_processed = preview_preprocessor.transform(data["X_val"])

    transform_decision = evaluate_target_transform_need(data["y_train"])
    validation_trials = {}

    raw_model, raw_predictions = _fit_linear_model(
        X_train_processed,
        data["train_target_capped"],
        X_val_processed,
        False,
        config,
    )
    raw_metrics = regression_metrics(data["y_val"], raw_predictions)
    validation_trials["without_log_transform"] = raw_metrics

    selected_use_log = False
    selected_reason = "Validation RMSE favored training on the original target scale."
    selected_validation_metrics = raw_metrics

    if transform_decision.use_log_transform:
        log_model, log_predictions = _fit_linear_model(
            X_train_processed,
            data["train_target_capped"],
            X_val_processed,
            True,
            config,
        )
        log_metrics = regression_metrics(data["y_val"], log_predictions)
        validation_trials["with_log_transform"] = log_metrics

        if log_metrics["RMSE"] + 1e-9 < raw_metrics["RMSE"]:
            selected_use_log = True
            selected_reason = "Validation RMSE improved after applying log1p to the target."
            selected_validation_metrics = log_metrics
        else:
            selected_reason = (
                "Target skewness was high enough for testing log1p, but validation RMSE did not improve, "
                "so the original target scale was kept."
            )
    else:
        validation_trials["with_log_transform"] = {
            "MAE": None,
            "MSE": None,
            "RMSE": None,
            "R2": None,
        }

    final_preprocessor, _, _ = build_preprocessor(data["X_train_val"])
    X_train_val_processed = final_preprocessor.fit_transform(data["X_train_val"])
    X_test_processed = final_preprocessor.transform(data["X_test"])
    final_target = cap_target_values(data["y_train_val"], data["target_bounds_train_val"]).to_numpy()

    final_model, test_predictions = _fit_linear_model(
        X_train_val_processed,
        final_target,
        X_test_processed,
        selected_use_log,
        config,
    )
    test_metrics = regression_metrics(data["y_test"], test_predictions)

    feature_names = get_feature_names(final_preprocessor, numeric_features, categorical_features)
    metadata["encoded_feature_names"] = feature_names
    metadata["number_of_extracted_features"] = len(feature_names)
    metadata["final_target_transform"] = {
        "used_log1p": selected_use_log,
        "reason": selected_reason,
        "train_skewness": transform_decision.skewness,
    }

    model_path = PROJECT_CONFIG["outputs"]["models_dir"] / "linear_regression_model.joblib"
    preprocessor_path = PROJECT_CONFIG["outputs"]["models_dir"] / "linear_regression_preprocessor.joblib"
    joblib.dump(final_preprocessor, preprocessor_path)
    joblib.dump(
        {
            "model": final_model,
            "preprocessor": final_preprocessor,
            "feature_columns": metadata["raw_feature_columns"],
            "target_column": metadata["target_column"],
            "target_transform": metadata["final_target_transform"],
        },
        model_path,
    )

    hyperparameters = dict(config)
    hyperparameters["selected_log1p_transform"] = selected_use_log
    hyperparameters["validation_trials"] = validation_trials

    return {
        "model_name": "Linear Regression",
        "model": final_model,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "test_predictions": np.asarray(test_predictions),
        "test_metrics": test_metrics,
        "validation_metrics": selected_validation_metrics,
        "validation_trials": validation_trials,
        "hyperparameters": hyperparameters,
    }
