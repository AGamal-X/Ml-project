from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from config import PROJECT_CONFIG
from src.numerical.data_preprocessing import (
    build_preprocessor,
    cap_target_values,
    evaluate_target_transform_need,
    get_feature_names,
)
from src.numerical.evaluate_regression import regression_metrics


def _make_search_pipeline(X_train, use_log_transform: bool) -> Pipeline:
    preprocessor, _, _ = build_preprocessor(X_train)
    regressor = KNeighborsRegressor()
    if use_log_transform:
        regressor = TransformedTargetRegressor(
            regressor=regressor,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def _make_param_grid(use_log_transform: bool) -> Dict[str, object]:
    config = PROJECT_CONFIG["numerical_models"]["knn_regressor"]
    prefix = "regressor__regressor__" if use_log_transform else "regressor__"
    return {
        f"{prefix}n_neighbors": config["candidate_k_values"],
        f"{prefix}weights": config["candidate_weights"],
        f"{prefix}metric": config["candidate_metrics"],
    }


def train_knn_regressor(data: Dict[str, object]) -> Dict[str, object]:
    config = PROJECT_CONFIG["numerical_models"]["knn_regressor"]
    metadata = data["metadata"]
    transform_decision = evaluate_target_transform_need(data["y_train"])
    cv_folds = int(config["cv_folds"])

    search_results = {}

    raw_search = GridSearchCV(
        estimator=_make_search_pipeline(data["X_train"], use_log_transform=False),
        param_grid=_make_param_grid(use_log_transform=False),
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
        refit=True,
    )
    raw_search.fit(data["X_train"], data["train_target_capped"])
    raw_validation_predictions = raw_search.best_estimator_.predict(data["X_val"])
    raw_validation_metrics = regression_metrics(data["y_val"], raw_validation_predictions)
    search_results["without_log_transform"] = {
        "best_params": raw_search.best_params_,
        "best_cv_rmse": float(-raw_search.best_score_),
        "validation_metrics": raw_validation_metrics,
    }

    selected_use_log = False
    selected_params = raw_search.best_params_
    selected_reason = "Validation RMSE favored KNN on the original target scale."
    selected_validation_metrics = raw_validation_metrics

    if transform_decision.use_log_transform:
        log_search = GridSearchCV(
            estimator=_make_search_pipeline(data["X_train"], use_log_transform=True),
            param_grid=_make_param_grid(use_log_transform=True),
            scoring="neg_root_mean_squared_error",
            cv=cv_folds,
            n_jobs=-1,
            refit=True,
        )
        log_search.fit(data["X_train"], data["train_target_capped"])
        log_validation_predictions = log_search.best_estimator_.predict(data["X_val"])
        log_validation_metrics = regression_metrics(data["y_val"], log_validation_predictions)
        search_results["with_log_transform"] = {
            "best_params": log_search.best_params_,
            "best_cv_rmse": float(-log_search.best_score_),
            "validation_metrics": log_validation_metrics,
        }
        if log_validation_metrics["RMSE"] + 1e-9 < raw_validation_metrics["RMSE"]:
            selected_use_log = True
            selected_params = log_search.best_params_
            selected_reason = "Validation RMSE improved after applying log1p to the KNN target."
            selected_validation_metrics = log_validation_metrics
    else:
        search_results["with_log_transform"] = {
            "best_params": {},
            "best_cv_rmse": None,
            "validation_metrics": {
                "MAE": None,
                "MSE": None,
                "RMSE": None,
                "R2": None,
            },
        }

    final_preprocessor, numeric_features, categorical_features = build_preprocessor(data["X_train_val"])
    X_train_val_processed = final_preprocessor.fit_transform(data["X_train_val"])
    X_test_processed = final_preprocessor.transform(data["X_test"])

    selected_prefix = "regressor__regressor__" if selected_use_log else "regressor__"
    selected_n_neighbors = int(selected_params[f"{selected_prefix}n_neighbors"])
    selected_weights = selected_params[f"{selected_prefix}weights"]
    selected_metric = selected_params[f"{selected_prefix}metric"]

    final_regressor = KNeighborsRegressor(
        n_neighbors=selected_n_neighbors,
        weights=selected_weights,
        metric=selected_metric,
    )
    if selected_use_log:
        final_model = TransformedTargetRegressor(
            regressor=final_regressor,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    else:
        final_model = final_regressor

    final_target = cap_target_values(data["y_train_val"], data["target_bounds_train_val"]).to_numpy()
    final_model.fit(X_train_val_processed, final_target)
    test_predictions = final_model.predict(X_test_processed)
    test_metrics = regression_metrics(data["y_test"], test_predictions)

    feature_names = get_feature_names(final_preprocessor, numeric_features, categorical_features)
    metadata["encoded_feature_names"] = feature_names
    metadata["number_of_extracted_features"] = len(feature_names)

    model_path = PROJECT_CONFIG["outputs"]["models_dir"] / "knn_regressor_model.joblib"
    preprocessor_path = PROJECT_CONFIG["outputs"]["models_dir"] / "knn_regressor_preprocessor.joblib"
    joblib.dump(final_preprocessor, preprocessor_path)
    joblib.dump(
        {
            "model": final_model,
            "preprocessor": final_preprocessor,
            "feature_columns": metadata["raw_feature_columns"],
            "target_column": metadata["target_column"],
            "target_transform": {
                "used_log1p": selected_use_log,
                "reason": selected_reason,
                "train_skewness": transform_decision.skewness,
            },
        },
        model_path,
    )

    hyperparameters = dict(config)
    hyperparameters["selected_n_neighbors"] = selected_n_neighbors
    hyperparameters["selected_weights"] = selected_weights
    hyperparameters["selected_metric"] = selected_metric
    hyperparameters["selected_log1p_transform"] = selected_use_log
    hyperparameters["selection_reason"] = selected_reason

    return {
        "model_name": "KNN Regressor",
        "model": final_model,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "test_predictions": np.asarray(test_predictions),
        "test_metrics": test_metrics,
        "validation_metrics": selected_validation_metrics,
        "validation_results_by_search": search_results,
        "hyperparameters": hyperparameters,
    }
