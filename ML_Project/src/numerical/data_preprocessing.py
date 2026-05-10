from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import PROJECT_CONFIG


@dataclass
class TargetTransformDecision:
    use_log_transform: bool
    skewness: float
    reason: str


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, multiplier: float = 1.5) -> None:
        self.multiplier = multiplier
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        q1 = frame.quantile(0.25)
        q3 = frame.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = (q1 - self.multiplier * iqr).to_numpy(dtype=float)
        self.upper_bounds_ = (q3 + self.multiplier * iqr).to_numpy(dtype=float)
        return self

    def transform(self, X):
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("IQRClipper must be fitted before calling transform().")
        array = np.asarray(X, dtype=float)
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)


def get_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def find_numerical_dataset_file() -> Path:
    dataset_config = PROJECT_CONFIG["datasets"]["numerical"]
    configured_path = Path(dataset_config["data_dir"]) / dataset_config["file_name"]
    if configured_path.exists():
        return configured_path

    csv_files = sorted(Path(dataset_config["data_dir"]).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found in datasets/numerical/. Place the provided numerical dataset there."
        )
    return csv_files[0]


def load_numerical_dataset() -> pd.DataFrame:
    dataset_path = find_numerical_dataset_file()
    return pd.read_csv(dataset_path)


def infer_target_column(df: pd.DataFrame) -> str:
    configured_target = PROJECT_CONFIG["datasets"]["numerical"].get("target_column")
    if configured_target and configured_target in df.columns:
        return configured_target

    priority_keywords = ["salary_in_usd", "target", "price", "salary", "income", "value", "score"]
    lower_columns = {column.lower(): column for column in df.columns}
    for keyword in priority_keywords:
        if keyword in lower_columns:
            return lower_columns[keyword]

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        return numeric_columns[-1]

    raise ValueError(
        "Could not infer a numerical regression target column. Set target_column in config.py."
    )


def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    preprocessing_config = PROJECT_CONFIG["numerical_models"]["preprocessing"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=preprocessing_config["numeric_imputation_strategy"])),
            ("clipper", IQRClipper(multiplier=preprocessing_config["feature_outlier_multiplier"])),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=preprocessing_config["categorical_imputation_strategy"])),
            ("onehot", get_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    feature_names = list(numeric_features)
    if categorical_features:
        onehot = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        encoded_names = onehot.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(encoded_names)
    return feature_names


def compute_target_outlier_bounds(target: pd.Series) -> Dict[str, float]:
    handling_config = PROJECT_CONFIG["numerical_models"]["target_handling"]
    q1 = float(target.quantile(0.25))
    q3 = float(target.quantile(0.75))
    iqr = q3 - q1
    multiplier = float(handling_config["target_outlier_multiplier"])
    return {
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "lower_bound": float(q1 - multiplier * iqr),
        "upper_bound": float(q3 + multiplier * iqr),
        "multiplier": multiplier,
    }


def cap_target_values(target: pd.Series, bounds: Dict[str, float]) -> pd.Series:
    return target.clip(bounds["lower_bound"], bounds["upper_bound"])


def inspect_numerical_dataset(
    df: pd.DataFrame,
    clean_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    target_column: str,
    drop_columns: List[str],
) -> Dict[str, object]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    missing_summary = df.isna().sum().to_dict()
    target_distribution = y.describe().to_dict()
    target_quantiles = y.quantile([0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]).to_dict()

    major_numeric_feature_summary = {}
    for column in numeric_features:
        major_numeric_feature_summary[column] = {
            "min": float(X[column].min()),
            "max": float(X[column].max()),
            "mean": float(X[column].mean()),
            "median": float(X[column].median()),
            "skewness": float(X[column].skew()) if len(X[column]) > 2 else 0.0,
        }

    return {
        "dataset_name": PROJECT_CONFIG["datasets"]["numerical"]["dataset_name"],
        "source": PROJECT_CONFIG["datasets"]["numerical"]["source"],
        "dataset_file": str(find_numerical_dataset_file()),
        "target_column": target_column,
        "dropped_columns": drop_columns,
        "original_shape": list(df.shape),
        "used_shape": list(clean_df.shape),
        "original_columns": df.columns.tolist(),
        "raw_feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dtype_summary": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {column: int(value) for column, value in missing_summary.items()},
        "target_distribution": {key: float(value) for key, value in target_distribution.items()},
        "target_quantiles": {str(key): float(value) for key, value in target_quantiles.items()},
        "target_skewness": float(y.skew()) if len(y) > 2 else 0.0,
        "major_numeric_feature_summary": major_numeric_feature_summary,
        "feature_dimensions": {},
        "split_counts": {},
    }


def evaluate_target_transform_need(y_train: pd.Series) -> TargetTransformDecision:
    handling_config = PROJECT_CONFIG["numerical_models"]["target_handling"]
    skewness = float(y_train.skew()) if len(y_train) > 2 else 0.0
    log_allowed = bool(handling_config["allow_log_transform"])
    threshold = float(handling_config["log_skew_threshold"])
    minimum_value = float(y_train.min())

    if not log_allowed:
        return TargetTransformDecision(False, skewness, "Log transform disabled in configuration.")
    if minimum_value <= -1.0:
        return TargetTransformDecision(False, skewness, "Target contains values <= -1, so log1p is not valid.")
    if abs(skewness) < threshold:
        return TargetTransformDecision(False, skewness, "Target skewness is not high enough to justify log1p.")
    return TargetTransformDecision(True, skewness, "Target is strongly skewed and eligible for log1p testing.")


def split_numerical_dataset() -> Dict[str, object]:
    df = load_numerical_dataset()
    target_column = infer_target_column(df)
    drop_columns = [
        column
        for column in PROJECT_CONFIG["datasets"]["numerical"].get("drop_columns", [])
        if column in df.columns and column != target_column
    ]

    clean_df = df.drop(columns=drop_columns).copy()
    clean_df = clean_df.dropna(subset=[target_column])

    X = clean_df.drop(columns=[target_column])
    y = clean_df[target_column].astype(float)

    metadata = inspect_numerical_dataset(df, clean_df, X, y, target_column, drop_columns)
    target_transform_decision = evaluate_target_transform_need(y)
    metadata["target_transform_decision"] = {
        "candidate_log_transform": target_transform_decision.use_log_transform,
        "skewness": target_transform_decision.skewness,
        "reason": target_transform_decision.reason,
    }

    splits = PROJECT_CONFIG["splits"]
    random_state = PROJECT_CONFIG["random_state"]
    test_size = splits["test_size"]
    validation_fraction_of_train_val = splits["validation_size"] / (
        splits["train_size"] + splits["validation_size"]
    )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction_of_train_val,
        random_state=random_state,
    )

    train_target_bounds = compute_target_outlier_bounds(y_train)
    train_val_target_bounds = compute_target_outlier_bounds(y_train_val)
    metadata["target_outlier_bounds"] = train_target_bounds
    metadata["split_counts"] = {
        "train": int(len(X_train)),
        "validation": int(len(X_val)),
        "test": int(len(X_test)),
        "total": int(len(clean_df)),
    }

    preview_preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    X_train_preview = preview_preprocessor.fit_transform(X_train)
    X_val_preview = preview_preprocessor.transform(X_val)
    X_test_preview = preview_preprocessor.transform(X_test)
    metadata["numeric_features"] = numeric_features
    metadata["categorical_features"] = categorical_features
    metadata["encoded_feature_names"] = get_feature_names(
        preview_preprocessor,
        numeric_features,
        categorical_features,
    )
    metadata["number_of_extracted_features"] = len(metadata["encoded_feature_names"])
    metadata["feature_dimensions"] = {
        "train": list(X_train_preview.shape),
        "validation": list(X_val_preview.shape),
        "test": list(X_test_preview.shape),
    }

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_val": X_val.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "X_train_val": X_train_val.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "y_train_val": y_train_val.reset_index(drop=True),
        "train_target_capped": cap_target_values(y_train, train_target_bounds).to_numpy(),
        "train_val_target_capped": cap_target_values(y_train_val, train_val_target_bounds).to_numpy(),
        "target_bounds_train": train_target_bounds,
        "target_bounds_train_val": train_val_target_bounds,
        "metadata": metadata,
    }
