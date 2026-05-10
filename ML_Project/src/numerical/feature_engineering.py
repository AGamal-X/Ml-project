from typing import Dict, List


def describe_numerical_feature_engineering(metadata: Dict[str, object]) -> str:
    numeric_features: List[str] = metadata["numeric_features"]
    categorical_features: List[str] = metadata["categorical_features"]
    encoded_features: List[str] = metadata["encoded_feature_names"]
    target_transform = metadata.get("final_target_transform", {})
    target_bounds = metadata.get("target_outlier_bounds", {})

    lines = [
        "Numerical feature engineering summary",
        "-------------------------------------",
        f"Target variable: {metadata['target_column']}",
        f"Numerical input columns: {', '.join(numeric_features) if numeric_features else 'None'}",
        f"Categorical input columns: {', '.join(categorical_features) if categorical_features else 'None'}",
        "Missing values were handled using median imputation for numerical columns and most-frequent imputation for categorical columns.",
        "Numerical columns were capped with an IQR-based clipper before scaling to reduce the influence of extreme values.",
        "Categorical variables were converted using one-hot encoding.",
        "Numerical variables were standardized using StandardScaler.",
        f"Target outlier bounds learned from the training split: {target_bounds}",
        f"Final target log1p transform used: {target_transform.get('used_log1p', False)}",
        f"Target transform decision: {target_transform.get('reason', 'Not available')}",
        f"Final number of extracted/encoded features: {len(encoded_features)}",
        f"Final feature names: {', '.join(encoded_features)}",
    ]
    return "\n".join(lines)
