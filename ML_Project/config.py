from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


PROJECT_CONFIG = {
    "random_state": 42,
    "datasets": {
        "numerical": {
            "dataset_name": "Cyber Security Salaries Dataset",
            "source": "User-provided CSV file placed in datasets/numerical/",
            "data_dir": PROJECT_ROOT / "datasets" / "numerical",
            "file_name": "salaries_cyber.csv",
            "target_column": "salary_in_usd",
            "drop_columns": ["salary", "salary_currency"],
        },
        "image": {
            "dataset_name": "Flowers Recognition Image Dataset",
            "source": "User-provided image folders placed in datasets/images/",
            "data_dir": PROJECT_ROOT / "datasets" / "images",
            "max_classes": 5,
            "selected_classes": [],
            "image_size": (128, 128),
            "supported_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
        },
    },
    "splits": {
        "train_size": 0.70,
        "validation_size": 0.15,
        "test_size": 0.15,
    },
    "numerical_models": {
        "target_handling": {
            "log_skew_threshold": 1.0,
            "allow_log_transform": True,
            "target_outlier_method": "iqr_capping",
            "target_outlier_multiplier": 1.5,
        },
        "preprocessing": {
            "numeric_imputation_strategy": "median",
            "categorical_imputation_strategy": "most_frequent",
            "feature_outlier_method": "iqr_capping",
            "feature_outlier_multiplier": 1.5,
        },
        "linear_regression": {
            "fit_intercept": True,
            "positive": False,
            "cv_folds": 5,
            "learning_rate": "Not applicable: closed-form ordinary least squares",
            "optimizer": "Not applicable: scikit-learn LinearRegression uses least-squares optimization",
            "regularization": "None",
            "batch_size": "Not applicable",
            "epochs": "Not applicable",
        },
        "knn_regressor": {
            "candidate_k_values": [3, 5, 7, 9, 11, 15, 21],
            "candidate_weights": ["uniform", "distance"],
            "candidate_metrics": ["euclidean", "manhattan"],
            "cv_folds": 5,
            "learning_rate": "Not applicable: instance-based method",
            "optimizer": "Not applicable",
            "regularization": "Not applicable",
            "batch_size": "Not applicable",
            "epochs": "Not applicable",
        },
    },
    "image_models": {
        "logistic_regression": {
            "penalty": "l2",
            "solver": "lbfgs",
            "multi_class": "auto",
            "candidate_C_values": [0.01, 0.1, 1.0, 10.0],
            "candidate_max_iter": [1000, 2000, 3000],
            "candidate_class_weight": [None, "balanced"],
            "candidate_pca_components": [None, 120, 180],
            "use_light_augmentation": False,
            "augmentation_limit": 400,
            "learning_rate": "Controlled internally by lbfgs solver",
            "optimizer": "lbfgs",
            "regularization": "L2 regularization with tuned C value",
            "batch_size": "Not applicable: full-batch deterministic solver",
            "epochs": "Not applicable: max_iter controls solver iterations",
        },
        "kmeans": {
            "candidate_pca_components": [None, 20, 40, 80],
            "candidate_n_init": [20, 30],
            "init": "k-means++",
            "n_init": 30,
            "max_iter": 400,
            "algorithm": "lloyd",
            "learning_rate": "Not applicable: centroid update algorithm",
            "optimizer": "KMeans iterative centroid minimization",
            "regularization": "None",
            "batch_size": "Not applicable",
            "epochs": "Not applicable: max_iter=300 controls iterations",
        },
    },
    "outputs": {
        "figures_dir": PROJECT_ROOT / "outputs" / "figures",
        "confusion_matrices_dir": PROJECT_ROOT / "outputs" / "confusion_matrices",
        "reports_dir": PROJECT_ROOT / "outputs" / "reports",
        "models_dir": PROJECT_ROOT / "outputs" / "trained_models",
        "docs_dir": PROJECT_ROOT / "docs",
    },
}
