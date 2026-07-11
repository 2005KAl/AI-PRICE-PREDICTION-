import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENTS_DIR.parent
DATA_DIR = EXPERIMENTS_DIR / "data"
PREPROCESS_DIR = EXPERIMENTS_DIR / "preprocessing"
MODEL_DIR = EXPERIMENTS_DIR / "models"
OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
FEATURE_IMPORTANCE_DIR = OUTPUT_DIR / "feature_importance"
SHAP_DIR = OUTPUT_DIR / "shap"
ERROR_ANALYSIS_DIR = OUTPUT_DIR / "error_analysis"
MASTER_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
CURRENT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
CURRENT_PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"


def ensure_directories() -> None:
    for path in [DATA_DIR, PREPROCESS_DIR, MODEL_DIR, OUTPUT_DIR, FEATURE_IMPORTANCE_DIR, SHAP_DIR, ERROR_ANALYSIS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_home_type(value: object) -> str:
    if pd.isna(value):
        return ""
    cleaned = str(value).strip().upper()
    cleaned = cleaned.replace("-", "_").replace(" ", "_")
    return cleaned


def get_group_mask(df: pd.DataFrame, group: str) -> pd.Series:
    normalized = df["home_type"].fillna("").astype(str).apply(normalize_home_type)
    if group == "condo":
        return normalized.isin(["CONDO"])
    if group == "lowdensity":
        return normalized.isin(["HOUSE", "DETACHED", "SEMI_DETACHED", "SEMI_DETACHED_HOUSE"])
    if group == "multiunit":
        return normalized.isin(["TOWNHOUSE", "DUPLEX_TRIPLEX_FOURPLEX", "DUPLEX", "TRIPLEX", "FOURPLEX"])
    raise ValueError(f"Unsupported group: {group}")


def load_master_dataset() -> pd.DataFrame:
    ensure_directories()
    df = pd.read_csv(MASTER_DATASET_PATH)
    return df.copy()


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    columns_to_drop = [
        "list_price",
        "sold_price",
        "neighbourhood",
        "_id",
        "AREA_ID",
        "AREA_ATTR_ID",
        "PARENT_AREA_ID",
        "AREA_SHORT_CODE",
        "AREA_LONG_CODE",
        "AREA_NAME",
        "AREA_DESC",
        "CLASSIFICATION",
        "CLASSIFICATION_CODE",
        "OBJECTID",
    ]
    work_df = work_df.drop(columns=[c for c in columns_to_drop if c in work_df.columns], errors="ignore")
    return work_df


def build_preprocessor(X: pd.DataFrame):
    categorical_features = [
        c for c in X.columns
        if X[c].dtype == "object" or pd.api.types.is_categorical_dtype(X[c])
    ]
    numerical_features = [
        c for c in X.columns
        if c not in categorical_features and pd.api.types.is_numeric_dtype(X[c])
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def split_dataset(df: pd.DataFrame, group: str) -> pd.DataFrame:
    filtered = df.loc[get_group_mask(df, group)].copy()
    if filtered.empty:
        raise ValueError(f"No rows available for group: {group}")
    return filtered


def save_dataset(df: pd.DataFrame, filename: str) -> None:
    output_path = DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved dataset -> {output_path}")


def save_preprocessor(preprocessor, filename: str) -> None:
    output_path = PREPROCESS_DIR / filename
    joblib.dump(preprocessor, output_path)
    print(f"Preprocessor saved -> {output_path}")


def load_preprocessor(path: Path):
    return joblib.load(path)


def create_processed_features(df: pd.DataFrame, preprocessor, train: bool = True):
    features = build_feature_frame(df)
    if train:
        return preprocessor.fit_transform(features.drop(columns=["price"]))
    return preprocessor.transform(features.drop(columns=["price"]))


def train_and_evaluate_model(name: str, dataset_path: Path, preprocessor_path: Path, model_path: Path, metrics_path: Path, output_dir: Path) -> dict:
    ensure_directories()
    df = pd.read_csv(dataset_path)
    feature_frame = build_feature_frame(df)

    X = feature_frame.drop(columns=["price"])
    y = feature_frame["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    joblib.dump(preprocessor, preprocessor_path)

    param_distributions = {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "n_estimators": [200, 400],
    }

    regressor = XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=-1)
    search = RandomizedSearchCV(
        regressor,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    start_train = time.time()
    search.fit(X_train_processed, y_train)
    end_train = time.time()

    model = search.best_estimator_
    start_pred = time.time()
    predictions = model.predict(X_test_processed)
    end_pred = time.time()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    metrics = {
        "Model": name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "MAPE (%)": round(float(mape), 4),
        "Training Time (s)": round(float(end_train - start_train), 4),
        "Prediction Time (s)": round(float(end_pred - start_pred), 4),
        "Training Samples": int(len(X_train)),
        "Testing Samples": int(len(X_test)),
        "Feature Count": int(X_train_processed.shape[1]),
        "Best Parameters": str(search.best_params_),
    }

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    joblib.dump(model, model_path)

    print(f"Dataset Size     : {len(feature_frame)}")
    print(f"Number of Features: {X_train_processed.shape[1]}")
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")
    print(f"Training Time    : {metrics['Training Time (s)']}s")
    print("Evaluation Metrics")
    print(metrics)
    print(f"Model Saved      : {model_path}")
    print(f"Output Location  : {metrics_path}")

    feature_names = preprocessor.get_feature_names_out()
    save_feature_importance(model, feature_names, output_dir / "feature_importance" / f"{name.lower()}_feature_importance.csv", output_dir / "feature_importance" / f"{name.lower()}_feature_importance.png")
    save_shap_plot(model, X_train_processed, X_test_processed, output_dir / "shap" / f"{name.lower()}_shap.png")
    save_error_analysis(y_test, predictions, X_test, output_dir / "error_analysis" / f"{name.lower()}_error_analysis.png")

    return metrics


def save_feature_importance(model, feature_names, csv_path: Path, plot_path: Path) -> None:
    importances = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importances = importances.sort_values(by="importance", ascending=False)
    importances.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importances.head(20)["feature"][::-1], importances.head(20)["importance"][::-1])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def save_shap_plot(model, X_train_processed, X_test_processed, plot_path: Path) -> None:
    try:
        explainer = shap.Explainer(model.predict, X_train_processed[:100])
        shap_values = explainer(X_test_processed[:100])
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_processed[:100], plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
    except Exception as exc:
        print(f"SHAP plot generation skipped for {plot_path.name}: {exc}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"SHAP unavailable for this model\n{exc}", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()


def save_error_analysis(y_true, predictions, X_test: pd.DataFrame, plot_path: Path) -> None:
    errors = y_true - predictions
    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(3, 2, 1)
    ax1.hist(errors, bins=30, color="steelblue", edgecolor="black")
    ax1.set_title("Prediction Error Distribution")
    ax1.set_xlabel("Error")
    ax1.set_ylabel("Frequency")

    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(predictions, errors, alpha=0.6)
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_title("Residual Plot")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")

    ax3 = plt.subplot(3, 2, 3)
    ax3.scatter(y_true, predictions, alpha=0.6)
    ax3.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--")
    ax3.set_title("Actual vs Predicted")
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")

    area_column = "Neighbourhood" if "Neighbourhood" in X_test.columns else X_test.columns[0]
    if area_column in X_test.columns:
        area_errors = pd.DataFrame({"area": X_test[area_column], "error": errors})
        area_summary = area_errors.groupby("area")["error"].mean().sort_values(ascending=False).head(10)
        ax4 = plt.subplot(3, 2, 4)
        area_summary.plot(kind="bar", ax=ax4, color="darkorange")
        ax4.set_title("Area-wise Error")
        ax4.set_ylabel("Average Error")
        ax4.tick_params(axis="x", rotation=45)

        neighbourhood_summary = area_errors.groupby("area")["error"].mean().sort_values(ascending=False).head(10)
        ax5 = plt.subplot(3, 2, 5)
        neighbourhood_summary.plot(kind="bar", ax=ax5, color="seagreen")
        ax5.set_title("Neighbourhood-wise Error")
        ax5.set_ylabel("Average Error")
        ax5.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def load_current_project_model_and_preprocessor():
    model = joblib.load(CURRENT_MODEL_PATH)
    preprocessor = joblib.load(CURRENT_PREPROCESSOR_PATH)
    return model, preprocessor


def generate_prediction_row(df: pd.DataFrame, home_type: str) -> pd.DataFrame:
    normalized = df["home_type"].fillna("").astype(str).apply(normalize_home_type)
    mask = normalized == normalize_home_type(home_type)
    if not mask.any():
        raise ValueError(f"No rows found for home type: {home_type}")
    return df.loc[mask].head(1).copy()
