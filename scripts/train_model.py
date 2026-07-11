import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

X_train = pd.read_csv(
    os.path.join(DATA_DIR, "X_train.csv")
)

X_test = pd.read_csv(
    os.path.join(DATA_DIR, "X_test.csv")
)

y_train = pd.read_csv(
    os.path.join(DATA_DIR, "y_train.csv")
).squeeze()

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
).squeeze()

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

print(f"Features         : {X_train.shape[1]}")

# ==========================================================
# MODELS
# ==========================================================

print("\nInitializing Models...\n")

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=20
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        random_state=42
    )

}

print("Models Ready")

for model in models:
    print(f"✓ {model}")

# ==========================================================
# RESULT STORAGE
# ==========================================================

results = []

trained_models = {}

print("\n")
print("=" * 60)
print("PART 1 COMPLETED")
print("=" * 60)

# ==========================================================
# PART 2 : TRAIN MODELS
# ==========================================================

print("\n")
print("=" * 60)
print("TRAINING MODELS")
print("=" * 60)

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # -------------------------
    # Training Time
    # -------------------------

    start_train = time.time()

    model.fit(X_train, y_train)

    end_train = time.time()

    train_time = end_train - start_train

    # -------------------------
    # Prediction Time
    # -------------------------

    start_pred = time.time()

    predictions = model.predict(X_test)

    end_pred = time.time()

    pred_time = end_pred - start_pred

    # -------------------------
    # Metrics
    # -------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mape = np.mean(

        np.abs(

            (y_test - predictions)

            / y_test

        )

    ) * 100

    # -------------------------
    # Store Results
    # -------------------------

    results.append({

        "Model": model_name,

        "MAE": round(mae, 2),

        "RMSE": round(rmse, 2),

        "R2": round(r2, 4),

        "MAPE (%)": round(mape, 2),

        "Training Time (s)": round(train_time, 3),

        "Prediction Time (s)": round(pred_time, 4)

    })

    trained_models[model_name] = model

    print(f"✓ Completed {model_name}")

    # ==========================================================
# PART 3 : MODEL COMPARISON & SAVE
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# ----------------------------------------------------------
# Results DataFrame
# ----------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

print(results_df)

# ----------------------------------------------------------
# Save Comparison
# ----------------------------------------------------------

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)

# ----------------------------------------------------------
# Save Individual Models
# ----------------------------------------------------------

filename_mapping = {

    "Linear Regression": "linear_regression.pkl",

    "Decision Tree": "decision_tree.pkl",

    "Random Forest": "random_forest.pkl",

    "XGBoost": "xgboost.pkl",

    "LightGBM": "lightgbm.pkl"

}

for model_name, model in trained_models.items():

    joblib.dump(

        model,

        os.path.join(

            MODEL_DIR,

            filename_mapping[model_name]

        )

    )

# ----------------------------------------------------------
# Save Best Model
# ----------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

joblib.dump(

    best_model,

    os.path.join(

        MODEL_DIR,

        "best_model.pkl"

    )

)

print("\n")

print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(best_model_name)

print(f"RMSE : {results_df.iloc[0]['RMSE']}")

print("\nAll models saved successfully.")

print("\n")

print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

df = pd.read_csv("ml_dataset.csv")

subset = df[
    (df["estimated_area_sqft"] >= 600) &
    (df["estimated_area_sqft"] <= 700) &
    (df["Neighbourhood"] == "Yonge-Bay Corridor") &
    (df["home_type"] == "Condo")
]

print(subset[["estimated_area_sqft", "price"]])

print(subset["price"].describe())