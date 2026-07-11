import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

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
print("LOADING DATA")
print("=" * 60)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

y_train = pd.read_csv(
    os.path.join(DATA_DIR, "y_train.csv")
)["price"]

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
)["price"]

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ==========================================================
# PARAMETER GRID
# ==========================================================

param_grid = {

    "n_estimators": [200, 300, 500, 700],

    "max_depth": [10, 20, 30, 40, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "max_features": ["sqrt", "log2", None],

    "bootstrap": [True, False]
}

# ==========================================================
# RANDOM FOREST
# ==========================================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# ==========================================================
# RANDOM SEARCH
# ==========================================================

print("\n")
print("=" * 60)
print("STARTING HYPERPARAMETER TUNING")
print("=" * 60)

search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_grid,

    n_iter=30,

    cv=5,

    scoring="neg_root_mean_squared_error",

    verbose=2,

    random_state=42,

    n_jobs=-1

)

start = time.time()

search.fit(
    X_train,
    y_train
)

elapsed = time.time() - start

print("\nTraining Finished")

print(f"Time Taken : {elapsed:.2f} seconds")

# ==========================================================
# BEST MODEL
# ==========================================================

best_model = search.best_estimator_

print("\n")
print("=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(search.best_params_)

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = best_model.predict(
    X_test
)

# ==========================================================
# METRICS
# ==========================================================

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

mape = mean_absolute_percentage_error(
    y_test,
    predictions
) * 100

metrics = pd.DataFrame({

    "Metric": [

        "MAE",

        "RMSE",

        "R2",

        "MAPE (%)"

    ],

    "Value": [

        mae,

        rmse,

        r2,

        mape

    ]

})

metrics.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "tuned_model_metrics.csv"
    ),

    index=False

)

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(

    best_model,

    os.path.join(
        MODEL_DIR,
        "best_model_tuned.pkl"
    )

)

# ==========================================================
# SAVE PARAMETERS
# ==========================================================

params = pd.DataFrame(
    [search.best_params_]
)

params.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "best_parameters.csv"
    ),

    index=False

)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("TUNED MODEL RESULTS")
print("=" * 60)

print(metrics)

print("\n")

print("Best Parameters")

print(search.best_params_)

print("\n")

print("Saved Files")

print("✓ best_model_tuned.pkl")

print("✓ tuned_model_metrics.csv")

print("✓ best_parameters.csv")