import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

X_test = pd.read_csv(
    os.path.join(DATA_DIR, "X_test.csv")
)

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
).squeeze()

print(f"Test Samples : {len(X_test)}")
print(f"Features     : {X_test.shape[1]}")

# ==========================================================
# LOAD BEST MODEL
# ==========================================================

print("\nLoading Best Model...")

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )
)

print("✓ Best Model Loaded")

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(X_test)

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

mape = np.mean(

    np.abs(

        (y_test - predictions)

        / y_test

    )

) * 100

# ==========================================================
# METRICS DATAFRAME
# ==========================================================

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

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(metrics)

# ==========================================================
# SAVE METRICS
# ==========================================================

metrics.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "evaluation_metrics.csv"

    ),

    index=False

)

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

prediction_df = pd.DataFrame({

    "Actual Price": y_test,

    "Predicted Price": predictions

})

prediction_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "predictions.csv"

    ),

    index=False

)

print("\nMetrics Saved")
print("Predictions Saved")

print("\n")
print("=" * 60)
print("PART 1 COMPLETED")
print("=" * 60)
# ==========================================================
# PART 2 : VISUALIZATIONS
# ==========================================================

print("\n")
print("=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# ----------------------------------------------------------
# Actual vs Predicted
# ----------------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()

print("✓ actual_vs_predicted.png")

# ----------------------------------------------------------
# Residual Plot
# ----------------------------------------------------------

residuals = y_test - predictions

plt.figure(figsize=(8,6))

plt.scatter(
    predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel("Predicted Price")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "residual_plot.png"
    ),
    dpi=300
)

plt.close()

print("✓ residual_plot.png")

print("\n")
print("=" * 60)
print("PART 2 COMPLETED")
print("=" * 60)

# ==========================================================
# PART 3 : FEATURE IMPORTANCE
# ==========================================================

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

# ----------------------------------------------------------
# LOAD FEATURE NAMES
# ----------------------------------------------------------

feature_names = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_names.pkl"
    )
)

# ----------------------------------------------------------
# EXTRACT IMPORTANCE
# ----------------------------------------------------------

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Save CSV

    importance_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),

        index=False

    )

    print("\nTop 20 Important Features\n")

    print(importance_df.head(20))

    # ------------------------------------------------------
    # Plot Top 20
    # ------------------------------------------------------

    plt.figure(figsize=(10,8))

    top20 = importance_df.head(20)

    plt.barh(

        top20["Feature"][::-1],

        top20["Importance"][::-1]

    )

    plt.xlabel("Importance")

    plt.title("Top 20 Feature Importance (XGBoost)")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "feature_importance.png"

        ),

        dpi=300

    )

    plt.close()

    print("\n✓ feature_importance.csv")

    print("✓ feature_importance.png")

else:

    print("Current model does not support feature importance.")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("Saved Files:")

print("✓ evaluation_metrics.csv")
print("✓ predictions.csv")
print("✓ actual_vs_predicted.png")
print("✓ residual_plot.png")
print("✓ feature_importance.csv")
print("✓ feature_importance.png")