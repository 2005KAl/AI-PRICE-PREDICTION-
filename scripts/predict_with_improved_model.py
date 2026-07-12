"""
predict_with_improved_model.py
------------------------------------------------------------
DROP-IN REPLACEMENT for the `predict_price()` function in app.py.

This reuses your EXISTING PREPROCESSOR (the same ColumnTransformer
your Flask app already loads) to turn a raw feature_row into the
original 155-column matrix - then converts that into the 21-column
matrix the improved model expects (merged home types + neighbourhood
target encoding), and returns a price prediction.

HOW TO USE IN app.py:
  1. Copy this file into your project's `scripts/` (or root) folder.
  2. In app.py, replace the existing `predict_price()` function with
     the `predict_price_improved()` function below (or simply import
     and call it instead).
  3. Make sure these 2 files are present in your models/ folder:
        - xgboost_final_model.pkl
        - neighbourhood_price_map.pkl
     (both already loaded by MODELS_DIR below - update the path if needed)
------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"  # adjust if needed

# ---- Load improved-model artifacts (load ONCE at app startup, like your other models) ----
IMPROVED_MODEL = joblib.load(MODELS_DIR / "xgboost_final_model.pkl")
NEIGHBOURHOOD_PRICE_MAP = joblib.load(MODELS_DIR / "neighbourhood_price_map.pkl")
GLOBAL_MEAN_LOG_PRICE = float(np.mean(list(NEIGHBOURHOOD_PRICE_MAP.values())))

FINAL_FEATURE_ORDER = [
    'num__latitude', 'num__longitude', 'num__bedrooms', 'num__bedrooms_plus',
    'num__bathrooms', 'num__estimated_area_sqft', 'num__Distance_to_School',
    'num__Distance_to_Hospital', 'num__Distance_to_Park', 'num__Distance_to_Library',
    'num__Distance_to_Bank', 'num__Distance_to_Pharmacy', 'num__Distance_to_Grocery',
    'num__Distance_to_Subway', 'num__Restaurants_Within_1km', 'num__Parks_Within_2km',
    'num__Schools_Within_2km', 'home_type_CONDO', 'home_type_HOUSE',
    'home_type_DUPLEX_TRIPLEX_FOURPLEX', 'neighbourhood_avg_logprice'
]

# Merge rule - must match training exactly
HOME_TYPE_MERGE_MAP = {
    "CONDO": "CONDO",
    "HOUSE": "HOUSE",
    "DUPLEX_TRIPLEX_FOURPLEX": "DUPLEX_TRIPLEX_FOURPLEX",
    "TOWNHOUSE": "DUPLEX_TRIPLEX_FOURPLEX",
    "OTHER_RESIDENTIAL": "DUPLEX_TRIPLEX_FOURPLEX",
}


def build_improved_features(transformed_155col_df: pd.DataFrame, neighbourhood_name: str) -> pd.DataFrame:
    """
    Convert the existing 155-column PREPROCESSOR output into the
    21-column improved feature matrix.

    Parameters
    ----------
    transformed_155col_df : pd.DataFrame
        Output of PREPROCESSOR.transform(frame), already turned into a
        DataFrame with get_feature_names_out() columns (exactly what
        app.py's current predict_price() already produces).
    neighbourhood_name : str
        The raw neighbourhood string looked up via build_neighbourhood_lookup()
        (already computed by app.py for the amenity features).
    """
    row = transformed_155col_df.iloc[[0]].copy()

    # Recover which home_type was set (one of the 5 original one-hot columns)
    hometype_cols = [
        "cat__home_type_CONDO", "cat__home_type_DUPLEX_TRIPLEX_FOURPLEX",
        "cat__home_type_HOUSE", "cat__home_type_OTHER_RESIDENTIAL", "cat__home_type_TOWNHOUSE"
    ]
    raw_home_type = row[hometype_cols].idxmax(axis=1).iloc[0].replace("cat__home_type_", "")
    merged_home_type = HOME_TYPE_MERGE_MAP.get(raw_home_type, "HOUSE")

    # Build final row
    final_row = {}
    numeric_cols = [c for c in FINAL_FEATURE_ORDER if c.startswith("num__")]
    for col in numeric_cols:
        final_row[col] = row[col].iloc[0] if col in row.columns else 0.0

    final_row["home_type_CONDO"] = 1 if merged_home_type == "CONDO" else 0
    final_row["home_type_HOUSE"] = 1 if merged_home_type == "HOUSE" else 0
    final_row["home_type_DUPLEX_TRIPLEX_FOURPLEX"] = 1 if merged_home_type == "DUPLEX_TRIPLEX_FOURPLEX" else 0

    final_row["neighbourhood_avg_logprice"] = NEIGHBOURHOOD_PRICE_MAP.get(
        neighbourhood_name, GLOBAL_MEAN_LOG_PRICE
    )

    final_df = pd.DataFrame([final_row])[FINAL_FEATURE_ORDER]
    return final_df


def predict_price_improved(feature_row: dict, preprocessor, make_prediction_payload_fn) -> float:
    """
    Full replacement for app.py's predict_price().

    Parameters
    ----------
    feature_row : dict
        Same feature_row dict your app already builds (from
        compute_amenity_features + user inputs).
    preprocessor : the existing loaded PREPROCESSOR object from app.py
    make_prediction_payload_fn : the existing make_prediction_payload()
        function from app.py (builds the raw frame before transform)
    """
    frame = make_prediction_payload_fn(feature_row)
    transformed = preprocessor.transform(frame)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed_df = pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=frame.index,
    )

    neighbourhood_name = feature_row.get("Neighbourhood") or feature_row.get("neighbourhood")
    final_features = build_improved_features(transformed_df, neighbourhood_name)

    pred_log = IMPROVED_MODEL.predict(final_features)[0]
    pred_log = float(np.clip(pred_log, 10.0, 18.0))  # safety guard, same as training
    prediction = float(np.expm1(pred_log))
    return prediction
