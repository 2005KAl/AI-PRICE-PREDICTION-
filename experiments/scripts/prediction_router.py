import joblib
from pathlib import Path

import pandas as pd
import numpy as np

from scripts.common import normalize_home_type


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MAP = {
    "condo": EXPERIMENT_ROOT / "models" / "Condo_Model.pkl",
    "lowdensity": EXPERIMENT_ROOT / "models" / "LowDensity_Model.pkl",
    "multiunit": EXPERIMENT_ROOT / "models" / "MultiUnit_Model.pkl",
}
PREPROCESSOR_MAP = {
    "condo": EXPERIMENT_ROOT / "preprocessing" / "condo_preprocessor.pkl",
    "lowdensity": EXPERIMENT_ROOT / "preprocessing" / "lowdensity_preprocessor.pkl",
    "multiunit": EXPERIMENT_ROOT / "preprocessing" / "multiunit_preprocessor.pkl",
}


def route_prediction(input_row: dict) -> float:
    home_type = normalize_home_type(input_row.get("home_type", ""))
    if home_type == "CONDO":
        group = "condo"
    elif home_type in {"DETACHED", "SEMI_DETACHED", "HOUSE"}:
        group = "lowdensity"
    elif home_type in {"TOWNHOUSE", "DUPLEX", "TRIPLEX", "FOURPLEX", "DUPLEX_TRIPLEX_FOURPLEX"}:
        group = "multiunit"
    else:
        raise ValueError(f"Unsupported home_type for experiment router: {home_type}")

    model = joblib.load(MODEL_MAP[group])
    preprocessor = joblib.load(PREPROCESSOR_MAP[group])

    # Ensure we pass a proper 2D table (DataFrame) to the preprocessor
    df = pd.DataFrame([input_row])
    # Align columns to what the preprocessor was fitted on so missing keys become NaN
    if hasattr(preprocessor, "feature_names_in_"):
        expected = list(preprocessor.feature_names_in_)
        df = df.reindex(columns=expected)

    # Some UI fields may be submitted as single-element lists (e.g. ['Downtown']).
    # Convert list/tuple cells to their first element so encoders receive scalars.
    df = df.applymap(lambda v: (v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else (np.nan if isinstance(v, (list, tuple)) and len(v) == 0 else v)))

    processed = preprocessor.transform(df)
    return float(model.predict(processed)[0])
