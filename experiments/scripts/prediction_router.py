import joblib
from pathlib import Path

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

    frame = {key: [value] for key, value in input_row.items()}
    processed = preprocessor.transform(frame)
    return float(model.predict(processed)[0])
