import sys
from pathlib import Path
import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.common import (
    ensure_directories,
    load_master_dataset,
    normalize_home_type,
    build_feature_frame,
    load_current_project_model_and_preprocessor,
)


def main() -> None:
    ensure_directories()
    df = load_master_dataset()
    home_types = ["Condo", "Detached", "Semi Detached", "House", "Townhouse", "Duplex", "Triplex", "Fourplex"]

    current_model, current_preprocessor = load_current_project_model_and_preprocessor()
    report_rows = []

    for home_type in home_types:
        normalized = normalize_home_type(home_type)
        mask = df["home_type"].fillna("").astype(str).str.upper().str.replace("-", "_").str.replace(" ", "_") == normalized
        sample = df.loc[mask].head(1)
        if sample.empty:
            continue

        feature_frame = build_feature_frame(sample)
        current_features = current_preprocessor.transform(feature_frame.drop(columns=["price"]))
        current_prediction = float(current_model.predict(current_features)[0])

        group = "condo" if normalized == "CONDO" else "lowdensity" if normalized in {"DETACHED", "SEMI_DETACHED", "HOUSE"} else "multiunit"
        model_path = Path(__file__).resolve().parents[1] / "models" / ({"condo": "Condo_Model.pkl", "lowdensity": "LowDensity_Model.pkl", "multiunit": "MultiUnit_Model.pkl"}[group])
        preprocessor_path = Path(__file__).resolve().parents[1] / "preprocessing" / ({"condo": "condo_preprocessor.pkl", "lowdensity": "lowdensity_preprocessor.pkl", "multiunit": "multiunit_preprocessor.pkl"}[group])
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        specialized_features = preprocessor.transform(feature_frame.drop(columns=["price"]))
        specialized_prediction = float(model.predict(specialized_features)[0])

        report_rows.append({
            "home_type": home_type,
            "actual_price": float(sample.iloc[0]["price"]),
            "current_model_prediction": current_prediction,
            "specialized_model_prediction": specialized_prediction,
            "absolute_difference": abs(current_prediction - specialized_prediction),
            "selected_model": "specialized" if abs(current_prediction - specialized_prediction) > 0 else "current",
        })

    output_path = Path(__file__).resolve().parents[1] / "outputs" / "test_predictions_report.csv"
    pd.DataFrame(report_rows).to_csv(output_path, index=False)
    print(f"Dataset Size     : {len(df)}")
    print(f"Testing Samples  : {len(report_rows)}")
    print(f"Report Saved     : {output_path}")
    print(pd.DataFrame(report_rows))


if __name__ == "__main__":
    main()
