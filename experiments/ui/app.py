from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd
import math

# Local experiment datasets
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DATA = ROOT / "experiments" / "data"

CONDODF = pd.read_csv(EXPERIMENT_DATA / "condo_dataset.csv")
LOWDF = pd.read_csv(EXPERIMENT_DATA / "lowdensity_dataset.csv")
MULTIDF = pd.read_csv(EXPERIMENT_DATA / "multiunit_dataset.csv")

# Ensure the experiments scripts package is importable
import sys
scripts_parent = str(ROOT / "experiments")
if scripts_parent not in sys.path:
    sys.path.insert(0, scripts_parent)

from scripts.prediction_router import route_prediction


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def index():
        return render_template("index.html", current_date=pd.Timestamp.now().strftime("%Y-%m-%d"), model_name="Experimental Multi-Model")

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json() or {}
        # Expecting keys: home_type, latitude, longitude, bedrooms, bathrooms, estimated_area_sqft
        home_type = payload.get("home_type") or payload.get("HomeType")
        try:
            lat = float(payload.get("latitude") or payload.get("Latitude"))
            lon = float(payload.get("longitude") or payload.get("Longitude"))
        except Exception:
            return jsonify({"error": "latitude and longitude required"}), 400

        htype_norm = str(home_type).strip().upper().replace(' ', '_').replace('-', '_')

        if htype_norm == "CONDO":
            df = CONDODF
        elif htype_norm in {"DETACHED", "SEMI_DETACHED", "HOUSE"}:
            df = LOWDF
        else:
            df = MULTIDF

        # find nearest row by lat/lon
        def dist(row):
            return (row.get('latitude', 0) - lat) ** 2 + (row.get('longitude', 0) - lon) ** 2

        # ensure numeric
        df2 = df.copy()
        df2['latitude'] = pd.to_numeric(df2['latitude'], errors='coerce')
        df2['longitude'] = pd.to_numeric(df2['longitude'], errors='coerce')
        df2 = df2.dropna(subset=['latitude','longitude'])
        df2['d'] = df2.apply(lambda r: (r.latitude - lat) ** 2 + (r.longitude - lon) ** 2, axis=1)
        nearest = df2.sort_values('d').head(1)
        if nearest.empty:
            return jsonify({"error": "No sample row available for this home type"}), 404

        sample_row = nearest.iloc[0].to_dict()

        # Use the existing prediction router for isolated models
        try:
            predicted = route_prediction(sample_row)
        except Exception as exc:
            return jsonify({"error": f"Prediction router failed: {exc}"}), 500

        response = {
            "predicted_price": predicted,
            "predicted_price_formatted": f"CAD ${int(predicted):,}",
            "neighbourhood": sample_row.get('Neighbourhood') or sample_row.get('neighbourhood') or 'Unknown',
            "home_type": sample_row.get('home_type'),
            "prediction_timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "nearby_points": {},
            "distance_school": sample_row.get('Distance_to_School', 0) / 1000 if sample_row.get('Distance_to_School') is not None else 0,
            "distance_hospital": sample_row.get('Distance_to_Hospital', 0) / 1000 if sample_row.get('Distance_to_Hospital') is not None else 0,
            "distance_park": sample_row.get('Distance_to_Park', 0) / 1000 if sample_row.get('Distance_to_Park') is not None else 0,
            "distance_library": sample_row.get('Distance_to_Library', 0) / 1000 if sample_row.get('Distance_to_Library') is not None else 0,
            "distance_bank": sample_row.get('Distance_to_Bank', 0) / 1000 if sample_row.get('Distance_to_Bank') is not None else 0,
            "distance_pharmacy": sample_row.get('Distance_to_Pharmacy', 0) / 1000 if sample_row.get('Distance_to_Pharmacy') is not None else 0,
            "distance_grocery": sample_row.get('Distance_to_Grocery', 0) / 1000 if sample_row.get('Distance_to_Grocery') is not None else 0,
            "distance_subway": sample_row.get('Distance_to_Subway', 0) / 1000 if sample_row.get('Distance_to_Subway') is not None else 0,
            "restaurants": int(sample_row.get('Restaurants_Within_1km', 0) or 0),
            "parks": int(sample_row.get('Parks_Within_2km', 0) or 0),
            "schools": int(sample_row.get('Schools_Within_2km', 0) or 0),
        }
        return jsonify(response)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(port=5001, debug=True)
