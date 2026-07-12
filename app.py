from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from shapely.geometry import Point
from sklearn.neighbors import BallTree


# ==========================================================
# PATHS AND CONSTANTS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

EARTH_RADIUS_KM = 6371.0088
HOME_TYPE_BUCKET = {

    "Condo": "CONDO",

    "Detached": "HOUSE",

    "Semi Detached": "HOUSE",

    "Townhouse": "TOWNHOUSE",   # ✅ Fixed

    "House": "HOUSE",

    "Duplex": "DUPLEX_TRIPLEX_FOURPLEX",

    "Triplex": "DUPLEX_TRIPLEX_FOURPLEX",

    "Fourplex": "DUPLEX_TRIPLEX_FOURPLEX"

}
DEFAULT_LATITUDE = 43.6532
DEFAULT_LONGITUDE = -79.3832

FEATURE_COLUMNS = [

    # Property Details
    "home_type",
    "home_type_bucket",

    "latitude",
    "longitude",

    "bedrooms",
    "bedrooms_plus",
    "bathrooms",

    "estimated_area_sqft",

    # Distance Features
    "Distance_to_School",
    "Distance_to_Hospital",
    "Distance_to_Park",
    "Distance_to_Library",
    "Distance_to_Bank",
    "Distance_to_Pharmacy",
    "Distance_to_Grocery",
    "Distance_to_Subway",

    # Count Features
    "Restaurants_Within_1km",
    "Parks_Within_2km",
    "Schools_Within_2km",

    # GIS Feature
    "Neighbourhood"

]
MODEL_NAME = "XGBoost Regression"
MODEL_ACCURACY_R2 = 0.8542
MODEL_ACCURACY_PERCENT = f"{MODEL_ACCURACY_R2 * 100:.2f}%"

# ==========================================================
# DATA MODELS
# ==========================================================


@dataclass(slots=True)
class AmenityIndex:
    """In-memory amenity store with BallTree support."""

    label: str
    dataframe: pd.DataFrame
    tree: BallTree
    coordinate_radians: np.ndarray
    name_column: str

    def nearest(self, latitude: float, longitude: float) -> dict[str, Any]:
        point_radians = np.radians([[latitude, longitude]])
        distances, indices = self.tree.query(point_radians, k=1)
        row = self.dataframe.iloc[int(indices[0][0])]

        return {
            "name": str(row[self.name_column]) if self.name_column in row and pd.notna(row[self.name_column]) else self.label,
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
            "distance_km": float(distances[0][0] * EARTH_RADIUS_KM),
        }

    def nearest_points(self, latitude: float, longitude: float, limit: int = 5) -> list[dict[str, Any]]:
        point_radians = np.radians([[latitude, longitude]])
        k_value = min(limit, len(self.dataframe))
        distances, indices = self.tree.query(point_radians, k=k_value)

        points: list[dict[str, Any]] = []
        for raw_distance, raw_index in zip(distances[0], indices[0], strict=False):
            row = self.dataframe.iloc[int(raw_index)]
            points.append(
                {
                    "name": str(row[self.name_column]) if self.name_column in row and pd.notna(row[self.name_column]) else self.label,
                    "latitude": float(row["Latitude"]),
                    "longitude": float(row["Longitude"]),
                    "distance_km": float(raw_distance * EARTH_RADIUS_KM),
                }
            )

        return points

    def count_within(self, latitude: float, longitude: float, radius_km: float) -> int:
        point_radians = np.radians([[latitude, longitude]])
        radius_radians = radius_km / EARTH_RADIUS_KM
        matches = self.tree.query_radius(point_radians, r=radius_radians, count_only=False)[0]
        return int(len(matches))


# ==========================================================
# HELPERS
# ==========================================================


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find any of the expected files: {[str(candidate) for candidate in candidates]}")


def coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def load_csv_amenity(path: Path, name_column: str) -> AmenityIndex:
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["Latitude", "Longitude"]).copy()
    frame["Latitude"] = pd.to_numeric(frame["Latitude"], errors="coerce")
    frame["Longitude"] = pd.to_numeric(frame["Longitude"], errors="coerce")
    frame = frame.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    coordinate_radians = np.radians(frame[["Latitude", "Longitude"]].to_numpy(dtype=float))
    tree = BallTree(coordinate_radians, metric="haversine")

    return AmenityIndex(
        label=path.stem,
        dataframe=frame,
        tree=tree,
        coordinate_radians=coordinate_radians,
        name_column=name_column,
    )


def load_geojson_points(path: Path, name_column: str) -> AmenityIndex:
    frame = gpd.read_file(path)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    else:
        frame = frame.to_crs("EPSG:4326")

    frame = frame.copy()
    frame["Latitude"] = frame.geometry.y
    frame["Longitude"] = frame.geometry.x
    frame = frame.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    coordinate_radians = np.radians(frame[["Latitude", "Longitude"]].to_numpy(dtype=float))
    tree = BallTree(coordinate_radians, metric="haversine")

    return AmenityIndex(
        label=path.stem,
        dataframe=frame,
        tree=tree,
        coordinate_radians=coordinate_radians,
        name_column=name_column,
    )


def load_neighbourhoods() -> gpd.GeoDataFrame:
    path = resolve_existing_path(
        DATA_DIR / "raw" / "Neighbourhoods - 4326.geojson",
        DATA_DIR / "raw" / "Neighbourhoods-4326.geojson",
    )

    frame = gpd.read_file(path)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    else:
        frame = frame.to_crs("EPSG:4326")

    return frame[["AREA_NAME", "geometry"]].copy()


def build_neighbourhood_lookup(latitude: float, longitude: float, neighbourhoods: gpd.GeoDataFrame) -> str:
    point = gpd.GeoDataFrame(
        {"Latitude": [latitude], "Longitude": [longitude]},
        geometry=[Point(longitude, latitude)],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(point, neighbourhoods, how="left", predicate="within")
    if not joined.empty:
        neighbourhood = joined.iloc[0].get("AREA_NAME")
        if pd.notna(neighbourhood):
            return str(neighbourhood)

    nearest = gpd.sjoin_nearest(point, neighbourhoods, how="left", distance_col="distance_to_neighbourhood")
    if not nearest.empty:
        neighbourhood = nearest.iloc[0].get("AREA_NAME")
        if pd.notna(neighbourhood):
            return str(neighbourhood)

    return "Unknown"


def format_currency(amount: float) -> str:
    return f"CAD ${amount:,.0f}"


def make_prediction_payload(feature_row: dict[str, Any]) -> pd.DataFrame:
    ordered_row = {column: feature_row.get(column) for column in FEATURE_COLUMNS}
    return pd.DataFrame([ordered_row], columns=FEATURE_COLUMNS)


FINAL_FEATURE_ORDER = [
    "num__latitude",
    "num__longitude",
    "num__bedrooms",
    "num__bedrooms_plus",
    "num__bathrooms",
    "num__estimated_area_sqft",
    "num__Distance_to_School",
    "num__Distance_to_Hospital",
    "num__Distance_to_Park",
    "num__Distance_to_Library",
    "num__Distance_to_Bank",
    "num__Distance_to_Pharmacy",
    "num__Distance_to_Grocery",
    "num__Distance_to_Subway",
    "num__Restaurants_Within_1km",
    "num__Parks_Within_2km",
    "num__Schools_Within_2km",
    "home_type_CONDO",
    "home_type_HOUSE",
    "home_type_DUPLEX_TRIPLEX_FOURPLEX",
    "neighbourhood_avg_logprice",
]

HOME_TYPE_MERGE_MAP = {
    "CONDO": "CONDO",
    "HOUSE": "HOUSE",
    "DUPLEX_TRIPLEX_FOURPLEX": "DUPLEX_TRIPLEX_FOURPLEX",
    "TOWNHOUSE": "DUPLEX_TRIPLEX_FOURPLEX",
    "OTHER_RESIDENTIAL": "DUPLEX_TRIPLEX_FOURPLEX",
}


def build_improved_features(transformed_155col_df: pd.DataFrame, neighbourhood_name: str) -> pd.DataFrame:
    row = transformed_155col_df.iloc[[0]].copy()

    hometype_cols = [
        "cat__home_type_CONDO",
        "cat__home_type_DUPLEX_TRIPLEX_FOURPLEX",
        "cat__home_type_HOUSE",
        "cat__home_type_OTHER_RESIDENTIAL",
        "cat__home_type_TOWNHOUSE",
    ]
    raw_home_type = row[hometype_cols].idxmax(axis=1).iloc[0].replace("cat__home_type_", "")
    merged_home_type = HOME_TYPE_MERGE_MAP.get(raw_home_type, "HOUSE")

    final_row: dict[str, Any] = {}
    numeric_cols = [column for column in FINAL_FEATURE_ORDER if column.startswith("num__")]
    for column in numeric_cols:
        final_row[column] = row[column].iloc[0] if column in row.columns else 0.0

    final_row["home_type_CONDO"] = 1 if merged_home_type == "CONDO" else 0
    final_row["home_type_HOUSE"] = 1 if merged_home_type == "HOUSE" else 0
    final_row["home_type_DUPLEX_TRIPLEX_FOURPLEX"] = 1 if merged_home_type == "DUPLEX_TRIPLEX_FOURPLEX" else 0

    final_row["neighbourhood_avg_logprice"] = NEIGHBOURHOOD_PRICE_MAP.get(
        neighbourhood_name,
        GLOBAL_MEAN_LOG_PRICE,
    )

    return pd.DataFrame([final_row])[FINAL_FEATURE_ORDER]


def compute_amenity_features(latitude: float, longitude: float) -> dict[str, Any]:
    nearest_features = {
        "Distance_to_School": AMENITY_INDEXES["schools"].nearest(latitude, longitude),
        "Distance_to_Hospital": AMENITY_INDEXES["hospitals"].nearest(latitude, longitude),
        "Distance_to_Park": AMENITY_INDEXES["parks"].nearest(latitude, longitude),
        "Distance_to_Library": AMENITY_INDEXES["libraries"].nearest(latitude, longitude),
        "Distance_to_Bank": AMENITY_INDEXES["banks"].nearest(latitude, longitude),
        "Distance_to_Pharmacy": AMENITY_INDEXES["pharmacies"].nearest(latitude, longitude),
        "Distance_to_Grocery": AMENITY_INDEXES["groceries"].nearest(latitude, longitude),
        "Distance_to_Subway": AMENITY_INDEXES["subway"].nearest(latitude, longitude),
    }

    counts = {
        "Restaurants_Within_1km": AMENITY_INDEXES["restaurants"].count_within(latitude, longitude, 1.0),
        "Schools_Within_2km": AMENITY_INDEXES["schools"].count_within(latitude, longitude, 2.0),
        "Parks_Within_2km": AMENITY_INDEXES["parks"].count_within(latitude, longitude, 2.0),
    }

    neighbourhood = build_neighbourhood_lookup(latitude, longitude, NEIGHBOURHOODS_GDF)

    feature_row = {
        "bedrooms": None,
        "bathrooms": None,
        "Latitude": latitude,
        "Longitude": longitude,
        "Distance_to_School": nearest_features["Distance_to_School"]["distance_km"],
        "Distance_to_Hospital": nearest_features["Distance_to_Hospital"]["distance_km"],
        "Distance_to_Park": nearest_features["Distance_to_Park"]["distance_km"],
        "Distance_to_Library": nearest_features["Distance_to_Library"]["distance_km"],
        "Distance_to_Bank": nearest_features["Distance_to_Bank"]["distance_km"],
        "Distance_to_Pharmacy": nearest_features["Distance_to_Pharmacy"]["distance_km"],
        "Distance_to_Grocery": nearest_features["Distance_to_Grocery"]["distance_km"],
        "Distance_to_Subway": nearest_features["Distance_to_Subway"]["distance_km"],
        "Restaurants_Within_1km": counts["Restaurants_Within_1km"],
        "Parks_Within_2km": counts["Parks_Within_2km"],
        "Schools_Within_2km": counts["Schools_Within_2km"],
        "Neighbourhood": neighbourhood,
    }

    return {
        "feature_row": feature_row,
        "nearest": nearest_features,
        "counts": counts,
        "neighbourhood": neighbourhood,
    }


def predict_price(feature_row: dict[str, Any]) -> float:
    frame = make_prediction_payload(feature_row)
    transformed = PREPROCESSOR.transform(frame)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    if hasattr(PREPROCESSOR, "get_feature_names_out"):
        transformed = pd.DataFrame(
            transformed,
            columns=PREPROCESSOR.get_feature_names_out(),
            index=frame.index,
        )

    try:
        neighbourhood_name = feature_row.get("Neighbourhood") or feature_row.get("neighbourhood")
        improved_features = build_improved_features(transformed, neighbourhood_name)
        prediction_log = IMPROVED_MODEL.predict(improved_features)[0]
        prediction_log = float(np.clip(prediction_log, 10.0, 18.0))
        prediction = float(np.expm1(prediction_log))
        return prediction
    except Exception as exc:
        print(f"Warning: improved model prediction failed ({exc}); falling back to legacy model.")
        prediction = MODEL.predict(transformed)[0]
        return float(prediction)


def validate_request_payload(payload: dict[str, Any]):

    bedrooms = int(coerce_float(payload.get("Bedrooms"), "Bedrooms"))

    bathrooms = int(coerce_float(payload.get("Bathrooms"), "Bathrooms"))

    estimated_area_sqft = float(

        coerce_float(

            payload.get("EstimatedArea"),

            "Estimated Area"

        )

    )

    home_type = str(

        payload.get("HomeType")

    ).strip()

    latitude = coerce_float(

        payload.get("Latitude"),

        "Latitude"

    )

    longitude = coerce_float(

        payload.get("Longitude"),

        "Longitude"

    )

    # ------------------------------------------
    # VALIDATION
    # ------------------------------------------

    if bedrooms < 0:
        raise ValueError("Bedrooms must be >= 0.")

    if bathrooms < 0:
        raise ValueError("Bathrooms must be >= 0.")

    if estimated_area_sqft <= 0:
        raise ValueError("Estimated Area must be greater than 0.")

    if home_type not in HOME_TYPE_BUCKET:
        raise ValueError("Invalid Home Type.")

    if not (-90 <= latitude <= 90):
        raise ValueError("Latitude must be between -90 and 90.")

    if not (-180 <= longitude <= 180):
        raise ValueError("Longitude must be between -180 and 180.")

    return (

        bedrooms,

        bathrooms,

        estimated_area_sqft,

        home_type,

        latitude,

        longitude

    )


def extract_payload() -> dict[str, Any]:
    json_payload = request.get_json(silent=True)
    if isinstance(json_payload, dict):
        return json_payload
    return request.form.to_dict()


# ==========================================================
# ARTIFACTS AND DATASETS
# ==========================================================


PREPROCESSOR = joblib.load(MODELS_DIR / "preprocessor.pkl")
MODEL = joblib.load(
    resolve_existing_path(
        MODELS_DIR / "best_model_tuned.pkl",
        MODELS_DIR / "best_model.pkl",
    )
)
IMPROVED_MODEL = joblib.load(MODELS_DIR / "xgboost_final_model.pkl")
NEIGHBOURHOOD_PRICE_MAP = joblib.load(MODELS_DIR / "neighbourhood_price_map.pkl")
GLOBAL_MEAN_LOG_PRICE = float(np.mean(list(NEIGHBOURHOOD_PRICE_MAP.values())))

ORIGINAL_FEATURES = joblib.load(MODELS_DIR / "original_features.pkl")

AMENITY_INDEXES = {
    "schools": load_csv_amenity(DATA_DIR / "cleaned data" / "schools_clean.csv", "NAME"),
    "parks": load_csv_amenity(DATA_DIR / "cleaned data" / "parks_clean.csv", "ASSET_NAME"),
    "libraries": load_csv_amenity(DATA_DIR / "cleaned data" / "libraries_clean.csv", "Library_Name"),
    "hospitals": load_csv_amenity(DATA_DIR / "cleaned data" / "hospitals_clean.csv", "Name"),
    "banks": load_csv_amenity(DATA_DIR / "cleaned data" / "banks_clean.csv", "Name"),
    "pharmacies": load_csv_amenity(DATA_DIR / "cleaned data" / "pharmacies_clean.csv", "Name"),
    "groceries": load_csv_amenity(DATA_DIR / "cleaned data" / "grocery_clean.csv", "Name"),
    "restaurants": load_csv_amenity(DATA_DIR / "cleaned data" / "restaurants_clean.csv", "Name"),
    "subway": load_geojson_points(
        resolve_existing_path(
            DATA_DIR / "raw" / "subway stations.geojson",
            DATA_DIR / "raw" / "subway stations .geojson",
            DATA_DIR / "raw" / "subway stations-geojson.geojson",
        ),
        "name",
    ),
}

NEIGHBOURHOODS_GDF = load_neighbourhoods()


# ==========================================================
# FLASK APP
# ==========================================================


app = Flask(__name__)


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        current_date=datetime.now().strftime("%B %d, %Y"),
        model_name=MODEL_NAME,
        model_accuracy_percent=MODEL_ACCURACY_PERCENT,
        original_features=ORIGINAL_FEATURES,
    )


@app.post("/predict")
def predict() -> tuple[Any, int]:
    try:
        payload = extract_payload()
        (
            bedrooms,
            bathrooms,
            estimated_area_sqft,
            home_type,
            latitude,
            longitude,
        ) = validate_request_payload(payload)

        amenity_data = compute_amenity_features(latitude, longitude)
        feature_row = amenity_data["feature_row"]

        # Property features
        feature_row["home_type"] = home_type
        feature_row["home_type_bucket"] = HOME_TYPE_BUCKET[home_type]
        feature_row["latitude"] = latitude
        feature_row["longitude"] = longitude
        feature_row["bedrooms"] = bedrooms
        feature_row["bedrooms_plus"] = 0
        feature_row["bathrooms"] = bathrooms
        feature_row["estimated_area_sqft"] = estimated_area_sqft

        predicted_price = predict_price(feature_row)
        price_per_sqft = predicted_price / estimated_area_sqft
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

        response = {
            "predicted_price": round(predicted_price, 2),
            "predicted_price_formatted": format_currency(predicted_price),
            "price_per_sqft": round(price_per_sqft, 2),
            "estimated_area_sqft": estimated_area_sqft,
            "home_type": home_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "prediction_accuracy": MODEL_ACCURACY_PERCENT,
            "model_name": MODEL_NAME,
            "neighbourhood": amenity_data["neighbourhood"],
            "distance_school": round(amenity_data["nearest"]["Distance_to_School"]["distance_km"], 3),
            "distance_hospital": round(amenity_data["nearest"]["Distance_to_Hospital"]["distance_km"], 3),
            "distance_park": round(amenity_data["nearest"]["Distance_to_Park"]["distance_km"], 3),
            "distance_library": round(amenity_data["nearest"]["Distance_to_Library"]["distance_km"], 3),
            "distance_bank": round(amenity_data["nearest"]["Distance_to_Bank"]["distance_km"], 3),
            "distance_pharmacy": round(amenity_data["nearest"]["Distance_to_Pharmacy"]["distance_km"], 3),
            "distance_grocery": round(amenity_data["nearest"]["Distance_to_Grocery"]["distance_km"], 3),
            "distance_subway": round(amenity_data["nearest"]["Distance_to_Subway"]["distance_km"], 3),
            "restaurants": amenity_data["counts"]["Restaurants_Within_1km"],
            "parks": amenity_data["counts"]["Parks_Within_2km"],
            "schools": amenity_data["counts"]["Schools_Within_2km"],
            "prediction_timestamp": timestamp,
            "model_name": MODEL_NAME,
            "property_latitude": latitude,
            "property_longitude": longitude,
            "nearby_points": {
                "schools": AMENITY_INDEXES["schools"].nearest_points(latitude, longitude, limit=5),
                "parks": AMENITY_INDEXES["parks"].nearest_points(latitude, longitude, limit=5),
                "hospitals": AMENITY_INDEXES["hospitals"].nearest_points(latitude, longitude, limit=5),
                "subway": AMENITY_INDEXES["subway"].nearest_points(latitude, longitude, limit=5),
                "libraries": AMENITY_INDEXES["libraries"].nearest_points(latitude, longitude, limit=5),
                "banks": AMENITY_INDEXES["banks"].nearest_points(latitude, longitude, limit=5),
                "pharmacies": AMENITY_INDEXES["pharmacies"].nearest_points(latitude, longitude, limit=5),
                "groceries": AMENITY_INDEXES["groceries"].nearest_points(latitude, longitude, limit=5),
                "restaurants": AMENITY_INDEXES["restaurants"].nearest_points(latitude, longitude, limit=5),
            },
        }

        return jsonify(response), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/health")
def health() -> tuple[Any, int]:
    return jsonify({"status": "ok", "model": MODEL_NAME}), 200


if __name__ == "__main__":
    app.run(debug=True)
