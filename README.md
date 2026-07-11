# Toronto House Price Prediction using AI

A Flask-based web application that predicts Toronto house prices using an XGBoost regression model, neighborhood intelligence, and nearby amenity features.

The app combines a responsive Bootstrap frontend, a Leaflet map, and a Python backend that performs feature engineering at prediction time.

## What this project does

Users enter a few property details, choose a home type, and set a location either manually or by clicking the map. The backend then:

- validates the request
- finds the nearest amenities around the property
- detects the neighborhood
- builds the model feature vector
- runs the XGBoost price prediction
- returns a detailed JSON response used by the frontend

## Main features

- Flask backend with `GET /health` and `POST /predict`
- XGBoost regression model for price estimation
- Glassmorphism Bootstrap UI
- Leaflet map with OpenStreetMap tiles
- Click-to-fill latitude and longitude
- Dark mode toggle
- Loading overlay and prediction feedback
- Nearby amenity cards and neighborhood display
- Responsive layout for desktop and mobile

## Current input fields

The prediction form uses these values:

- Bedrooms
- Bathrooms
- Estimated Area (sqft)
- Home Type
- Latitude
- Longitude

Latitude and longitude are still required because the app uses them for the map, neighborhood lookup, and amenity distance calculations.

## Prediction output

The frontend displays:

- Estimated Price
- Price / Sq.Ft
- Prediction Accuracy
- Home Type
- Neighborhood
- Nearest amenity distances
- Amenity counts
- Prediction timestamp
- Nearby points used by the map

## How the backend works

The main application logic lives in [app.py](app.py).

At prediction time the app:

1. reads the request payload
2. validates bedrooms, bathrooms, estimated area, home type, latitude, and longitude
3. computes nearest-school, hospital, park, library, bank, pharmacy, grocery, subway, and restaurant features
4. resolves the Toronto neighborhood polygon
5. builds the model input frame in the expected feature order
6. applies the saved preprocessor
7. runs the XGBoost model
8. returns the prediction as JSON

## Project structure

```text
IMPROVED ACCURACY/
├── app.py
├── README.md
├── PROJECT_SUMMARY.md
├── requirements.txt
├── data/
│   ├── toronto_real_estate_public_area.csv
│   ├── cleaned data/
│   │   ├── banks_clean.csv
│   │   ├── grocery_clean.csv
│   │   ├── hospitals_clean.csv
│   │   ├── libraries_clean.csv
│   │   ├── parks_clean.csv
│   │   ├── pharmacies_clean.csv
│   │   ├── properties_clean.csv
│   │   ├── restaurants_clean.csv
│   │   └── schools_clean.csv
│   ├── processed/
│   │   ├── ml_dataset.csv
│   │   ├── X_test.csv
│   │   ├── X_train.csv
│   │   ├── y_test.csv
│   │   └── y_train.csv
│   └── raw/
│       ├── banks.geojson
│       ├── grocery_stores.geojson
│       ├── hospitals.geojson
│       ├── Neighbourhoods - 4326.geojson
│       ├── Parks and Recreation Facilities - 4326.csv
│       ├── pharmacies.geojson
│       ├── restaurant.geojson
│       ├── School locations-all types data - 4326.csv
│       ├── subway stations .geojson
│       └── tpl-branch-general-information - 4326.csv
├── models/
├── outputs/
├── output/
├── scripts/
├── static/
│   ├── css/style.css
│   └── js/script.js
└── templates/
    └── index.html
```

## Required model files

The app loads saved artifacts from `models/`.

At minimum, the following files must exist:

- `preprocessor.pkl`
- `best_model.pkl` or `best_model_tuned.pkl`
- `original_features.pkl`

Depending on the training pipeline, other supporting files may also be present, such as `feature_names.pkl`.

## Required data files

The backend expects the cleaned amenity datasets and Toronto neighborhood geometry to be available in `data/`.

Most importantly, these files are used for nearest-amenity lookups and neighborhood detection:

- `data/cleaned data/schools_clean.csv`
- `data/cleaned data/parks_clean.csv`
- `data/cleaned data/libraries_clean.csv`
- `data/cleaned data/hospitals_clean.csv`
- `data/cleaned data/banks_clean.csv`
- `data/cleaned data/pharmacies_clean.csv`
- `data/cleaned data/grocery_clean.csv`
- `data/cleaned data/restaurants_clean.csv`
- `data/raw/Neighbourhoods - 4326.geojson`
- `data/raw/subway stations .geojson`

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

Run the Flask application from the project root:

```bash
python app.py
```

The app will usually be available at:

```text
http://127.0.0.1:5000/
```

## API

### `GET /health`

Returns a simple status response for checking whether the backend is alive.

### `POST /predict`

Example request body:

```json
{
  "Bedrooms": 3,
  "Bathrooms": 2,
  "EstimatedArea": 1500,
  "HomeType": "Detached",
  "Latitude": 43.6532,
  "Longitude": -79.3832
}
```

Example response fields include:

- `predicted_price`
- `predicted_price_formatted`
- `price_per_sqft`
- `prediction_accuracy`
- `home_type`
- `neighbourhood`
- distance values for the nearest amenities
- nearby point lists for map use

## Frontend notes for collaborators

The current UI is intentionally kept as a glassmorphism-style dashboard. Do not redesign the layout unless the task specifically asks for it.

The existing frontend includes:

- navbar
- property details form
- prediction result section
- amenity cards
- Leaflet map
- legend
- footer
- loading overlay
- dark mode toggle

## Working with the scripts folder

The `scripts/` directory contains the data preparation and modeling pipeline used to build the cleaned datasets and model artifacts. These scripts are useful if you need to retrain the model or reproduce the processed files.

## Troubleshooting

- If the app fails to start because of a missing package, reinstall the dependencies with `pip install -r requirements.txt`.
- If the app cannot load a model artifact, check the contents of `models/` and make sure `best_model.pkl` or `best_model_tuned.pkl` exists.
- If neighborhood or amenity lookups fail, confirm the files in `data/cleaned data/` and `data/raw/` are present.

## Summary

This repository contains a complete end-to-end Toronto house price prediction app: data files, preprocessing artifacts, trained model, Flask backend, and the browser UI used to explore predictions.
