# Project Summary

## What Was Built

Built a complete AI-powered Toronto house price prediction web application with a Flask backend and a modern real-estate style frontend.

## Backend

- Added [app.py](app.py) as the main Flask application.
- Loads the trained model artifacts from `models/`:
  - `best_model_tuned.pkl`
  - `preprocessor.pkl`
  - `feature_names.pkl`
  - `original_features.pkl`
- Exposes `GET /health` and `POST /predict`.
- Accepts only user-provided inputs:
  - Bedrooms
  - Bathrooms
  - Latitude
  - Longitude
- Automatically computes all engineered features:
  - Nearest school, hospital, park, library, bank, pharmacy, grocery, and subway distances
  - Restaurants within 1 km
  - Schools within 2 km
  - Parks within 2 km
  - Neighbourhood via GeoPandas spatial join
- Uses BallTree with haversine distance for nearest-amenity lookup.
- Returns JSON with predicted price, neighbourhood, amenity distances, counts, timestamp, and nearby map points.

## Frontend

- Added a polished Bootstrap 5 dashboard UI in:
  - [templates/index.html](templates/index.html)
  - [static/css/style.css](static/css/style.css)
  - [static/js/script.js](static/js/script.js)
- Includes:
  - Hero section
  - Interactive Leaflet/OpenStreetMap map
  - Click-to-set property location
  - Property inputs card
  - Prediction result panel
  - Loading spinner
  - Prediction success animation
  - Reset button
  - Dark mode toggle
  - Footer with project info
  - Current date and prediction timestamp display

## Project Files Added

- [requirements.txt](requirements.txt)
- [README.md](README.md)

## Validation

- Verified [app.py](app.py) compiles successfully.
- Verified `GET /health` returns `200`.
- Verified `POST /predict` returns a valid CAD price and neighbourhood.
- Confirmed the frontend runs locally at `http://127.0.0.1:5000/`.

## Notes

- The app is directly runnable after installing dependencies in the provided virtual environment.
- The frontend server is currently running in the workspace.
