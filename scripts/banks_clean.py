import os
import json
import pandas as pd

# ==========================================
# File Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "banks.geojson"      # Rename if necessary
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned data",
    "banks_clean.csv"
)

# ==========================================
# Load GeoJSON
# ==========================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    geojson = json.load(f)

rows = []

# ==========================================
# Extract Data
# ==========================================

for feature in geojson["features"]:

    props = feature.get("properties", {})
    geom = feature.get("geometry", {})

    if geom.get("type") != "Point":
        continue

    lon, lat = geom["coordinates"]

    rows.append({
        "Name": props.get("name"),
        "Latitude": lat,
        "Longitude": lon
    })

# ==========================================
# Create DataFrame
# ==========================================

df = pd.DataFrame(rows)

df.dropna(subset=["Name"], inplace=True)

df.drop_duplicates(
    subset=["Name", "Latitude", "Longitude"],
    inplace=True
)

df.reset_index(drop=True, inplace=True)

df.to_csv(OUTPUT_FILE, index=False)

print("Banks Cleaning Completed Successfully!")
print(df.shape)
print(df.head())