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
    "restaurant.geojson"      # Rename to restaurants.geojson if you prefer
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned data",
    "restaurants_clean.csv"
)

# ==========================================
# Load GeoJSON
# ==========================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    geojson = json.load(f)

rows = []

# ==========================================
# Extract Restaurant Data
# ==========================================

for feature in geojson["features"]:

    props = feature.get("properties", {})
    geom = feature.get("geometry", {})

    if geom.get("type") != "Point":
        continue

    lon, lat = geom["coordinates"]

    rows.append({
        "Name": props.get("name"),
        "Cuisine": props.get("cuisine"),
        "Latitude": lat,
        "Longitude": lon
    })

# ==========================================
# Create DataFrame
# ==========================================

df = pd.DataFrame(rows)

# Remove restaurants without names
df.dropna(subset=["Name"], inplace=True)

# Remove duplicate restaurants
df.drop_duplicates(
    subset=["Name", "Latitude", "Longitude"],
    inplace=True
)

# Reset Index
df.reset_index(drop=True, inplace=True)

# Save CSV
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaning Completed Successfully!")
print(f"Final Shape: {df.shape}")

print(f"\nSaved To:\n{OUTPUT_FILE}")

print("\nFirst 5 Rows:\n")
print(df.head())