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
    "Parks and Recreation Facilities - 4326.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned data",
    "parks_clean.csv"
)

# ==========================================
# Check File
# ==========================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"\nInput file not found:\n{INPUT_FILE}")

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("Original Shape:", df.shape)

# ==========================================
# Keep Required Columns
# ==========================================

df = df[
    [
        "ASSET_NAME",
        "TYPE",
        "AMENITIES",
        "geometry"
    ]
]

# ==========================================
# Extract Coordinates
# ==========================================

def extract_coordinates(geometry):
    try:
        geom = json.loads(geometry.replace("'", '"'))
        lon, lat = geom["coordinates"][0]
        return pd.Series([lat, lon])
    except:
        return pd.Series([None, None])

df[["Latitude", "Longitude"]] = df["geometry"].apply(extract_coordinates)

# Remove geometry
df.drop(columns=["geometry"], inplace=True)

# Remove missing coordinates
df.dropna(subset=["Latitude", "Longitude"], inplace=True)

# Remove duplicates
df.drop_duplicates(
    subset=["ASSET_NAME", "Latitude", "Longitude"],
    inplace=True
)

# Reset index
df.reset_index(drop=True, inplace=True)

# ==========================================
# Save Dataset
# ==========================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaning Completed Successfully!")
print("Final Shape:", df.shape)
print("\nSaved To:")
print(OUTPUT_FILE)

print("\nFacility Types:\n")
print(df["TYPE"].value_counts())