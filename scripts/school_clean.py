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
    "School locations-all types data - 4326.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned data",
    "schools_clean.csv"
)

# ==========================================
# Check File Exists
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

required_columns = [
    "NAME",
    "ADDRESS_FULL",
    "BOARD_NAME",
    "SCHOOL_TYPE_DESC",
    "geometry"
]

df = df[required_columns]

# ==========================================
# Extract Latitude & Longitude
# ==========================================

def extract_coordinates(geometry):
    try:
        geom = json.loads(geometry.replace("'", '"'))
        lon, lat = geom["coordinates"][0]
        return pd.Series([lat, lon])
    except:
        return pd.Series([None, None])

df[["Latitude", "Longitude"]] = df["geometry"].apply(extract_coordinates)

# ==========================================
# Remove Geometry
# ==========================================

df.drop(columns=["geometry"], inplace=True)

# ==========================================
# Remove Missing Coordinates
# ==========================================

df.dropna(subset=["Latitude", "Longitude"], inplace=True)

# ==========================================
# Remove Duplicates
# ==========================================

df.drop_duplicates(
    subset=["NAME", "Latitude", "Longitude"],
    inplace=True
)

# ==========================================
# Reset Index
# ==========================================

df.reset_index(drop=True, inplace=True)

# ==========================================
# Save Dataset
# ==========================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaning Completed Successfully!")
print("Final Shape:", df.shape)

print("\nSaved To:")
print(OUTPUT_FILE)

print("\nSchool Types:\n")
print(df["SCHOOL_TYPE_DESC"].value_counts())

print("\nSchool Boards:\n")
print(df["BOARD_NAME"].value_counts())

print("\nFirst 5 Rows:")
print(df.head())