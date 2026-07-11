# ==========================================================
# PROJECT PATHS
# ==========================================================
import os 
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DIR = os.path.join(DATA_DIR, "raw")

CLEAN_DIR = os.path.join(DATA_DIR, "cleaned data")

os.makedirs(CLEAN_DIR, exist_ok=True)

INPUT_FILE = os.path.join(
    DATA_DIR,
    "toronto_real_estate_public_area.csv"
)

OUTPUT_FILE = os.path.join(
    CLEAN_DIR,
    "properties_clean.csv"
)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("LOADING REAL ESTATE DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Original Shape : {df.shape}")

# ==========================================================
# KEEP REQUIRED COLUMNS
# ==========================================================

columns = [
    "price",
    "list_price",
    "sold_price",
    "home_type",
    "home_type_bucket",
    "neighbourhood",
    "latitude",
    "longitude",
    "bedrooms",
    "bedrooms_plus",
    "bathrooms",
    "estimated_area_sqft"
]

df = df[columns]

print("\nColumns Selected:")
print(df.columns.tolist())

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print(f"\nDuplicates Removed : {before-after}")

# ==========================================================
# CONVERT DATA TYPES
# ==========================================================

numeric_cols = [
    "price",
    "list_price",
    "sold_price",
    "latitude",
    "longitude",
    "bedrooms",
    "bedrooms_plus",
    "bathrooms",
    "estimated_area_sqft"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

print("\nMissing Values Before Cleaning")

print(df.isnull().sum())

# Mandatory columns

mandatory = [
    "price",
    "latitude",
    "longitude",
    "bathrooms",
    "estimated_area_sqft",
    "home_type",
    "home_type_bucket",
    "neighbourhood"
]

df = df.dropna(subset=mandatory)

# Fill bedroom values

df["bedrooms"] = df["bedrooms"].fillna(
    df["bedrooms"].median()
)

df["bedrooms_plus"] = df["bedrooms_plus"].fillna(0)

# If sold price missing use list price

df["sold_price"] = df["sold_price"].fillna(
    df["list_price"]
)

# If list price missing use price

df["list_price"] = df["list_price"].fillna(
    df["price"]
)

# ==========================================================
# REMOVE INVALID VALUES
# ==========================================================

df = df[df["price"] > 0]

df = df[df["bathrooms"] > 0]

df = df[df["estimated_area_sqft"] > 0]

df = df[
    (df["latitude"] > 43.0)
    &
    (df["latitude"] < 44.0)
]

df = df[
    (df["longitude"] < -79.0)
    &
    (df["longitude"] > -80.0)
]

# ==========================================================
# CONVERT TO INTEGER
# ==========================================================

df["bedrooms"] = df["bedrooms"].astype(int)

df["bedrooms_plus"] = df["bedrooms_plus"].astype(int)

df["bathrooms"] = df["bathrooms"].astype(int)

# ==========================================================
# RESET INDEX
# ==========================================================

df = df.reset_index(drop=True)

# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print(f"Final Shape : {df.shape}")

print("\nMissing Values After Cleaning")

print(df.isnull().sum())

print("\nSaved File")

print(OUTPUT_FILE)

print("\nFirst 5 Rows")

print(df.head())