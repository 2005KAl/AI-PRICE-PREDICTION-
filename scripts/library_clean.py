import os
import pandas as pd

# ==========================================
# File Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "tpl-branch-general-information - 4326.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned data",
    "libraries_clean.csv"
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

df = df[
    [
        "BranchName",
        "Address",
        "Lat",
        "Long"
    ]
]

# Rename Columns
df.rename(
    columns={
        "BranchName": "Library_Name",
        "Lat": "Latitude",
        "Long": "Longitude"
    },
    inplace=True
)

# Remove Missing Coordinates
df.dropna(subset=["Latitude", "Longitude"], inplace=True)

# Remove Duplicates
df.drop_duplicates(
    subset=["Library_Name", "Latitude", "Longitude"],
    inplace=True
)

# Reset Index
df.reset_index(drop=True, inplace=True)

# Save Dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaning Completed Successfully!")
print("Final Shape:", df.shape)
print(f"\nSaved To:\n{OUTPUT_FILE}")

print("\nFirst 5 Rows:")
print(df.head())