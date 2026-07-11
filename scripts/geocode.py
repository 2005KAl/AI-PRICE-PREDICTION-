import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
import os

# ==========================================================
# FILE PATHS
# ==========================================================

INPUT_FILE = r"C:\Users\Dell\Downloads\e88186124ec611f1\dataset\NEARBY\data\clean_combined_toronto_property_data.xlsx"

OUTPUT_FILE = r"C:\Users\Dell\Downloads\e88186124ec611f1\dataset\NEARBY\output\toronto_properties_geocoded.xlsx"

# ==========================================================
# READ DATASET
# ==========================================================

print("Reading dataset...")

df = pd.read_excel(INPUT_FILE)

print(f"\nTotal Properties Before Filtering : {len(df)}")

# ==========================================================
# FILTER ONLY TORONTO
# ==========================================================

df = df[df["region"].str.contains("Toronto", case=False, na=False)]

print(f"Toronto Properties : {len(df)}")

# Uncomment this for testing
# df = df.head(20)

# ==========================================================
# CREATE LATITUDE & LONGITUDE COLUMNS
# ==========================================================

if "Latitude" not in df.columns:
    df["Latitude"] = None

if "Longitude" not in df.columns:
    df["Longitude"] = None

# ==========================================================
# INITIALIZE GEOCODER
# ==========================================================

geolocator = Nominatim(user_agent="toronto_real_estate_project")

geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1
)

# ==========================================================
# CACHE TO AVOID DUPLICATE REQUESTS
# ==========================================================

address_cache = {}

success = 0
failed = 0

print("\nStarting Geocoding...\n")

# ==========================================================
# GEOCODING LOOP
# ==========================================================

for index, row in tqdm(df.iterrows(), total=len(df)):

    address = str(row["address"]).strip()

    # Skip invalid addresses
    if (
        address == ""
        or address.lower() == "address not available"
        or address.lower() == "nan"
    ):
        failed += 1
        continue

    # Address already contains city/province
    full_address = f"{address}, Canada"

    try:

        if full_address in address_cache:

            location = address_cache[full_address]

        else:

            location = geocode(full_address)
            address_cache[full_address] = location

        if location:

            df.at[index, "Latitude"] = location.latitude
            df.at[index, "Longitude"] = location.longitude

            success += 1

        else:

            print(f"Address Not Found: {full_address}")
            failed += 1

    except Exception as e:

        print(f"Error: {full_address}")
        print(e)

        failed += 1

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ==========================================================
# SAVE OUTPUT
# ==========================================================

df.to_excel(OUTPUT_FILE, index=False)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n======================================")
print("      GEOCODING COMPLETED")
print("======================================")

print(f"Toronto Properties        : {len(df)}")
print(f"Successfully Geocoded     : {success}")
print(f"Failed                    : {failed}")
print(f"Success Rate              : {(success/len(df))*100:.2f}%")

print(f"\nOutput Saved To:\n{OUTPUT_FILE}")