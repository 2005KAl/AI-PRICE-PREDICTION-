import pandas as pd
import re
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

# ======================================================
# FILE PATHS
# ======================================================

INPUT_FILE = r"C:\Users\Dell\Downloads\e88186124ec611f1\dataset\NEARBY\output\failed_addresses.xlsx"

OUTPUT_FILE = r"C:\Users\Dell\Downloads\e88186124ec611f1\dataset\NEARBY\output\failed_addresses_fixed.xlsx"

# ======================================================
# READ DATASET
# ======================================================

print("Reading failed addresses...")

df = pd.read_excel(INPUT_FILE)

print(f"Failed Addresses : {len(df)}")

# ======================================================
# GEOCODER
# ======================================================

geolocator = Nominatim(user_agent="toronto_retry_geocoder")

geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1
)

# ======================================================
# CREATE LATITUDE/LONGITUDE IF NOT PRESENT
# ======================================================

if "Latitude" not in df.columns:
    df["Latitude"] = None

if "Longitude" not in df.columns:
    df["Longitude"] = None

# ======================================================
# CLEAN ADDRESS FUNCTION
# ======================================================

def clean_address(address):

    address = str(address)

    # Remove Unit# 708
    address = re.sub(r'Unit#\s*\w+', '', address, flags=re.IGNORECASE)

    # Remove "#3112 -"
    address = re.sub(r'^#\d+\s*-\s*', '', address)

    # Remove apartment/unit numbers like "1803 151 Dan Leckie Way"
    address = re.sub(r'^\d{3,5}\s+(?=\d)', '', address)

    # Remove extra spaces
    address = " ".join(address.split())

    return address

# ======================================================
# RETRY GEOCODING
# ======================================================

success = 0
failed = 0

print("\nRetrying Failed Addresses...\n")

for index, row in tqdm(df.iterrows(), total=len(df)):

    address = clean_address(row["address"])

    full_address = f"{address}, Canada"

    try:

        location = geocode(full_address)

        if location:

            df.at[index, "Latitude"] = location.latitude
            df.at[index, "Longitude"] = location.longitude

            success += 1

            print(f"SUCCESS : {full_address}")

        else:

            failed += 1

            print(f"FAILED  : {full_address}")

    except Exception as e:

        failed += 1

        print(f"ERROR : {full_address}")
        print(e)

# ======================================================
# SAVE
# ======================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df.to_excel(OUTPUT_FILE, index=False)

# ======================================================
# SUMMARY
# ======================================================

print("\n===================================")
print("RETRY COMPLETED")
print("===================================")

print(f"Total Failed Addresses : {len(df)}")
print(f"Recovered              : {success}")
print(f"Still Failed           : {failed}")
print(f"Recovery Rate          : {(success/len(df))*100:.2f}%")

print(f"\nSaved to:\n{OUTPUT_FILE}")