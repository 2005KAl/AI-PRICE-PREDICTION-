import pandas as pd

# Read the geocoded dataset
FILE_PATH = r"C:\Users\Dell\Downloads\e88186124ec611f1\dataset\NEARBY\output\toronto_properties_geocoded.xlsx"

df = pd.read_excel(FILE_PATH)

# Find failed geocoding rows
failed_df = df[df["Latitude"].isna()]

print("=" * 50)
print("FAILED ADDRESSES")
print("=" * 50)

print(f"Total Failed Rows : {len(failed_df)}")
print(f"Unique Failed Addresses : {failed_df['address'].nunique()}")

print("\nFirst 20 Failed Addresses:\n")
print(failed_df[["address", "region"]].head(20))

# Save failed addresses
OUTPUT = r"C:\Users\Dell\Downloads\e88186124ec611f1\dataset\NEARBY\output\failed_addresses.xlsx"

failed_df.to_excel(OUTPUT, index=False)

print(f"\nFailed addresses saved to:\n{OUTPUT}")