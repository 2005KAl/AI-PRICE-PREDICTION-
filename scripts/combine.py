import pandas as pd

# File paths
file1 = r"C:\Users\Dell\Downloads\e88186124ec611f1\dataset\NEARBY\output\failed_addresses_fixed.xlsx"   # retry/recovered file
file2 = r"C:\Users\Dell\Downloads\e88186124ec611f1\dataset\NEARBY\output\toronto_properties_geocoded.xlsx"   # original file


# Load datasets
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)


# Combine both datasets
df = pd.concat([df1, df2], ignore_index=True)


print("Before Cleaning:")
print(df.shape)


# ---------------------------
# DATA CLEANING
# ---------------------------

# Remove completely empty rows
df.dropna(how="all", inplace=True)


# Remove duplicate properties
# Using address because it identifies the property
df.drop_duplicates(
    subset=["address"],
    keep="first",
    inplace=True
)


# Clean address text
df["address"] = (
    df["address"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# Clean region column
df["region"] = (
    df["region"]
    .astype(str)
    .str.strip()
)


# Convert numerical columns
numeric_columns = [
    "price",
    "bedrooms",
    "bathrooms",
    "pricem",
    "Latitude",
    "Longitude"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# Remove rows without coordinates
# (important for mapping)
df = df.dropna(
    subset=["Latitude", "Longitude"]
)


# Remove invalid coordinates
df = df[
    (df["Latitude"].between(-90,90)) &
    (df["Longitude"].between(-180,180))
]


# Handle missing bedrooms/bathrooms
df["bedrooms"] = df["bedrooms"].fillna(
    df["bedrooms"].median()
)

df["bathrooms"] = df["bathrooms"].fillna(
    df["bathrooms"].median()
)


# Remove duplicate coordinates
# Prevent same property appearing twice
df.drop_duplicates(
    subset=["Latitude","Longitude"],
    keep="first",
    inplace=True
)


# Reset index
df.reset_index(drop=True, inplace=True)


# Save final cleaned dataset
df.to_excel(
    "final_cleaned_toronto_properties.xlsx",
    index=False
)


print("\nAfter Cleaning:")
print(df.shape)

print("\nFinal file created successfully!")