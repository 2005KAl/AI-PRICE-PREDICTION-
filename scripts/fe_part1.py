import os
import pandas as pd
import geopandas as gpd

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DIR = os.path.join(DATA_DIR, "raw")

CLEAN_DIR = os.path.join(DATA_DIR, "cleaned data")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==========================================================
# PROPERTY DATASET
# ==========================================================

PROPERTY_FILE = os.path.join(
    CLEAN_DIR,
    "properties_clean.csv"
)

# ==========================================================
# CLEANED AMENITY DATASETS
# ==========================================================

SCHOOLS_FILE = os.path.join(CLEAN_DIR, "schools_clean.csv")

PARKS_FILE = os.path.join(CLEAN_DIR, "parks_clean.csv")

LIBRARIES_FILE = os.path.join(CLEAN_DIR, "libraries_clean.csv")

HOSPITALS_FILE = os.path.join(CLEAN_DIR, "hospitals_clean.csv")

BANKS_FILE = os.path.join(CLEAN_DIR, "banks_clean.csv")

PHARMACIES_FILE = os.path.join(CLEAN_DIR, "pharmacies_clean.csv")

GROCERY_FILE = os.path.join(CLEAN_DIR, "grocery_clean.csv")

RESTAURANTS_FILE = os.path.join(CLEAN_DIR, "restaurants_clean.csv")

# ==========================================================
# GEOJSON FILES
# ==========================================================

NEIGHBOURHOODS_FILE = os.path.join(
    RAW_DIR,
    "Neighbourhoods - 4326.geojson"
)

SUBWAY_FILE = os.path.join(
    RAW_DIR,
    "subway stations .geojson"
)

# ==========================================================
# VERIFY FILES
# ==========================================================

print("=" * 60)
print("VERIFYING DATASETS")
print("=" * 60)

files = [
    PROPERTY_FILE,
    SCHOOLS_FILE,
    PARKS_FILE,
    LIBRARIES_FILE,
    HOSPITALS_FILE,
    BANKS_FILE,
    PHARMACIES_FILE,
    GROCERY_FILE,
    RESTAURANTS_FILE,
    NEIGHBOURHOODS_FILE,
    SUBWAY_FILE
]

for file in files:

    if os.path.exists(file):
        print(f"✓ {os.path.basename(file)}")

    else:
        raise FileNotFoundError(f"\nMissing File:\n{file}")

# ==========================================================
# LOAD PROPERTY DATASET
# ==========================================================

print("\nLoading Property Dataset...")

properties = pd.read_csv(PROPERTY_FILE)

print(properties.shape)

# ==========================================================
# CONVERT TO GEODATAFRAME
# ==========================================================

properties = gpd.GeoDataFrame(

    properties,

    geometry=gpd.points_from_xy(
        properties["longitude"],
        properties["latitude"]
    ),

    crs="EPSG:4326"

)

# ==========================================================
# LOAD CSV DATASETS
# ==========================================================

def load_dataset(file):

    df = pd.read_csv(file)

    if "Latitude" not in df.columns:
        raise Exception(f"{file} missing Latitude")

    if "Longitude" not in df.columns:
        raise Exception(f"{file} missing Longitude")

    return gpd.GeoDataFrame(

        df,

        geometry=gpd.points_from_xy(

            df["Longitude"],
            df["Latitude"]

        ),

        crs="EPSG:4326"

    )

print("\nLoading Amenity Datasets...")

schools = load_dataset(SCHOOLS_FILE)

parks = load_dataset(PARKS_FILE)

libraries = load_dataset(LIBRARIES_FILE)

hospitals = load_dataset(HOSPITALS_FILE)

banks = load_dataset(BANKS_FILE)

pharmacies = load_dataset(PHARMACIES_FILE)

grocery = load_dataset(GROCERY_FILE)

restaurants = load_dataset(RESTAURANTS_FILE)

# ==========================================================
# LOAD GEOJSON
# ==========================================================

print("\nLoading GeoJSON Files...")

neighbourhoods = gpd.read_file(NEIGHBOURHOODS_FILE)

subway = gpd.read_file(SUBWAY_FILE)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")

print("=" * 60)

print("DATASET SUMMARY")

print("=" * 60)

print(f"Properties      : {len(properties)}")

print(f"Schools         : {len(schools)}")

print(f"Parks           : {len(parks)}")

print(f"Libraries       : {len(libraries)}")

print(f"Hospitals       : {len(hospitals)}")

print(f"Banks           : {len(banks)}")

print(f"Pharmacies      : {len(pharmacies)}")

print(f"Grocery Stores  : {len(grocery)}")

print(f"Restaurants     : {len(restaurants)}")

print(f"Subway Stations : {len(subway)}")

print(f"Neighbourhoods  : {len(neighbourhoods)}")

print("\n")

print("=" * 60)

print("PART 1 COMPLETED SUCCESSFULLY")

print("=" * 60)


# ==========================================================
# PART 2 : DISTANCE FEATURES
# ==========================================================

import numpy as np
from sklearn.neighbors import BallTree

print("\n")
print("=" * 60)
print("PART 2 : DISTANCE FEATURES")
print("=" * 60)

EARTH_RADIUS = 6371000  # meters

# ----------------------------------------------------------
# Function to calculate nearest distance
# ----------------------------------------------------------

def add_nearest_distance(properties, amenity, column_name):

    if amenity.empty:
        properties[column_name] = np.nan
        print(f"{column_name} : Dataset Empty")
        return properties

    property_coords = np.radians(
        properties[["latitude", "longitude"]].values
    )

    amenity_coords = np.radians(
        amenity[["Latitude", "Longitude"]].values
    )

    tree = BallTree(
        amenity_coords,
        metric="haversine"
    )

    distances, _ = tree.query(
        property_coords,
        k=1
    )

    properties[column_name] = (
        distances.flatten() * EARTH_RADIUS
    ).round(2)

    print(f"✓ {column_name}")

    return properties

# ----------------------------------------------------------
# Subway Coordinates
# ----------------------------------------------------------

subway["Latitude"] = subway.geometry.y
subway["Longitude"] = subway.geometry.x

# ----------------------------------------------------------
# Distance Features
# ----------------------------------------------------------

amenity_mapping = {

    "Distance_to_School": schools,

    "Distance_to_Hospital": hospitals,

    "Distance_to_Park": parks,

    "Distance_to_Library": libraries,

    "Distance_to_Bank": banks,

    "Distance_to_Pharmacy": pharmacies,

    "Distance_to_Grocery": grocery,

    "Distance_to_Subway": subway

}

for column, dataset in amenity_mapping.items():

    properties = add_nearest_distance(
        properties,
        dataset,
        column
    )

print("\n")

print("=" * 60)
print("DISTANCE FEATURES COMPLETED")
print("=" * 60)

print(

    properties[

        [

            "Distance_to_School",

            "Distance_to_Hospital",

            "Distance_to_Park",

            "Distance_to_Library",

            "Distance_to_Bank",

            "Distance_to_Pharmacy",

            "Distance_to_Grocery",

            "Distance_to_Subway"

        ]

    ].head()

)

# ==========================================================
# PART 3 : NEARBY COUNT FEATURES
# ==========================================================

print("\n")
print("=" * 60)
print("PART 3 : NEARBY COUNT FEATURES")
print("=" * 60)

from sklearn.neighbors import BallTree

EARTH_RADIUS = 6371000


def add_count_feature(properties, amenity, radius_meters, column_name):

    property_coords = np.radians(
        properties[["latitude", "longitude"]].values
    )

    amenity_coords = np.radians(
        amenity[["Latitude", "Longitude"]].values
    )

    tree = BallTree(
        amenity_coords,
        metric="haversine"
    )

    radius = radius_meters / EARTH_RADIUS

    counts = tree.query_radius(
        property_coords,
        r=radius,
        count_only=True
    )

    properties[column_name] = counts

    print(f"✓ {column_name}")

    return properties


# ----------------------------------------------------------
# Count Features
# ----------------------------------------------------------

properties = add_count_feature(
    properties,
    restaurants,
    1000,
    "Restaurants_Within_1km"
)

properties = add_count_feature(
    properties,
    parks,
    2000,
    "Parks_Within_2km"
)

properties = add_count_feature(
    properties,
    schools,
    2000,
    "Schools_Within_2km"
)

print("\n")

print("=" * 60)
print("COUNT FEATURES COMPLETED")
print("=" * 60)

print(

    properties[

        [

            "Restaurants_Within_1km",

            "Parks_Within_2km",

            "Schools_Within_2km"

        ]

    ].head()

)

# ==========================================================
# PART 3B : NEIGHBOURHOOD ASSIGNMENT
# ==========================================================

print("\n")
print("=" * 60)
print("ASSIGNING NEIGHBOURHOODS")
print("=" * 60)

# Make CRS consistent
properties = properties.to_crs("EPSG:4326")
neighbourhoods = neighbourhoods.to_crs("EPSG:4326")

# Spatial Join
properties = gpd.sjoin(
    properties,
    neighbourhoods,
    how="left",
    predicate="within"
)

# Find neighbourhood column automatically
possible_columns = [
    "AREA_NAME",
    "AREA_NAME_LONG",
    "Neighbourhood",
    "AREA_SHORT_CODE",
    "NAME"
]

neighbourhood_column = None

for col in possible_columns:
    if col in properties.columns:
        neighbourhood_column = col
        break

if neighbourhood_column is None:
    print("\nAvailable Columns:")
    print(properties.columns.tolist())
    raise Exception("Neighbourhood column not found!")

properties["Neighbourhood"] = properties[neighbourhood_column]

print("\nFirst 10 Neighbourhoods:\n")

print(

    properties[
        [
            "neighbourhood",
            "Neighbourhood"
        ]
    ].head(10)

)

# ==========================================================
# PART 4 : FINAL DATASET
# ==========================================================

print("\n")
print("=" * 60)
print("PART 4 : FINAL DATASET")
print("=" * 60)

# ----------------------------------------------------------
# Remove unwanted columns created by spatial join
# ----------------------------------------------------------

columns_to_drop = [

    "geometry",

    "index_right"

]

for col in columns_to_drop:

    if col in properties.columns:

        properties.drop(
            columns=col,
            inplace=True
        )

# ----------------------------------------------------------
# Save Dataset
# ----------------------------------------------------------

OUTPUT_FILE = os.path.join(

    PROCESSED_DIR,

    "ml_dataset.csv"

)

properties.to_csv(

    OUTPUT_FILE,

    index=False

)

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\nDataset Shape")

print(properties.shape)

print("\nColumns")

print(properties.columns.tolist())

print("\nSaved To")

print(OUTPUT_FILE)

print("\n")

print("=" * 60)

print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")

print("=" * 60)